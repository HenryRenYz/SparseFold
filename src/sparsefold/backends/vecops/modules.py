# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""SparseFold module replacements backed by vecops JIT kernels."""

import contextlib
import fnmatch
import functools
import math
import os
import platform

import torch
import einops
from sparsefold.backends.vecops.distributed import evoformer_single_act_update_distributed
from sparsefold.backends.vecops.distributed import grid_self_attention_distributed
from sparsefold.backends.vecops.distributed import msa_attention_distributed
from sparsefold.backends.vecops.distributed import outer_product_mean_distributed
from sparsefold.backends.vecops.distributed import triangle_multiplication_distributed
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import trace
from sparsefold.runtime import cache
from sparsefold.runtime import util

import sparsefold.model.trunk.blocks as _modules
import sparsefold.model.diffusion.transformer as _diffusion_transformer


_OriginalLinear = fnn.Linear


@functools.cache
def _kernels():
  """Load the native JIT facade only when a native path is first used."""
  from sparsefold.backends.vecops import kernels

  return kernels


def _is_precompiling() -> bool:
  import vecops

  return vecops.is_precompiling()


def _native_pool_region_enabled(*qualified_ops: str) -> bool:
  """Whether all operators in a bounded region use the private pool."""
  # Diffusion uses dynamic denoising batches.  A scoped native team changes
  # the prepared-kernel parallelism specialization, and multi-rank runs can
  # then reuse a smaller planning capacity for a later dynamic view.  Keep the
  # per-operator native pool routing, but do not enter the fused region until
  # vecops can re-plan that capacity without terminating the process.
  if any(name.startswith("sparsefold_native::diffusion_") for name in qualified_ops):
    return False
  enabled = (
      os.environ.get("VECOPS_THREAD_POOL", "torch").strip().lower() == "native"
      and os.environ.get("VECOPS_NATIVE_ACTIVE_REGION", "0") != "0"
  )
  configured = os.environ.get("VECOPS_NATIVE_THREAD_POOL_OPS")
  if enabled and configured:
    patterns = [item.strip() for item in configured.split(",") if item.strip()]
    enabled = all(
        any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        for name in qualified_ops
    )
  return enabled


def _native_pool_region(*qualified_ops: str):
  """Keep one private vecops team active only across all-native model regions."""
  if not _native_pool_region_enabled(*qualified_ops):
    return contextlib.nullcontext()

  import vecops

  return vecops.native_thread_pool_region(
      torch.get_num_threads(),
  )


def _begin_native_preparation(module):
  module._native_prepared = False
  devices = {parameter.device.type for parameter in module.parameters()}
  if devices - {"cpu", "meta"}:
    raise RuntimeError("vecops fused kernels support CPU only; use the Torch backend for CUDA")


def _require_native_preparation(function):
  @functools.wraps(function)
  def wrapped(self, *args, **kwargs):
    if not getattr(self, "_native_prepared", False):
      raise RuntimeError("Native buffers are not prepared; call backends.prepare(model) after strict checkpoint loading")
    return function(self, *args, **kwargs)
  return wrapped


class LinearVecops(_OriginalLinear):
  """AF3 Linear backed by a persistent vecops-packed weight."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.register_buffer("_packed_weight", None, persistent=False)
    self.register_buffer("_native_bias", None, persistent=False)

  @property
  def fp32_matmul_precision(self) -> str:
    return self.precision or util.DEFAULT_FP32_MATMUL_PRECISION

  def prepare_native(self) -> bool:
    """Pack the loaded weight when this Linear is supported natively."""
    if self.num_input_dims == 0 or self.slice_order:
      return False
    if self.weight.device.type not in {"cpu", "meta"}:
      return False
    if self.weight.dtype not in {
        torch.bfloat16, torch.float16, torch.float32
    }:
      return False
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "x64"}:
      if (
          self.fp32_matmul_precision != "medium"
          or self.weight.dtype == torch.float16
      ):
        return False
    if machine not in {
        "x86_64", "amd64", "x64", "aarch64", "arm64"
    }:
      return False
    in_features = math.prod(self.input_shape)
    out_features = math.prod(self.output_shape)
    if self.transpose_weights:
      logical_weight = self.weight.reshape(out_features, in_features)
    else:
      logical_weight = self.weight.reshape(in_features, out_features).T
    packed = _kernels().pack_linear_b(
        logical_weight.contiguous(),
        fp32_matmul_precision=self.fp32_matmul_precision,
    )
    _set_native_buffer(self, "_packed_weight", packed)
    # Cached for the per-call fast path below: attribute reads are far
    # cheaper than math.prod / property resolution on every forward.
    self._native_out_features = out_features
    self._native_in_features = in_features
    self._native_precision = self.fp32_matmul_precision
    native_bias = (
        self.bias.reshape(-1).to(torch.float32)
        if self.use_bias
        else torch.zeros(
            (out_features,), dtype=torch.float32, device=self.weight.device
        )
    )
    _set_native_buffer(self, "_native_bias", native_bias)
    return True

  def forward(
      self,
      inputs: torch.Tensor,
      *,
      post_op=None,
      residual: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if self._packed_weight is None:
      return super().forward(inputs, post_op=post_op, residual=residual)

    input_dtype = inputs.dtype
    out_features = self._native_out_features
    packed_dtype = self._packed_weight.dtype
    compute_input = (
        inputs if input_dtype is packed_dtype else inputs.to(packed_dtype)
    )
    input_leading_shape = compute_input.shape[: -self.num_input_dims]
    flat_input = compute_input.reshape(
        input_leading_shape + (self._native_in_features,)
    )
    output_shape = input_leading_shape + self.output_shape
    if post_op is None:
      normalized_post_op = fnn.LinearPostOp.NONE
    else:
      normalized_post_op = fnn.normalize_linear_post_op(post_op)
    fuse_residual = residual is None or (
        residual.shape == output_shape and residual.dtype == input_dtype
    )
    kernel_post_op = (
        normalized_post_op if fuse_residual else fnn.LinearPostOp.NONE
    )
    output = _kernels().linear_packed(
        flat_input,
        self._packed_weight,
        out_features=out_features,
        bias=self._native_bias,
        residual=(
            residual.reshape(input_leading_shape + (-1,))
            if residual is not None and fuse_residual else None
        ),
        post_op=int(kernel_post_op),
        fp32_matmul_precision=self._native_precision,
        out_dtype=input_dtype if fuse_residual else self.weight.dtype,
    )
    output = output.reshape(output_shape)
    if residual is not None and not fuse_residual:
      # Preserve Torch broadcasting without materializing an expanded CInput.
      # The projection remains native; only the broadcast epilogue stays ATen.
      output = output + residual.to(output.dtype)
      output = fnn.apply_linear_post_op(output, normalized_post_op)
    return output.to(input_dtype)


class TransitionBlockVecops(_modules.TransitionBlock):
  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    _set_native_buffer(self.input_layer_norm, '_native_weight', _source_tensor(self.input_layer_norm, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(self.input_layer_norm, '_native_bias', _source_tensor(self.input_layer_norm, 'bias').to(torch.float32).contiguous())
    _set_native_buffer(self.transition1, '_native_weight', _source_tensor(self.transition1, 'weight').to(torch.bfloat16).contiguous())
    _set_native_buffer(self.transition2, '_native_weight', _source_tensor(self.transition2, 'weight').to(torch.bfloat16).contiguous())
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(
      self,
      act: torch.Tensor,  # (*B, N, C)
      *,
      broadcast_dim: int = 0,
  ):
    if act.stride(-1) != 1:
      trace.log_debug(f"Weird act is not last-contiguous: {act.stride()}")
      act = act.contiguous()

    if act.dtype != torch.bfloat16:
      trace.log_debug(f"act is not bfloat16: {act.dtype}")
      act = act.to(torch.bfloat16)

    with _native_pool_region("sparsefold_native::transition_block"):
      return _kernels().transition_block(
          act,
          _native_tensor(self.input_layer_norm, 'weight'),
          _native_tensor(self.input_layer_norm, 'bias'),
          _native_tensor(self.transition1, 'weight'),
          _native_tensor(self.transition2, 'weight'),
      )


class MSAAttentionVecops(_modules.MSAAttention):
  """Native non-distributed MSA attention with persistent packed weights."""

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    _set_native_buffer(self.act_norm, '_native_weight', _source_tensor(self.act_norm, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(self.act_norm, '_native_bias', _source_tensor(self.act_norm, 'bias').to(torch.float32).contiguous())
    _set_native_buffer(self.pair_norm, '_native_weight', _source_tensor(self.pair_norm, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(self.pair_norm, '_native_bias', _source_tensor(self.pair_norm, 'bias').to(torch.float32).contiguous())
    _set_native_buffer(self.pair_logits, '_native_weight', _bf16_tensor(_source_tensor(self.pair_logits, 'weight')))
    _set_native_buffer(self.v_projection, '_native_weight', _bf16_tensor(_source_tensor(self.v_projection, 'weight')))
    _set_native_buffer(self.gating_query, '_native_weight', _bf16_tensor(_source_tensor(self.gating_query, 'weight')))
    _set_native_buffer(self.output_projection, '_native_weight', _bf16_tensor(_source_tensor(self.output_projection, 'weight')))

    heads = _native_tensor(self.pair_logits, 'weight').shape[-1]
    head_dim = _native_tensor(self.v_projection, 'weight').shape[-1]
    channels = self.num_channels
    _set_native_buffer(
        self,
        "_packed_msa_pair_logits",
        _kernels().pack_matmul_b(
            _native_tensor(self.pair_logits, 'weight').transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_msa_value",
        _kernels().pack_matmul_b(
            _native_tensor(self.v_projection, 'weight').reshape(channels, heads * head_dim)
            .transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_msa_gate",
        _kernels().pack_matmul_b(
            _native_tensor(self.gating_query, 'weight').reshape(channels, heads * head_dim)
            .transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_msa_output",
        _kernels().pack_matmul_b(
            _native_tensor(self.output_projection, 'weight').transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(
      self,
      act: torch.Tensor,
      mask: torch.Tensor,
      pair_act: torch.Tensor,
  ):
    original_dtype = act.dtype
    act = act.contiguous().to(torch.bfloat16)
    mask = mask.contiguous().bool()
    pair_act = pair_act.contiguous().to(torch.bfloat16)
    heads = _native_tensor(self.pair_logits, 'weight').shape[-1]
    head_dim = _native_tensor(self.v_projection, 'weight').shape[-1]
    fn = (
        msa_attention_distributed
        if self.distributed
        else _kernels().msa_attention_packed
    )
    region = (
        contextlib.nullcontext()
        if self.distributed
        else _native_pool_region("sparsefold_native::msa_attention")
    )
    with region:
      output = fn(
          act, mask, pair_act,
          _native_tensor(self.act_norm, 'weight'), _native_tensor(self.act_norm, 'bias'),
          _native_tensor(self.pair_norm, 'weight'), _native_tensor(self.pair_norm, 'bias'),
          self._packed_msa_pair_logits,
          self._packed_msa_value,
          self._packed_msa_gate,
          self._packed_msa_output,
          heads=heads,
          head_dim=head_dim,
      )
    return output.to(original_dtype)


class OuterProductMeanVecops(_modules.OuterProductMean):
  """Native non-distributed outer-product mean with packed projections."""

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    _set_native_buffer(self.layer_norm_input, '_native_weight', _source_tensor(self.layer_norm_input, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(self.layer_norm_input, '_native_bias', _source_tensor(self.layer_norm_input, 'bias').to(torch.float32).contiguous())
    _set_native_buffer(self.left_projection, '_native_weight', _bf16_tensor(_source_tensor(self.left_projection, 'weight')))
    _set_native_buffer(self.right_projection, '_native_weight', _bf16_tensor(_source_tensor(self.right_projection, 'weight')))
    _set_native_buffer(self, '_native_output_w', _source_tensor(self, 'output_w').to(torch.bfloat16).contiguous())
    _set_native_buffer(self, '_native_output_b', _source_tensor(self, 'output_b').to(torch.bfloat16).contiguous())

    outer_channels = _native_tensor(self.left_projection, 'weight').shape[-1]
    output_channels = self._native_output_b.shape[-1]
    _set_native_buffer(
        self,
        "_packed_outer_left",
        _kernels().pack_matmul_b(
            _native_tensor(self.left_projection, 'weight').transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_outer_right",
        _kernels().pack_matmul_b(
            _native_tensor(self.right_projection, 'weight').transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_outer_output",
        _kernels().pack_matmul_b(
            self._native_output_w.permute(2, 1, 0).reshape(
                output_channels, outer_channels * outer_channels
            ).contiguous()
        ).unsqueeze(0),
    )
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(self, act: torch.Tensor, mask: torch.Tensor):
    original_dtype = act.dtype
    act = act.contiguous().to(torch.bfloat16)
    mask = mask.contiguous().bool()
    fn = (
        outer_product_mean_distributed
        if self.distributed
        else _kernels().outer_product_mean_packed
    )
    region = (
        contextlib.nullcontext()
        if self.distributed
        else _native_pool_region("sparsefold_native::outer_product_mean")
    )
    with region:
      output = fn(
          act, mask,
          _native_tensor(self.layer_norm_input, 'weight'), _native_tensor(self.layer_norm_input, 'bias'),
          self._packed_outer_left, self._packed_outer_right,
          self._packed_outer_output, self._native_output_b,
          outer_channels=_native_tensor(self.left_projection, 'weight').shape[-1],
      )
    return output.to(original_dtype)


def _bf16_tensor(value: torch.Tensor) -> torch.Tensor:
  return value.to(torch.bfloat16).contiguous()


def _source_tensor(module: torch.nn.Module, name: str):
  """Reference checkpoint parameter, never a derived native buffer."""
  return getattr(module, name)


def _native_tensor(module: torch.nn.Module, name: str):
  """Prepared dtype adaptation or the unchanged reference parameter."""
  prepared = module._buffers.get("_native_" + name)
  return prepared if prepared is not None else getattr(module, name)


def _prepared_visitor(scopes):
  # Values come from registered reference Parameters, after the strict load.
  from sparsefold.weights.legacy_scopes import ParamVisitor
  return ParamVisitor(scopes)


def _norm_scope(module):
  return {"scale": module.weight,
          "offset": module.bias if module.bias is not None else torch.zeros_like(module.weight)}


def _set_native_buffer(
    module: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
  """Install or refresh an architecture-specific, non-serialized buffer."""
  value = value.detach().contiguous()
  if name in module._buffers:
    module._buffers[name] = value
  else:
    module.register_buffer(name, value, persistent=False)


def _prepare_adaptive_layer_norm(module) -> None:
  if module.single_cond:
    _set_native_buffer(module.single_cond_layer_norm, '_native_weight', _source_tensor(module.single_cond_layer_norm, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(module.single_cond_scale, '_native_weight', _bf16_tensor(_source_tensor(module.single_cond_scale, 'weight')))
    _set_native_buffer(module.single_cond_scale, '_native_bias', _bf16_tensor(_source_tensor(module.single_cond_scale, 'bias')))
    _set_native_buffer(module.single_cond_bias, '_native_weight', _bf16_tensor(_source_tensor(module.single_cond_bias, 'weight')))
  else:
    _set_native_buffer(module.layer_norm, '_native_weight', _source_tensor(module.layer_norm, 'weight').to(torch.float32).contiguous())
    _set_native_buffer(module.layer_norm, '_native_bias', _source_tensor(module.layer_norm, 'bias').to(torch.float32).contiguous())


def _prepare_adaptive_zero(module) -> None:
  _set_native_buffer(module.transition2, '_native_weight', _bf16_tensor(_source_tensor(module.transition2, 'weight')))
  if module.single_cond:
    _set_native_buffer(module.adaptive_zero_cond, '_native_weight', _bf16_tensor(_source_tensor(module.adaptive_zero_cond, 'weight')))
    _set_native_buffer(module.adaptive_zero_cond, '_native_bias', _bf16_tensor(_source_tensor(module.adaptive_zero_cond, 'bias')))


def _pack_head_projection(
    owner: torch.nn.Module,
    buffer_name: str,
    module: torch.nn.Module,
    heads: int,
    head_dim: int,
) -> None:
  weight = _native_tensor(module, 'weight')
  channels = weight.shape[0]
  assert weight.shape == (channels, heads, head_dim)
  _set_native_buffer(
      owner,
      buffer_name,
      _kernels().pack_matmul_b(weight.permute(1, 2, 0)),
  )
  if _native_tensor(module, 'bias') is not None:
    _set_native_buffer(module, '_native_bias', _bf16_tensor(_source_tensor(module, 'bias')))


_DIFFUSION_TRANSITION_PANEL = 128


def _use_native_diffusion_transition(sequence: int) -> bool:
  """Use the fused transition, including its BF16 rounding boundaries."""
  return True


def _transition_panel_size(size: int) -> int:
  """Choose the widest panel up to 128 that exactly divides ``size``."""
  for panel in range(min(_DIFFUSION_TRANSITION_PANEL, size), 0, -1):
    if size % panel == 0:
      return panel
  raise AssertionError("positive dimensions always have a divisor")


def _pack_diffusion_transition_weights(
    transition1: torch.Tensor,
    transition2: torch.Tensor,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
  """Pack FFN weights as independent output-channel panels.

  Panel packing lets the native kernel schedule GEMMs over both token tiles
  and output panels. Padding is private to the packed representation; the
  readable Torch parameters retain their original shapes.
  """
  channels = transition1.shape[0]
  hidden_panel = _transition_panel_size(hidden)
  output_panel = _transition_panel_size(channels)
  up = transition1.reshape(channels, 2, hidden).permute(1, 2, 0)
  packed_up = _kernels().pack_matmul_b(
      up.reshape(2, hidden // hidden_panel, hidden_panel, channels)
  )

  down = transition2.transpose(0, 1)
  packed_down = _kernels().pack_matmul_b(
      down.reshape(channels // output_panel, output_panel, hidden)
  )
  return packed_up, packed_down, hidden_panel, output_panel


class DiffusionTransitionBlockVecops(_diffusion_transformer.TransitionBlock):
  """Native diffusion transition, distinct from the Evoformer block."""

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    self._prepare_native_parameters()
    self._native_prepared = True
    return True

  def _prepare_native_parameters(self):
    _prepare_adaptive_layer_norm(self.layer_norm)
    _set_native_buffer(self.transition, '_native_weight', _bf16_tensor(_source_tensor(self.transition, 'weight')))
    _prepare_adaptive_zero(self.zero_init)
    self._native_weights_packed = False
    if self.layer_norm.single_cond:
      _set_native_buffer(
          self,
          "_packed_cond_scale",
          _kernels().pack_matmul_b(
              _native_tensor(self.layer_norm.single_cond_scale, 'weight').transpose(0, 1)
          ).unsqueeze(0),
      )
      _set_native_buffer(
          self,
          "_packed_cond_bias",
          _kernels().pack_matmul_b(
              _native_tensor(self.layer_norm.single_cond_bias, 'weight').transpose(0, 1)
          ).unsqueeze(0),
      )
      (
          packed_transition,
          packed_transition2,
          self._native_hidden_panel,
          self._native_output_panel,
      ) = (
          _pack_diffusion_transition_weights(
              _native_tensor(self.transition, 'weight'),
              _native_tensor(self.zero_init.transition2, 'weight'),
              self.num_intermediates,
          )
      )
      _set_native_buffer(
          self, "_packed_transition", packed_transition
      )
      _set_native_buffer(
          self,
          "_packed_transition2",
          packed_transition2,
      )
      _set_native_buffer(
          self,
          "_packed_zero_cond",
          _kernels().pack_matmul_b(
              _native_tensor(self.zero_init.adaptive_zero_cond, 'weight').transpose(0, 1)
          ).unsqueeze(0),
      )
      self._native_weights_packed = True

  def _forward_native(
      self,
      x: torch.Tensor,
      *,
      single_cond: torch.Tensor | None,
  ):
    if not _use_native_diffusion_transition(x.shape[-2]):
      # The denoising loop suspends the model-wide TorchDispatchMode because
      # most of its matmuls are explicit vecops calls. Reinstall the router
      # around this Torch fallback so ARM's slow default GEMM is never used.
      from sparsefold.backends.vecops.matmul_dispatch import execution_context

      with execution_context():
        return super().forward(x, single_cond=single_cond)
    orig_dtype = x.dtype
    original = x
    x = x.contiguous().to(torch.bfloat16)
    if single_cond is None:
      if self.layer_norm.single_cond:
        raise ValueError("single_cond is required by this DiffusionTransitionBlock")
      output = _kernels().transition_block(
          x,
          _native_tensor(self.layer_norm.layer_norm, 'weight'),
          _native_tensor(self.layer_norm.layer_norm, 'bias'),
          _native_tensor(self.transition, 'weight'),
          _native_tensor(self.zero_init.transition2, 'weight'),
      )
      # The generic transition kernel already adds its input residual.
      output = output.to(orig_dtype)
    else:
      if not self.layer_norm.single_cond:
        raise ValueError("single_cond was supplied to an unconditional block")
      single_cond = single_cond.contiguous().to(torch.bfloat16)
      if not self._native_weights_packed:
        raise RuntimeError("native diffusion transition weights are not packed")
      output = _kernels().diffusion_transition_block_packed(
          x,
          single_cond,
          _native_tensor(self.layer_norm.single_cond_layer_norm, 'weight'),
          self._packed_cond_scale,
          _native_tensor(self.layer_norm.single_cond_scale, 'bias'),
          self._packed_cond_bias,
          self._packed_transition,
          self._packed_transition2,
          self._packed_zero_cond,
          _native_tensor(self.zero_init.adaptive_zero_cond, 'bias'),
          hidden=self.num_intermediates,
          hidden_panel=self._native_hidden_panel,
          output_panel=self._native_output_panel,
      )
    return output.to(orig_dtype)

  @_require_native_preparation
  def forward(self, x: torch.Tensor, *, single_cond: torch.Tensor | None = None):
    return self._forward_native(x, single_cond=single_cond)

  @_require_native_preparation
  def forward_prepared(
      self,
      x: torch.Tensor,
      adaptive_scale: torch.Tensor,
      adaptive_bias: torch.Tensor,
      zero_gate: torch.Tensor,
  ) -> torch.Tensor:
    """Run this block with conditioning prepared by its parent stack."""
    orig_dtype = x.dtype
    output = _kernels().diffusion_transition_block_prepared_packed(
        x.contiguous().to(torch.bfloat16),
        adaptive_scale,
        adaptive_bias,
        zero_gate,
        self._packed_transition,
        self._packed_transition2,
        hidden=self.num_intermediates,
        hidden_panel=self._native_hidden_panel,
        output_panel=self._native_output_panel,
    )
    return output.to(orig_dtype)

class DiffusionSelfAttentionVecops(_diffusion_transformer.SelfAttention):
  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    self._prepare_native_parameters()
    self._native_prepared = True
    return True

  def _prepare_native_parameters(self):
    if not self.layer_norm.single_cond or self.distributed:
      return
    _prepare_adaptive_layer_norm(self.layer_norm)
    heads = _native_tensor(self.q_projection, 'weight').shape[1]
    _set_native_buffer(
        self,
        "_packed_cond_scale",
        _kernels().pack_matmul_b(
            _native_tensor(self.layer_norm.single_cond_scale, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_cond_bias",
        _kernels().pack_matmul_b(
            _native_tensor(self.layer_norm.single_cond_bias, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _pack_head_projection(
        self, "_packed_q_projection", self.q_projection, heads, self.key_dim
    )
    _pack_head_projection(
        self, "_packed_k_projection", self.k_projection, heads, self.key_dim
    )
    _pack_head_projection(
        self, "_packed_v_projection", self.v_projection, heads, self.value_dim
    )
    gate = _native_tensor(self.gating_query, 'weight').reshape(
        _native_tensor(self.gating_query, 'weight').shape[0], heads, self.value_dim
    )
    _set_native_buffer(
        self,
        "_packed_gate_projection",
        _kernels().pack_matmul_b(gate.permute(1, 2, 0)),
    )
    _prepare_adaptive_zero(self.zero_init)
    _set_native_buffer(
        self,
        "_packed_output_projection",
        _kernels().pack_matmul_b(
            _native_tensor(self.zero_init.transition2, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_zero_cond",
        _kernels().pack_matmul_b(
            _native_tensor(self.zero_init.adaptive_zero_cond, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )

  def _forward_native(
      self,
      x: torch.Tensor,
      mask: torch.BoolTensor,
      *,
      pair_logits: torch.Tensor | None = None,
      single_cond: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if not self.layer_norm.single_cond or self.distributed:
      output = super().forward(
          x, mask, pair_logits=pair_logits, single_cond=single_cond
      )
      return output
    if single_cond is None:
      raise ValueError("single_cond is required by DiffusionSelfAttentionVecops")
    orig_dtype = x.dtype
    x = x.contiguous().to(torch.bfloat16)
    single_cond = single_cond.contiguous().to(torch.bfloat16)
    if pair_logits is not None:
      pair_logits = pair_logits.contiguous().to(torch.bfloat16)
    output = _kernels().diffusion_self_attention_packed(
        x,
        mask.to(torch.bool).contiguous(),
        pair_logits,
        single_cond,
        _native_tensor(self.layer_norm.single_cond_layer_norm, 'weight'),
        self._packed_cond_scale,
        _native_tensor(self.layer_norm.single_cond_scale, 'bias'),
        self._packed_cond_bias,
        self._packed_q_projection,
        _native_tensor(self.q_projection, 'bias'),
        self._packed_k_projection,
        self._packed_v_projection,
        self._packed_gate_projection,
        self._packed_output_projection,
        self._packed_zero_cond,
        _native_tensor(self.zero_init.adaptive_zero_cond, 'bias'),
    )
    return output.to(orig_dtype)

  @_require_native_preparation
  def forward(
      self,
      x: torch.Tensor,
      mask: torch.BoolTensor,
      *,
      pair_logits: torch.Tensor | None = None,
      single_cond: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return self._forward_native(
        x,
        mask,
        pair_logits=pair_logits,
        single_cond=single_cond,
    )

  @_require_native_preparation
  def forward_prepared(
      self,
      x: torch.Tensor,
      mask: torch.BoolTensor,
      *,
      pair_logits: torch.Tensor | None,
      adaptive_scale: torch.Tensor,
      adaptive_bias: torch.Tensor,
      zero_gate: torch.Tensor,
  ) -> torch.Tensor:
    """Run this attention with conditioning prepared by its parent stack."""
    orig_dtype = x.dtype
    if pair_logits is not None:
      pair_logits = pair_logits.contiguous().to(torch.bfloat16)
    output = _kernels().diffusion_self_attention_prepared_packed(
        x.contiguous().to(torch.bfloat16),
        mask.to(torch.bool).contiguous(),
        pair_logits,
        adaptive_scale,
        adaptive_bias,
        zero_gate,
        self._packed_q_projection,
        _native_tensor(self.q_projection, 'bias'),
        self._packed_k_projection,
        self._packed_v_projection,
        self._packed_gate_projection,
        self._packed_output_projection,
    )
    return output.to(orig_dtype)

class DiffusionCrossAttentionVecops(_diffusion_transformer.CrossAttention):
  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    self._prepare_native_parameters()
    self._native_prepared = True
    return True

  def _prepare_native_parameters(self):
    _prepare_adaptive_layer_norm(self.q_layernorm)
    _prepare_adaptive_layer_norm(self.k_layernorm)
    heads = _native_tensor(self.q_projection, 'weight').shape[1]
    _set_native_buffer(
        self,
        "_packed_q_cond_scale",
        _kernels().pack_matmul_b(
            _native_tensor(self.q_layernorm.single_cond_scale, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_q_cond_bias",
        _kernels().pack_matmul_b(
            _native_tensor(self.q_layernorm.single_cond_bias, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_k_cond_scale",
        _kernels().pack_matmul_b(
            _native_tensor(self.k_layernorm.single_cond_scale, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_k_cond_bias",
        _kernels().pack_matmul_b(
            _native_tensor(self.k_layernorm.single_cond_bias, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _pack_head_projection(
        self, "_packed_q_projection", self.q_projection, heads, self.key_dim
    )
    _pack_head_projection(
        self, "_packed_k_projection", self.k_projection, heads, self.key_dim
    )
    _pack_head_projection(
        self, "_packed_v_projection", self.v_projection, heads, self.value_dim
    )
    gate = _native_tensor(self.gating_query, 'weight').reshape(
        _native_tensor(self.gating_query, 'weight').shape[0], heads, self.value_dim
    )
    _set_native_buffer(
        self,
        "_packed_gate_projection",
        _kernels().pack_matmul_b(gate.permute(1, 2, 0)),
    )
    _prepare_adaptive_zero(self.zero_init)
    _set_native_buffer(
        self,
        "_packed_output_projection",
        _kernels().pack_matmul_b(
            _native_tensor(self.zero_init.transition2, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self,
        "_packed_zero_cond",
        _kernels().pack_matmul_b(
            _native_tensor(self.zero_init.adaptive_zero_cond, 'weight').transpose(0, 1)
        ).unsqueeze(0),
    )

  @_require_native_preparation
  def forward(
      self,
      x_q: torch.Tensor,
      x_k: torch.Tensor,
      mask_q: torch.Tensor,
      mask_k: torch.Tensor,
      pair_logits: torch.Tensor | None = None,
      single_cond_q: torch.Tensor | None = None,
      single_cond_k: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if single_cond_q is None or single_cond_k is None:
      raise ValueError("both single_cond_q and single_cond_k are required")
    orig_dtype = x_q.dtype
    x_q = x_q.contiguous().to(torch.bfloat16)
    x_k = x_k.contiguous().to(torch.bfloat16)
    single_cond_q = single_cond_q.contiguous().to(torch.bfloat16)
    single_cond_k = single_cond_k.contiguous().to(torch.bfloat16)
    if pair_logits is not None:
      pair_logits = pair_logits.contiguous().to(torch.bfloat16)
    output = _kernels().diffusion_cross_attention_packed(
        x_q,
        x_k,
        mask_q.to(torch.bool).contiguous(),
        mask_k.to(torch.bool).contiguous(),
        pair_logits,
        single_cond_q,
        single_cond_k,
        _native_tensor(self.q_layernorm.single_cond_layer_norm, 'weight'),
        self._packed_q_cond_scale,
        _native_tensor(self.q_layernorm.single_cond_scale, 'bias'),
        self._packed_q_cond_bias,
        _native_tensor(self.k_layernorm.single_cond_layer_norm, 'weight'),
        self._packed_k_cond_scale,
        _native_tensor(self.k_layernorm.single_cond_scale, 'bias'),
        self._packed_k_cond_bias,
        self._packed_q_projection,
        _native_tensor(self.q_projection, 'bias'),
        self._packed_k_projection,
        self._packed_v_projection,
        self._packed_gate_projection,
        self._packed_output_projection,
        self._packed_zero_cond,
        _native_tensor(self.zero_init.adaptive_zero_cond, 'bias'),
    )
    return output.to(orig_dtype)

  @_require_native_preparation
  def forward_prepared(
      self,
      x_q: torch.Tensor,
      x_k: torch.Tensor,
      mask_q: torch.Tensor,
      mask_k: torch.Tensor,
      pair_logits: torch.Tensor | None,
      q_adaptive_scale: torch.Tensor,
      q_adaptive_bias: torch.Tensor,
      k_adaptive_scale: torch.Tensor,
      k_adaptive_bias: torch.Tensor,
      zero_gate: torch.Tensor,
  ) -> torch.Tensor:
    """Run this layer with static conditioning prepared by its parent."""
    orig_dtype = x_q.dtype
    if pair_logits is not None:
      pair_logits = pair_logits.contiguous().to(torch.bfloat16)
    output = _kernels().diffusion_cross_attention_prepared_packed(
        x_q.contiguous().to(torch.bfloat16),
        x_k.contiguous().to(torch.bfloat16),
        mask_q.to(torch.bool).contiguous(),
        mask_k.to(torch.bool).contiguous(),
        pair_logits,
        q_adaptive_scale,
        q_adaptive_bias,
        k_adaptive_scale,
        k_adaptive_bias,
        zero_gate,
        self._packed_q_projection,
        _native_tensor(self.q_projection, 'bias'),
        self._packed_k_projection,
        self._packed_v_projection,
        self._packed_gate_projection,
        self._packed_output_projection,
    )
    return output.to(orig_dtype)

  @_require_native_preparation
  def forward_indexed_prepared(
      self,
      queries_act: torch.Tensor,
      query_mask: torch.Tensor,
      key_indices: torch.Tensor,
      key_mask: torch.Tensor,
      pair_logits: torch.Tensor,
      q_adaptive_scale: torch.Tensor,
      q_adaptive_bias: torch.Tensor,
      k_adaptive_scale: torch.Tensor,
      k_adaptive_bias: torch.Tensor,
      zero_gate: torch.Tensor,
  ) -> torch.Tensor:
    """Project unique query atoms before indexed K/V window assembly."""
    orig_dtype = queries_act.dtype
    output = _kernels().diffusion_cross_attention_indexed_prepared_packed(
        queries_act.contiguous().to(torch.bfloat16),
        query_mask.to(torch.bool).contiguous(),
        key_indices.to(torch.int32).contiguous(),
        key_mask.to(torch.bool).contiguous(),
        pair_logits.contiguous().to(torch.bfloat16),
        q_adaptive_scale,
        q_adaptive_bias,
        k_adaptive_scale,
        k_adaptive_bias,
        zero_gate,
        self._packed_q_projection,
        _native_tensor(self.q_projection, 'bias'),
        self._packed_k_projection,
        self._packed_v_projection,
        self._packed_gate_projection,
        self._packed_output_projection,
    )
    return output.to(orig_dtype)


class TransformerVecops(_diffusion_transformer.Transformer):
  """Native Diffusion Transformer with superblock conditioning plans.

  The readable Torch Transformer remains unchanged.  This replacement owns the
  backend-specific reordering that prepares the four attention and four
  transition conditioning branches of a superblock in one native call.
  """

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    self._prepare_native_superblocks()
    self._native_prepared = True
    return True

  def _prepare_native_superblocks(self) -> None:
    for superblock, (attentions, transitions) in enumerate(
        zip(self.self_attention, self.transition_block, strict=True)
    ):
      ordered = []
      for attention, transition in zip(attentions, transitions, strict=True):
        if not isinstance(attention, DiffusionSelfAttentionVecops):
          raise TypeError("TransformerVecops requires DiffusionSelfAttentionVecops children")
        if not isinstance(transition, DiffusionTransitionBlockVecops):
          raise TypeError("TransformerVecops requires DiffusionTransitionBlockVecops children")
        ordered.extend((attention, transition))

      _set_native_buffer(
          self,
          f"_sb{superblock}_cond_norm_scale",
          torch.stack([
              _native_tensor(module.layer_norm.single_cond_layer_norm, 'weight')
              for module in ordered
          ]),
      )
      _set_native_buffer(
          self,
          f"_sb{superblock}_packed_cond_scale",
          torch.cat([module._packed_cond_scale for module in ordered], dim=0),
      )
      _set_native_buffer(
          self,
          f"_sb{superblock}_cond_scale_bias",
          torch.stack([
              _native_tensor(module.layer_norm.single_cond_scale, 'bias') for module in ordered
          ]),
      )
      _set_native_buffer(
          self,
          f"_sb{superblock}_packed_cond_bias",
          torch.cat([module._packed_cond_bias for module in ordered], dim=0),
      )
      _set_native_buffer(
          self,
          f"_sb{superblock}_packed_zero_cond",
          torch.cat([module._packed_zero_cond for module in ordered], dim=0),
      )
      _set_native_buffer(
          self,
          f"_sb{superblock}_zero_cond_bias",
          torch.stack([
              _native_tensor(module.zero_init.adaptive_zero_cond, 'bias') for module in ordered
          ]),
      )

  def _prepare_conditioning(self, superblock: int, single_cond: torch.Tensor):
    return _kernels().diffusion_conditioning_packed(
        single_cond.contiguous().to(torch.bfloat16),
        getattr(self, f"_sb{superblock}_cond_norm_scale"),
        getattr(self, f"_sb{superblock}_packed_cond_scale"),
        getattr(self, f"_sb{superblock}_cond_scale_bias"),
        getattr(self, f"_sb{superblock}_packed_cond_bias"),
        getattr(self, f"_sb{superblock}_packed_zero_cond"),
        getattr(self, f"_sb{superblock}_zero_cond_bias"),
    )

  @_require_native_preparation
  def forward(
      self,
      act: torch.Tensor,
      mask: torch.Tensor,
      single_cond: torch.Tensor,
      pair_cond: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if not _use_native_diffusion_transition(act.shape[-2]):
      return super().forward(act, mask, single_cond, pair_cond)
    if pair_cond is not None:
      def _compute_pair_logits():
        pair_act = self.pair_input_layer_norm(pair_cond)
        pack = []
        for projection in self.pair_logits_projection:
          pair_logits = projection(pair_act)
          pair_logits = einops.rearrange(
              pair_logits,
              "b n s h -> s h b n",
              s=self.config.super_block_size,
          )
          pack.append(pair_logits)
        return torch.stack(pack).to(self.dtype).contiguous()

      # pair_cond is invariant across the denoising loop but is re-cast to a
      # fresh Tensor object at every step. Keying by identity recomputes all six
      # large projections 200 times; the base model intentionally caches this
      # value by purpose for the lifetime of one example.
      pair_logits_pack = self._cache.get(
          name="pair_logits", factory=_compute_pair_logits
      )
    else:
      pair_logits_pack = None

    input_dtype = act.dtype
    act = act.to(self.dtype)
    use_native_region = _native_pool_region_enabled(
        "sparsefold_native::diffusion_conditioning",
        "sparsefold_native::diffusion_self_attention",
        "sparsefold_native::diffusion_transition_block",
    )
    if use_native_region:
      # Keep the inputs stable while the private pool is active.  The
      # prepared wrappers below still own flattening/broadcasting and dynamic
      # shape rebinding; bypassing them with raw bridges can reuse a stale
      # planning pattern and abort when a later denoising shape differs.
      act = act.contiguous()
      mask = mask.to(torch.bool).contiguous()
      single_cond = single_cond.contiguous().to(torch.bfloat16)
    with _native_pool_region(
        "sparsefold_native::diffusion_conditioning",
        "sparsefold_native::diffusion_self_attention",
        "sparsefold_native::diffusion_transition_block",
    ):
      for superblock, (attentions, transitions) in enumerate(
          zip(self.self_attention, self.transition_block, strict=True)
      ):
        adaptive_scale, adaptive_bias, zero_gate = self._prepare_conditioning(
            superblock, single_cond
        )
        superblock_pair = (
            None if pair_logits_pack is None else pair_logits_pack[superblock]
        )
        for block, (attention, transition) in enumerate(
            zip(attentions, transitions, strict=True)
        ):
          attention_state = 2 * block
          transition_state = attention_state + 1
          # Always use the prepared wrappers here.  Besides keeping the
          # public call contract in one place, they flatten shared leading
          # batch dimensions and rebind every dynamic tensor against the
          # current invocation.  The private pool is supplied by the context
          # manager and does not require bypassing this safety boundary.
          act = attention.forward_prepared(
              act,
              mask,
              pair_logits=(
                  None if superblock_pair is None else superblock_pair[block]
              ),
              adaptive_scale=adaptive_scale[attention_state],
              adaptive_bias=adaptive_bias[attention_state],
              zero_gate=zero_gate[attention_state],
          )
          act = transition.forward_prepared(
              act,
              adaptive_scale[transition_state],
              adaptive_bias[transition_state],
              zero_gate[transition_state],
          )
    return act.to(input_dtype)


class CrossAttTransformerVecops(_diffusion_transformer.CrossAttTransformer):
  """Native atom Transformer with per-input conditioning cached internally."""

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    self._prepare_native_conditioning_weights()
    self._native_prepared = True
    return True

  def _prepare_native_conditioning_weights(self) -> None:
    q_ordered = []
    for attention, transition in zip(
        self.cross_attention, self.transition_block, strict=True
    ):
      if not isinstance(attention, DiffusionCrossAttentionVecops):
        raise TypeError(
            "CrossAttTransformerVecops requires DiffusionCrossAttentionVecops children"
        )
      if not isinstance(transition, DiffusionTransitionBlockVecops):
        raise TypeError(
            "CrossAttTransformerVecops requires DiffusionTransitionBlockVecops children"
        )
      q_ordered.extend((attention, transition))

    def q_attr(module, cross_name, transition_name):
      if isinstance(module, DiffusionCrossAttentionVecops):
        return getattr(module, cross_name)
      return getattr(module, transition_name)

    _set_native_buffer(
        self,
        "_q_cond_norm_scale",
        torch.stack([
            _native_tensor(module.q_layernorm.single_cond_layer_norm, 'weight')
            if isinstance(module, DiffusionCrossAttentionVecops)
            else _native_tensor(module.layer_norm.single_cond_layer_norm, 'weight')
            for module in q_ordered
        ]),
    )
    for target, cross_name, transition_name in (
        ("_q_packed_cond_scale", "_packed_q_cond_scale", "_packed_cond_scale"),
        ("_q_packed_cond_bias", "_packed_q_cond_bias", "_packed_cond_bias"),
        ("_q_packed_zero_cond", "_packed_zero_cond", "_packed_zero_cond"),
    ):
      _set_native_buffer(
          self,
          target,
          torch.cat([
              q_attr(module, cross_name, transition_name)
              for module in q_ordered
          ], dim=0),
      )
    _set_native_buffer(
        self,
        "_q_cond_scale_bias",
        torch.stack([
            _native_tensor(module.q_layernorm.single_cond_scale, 'bias')
            if isinstance(module, DiffusionCrossAttentionVecops)
            else _native_tensor(module.layer_norm.single_cond_scale, 'bias')
            for module in q_ordered
        ]),
    )
    _set_native_buffer(
        self,
        "_q_zero_cond_bias",
        torch.stack([
            _native_tensor(module.zero_init.adaptive_zero_cond, 'bias') for module in q_ordered
        ]),
    )

    attentions = list(self.cross_attention)
    _set_native_buffer(
        self,
        "_k_cond_norm_scale",
        torch.stack([
            _native_tensor(module.k_layernorm.single_cond_layer_norm, 'weight')
            for module in attentions
        ]),
    )
    _set_native_buffer(
        self,
        "_k_packed_cond_scale",
        torch.cat([module._packed_k_cond_scale for module in attentions], dim=0),
    )
    _set_native_buffer(
        self,
        "_k_cond_scale_bias",
        torch.stack([
            _native_tensor(module.k_layernorm.single_cond_scale, 'bias') for module in attentions
        ]),
    )
    _set_native_buffer(
        self,
        "_k_packed_cond_bias",
        torch.cat([module._packed_k_cond_bias for module in attentions], dim=0),
    )
    # The generic conditioning kernel has a three-output contract.  Its K zero
    # result is deliberately discarded; computing it once per input is cheaper
    # than carrying a second specialized preparation ABI.
    _set_native_buffer(
        self,
        "_k_packed_zero_cond",
        torch.cat([module._packed_zero_cond for module in attentions], dim=0),
    )
    _set_native_buffer(
        self,
        "_k_zero_cond_bias",
        torch.stack([
            _native_tensor(module.zero_init.adaptive_zero_cond, 'bias') for module in attentions
        ]),
    )

  def _prepare_static_conditioning(
      self,
      queries_single_cond: torch.Tensor,
  ):
    q_state = _kernels().diffusion_conditioning_packed(
        queries_single_cond.contiguous().to(torch.bfloat16),
        self._q_cond_norm_scale,
        self._q_packed_cond_scale,
        self._q_cond_scale_bias,
        self._q_packed_cond_bias,
        self._q_packed_zero_cond,
        self._q_zero_cond_bias,
    )
    k_state = _kernels().diffusion_conditioning_packed(
        queries_single_cond.contiguous().to(torch.bfloat16),
        self._k_cond_norm_scale,
        self._k_packed_cond_scale,
        self._k_cond_scale_bias,
        self._k_packed_cond_bias,
        self._k_packed_zero_cond,
        self._k_zero_cond_bias,
    )
    return q_state, k_state

  @_require_native_preparation
  def forward(
      self,
      queries_act: torch.Tensor,
      queries_mask: torch.Tensor,
      queries_to_keys,
      keys_mask: torch.Tensor,
      queries_single_cond: torch.Tensor,
      keys_single_cond: torch.Tensor,
      pair_cond: torch.Tensor,
  ) -> torch.Tensor:
    def _compute_pair_logits():
      pair_act = self.pair_input_layer_norm(pair_cond)
      pair_logits = self.pair_logits_projection(pair_act)
      return einops.rearrange(
          pair_logits, "s q k b h -> b s h q k"
      ).contiguous()

    pair_logits = self._cache.get(
        pair_cond, name="pair_logits", factory=_compute_pair_logits
    )
    q_state, k_state = self._cache.get(
        queries_single_cond,
        name="native_conditioning",
        factory=lambda: self._prepare_static_conditioning(queries_single_cond),
    )
    q_scale, q_bias, q_zero = q_state
    k_scale, k_bias, _ = k_state
    use_native_region = _native_pool_region_enabled(
        "sparsefold_native::diffusion_cross_attention_indexed",
        "sparsefold_native::diffusion_transition_block",
    )
    if use_native_region:
      input_dtype = queries_act.dtype
      queries_act = queries_act.contiguous().to(torch.bfloat16)
      queries_mask = queries_mask.to(torch.bool).contiguous()
      key_indices = queries_to_keys.gather_idxs.to(torch.int32).contiguous()
      keys_mask = keys_mask.to(torch.bool).contiguous()
      pair_logits = pair_logits.contiguous().to(torch.bfloat16)
    else:
      input_dtype = None
      key_indices = queries_to_keys.gather_idxs

    with _native_pool_region(
        "sparsefold_native::diffusion_cross_attention_indexed",
        "sparsefold_native::diffusion_transition_block",
    ):
      for block, (attention, transition) in enumerate(
          zip(self.cross_attention, self.transition_block, strict=True)
      ):
        q_state_index = 2 * block
        queries_act = attention.forward_indexed_prepared(
            queries_act,
            queries_mask,
            key_indices,
            keys_mask,
            pair_logits[block],
            q_scale[q_state_index],
            q_bias[q_state_index],
            k_scale[block],
            k_bias[block],
            q_zero[q_state_index],
        )
        transition_state_index = q_state_index + 1
        queries_act = transition.forward_prepared(
            queries_act,
            q_scale[transition_state_index],
            q_bias[transition_state_index],
            q_zero[transition_state_index],
        )
    return queries_act if input_dtype is None else queries_act.to(input_dtype)


# Fast GridSelfAttention implemented in C++, using IBSAttention (aka SparseAttention)
class GridSelfAttentionVecops(_modules.GridSelfAttention):
  def __init__(
      self,
      *args,
      index_reuse_policy: _modules.IndexReusePolicy = _modules.default_reuse_policy, 
      ibsa_params: _modules.IBSAParametersOrProvider = _modules.IBSAParameters.default,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    if isinstance(ibsa_params, _modules.IBSAParameters):
      def constant_provide(_):
        return ibsa_params
      self.ibsa_params_provider = constant_provide
    else:
      self.ibsa_params_provider = ibsa_params
    self.index_reuse_policy = index_reuse_policy
    self.cached_index_reuser = cache.ValueCache()
    # (index_map: Tensor, weight_map: Tensor) | None
    self.cached_maps = None
    
  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    visitor = _prepared_visitor({
        "q_projection": {"weights": self.q_projection.weight},
        "k_projection": {"weights": self.k_projection.weight},
        "v_projection": {"weights": self.v_projection.weight},
        "gating_query": {"weights": self.gating_query.weight},
        "output_projection": {"weights": self.output_projection.weight},
        "pair_bias_projection": {"weights": self.pair_bias_projection.weight},
        "act_norm": _norm_scope(self.act_norm),
    })
    # W: layer stack
    H = self.config.num_head
    C = self.num_channels
    D = self.qkv_dim

    # (W, H, C, D)
    with visitor.into("q_projection"):
      q_proj = visitor["weights"]
      assert q_proj.shape == (H, D, C)
      _set_native_buffer(self, '_native_q_projection', torch.transpose_copy(q_proj, -1, -2).to(torch.bfloat16))

    # (W, H, C, D)
    with visitor.into("k_projection"):
      k_proj = visitor["weights"]
      assert k_proj.shape == (H, D, C)
      _set_native_buffer(self, '_native_k_projection', torch.transpose_copy(k_proj, -1, -2).to(torch.bfloat16))

    # (W, H, C, D)
    with visitor.into("v_projection"):
      v_proj = visitor["weights"]
      assert v_proj.shape == (C, H, D)
      _set_native_buffer(self, '_native_v_projection', torch.transpose_copy(v_proj, -2, -3).to(torch.bfloat16))

    # (W, H, C, D)
    with visitor.into("gating_query"):
      gating_query = visitor["weights"]
      gating_query = gating_query.view((H, D, C))
      assert gating_query.shape == (H, D, C)
      _set_native_buffer(self, '_native_gating_query', torch.transpose_copy(gating_query, -1, -2).to(torch.bfloat16))

    # (W, H * D, C)
    with visitor.into("output_projection"):
      o_proj = visitor["weights"]
      assert o_proj.shape == (H * D, C)
      _set_native_buffer(self, '_native_output_projection', o_proj.to(torch.bfloat16))

    # (H, C)
    with visitor.into("pair_bias_projection"):
      b_proj = visitor["weights"]
      assert b_proj.shape == (C, H)
      _set_native_buffer(self, '_native_pair_bias_projection', torch.transpose_copy(b_proj, -1, -2).to(torch.bfloat16))

    # (C,)
    with visitor.into("act_norm"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (C,)
      _set_native_buffer(self, '_native_act_norm_weight', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (C,)
      _set_native_buffer(self, '_native_act_norm_offset', ln_offset.to(torch.float32))

    _set_native_buffer(
        self, "_packed_q_projection",
        _kernels().pack_matmul_b(self._native_q_projection.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_k_projection",
        _kernels().pack_matmul_b(self._native_k_projection.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_v_projection",
        _kernels().pack_matmul_b(self._native_v_projection.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_g_projection",
        _kernels().pack_matmul_b(self._native_gating_query.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_output_projection",
        _kernels().pack_matmul_b(
            self._native_output_projection.transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self, "_packed_bias_projection",
        _kernels().pack_matmul_b(self._native_pair_bias_projection).unsqueeze(0),
    )
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(
      self,
      act: torch.Tensor,  # (*, B, N, C)
      seq_mask: torch.Tensor,  # (N,)
  ):
    assert len(act.shape) >= 3
    B, N, C = act.shape[-3:]
    Batch = act.shape[:-3]
    assert C == self.num_channels
    assert seq_mask.shape == (N,) and seq_mask.dtype == torch.bool

    if act.stride(-1) != 1:
      trace.log_debug(f"Weird act is not last-contiguous: {act.stride()}")
      act = act.contiguous()

    if act.dtype != torch.bfloat16:
      trace.log_debug(f"act is not bfloat16: {act.dtype}")
      act = act.to(torch.bfloat16)
    
    seq_mask = util.align_batch(seq_mask, Batch)

    if self.distributed:
      fn = grid_self_attention_distributed
      projection_args = (
          self._packed_q_projection,
          self._packed_k_projection,
          self._packed_v_projection,
          self._packed_output_projection,
          self._packed_g_projection,
          self._packed_bias_projection,
      )
      projection_kwargs = {"head_dim": self.qkv_dim}
    else:
      fn = _kernels().grid_self_attention_packed
      projection_args = (
          self._packed_q_projection,
          self._packed_k_projection,
          self._packed_v_projection,
          self._packed_output_projection,
          self._packed_g_projection,
          self._packed_bias_projection,
      )
      projection_kwargs = {"head_dim": self.qkv_dim}

    if _is_precompiling():
      ibsa_params = self.ibsa_params_provider(fnn.layer_stack_indices)
      region = (
          contextlib.nullcontext()
          if self.distributed
          else _native_pool_region("sparsefold_native::grid_self_attention")
      )
      with region:
        traced_act, _, _ = fn(
            act,
            seq_mask,
            *projection_args,
            self._native_act_norm_weight,
            self._native_act_norm_offset,
            None,
            None,
            ibsa_params.num_samples,
            ibsa_params.num_rand_blocks,
            ibsa_params.t_prob,
            ibsa_params.tr_prob,
            ibsa_params.u_prob,
            self.transpose,
            **projection_kwargs,
        )
      return traced_act

    @torch.compiler.disable
    def _call_with_cache(act) -> torch.Tensor:

      def make_reuser():
        return _modules.IndexReuser(self.index_reuse_policy)
      
      ibsa_params = self.ibsa_params_provider(fnn.layer_stack_indices)
      # Available if shape of act & seq_mask is not changed
      # Note: doesn't check for data in seq_mask, this could be an exploit
      reuser: _modules.IndexReuser = self.cached_index_reuser.get(act.shape, seq_mask.shape, name="reuser", factory=make_reuser)
      maps, reuse_indices = reuser.get(fnn.layer_stack_indices)
      is_reuse = maps is not None

      if not is_reuse:
        maps = None, None
      
      index_map, weight_map = maps
      region = (
          contextlib.nullcontext()
          if self.distributed
          else _native_pool_region("sparsefold_native::grid_self_attention")
      )
      with region:
        act, index_map, weight_map = fn(
          act,
          seq_mask,
          *projection_args,
          self._native_act_norm_weight,
          self._native_act_norm_offset,
          index_map,
          weight_map,
          ibsa_params.num_samples,
          ibsa_params.num_rand_blocks,
          ibsa_params.t_prob,
          ibsa_params.tr_prob,
          ibsa_params.u_prob,
          self.transpose,
          **projection_kwargs,
        )
      if not is_reuse:
        reuser.set(fnn.layer_stack_indices, (index_map, weight_map))
        try:
          from vecops.graph import mark_retained

          mark_retained(
              (index_map, weight_map),
              reason="sparsefold.GridSelfAttention.index_reuse",
          )
        except ImportError:
          pass

      if trace.LOG_DEBUG:
        if is_reuse:
          trace.log_debug(f"Reusing index & weight map from {reuse_indices}.")
        else:
          # TODO not accurate, requires internal profiling in native ops
          sparsity = (index_map >= 0).sum() / (index_map.numel() / ibsa_params.num_samples * index_map.shape[-2])
          trace.log_debug(f"Creating new index & weight map (sparsity={1 - sparsity:.04f})")

      return act
    
    act = _call_with_cache(act)

    if act.stride(-1) != 1:
      trace.log_debug("Weird return is not last-contiguous...")
      act = act.contiguous()

    return act


class TriangleMultiplicationVecops(_modules.TriangleMultiplication):
  def __init__(
      self,
      *args,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.transpose = self.equation == "...ckj,...cki->...cij"

  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    visitor = _prepared_visitor({
        "left_norm_input": _norm_scope(self.left_norm_input),
        "gate": {"weights": self.gate_projection.weight},
        "projection": {"weights": self.projection.weight},
        "center_norm": _norm_scope(self.center_norm),
        "output_projection": {"weights": self.output_projection.weight},
        "gating_linear": {"weights": self.gating_linear.weight},
    })
    C = self.num_channels

    # (C,)
    with visitor.into("left_norm_input"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (C,)
      _set_native_buffer(self, '_native_left_norm_scale', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (C,)
      _set_native_buffer(self, '_native_left_norm_offset', ln_offset.to(torch.float32))

    # (2 * C, C)
    with visitor.into("gate"):
      g_proj = visitor["weights"]
      assert g_proj.shape == (C, 2 * C)
      g_proj = torch.transpose_copy(g_proj, -1, -2)
    # (2 * C, C)
    with visitor.into("projection"):
      w_proj = visitor["weights"]
      assert w_proj.shape == (C, 2 * C)
      w_proj = torch.transpose_copy(w_proj, -1, -2)

    # Complex Transpose:
    #   Wg: [a,b,a,b,...], Wp:[c,d,c,d,...] -> Wgp: [[a,c,a,c,...], [b,d,b,d,...]]
    Wu = torch.cat((g_proj, w_proj), dim=-1)
    Wu = torch.reshape(Wu, (C, 2, 2 * C))
    Wu = torch.transpose(Wu, -3, -2)
    gw_proj = torch.reshape(Wu, (2, 2 * C, C)).contiguous()
    _set_native_buffer(self, '_native_gw_proj', gw_proj.to(torch.bfloat16))

    # (C,)
    with visitor.into("center_norm"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (C,)
      _set_native_buffer(self, '_native_center_norm_scale', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (C,)
      _set_native_buffer(self, '_native_center_norm_offset', ln_offset.to(torch.float32))

    # (C, C)
    with visitor.into("output_projection"):
      o_proj = visitor["weights"]
      assert o_proj.shape == (C, C)
      _set_native_buffer(self, '_native_output_projection', o_proj.to(torch.bfloat16))

    # (C, C)
    with visitor.into("gating_linear"):
      gl_proj = visitor["weights"]
      assert gl_proj.shape == (C, C)
      _set_native_buffer(self, '_native_gate_linear', gl_proj.to(torch.bfloat16))

    _set_native_buffer(
        self, "_packed_gw_projection",
        _kernels().pack_matmul_b(
            self._native_gw_proj.reshape(2, C, 2, C)
            .permute(0, 2, 1, 3)
            .reshape(4, C, C)
            .contiguous()
        ),
    )
    _set_native_buffer(
        self, "_packed_triangle_output",
        _kernels().pack_matmul_b(
            self._native_output_projection.transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    _set_native_buffer(
        self, "_packed_triangle_gate",
        _kernels().pack_matmul_b(
            self._native_gate_linear.transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(
      self,
      act: torch.Tensor,  # (*, B, N, C), where B = N, may have batch dim
      mask: torch.Tensor,  # (N,), bool
  ):
    C = self.num_channels
    Batch = act.shape[:-3]
    B, N, C = act.shape[-3:]
    assert C == self.num_channels
    assert mask.shape == (N,) and mask.dtype == torch.bool

    if act.stride(-1) != 1:
      trace.log_debug(f"Weird act is not last-contiguous: {act.stride()}")
      act = act.contiguous()

    if act.dtype != torch.bfloat16:
      trace.log_debug(f"act is not bfloat16: {act.dtype}")
      act = act.to(torch.bfloat16)
    
    mask = util.align_batch(mask, Batch)

    if self.distributed:
      fn = triangle_multiplication_distributed
      projection_args = (
          self._packed_gw_projection,
          self._native_center_norm_scale,
          self._native_center_norm_offset,
          self._packed_triangle_output,
          self._packed_triangle_gate,
      )
    else:
      fn = _kernels().triangle_multiplication_packed
      projection_args = (
          self._native_gw_proj,
          self._packed_gw_projection,
          self._native_center_norm_scale,
          self._native_center_norm_offset,
          self._packed_triangle_output,
          self._packed_triangle_gate,
      )
    region = (
        contextlib.nullcontext()
        if self.distributed
        else _native_pool_region("sparsefold_native::triangle_multiplication")
    )
    with region:
      act = fn(
          act,
          mask,
          self._native_left_norm_scale,
          self._native_left_norm_offset,
          *projection_args,
          self.transpose,
      )

    if act.stride(-1) != 1:
      trace.log_debug("Weird return is not last-contiguous...")
      act = act.contiguous()

    return act


class SingleActUpdateVecops(_modules.SingleActUpdate):
  def set_parameters(self, visitor, **kwargs):
    self._native_prepared = False
    super().set_parameters(visitor, **kwargs)

  def prepare_native(self) -> bool:
    _begin_native_preparation(self)
    visitor = _prepared_visitor({
        "single_pair_logits_norm": _norm_scope(self.single_pair_logits_norm),
        "single_pair_logits_projection": {"weights": self.single_pair_logits_projection.weight},
        "single_attention_layer_norm": _norm_scope(self.single_attention.layer_norm.layer_norm),
        "single_attention_q_projection": {"weights": self.single_attention.q_projection.weight,
                                          "bias": self.single_attention.q_projection.bias},
        "single_attention_k_projection": {"weights": self.single_attention.k_projection.weight},
        "single_attention_v_projection": {"weights": self.single_attention.v_projection.weight},
        "single_attention_gating_query": {"weights": self.single_attention.gating_query.weight},
        "single_attention_transition2": {"weights": self.single_attention.zero_init.transition2.weight},
        "single_transition/input_layer_norm": _norm_scope(self.single_transition.input_layer_norm),
        "single_transition/transition1": {"weights": self.single_transition.transition1.weight},
        "single_transition/transition2": {"weights": self.single_transition.transition2.weight},
    })

    Q = self.num_channels
    C = self.seq_channels
    H = self.num_head
    D = self.qkv_dim
    M = self.num_intermediate

    # (Q,)
    with visitor.into("single_pair_logits_norm"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (Q,)
      _set_native_buffer(self, '_native_pair_logits_norm_scale', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (Q,)
      _set_native_buffer(self, '_native_pair_logits_norm_offset', ln_offset.to(torch.float32))

    # (H, Q)
    with visitor.into("single_pair_logits_projection"):
      proj = visitor["weights"]
      assert proj.shape == (Q, H)
      _set_native_buffer(self, '_native_pair_logits_proj', torch.transpose_copy(proj, -1, -2).to(torch.bfloat16))
    # (C,)
    with visitor.into("single_attention_layer_norm"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (C,)
      _set_native_buffer(self, '_native_attn_norm_scale', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (C,)
      _set_native_buffer(self, '_native_attn_norm_offset', ln_offset.to(torch.float32))
    # (H, C, D) & (H, D)
    with visitor.into("single_attention_q_projection"):
      proj = visitor["weights"]
      assert proj.shape == (C, H, D)
      _set_native_buffer(self, '_native_q_proj', torch.transpose_copy(proj, -2, -3).to(torch.bfloat16))
      bias = visitor["bias"]
      assert bias.shape == (H, D)
      _set_native_buffer(self, '_native_q_bias', bias.contiguous().to(torch.bfloat16))
    # (H, C, D)
    with visitor.into("single_attention_k_projection"):
      proj = visitor["weights"]
      assert proj.shape == (C, H, D)
      _set_native_buffer(self, '_native_k_proj', torch.transpose_copy(proj, -2, -3).to(torch.bfloat16))
    # (H, C, D)
    with visitor.into("single_attention_v_projection"):
      proj = visitor["weights"]
      assert proj.shape == (C, H, D)
      _set_native_buffer(self, '_native_v_proj', torch.transpose_copy(proj, -2, -3).to(torch.bfloat16))
    # (H, C, D)
    with visitor.into("single_attention_gating_query"):
      proj = visitor["weights"]
      assert proj.shape == (C, C)
      _set_native_buffer(self, '_native_g_proj', torch.transpose_copy(proj.view((C, H, D)), -2, -3).to(torch.bfloat16))
    # (H, C, D)
    with visitor.into("single_attention_transition2"):
      proj = visitor["weights"]
      assert proj.shape == (C, C)
      _set_native_buffer(self, '_native_o_proj', torch.transpose_copy(proj.view((H, D, C)), -1, -2).to(torch.bfloat16))
    # (C,)
    with visitor.into("single_transition/input_layer_norm"):
      ln_scale = visitor["scale"]
      assert ln_scale.shape == (C,)
      _set_native_buffer(self, '_native_transition_norm_scale', ln_scale.to(torch.float32))
      ln_offset = visitor["offset"]
      assert ln_offset.shape == (C,)
      _set_native_buffer(self, '_native_transition_norm_offset', ln_offset.to(torch.float32))
    # (C, 2 * M)
    with visitor.into("single_transition/transition1"):
      proj = visitor["weights"]
      assert proj.shape == (C, 2 * M)
      _set_native_buffer(self, '_native_transition1_proj', proj.contiguous().to(torch.bfloat16))
    # (M, C)
    with visitor.into("single_transition/transition2"):
      proj = visitor["weights"]
      assert proj.shape == (M, C)
      _set_native_buffer(self, '_native_transition2_proj', proj.contiguous().to(torch.bfloat16))

    _set_native_buffer(
        self, "_packed_pair_logits_projection",
        _kernels().pack_matmul_b(self._native_pair_logits_proj).unsqueeze(0),
    )
    _set_native_buffer(
        self, "_packed_single_q",
        _kernels().pack_matmul_b(self._native_q_proj.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_single_k",
        _kernels().pack_matmul_b(self._native_k_proj.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_single_v",
        _kernels().pack_matmul_b(self._native_v_proj.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_single_g",
        _kernels().pack_matmul_b(self._native_g_proj.permute(0, 2, 1).contiguous()),
    )
    _set_native_buffer(
        self, "_packed_single_o", _kernels().pack_matmul_b(self._native_o_proj)
    )
    _set_native_buffer(
        self, "_packed_single_transition1",
        _kernels().pack_matmul_b(
            self._native_transition1_proj.reshape(C, 2, M)
            .permute(1, 2, 0)
            .contiguous()
        ),
    )
    _set_native_buffer(
        self, "_packed_single_transition2",
        _kernels().pack_matmul_b(
            self._native_transition2_proj.transpose(0, 1).contiguous()
        ).unsqueeze(0),
    )
    self._native_prepared = True
    return True

  @_require_native_preparation
  def forward(
      self,
      pair_act: torch.Tensor,  # (*, B, N, Q)
      single_act: torch.Tensor,  # (*, B, C)
      seq_mask: torch.Tensor,  # (N)
  ):
    Batch = pair_act.shape[:-3]
    assert single_act.shape[:-2] == Batch, f"Batch mismatch: pair_act={pair_act.shape}, single_act={single_act.shape}"

    if pair_act.stride(-1) != 1:
      trace.log_debug(f"Weird non-contiguous pair_act: {pair_act.stride()}")
      pair_act = pair_act.contiguous()

    if pair_act.dtype != torch.bfloat16:
      trace.log_debug(f"pair_act is not bfloat16: {pair_act.dtype}")
      pair_act = pair_act.to(torch.bfloat16)

    if single_act.stride(-1) != 1:
      trace.log_debug(f"Weird non-contiguous single_act: {single_act.stride()}")
      single_act = single_act.contiguous()

    if single_act.dtype != torch.bfloat16:
      trace.log_debug(f"single_act is not bfloat16: {single_act.dtype}")
      single_act = single_act.to(torch.bfloat16)

    if self.distributed:
      fn = evoformer_single_act_update_distributed
      projection_args = (
          self._packed_pair_logits_projection,
          self._native_attn_norm_scale,
          self._native_attn_norm_offset,
          self._packed_single_q,
          self._native_q_bias,
          self._packed_single_k,
          self._packed_single_v,
          self._packed_single_g,
          self._packed_single_o,
          self._native_transition_norm_scale,
          self._native_transition_norm_offset,
          self._packed_single_transition1,
          self._packed_single_transition2,
      )
      projection_kwargs = {"hidden": self.num_intermediate}
    else:
      fn = _kernels().evoformer_single_act_update_packed
      projection_args = (
          self._packed_pair_logits_projection,
          self._native_attn_norm_scale,
          self._native_attn_norm_offset,
          self._packed_single_q,
          self._native_q_bias,
          self._packed_single_k,
          self._packed_single_v,
          self._packed_single_g,
          self._packed_single_o,
          self._native_transition_norm_scale,
          self._native_transition_norm_offset,
          self._packed_single_transition1,
          self._packed_single_transition2,
      )
      projection_kwargs = {"hidden": self.num_intermediate}

    seq_mask = util.align_batch(seq_mask, Batch)

    region = (
        contextlib.nullcontext()
        if self.distributed
        else _native_pool_region("sparsefold_native::evoformer_single_act_update")
    )
    with region:
      output = fn(
          pair_act,
          single_act,
          seq_mask,
          self._native_pair_logits_norm_scale,
          self._native_pair_logits_norm_offset,
          *projection_args,
          **projection_kwargs,
      )

    return output
