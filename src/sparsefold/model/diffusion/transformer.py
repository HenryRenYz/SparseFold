# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Diffusion transformer model."""
from typing import Sequence, Optional
import torch
import einops

from sparsefold.weights import legacy_scopes as pv
from sparsefold.model import legacy_config as base_config
from sparsefold.model import config as model_config
from sparsefold.schema import atom_layout
from sparsefold.schema import layout
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import util
from sparsefold.runtime import compiler
from sparsefold.runtime.cache import ValueCache
from sparsefold.backends.torch import attention
from sparsefold.runtime import trace
from sparsefold.runtime import dist

_NUM_KEYS = layout.ATOM_ATTENTION_KEYS
_NUM_QUERIES = layout.ATOM_ATTENTION_QUERIES


class AdaptiveLayerNorm(fnn.Module):
  """Adaptive LayerNorm."""

  # Adopted from Scalable Diffusion Models with Transformers
  # https://arxiv.org/abs/2212.09748

  def __init__(
      self,
      hidden_size: int | Sequence[int],
      single_cond_size: int | Sequence[int] | None,
      *,
      dtype: torch.dtype | None = None,
      name: str
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype
    self.single_cond = single_cond_size is not None

    # ///// MODULE STRUCTURE ///// #
    if self.single_cond:
      # upcast to float32
      # No learneable parameter
      self.layer_norm = fnn.LayerNorm(
        hidden_size, elementwise_affine=False, dtype=torch.float32,
        name='layer_norm'
      )

      # upcast to float32
      self.single_cond_layer_norm = fnn.LayerNorm(
        single_cond_size, bias=False, dtype=torch.float32,
        name='single_cond_layer_norm'
      )
      # zero-initialized
      self.single_cond_scale = fnn.Linear(
        single_cond_size, hidden_size, bias=True, dtype=dtype,
        name='single_cond_scale'
      )
      # zero-initialized
      self.single_cond_bias = fnn.Linear(
        single_cond_size, hidden_size, bias=False, dtype=dtype,
        name='single_cond_bias'
      )
    else:
      # upcast to float32
      self.layer_norm = fnn.LayerNorm(
        hidden_size, dtype=torch.float32,
        name='layer_norm'
      )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='', **kwargs)

  def forward(
      self,
      x: torch.Tensor,  # (..., C)
      single_cond: torch.Tensor | None,  # (..., S)
  ):
    if self.single_cond:
      orig_type = x.dtype
      x = self.layer_norm(x)
      single_cond = self.single_cond_layer_norm(single_cond).to(self.dtype)
      single_scale = self.single_cond_scale(
          single_cond, post_op="sigmoid"
      ).to(orig_type)
      single_bias = self.single_cond_bias(single_cond).to(orig_type)
      x = single_scale * x + single_bias
    else:
      x = self.layer_norm(x)
    return x


class AdaptiveZeroInit(fnn.Module):
  """Adaptive zero init, from AdaLN-zero."""

  def __init__(
      self,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      hidden_size: int,
      single_cond_size: int | None,
      dtype: torch.dtype | None = None,
      name: str = ''
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype
    self.single_cond = single_cond_size is not None

    # ///// MODULE STRUCTURES ///// #
    if self.single_cond:
      self.transition2 = fnn.Linear(
        hidden_size, num_channels, bias=False, dtype=dtype,
        name='transition2'
      )
      # zero-init, bias-init=-2.0
      self.adaptive_zero_cond = fnn.Linear(
        single_cond_size, num_channels, bias=True, dtype=dtype,
        name='adaptive_zero_cond'
      )
    else:
      # initialize from global_config.final_init
      self.transition2 = fnn.Linear(
        hidden_size, num_channels, bias=False, dtype=dtype,
        name='transition2'
      )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='')

  def forward(
      self,
      x: torch.Tensor,  # (..., I)
      single_cond: torch.Tensor | None,  # (..., S)
  ):
    orig_type = x.dtype
    x = x.to(self.dtype)
    if self.single_cond:
      output = self.transition2(x)
      # Init to a small gain, sigmoid(-2) ~ 0.1
      cond = self.adaptive_zero_cond(
          single_cond.to(self.dtype), post_op="sigmoid"
      )
      output = cond * output
    else:
      output = self.transition2(x)
    return output.to(orig_type)


class TransitionBlock(fnn.Module):
  """Transition Block."""

  def __init__(
      self,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      num_intermediate_factor: int,
      single_cond_size: int | None = None,
      use_glu_kernel: bool = True,
      dtype: torch.dtype | None = None,
      name: str = ''
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype
    self.use_glu_kernel = use_glu_kernel

    num_intermediates = num_intermediate_factor * num_channels
    self.num_intermediates = num_intermediates

    # ///// MODULE STRUCTURE ///// #
    self.layer_norm = AdaptiveLayerNorm(
      hidden_size=num_channels,
      single_cond_size=single_cond_size,
      dtype=dtype,
      name='ffw_'
    )
    # ASC: ReLU initialized
    self.transition = fnn.Linear(
      num_channels, num_intermediates * 2, bias=False,
      dtype=dtype,
      name='ffw_transition1'
    )
    self.zero_init = AdaptiveZeroInit(
      global_config,
      num_channels=num_channels,
      hidden_size=num_intermediates,
      single_cond_size=single_cond_size,
      dtype=dtype,
      name='ffw_'
    )
    if use_glu_kernel:
      self.glu_kernel = fnn.GatedLinearUnit(
        fnn.Swish()
      )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='', **kwargs)

  def forward(
      self,
      x: torch.Tensor,  # (..., C)
      *,
      single_cond: torch.Tensor | None = None,  # (..., S)
  ) -> torch.Tensor:
    residual = x
    orig_dtype = x.dtype
    x = self.layer_norm(x, single_cond).to(self.dtype)

    if self.use_glu_kernel:
      # For PyTorch linear the weight is the transpose as the Jax's
      weights = self.transition.weight
      weights = einops.rearrange(weights, "n (i2 c) -> n i2 c", i2=2, c=self.num_intermediates)

      c = self.glu_kernel(x, weights)
    else:
      x = self.transition(x)
      a, b = einops.rearrange(x, "... (i2 c) -> i2 ... c")
      a, b = torch.tensor_split(x, 2, dim=-1)
      c = fnn.swish(a) * b

    output = self.zero_init(c.to(orig_dtype), single_cond)
    return residual + output


class SelfAttention(fnn.Module):
  """Multihead self-attention."""

  class Config(base_config.BaseConfig):
    num_head: int = 16
    key_dim: int | None = None
    value_dim: int | None = None

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      single_cond_size: int | None = None,
      dtype: torch.dtype | None = None,
      distributed: bool = False,
      name: str = ''
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype
    self.distributed = distributed and dist.is_enabled()

    # ///// MODULE STRUCTURES ///// #
    self.layer_norm = AdaptiveLayerNorm(
      num_channels, single_cond_size, dtype=dtype,
      name=''
    )

    # Sensible default for when the config keys are missing
    key_dim = config.key_dim if config.key_dim is not None else num_channels
    value_dim = config.value_dim if config.value_dim is not None else num_channels
    num_head = config.num_head
    assert key_dim % num_head == 0, f'{key_dim=} % {num_head=} != 0'
    assert value_dim % num_head == 0, f'{value_dim=} % {num_head=} != 0'
    self.key_dim = key_dim = key_dim // num_head
    self.value_dim = value_dim = value_dim // num_head

    qk_shape = (num_head, key_dim)
    self.q_projection = fnn.Linear(
      num_channels, qk_shape, bias=True, dtype=dtype,
      name='q_projection'
    )
    self.k_projection = fnn.Linear(
      num_channels, qk_shape, bias=False, dtype=dtype,
      name='k_projection'
    )

    v_shape = (num_head, value_dim)
    self.v_projection = fnn.Linear(
      num_channels, v_shape, bias=False, dtype=dtype,
      name='v_projection'
    )
    assert key_dim == value_dim

    # ASC: initializer=zeros, bias_init=1.0
    self.gating_query = fnn.Linear(
      num_channels, num_head * value_dim, bias=False,dtype=dtype,
      name='gating_query'
    )

    self.zero_init = AdaptiveZeroInit(
      global_config, num_channels=num_channels, hidden_size=num_channels,
      single_cond_size=single_cond_size,
      dtype=dtype,
      name=''
    )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='', **kwargs)

  def forward(
      self,
      x: torch.Tensor,  # (..., B, C)
      mask: torch.BoolTensor,  # (..., N)
      *,
      pair_logits: torch.Tensor | None = None,  # (..., H, B, N)
      single_cond: torch.Tensor | None = None,  # (..., B, S)
  ) -> torch.Tensor:
    residual = x
    orig_dtype = x.dtype

    x = self.layer_norm(x, single_cond).to(self.dtype)
    q = self.q_projection(x)
    k = self.k_projection(x)  # (*, B, H, D)
    v = self.v_projection(x)  # (*, B, H, D)

    if self.distributed:
      # (*, N, H, D)
      k = dist.current_shard.all_gather_into_tensor(
          input_tensor=k, dim=-3
      )
      # (*, N, H, D)
      v = dist.current_shard.all_gather_into_tensor(
          input_tensor=v, dim=-3
      )

    # In some situations the gradient norms can blow up without running this
    # einsum in float32.
    weighted_avg = attention.dot_product_attention(
        q, k, v, 
        mask_k=mask, 
        bias=pair_logits
    )
    
    weighted_avg = einops.rearrange(weighted_avg, "... b h d -> ... b (h d)")
    gate = self.gating_query(x.to(self.dtype), post_op="sigmoid")
    weighted_avg = weighted_avg * gate

    output = self.zero_init(weighted_avg, single_cond).to(orig_dtype)
    return residual + output


class Transformer(fnn.Module):
  """Simple transformer stack."""

  class Config(base_config.BaseConfig):
    attention: SelfAttention.Config = base_config.autocreate()
    num_blocks: int = 24
    block_remat: bool = False
    super_block_size: int = 4
    num_intermediate_factor: int = 2

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      single_cond_size: int,
      pair_cond_size: int | None,
      dtype: torch.dtype | None = None,
      name: str = 'transformer',
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype
    self.config = config

    # ///// MODULE STRUCTURES ///// #
    assert config.num_blocks % config.super_block_size == 0
    num_super_blocks = config.num_blocks // config.super_block_size

    self.self_attention = fnn.LayerStack(
      [fnn.LayerStack(
        [SelfAttention(
          config.attention, global_config,
          num_channels=num_channels,
          single_cond_size=single_cond_size,
          dtype = dtype,
          name=name
        ) for _ in range(config.super_block_size)],
        with_per_layer_inputs=True
      ) for _ in range(num_super_blocks)],
      with_per_layer_inputs=True
    )
    self.transition_block = fnn.LayerStack(
      [fnn.LayerStack(
        [TransitionBlock(
          global_config,
          num_channels=num_channels,
          num_intermediate_factor=config.num_intermediate_factor,
          single_cond_size=single_cond_size,
          dtype = dtype,
          name=name
        ) for _ in range(config.super_block_size)],
        with_per_layer_inputs=True
      ) for _ in range(num_super_blocks)],
      with_per_layer_inputs=True
    )
    self.pair_logits_projection = fnn.LayerStack(
      [fnn.Linear(
        pair_cond_size, (config.super_block_size, config.attention.num_head),
        bias=False,
        name='pair_logits_projection',
      ) for _ in range(num_super_blocks)],
      with_per_layer_inputs=True
    )

    if pair_cond_size is not None:
      self.pair_input_layer_norm = fnn.LayerNorm(
        pair_cond_size, bias=False,
        name='pair_input_layer_norm',
      )

    self._cache = ValueCache()

  def forward(
      self,
      act: torch.Tensor,  # (..., N, C)
      mask: torch.Tensor,  # (N,)
      single_cond: torch.Tensor,  # (N, S); constant across intra-super-blocks
      pair_cond: torch.Tensor | None = None,  # (N, N, P); constant across denoising loops
  ) -> torch.Tensor:
    # Precompute pair logits for performance
    if pair_cond is not None:
      def _compute_cache():
        pair_act = self.pair_input_layer_norm(pair_cond)
        pack = []
        for projection in self.pair_logits_projection:
          pair_logits = projection(pair_act)
          pair_logits = einops.rearrange(pair_logits, "b n s h -> s h b n", s=self.config.super_block_size)
          pack.append(pair_logits)
        return torch.stack(pack).to(self.dtype).contiguous()

      # ``pair_cond`` may be re-cast to a fresh Tensor object at every
      # denoising step even though its value is invariant. Model.forward resets
      # ValueCache once per example, so key this invariant by purpose instead
      # of object identity; otherwise all six FP32 projections are recomputed
      # for every diffusion step.
      pair_logits_pack = self._cache.get(
          name="pair_logits", factory=_compute_cache
      )
    else:
      pair_logits_pack = None

    def super_block_fn(mod, act_: torch.Tensor, pair_logits: torch.Tensor | None = None):
      @compiler.mark_compile
      def block_fn(mod, act__: torch.Tensor, pair_logits_: Optional[torch.Tensor] = None):
        self_attention, transition_block = mod
        act__ = self_attention(
            act__,
            mask,
            pair_logits=pair_logits_,
            # TODO batch compute, TODO analyze compute cost (is it worth extract out to make a batch compute?)
            single_cond=single_cond,  # TODO batch compute
        )
        act__ = transition_block(
            act__, 
            single_cond=single_cond, 
        )
        return act__

      act_ = fnn.layer_stack(
          block_fn, mod, with_per_layer_inputs=True
      )(act_, pair_logits)
      return act_
    
    input_dtype = act.dtype
    with trace.trace(enable=False):
      return fnn.layer_stack(
          super_block_fn, (self.self_attention, self.transition_block),  with_per_layer_inputs=True
      )(act.to(self.dtype), pair_logits_pack).to(input_dtype)


class CrossAttention(fnn.Module):
  class Config(base_config.BaseConfig):
    num_head: int = 4
    key_dim: int = 128
    value_dim: int = 128

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,  # C
      dtype: torch.dtype | None = None,
      name: str
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype

    # ///// MODULE STRUCTURE ///// #
    self.q_layernorm = AdaptiveLayerNorm(
      num_channels, num_channels, dtype=dtype,
      name='q'
    )
    self.k_layernorm = AdaptiveLayerNorm(
      num_channels, num_channels, dtype=dtype,
      name='k'
    )

    assert config.key_dim % config.num_head == 0
    assert config.value_dim % config.num_head == 0
    key_dim = config.key_dim // config.num_head
    value_dim = config.value_dim // config.num_head
    self.key_dim = key_dim
    self.value_dim = value_dim

    self.q_projection = fnn.Linear(
      num_channels, (config.num_head, key_dim), bias=True, 
      dtype=dtype,
      name='q_projection'
    )
    self.k_projection = fnn.Linear(
      num_channels, (config.num_head, key_dim), bias=False,
      dtype=dtype,
      name='k_projection'
    )
    self.v_projection = fnn.Linear(
      num_channels, (config.num_head, value_dim), bias=False,
      dtype=dtype,
      name='v_projection'
    )
    assert key_dim == value_dim

    # ASC: zero-initialized
    self.gating_query = fnn.Linear(
      num_channels, config.num_head * value_dim, bias=False,
      dtype=dtype,
      name='gating_query'
    )
    self.zero_init = AdaptiveZeroInit(
      global_config, num_channels=num_channels,
      hidden_size=num_channels, single_cond_size=num_channels,
      dtype=dtype,
      name=''
    )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='', **kwargs)

  def forward(
      self,
      x_q: torch.Tensor,  # (..., Q, C)
      x_k: torch.Tensor,  # (..., K, C)
      mask_q: torch.Tensor,  # (..., Q)
      mask_k: torch.Tensor,  # (..., K)
      pair_logits: torch.Tensor | None = None,  # (..., H, Q, K)
      single_cond_q: torch.Tensor | None = None,  # (..., Q, C)
      single_cond_k: torch.Tensor | None = None,  # (..., K, C)
  ) -> torch.Tensor:
    residual = x_q
    orig_dtype = x_q.dtype

    x_q = self.q_layernorm(x_q, single_cond_q).to(self.dtype)
    x_k = self.k_layernorm(x_k, single_cond_k).to(self.dtype)

    q = self.q_projection(x_q)
    k = self.k_projection(x_k)
    v = self.v_projection(x_k)

    # In some situations the gradient norms can blow up without running this
    # einsum in float32.
    weighted_avg = attention.dot_product_attention(
        q, k, v, 
        mask_q=mask_q, 
        mask_k=mask_k, 
        bias=pair_logits
    ).to(orig_dtype)

    weighted_avg = einops.rearrange(weighted_avg, "... h d -> ... (h d)")
    gate = self.gating_query(
        x_q.to(self.dtype), post_op="sigmoid"
    ).to(orig_dtype)
    weighted_avg = weighted_avg * gate

    output = self.zero_init(weighted_avg, single_cond_q)
    return residual + output


class CrossAttTransformer(fnn.Module):
  """Transformer that applies cross attention between two sets of subsets."""

  class Config(base_config.BaseConfig):
    num_intermediate_factor: int
    num_blocks: int
    attention: CrossAttention.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      pair_cond_size: int,
      num_channels: int,
      super_name: str,  # stupid dump naming
      dtype: torch.dtype | None = None,
      name: str = 'transformer',
  ):
    super().__init__()
    self.name = name
    self.dtype = dtype

    # ///// MODULE STRUCTURE ///// #
    self.pair_input_layer_norm = fnn.LayerNorm(
      pair_cond_size, elementwise_affine=True, bias=False,
      name='pair_input_layer_norm'
    )
    self.pair_logits_projection = fnn.Linear(
      pair_cond_size, (config.num_blocks, config.attention.num_head),
      bias=False,
      name='pair_logits_projection'
    )

    self.cross_attention = fnn.LayerStack(
      [CrossAttention(
        config.attention, global_config,
        num_channels=num_channels,
        dtype = dtype,
        name=f"{super_name}_{name}"
      ) for _ in range(config.num_blocks)],
      with_per_layer_inputs=True
    )
    self.transition_block = fnn.LayerStack(
      [TransitionBlock(
        global_config, num_channels=num_channels,
        num_intermediate_factor=config.num_intermediate_factor,
        single_cond_size=num_channels,  # TODO
        dtype = dtype,
        name=f"{super_name}_{name}"
      ) for _ in range(config.num_blocks)],
      with_per_layer_inputs=True
    )
    
    self._cache = ValueCache()

  def forward(
      self,
      queries_act: torch.Tensor,  # (..., num_subsets, num_queries, ch)
      # The only variable between denoising loops in the diffusion head.
      queries_mask: torch.Tensor,  # (num_subsets, num_queries)
      queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
      keys_mask: torch.Tensor,  # (num_subsets, num_keys)
      queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
      keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
      pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
  ) -> torch.Tensor:
    def _compute_cache():
      # Precompute pair logits for performance
      pair_act = self.pair_input_layer_norm(pair_cond)
      # (num_subsets, num_queries, num_keys, num_blocks, num_heads)
      pair_logits = self.pair_logits_projection(pair_act)
      # (num_block, num_subsets, num_heads, num_queries, num_keys)
      pair_logits = einops.rearrange(pair_logits, "s q k b h -> b s h q k")
      return pair_logits.contiguous()  # requires contiguous
    
    pair_logits = self._cache.get(
        pair_cond, name="pair_logits", factory=_compute_cache
    )

    @compiler.mark_compile
    def block(mod, queries_act: torch.Tensor, pair_logits: torch.Tensor):
      cross_attention, transition_block = mod
      keys_act = atom_layout.convert(
          queries_to_keys, queries_act, layout_axes=(-3, -2)
      )
      # cross attention
      queries_act = cross_attention(
          x_q=queries_act,
          x_k=keys_act,
          mask_q=queries_mask,
          mask_k=keys_mask,
          pair_logits=pair_logits,
          single_cond_q=queries_single_cond,
          single_cond_k=keys_single_cond,
      )
      queries_act = transition_block(
          queries_act,
          single_cond=queries_single_cond,
      )
      return queries_act

    with trace.trace(enable=False):
      return fnn.layer_stack(
          block, (self.cross_attention, self.transition_block), with_per_layer_inputs=True
      )(queries_act, pair_logits)
