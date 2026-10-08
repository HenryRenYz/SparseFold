# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Haiku modules for the Diffuser model."""

from typing import Literal, List, Tuple, Optional, Mapping, Callable, TypeAlias, Sequence, Any
from functools import partial
import dataclasses
import einops
import torch
import torch.nn

from sparsefold.model import legacy_config as base_config
from sparsefold.model import config as model_config
from sparsefold.model.diffusion import transformer as diffusion_transformer
from sparsefold.model.layers import mapping
from sparsefold.backends.torch import attention
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import util
from sparsefold.runtime import config
from sparsefold.runtime import trace
from sparsefold.runtime.cache import ValueCache
from sparsefold.runtime import dist


def get_shard_size(
    num_residues: int, shard_spec: List[Tuple[Optional[int], Optional[int]]]
) -> Optional[int]:
  shard_size = shard_spec[0][-1]
  for num_residues_upper_bound, num_residues_shard_size in shard_spec:
    shard_size = num_residues_shard_size
    if num_residues_upper_bound is None or num_residues <= num_residues_upper_bound:
      break
  return shard_size


class TransitionBlock(fnn.Module):
  """Transition block for transformer."""

  class Config(base_config.BaseConfig):
    num_intermediate_factor: int = 4
    use_glu_kernel: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      name: str,
      **kwargs,
  ):
    super().__init__()
    self.name = name
    self.num_channels = num_channels
    self.num_intermediate_factor = config.num_intermediate_factor
    self.use_glu_kernel = config.use_glu_kernel

    self.num_intermediate = int(num_channels * config.num_intermediate_factor)

    # ///// MODULE STRUCTURES ///// #
    self.input_layer_norm = fnn.LayerNorm(
      num_channels,
      name="input_layer_norm",
    )
    # ASC: initializer=relu
    self.transition1 = fnn.Linear(
      num_channels, self.num_intermediate * 2, name="transition1"
    )
    # ASC: initializer=self.global_config.final_init
    self.transition2 = fnn.Linear(
      self.num_intermediate, num_channels, name="transition2"
    )

    if config.use_glu_kernel:
      self.glu_kernel = fnn.GatedLinearUnit(fnn.Swish())

  def forward(self, act: torch.Tensor, *, broadcast_dim: int = 0):
    num_channels = act.shape[-1]

    num_intermediate = int(num_channels * self.num_intermediate_factor)

    old_act = act
    act = self.input_layer_norm(act)

    if self.use_glu_kernel:
      weights = self.transition1.weight
      weights = einops.rearrange(weights, "a (i2 n) -> a i2 n", i2=2, n=num_intermediate)
      c = self.glu_kernel(act, weights)
    else:
      act = self.transition1(act)
      a, b = einops.rearrange(act, "... (i2 c) -> i2 ... c")
      c = fnn.swish(a) * b

    return self.transition2(c, residual=old_act)


class MSAAttention(fnn.Module):
  """MSA Attention."""

  class Config(base_config.BaseConfig):
    num_head: int = 8

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      num_output_channel: int,
      distributed: bool = False,
      name: str,
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.num_channels = num_channels
    self.num_output_channel = num_output_channel

    # ///// MODULE STRUCTURES ///// #
    self.act_norm = fnn.LayerNorm(
      num_channels,
      name="act_norm",
    )
    self.pair_norm = fnn.LayerNorm(
      num_output_channel,
      name="pair_norm",
    )
    self.pair_logits = fnn.Linear(
      num_output_channel, config.num_head, bias=False, name="pair_logits"
    )

    value_dim = num_channels // config.num_head
    self.v_projection = fnn.Linear(
      num_channels, (config.num_head, value_dim), bias=False, name="v_projection"
    )
    # ASC: initializer=zeros, bias_init=1.0
    self.gating_query = fnn.Linear(
      num_channels,
      config.num_head
      * value_dim,  # might not equals to num_channels. However in weight file these equals
      bias=False,
      name="gating_query",
    )
    # ASC: initializer=self.global_config.final_init
    self.output_projection = fnn.Linear(
      config.num_head * value_dim,
      num_channels,
      bias=False,
      name="output_projection",
    )
    self._cache = ValueCache()

  def forward(
      self,
      act: torch.Tensor,  # (M, B, C)
      mask: torch.Tensor,  # (M, N)
      pair_act: torch.Tensor,  # (B, N, C_2)
  ):
    act = self.act_norm(act)
    pair_act = self.pair_norm(pair_act)
    logits = self.pair_logits(pair_act)
    logits = einops.rearrange(logits, "... b n h -> ... h b n")
    
    def _compute_bias():
      return 1e9 * (torch.max(mask, dim=0)[0].bfloat16() - 1.0)

    logits += self._cache.get(mask, name="bias", factory=_compute_bias)
    weights = torch.softmax(logits, dim=-1)

    # (M, B, H, D)
    v = self.v_projection(act)
    if self.distributed:
      # (M, N, H, D)
      v = dist.current_shard.all_gather_into_tensor(
          input_tensor=v, dim=-3
      )
    v_avg = einops.einsum(weights, v, "... h q k, ... b k h d -> ... b q h d")
    v_avg = einops.rearrange(v_avg, "... h d -> ... (h d)")

    gate_values = self.gating_query(act, post_op="sigmoid")
    v_avg *= gate_values

    return self.output_projection(v_avg)


class GridSelfAttention(fnn.Module):
  """Self attention that is either per-sequence or per-residue."""

  class Config(base_config.BaseConfig):
    num_head: int = 4

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      transpose: bool,
      *,
      num_channels: int,
      distributed: bool = False,
      name: str,
      **kwargs,
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.transpose = transpose
    self.config = config
    self.num_channels = num_channels
    self.pair_attention_chunk_size = global_config.pair_attention_chunk_size

    # ///// MODULE STRUCTURES ///// #
    assert num_channels % config.num_head == 0
    # Triton requires a minimum dimension of 16 for doing matmul.
    qkv_dim = max(num_channels // config.num_head, 16)
    self.qkv_dim = qkv_dim

    qkv_shape = (config.num_head, qkv_dim)
    # (H, D, C)
    self.q_projection = fnn.Linear(
      num_channels,
      qkv_shape,
      bias=False,
      transpose_weights=True,
      name="q_projection",
    )
    # (H, D, C)
    self.k_projection = fnn.Linear(
      num_channels,
      qkv_shape,
      bias=False,
      transpose_weights=True,
      name="k_projection",
    )
    # (C, H, D)
    self.v_projection = fnn.Linear(
      num_channels, qkv_shape, bias=False, name="v_projection"
    )

    # ASC: initializer=zeros, bias_init=1.0
    # (H * D, C)
    self.gating_query = fnn.Linear(
      num_channels,
      config.num_head * qkv_dim,
      bias=False,
      transpose_weights=True,
      name="gating_query",
    )

    # ASC: initializer=self.global_config.final_init
    # (H * D, C)
    self.output_projection = fnn.Linear(
      config.num_head * qkv_dim,
      num_channels,
      bias=False,
      name="output_projection",
    )

    self.act_norm = fnn.LayerNorm(
      num_channels,
      bias=False,
      name="act_norm",
    )
    # (C, H)
    self.pair_bias_projection = fnn.Linear(
      num_channels, config.num_head, bias=False, name="pair_bias_projection"
    )

  def _attention(
      self,
      act: torch.Tensor,  # (*B, N, C)
      pair_mask: torch.Tensor,  # (B, N), bool
      bias: torch.Tensor,  # (*#B, #H, #N, #N)
  ):
    q = self.q_projection(act)
    k = self.k_projection(act)
    v = self.v_projection(act)

    # Filters (:N, :, :, :N), where for first N elems in bias, only key is filtered
    weighted_avg = attention.dot_product_attention(
        q, k, v, 
        mask_k=pair_mask, 
        bias=bias
    )
    weighted_avg = einops.rearrange(weighted_avg, "... h d -> ... (h d)")

    gate_values = self.gating_query(act, post_op="sigmoid")
    weighted_avg *= gate_values

    return self.output_projection(weighted_avg)

  def forward(self, act: torch.Tensor, seq_mask: torch.Tensor):
    """Builds a module.

    Arguments:
      act: [*, shard_size, num_res, channels] activations tensor
      seq_mask: [num_res] mask of non-padded regions in the tensor.
        Only used in inducing points attention currently.

    Returns:
      The new activations, that is the act + GridSelfAttention(act, mask)
    """
    assert len(act.shape) >= 3
    num_res = act.shape[-2]
    assert act.shape[-1] == self.num_channels
    assert seq_mask.shape == (num_res,) and seq_mask.dtype == torch.bool

    if self.distributed:
      seq_mask_shard = dist.current_shard.shard(seq_mask, dim=0)
    else:
      seq_mask_shard = seq_mask
    # (B, N)
    pair_mask = (seq_mask_shard[:, None] & seq_mask[None, :])
    # (*, B, N, C)
    old_act = act
    act = self.act_norm(act)

    # (*, B, N, H)
    nonbatched_bias = self.pair_bias_projection(act)
    if self.distributed:
      # (*, N, N, H)
      nonbatched_bias = dist.current_shard.all_gather_into_tensor(
          input_tensor=nonbatched_bias, dim=-3
      )
    # Intersect with Batch dim.
    # (*, 1, H, N, N)
    # Dot product attention requires the bias term to have a batch dimension.
    nonbatched_bias = einops.rearrange(nonbatched_bias, "... b n h -> ... 1 h b n").contiguous()

    chunk_size = get_shard_size(
        num_res, self.pair_attention_chunk_size
    )

    def reshard(act_: torch.Tensor):
      if self.distributed:
        # (*, N, B, C)
        act_ = dist.current_shard.reshard(
            input=act_, dim1=-3, dim2=-2
        )
      # (*, N, B, C) transpose to (*, B, N, C)
      act_ = torch.transpose(act_, -2, -3)
      return act_

    if self.transpose:
      act = reshard(act)

    n_batch_dim = act.ndim - 3
    if n_batch_dim > 0:
      pair_mask = util.align_batch(pair_mask, act.shape[:-3])

    attention = partial(self._attention)
    # (B, *, a, N, C)
    act = mapping.inference_subbatch(
        attention,
        chunk_size,
        batched_args=[act, pair_mask],
        nonbatched_args=[nonbatched_bias],
        input_subbatch_dim=n_batch_dim,
        input_subbatch_dim_is_partitioned=False,
    )

    if self.transpose:
      act = reshard(act)

    return old_act + act

IndexReusePolicy: TypeAlias = Callable[[util.Slice], util.Slice | None]

def _normalize_slice(slice: util.Slice):
    if isinstance(slice, Sequence):
      return tuple(int(x) for x in slice)
    else:
      return (int(slice),)

def default_reuse_policy(slice: util.Slice) -> util.Slice | None:
  return None  # no reuse

class IndexReuser:
  def __init__(
      self,
      policy: IndexReusePolicy
  ):
    self.policy = policy
    self.cached_indices: dict[util.Slice, Any] = {}

  @torch.compiler.disable
  def get(self, slice: util.Slice) -> tuple[Any | None, util.Slice | None]:
    """
    :param args: guard variable for `self.cache`
    """
    if not config.enable_gsa_index_reuse:
      return None, None
    slice = _normalize_slice(slice)
    target_index = self.policy(slice)
    if target_index is None:
      return None, None
    return self.cached_indices.get(target_index), target_index
  
  @torch.compiler.disable
  def set(self, slice: util.Slice, masks: Any):
    slice = _normalize_slice(slice)
    self.cached_indices[slice] = masks

@dataclasses.dataclass(frozen=True)
class IBSAParameters:
  """
  Parameters for Index-based Sparse Attention (IBSA)
  """
  num_samples: int = 16
  num_rand_blocks: int = 6
  t_prob: float = 0.90
  tr_prob: float = 0.95
  u_prob: float = 0.02

  @classmethod
  def default(cls, *args, **kwargs) -> "IBSAParameters":
    return cls(**config.sparse_attention_parameters)
    

IBSAParametersProvider: TypeAlias = Callable[[util.Slice], IBSAParameters]
IBSAParametersOrProvider = IBSAParameters | IBSAParametersProvider


class TriangleMultiplication(fnn.Module):
  """Triangle Multiplication."""

  class Config(base_config.BaseConfig):
    equation: Literal["...ikc,...jkc->...ijc", "...kjc,...kic->...ijc"]
    use_glu_kernel: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      distributed: bool=False,
      name: str,
      **kwargs,
  ):
    super().__init__()
    self.name = name
    self.num_channels = num_channels
    self.distributed = distributed and dist.is_enabled()

    equation = {
      "...ikc,...jkc->...ijc": "...cik,...cjk->...cij",
      "...kjc,...kic->...ijc": "...ckj,...cki->...cij",
    }[config.equation]
    self.equation = equation
    self.use_glu_kernel = config.use_glu_kernel

    # ///// MODULE STRUCTURE ///// #
    self.left_norm_input = fnn.LayerNorm(
      num_channels,
      bias=True,
      name="left_norm_input",
    )
    self.projection = fnn.Linear(
      num_channels, num_channels * 2, bias=False, name="projection"
    )
    # ASC: initializer=self.global_config.final_init
    self.gate_projection = fnn.Linear(
      num_channels, num_channels * 2, bias=False, name="gate"
    )
    self.center_norm = fnn.LayerNorm(
      num_channels,
      bias=True,
      dim=-3,
      name="center_norm",
    )
    # ASC: initializer=self.global_config.final_init
    self.output_projection = fnn.Linear(
      num_channels, num_channels, bias=False, name="output_projection"
    )
    # ASC: initializer=self.global_config.final_init
    self.gating_linear = fnn.Linear(
      num_channels, num_channels, bias=False, name="gating_linear"
    )
    if config.use_glu_kernel:
      self.glu_kernel = fnn.GatedLinearUnit(torch.nn.Sigmoid())

  def forward(
      self,
      act: torch.Tensor,  # (*, B, N, C)
      mask: torch.Tensor,  # (N,), bool
  ):
    """Applies Module.

    Args:
      act: The activation.
      mask: The mask.

    Returns:
      act + TriangleMultiplication(act)
    """
    # (1, B, N)
    if self.distributed:
      mask_shard = dist.current_shard.shard(mask, dim=0)  # (B,)
    else:
      mask_shard = mask
    mask = (mask_shard[:, None] & mask[None, :])[None]
    C = self.num_channels
    # (B, N, C)
    original_act = act
    act = self.left_norm_input(act)
    input_act = act

    # projection = sigmoid(act @ Wg) * (act @ Wp)
    if self.use_glu_kernel:
      # (C, C * 2)
      weights_projection = self.projection.weight
      # (C, C * 2)
      weights_gate = self.gate_projection.weight
      # (C, 2, C * 2)
      weights_glu = torch.stack((weights_gate, weights_projection), dim=1)

      # (B, N, C * 2)
      projection = self.glu_kernel(act, weights_glu)
      # (C * 2, B, N)
      projection = einops.rearrange(projection, "... b n c2 -> ... c2 b n")
      projection = projection.masked_fill_(~mask, 0.0)
    else:
      # (B, N, C * 2)
      projection = self.projection(act)
      # (C * 2, B, N)
      projection = einops.rearrange(projection, "... b n c2 -> ... c2 b n")
      projection = projection.masked_fill_(~mask, 0.0)

      # (B, N, C * 2)
      gate = self.gate_projection(act, post_op="sigmoid")
      # (C * 2, B, N)
      gate = einops.rearrange(gate, "... b n c2 -> ... c2 b n")
      projection = projection * gate

    # (C, B, N)
    a, b = einops.rearrange(projection, "... (c i2) b n -> i2 ... c b n", i2=2)

    # TODO DAP
    if self.distributed:
      if self.equation == "...cik,...cjk->...cij":
        # (C, N, N)
        b = dist.current_shard.all_gather_into_tensor(
            input_tensor=b, dim=-2
        )
      elif self.equation == "...ckj,...cki->...cij":
        # (C, N, N)
        a = dist.current_shard.all_gather_into_tensor(
            input_tensor=a, dim=-2
        )
        # (C, N, B)
        b = dist.current_shard.reshard(
            input=b, dim1=-2, dim2=-1
        )

    # (C, B, N)
    act = torch.einsum(self.equation, a, b)
    act = self.center_norm(act)

    # (B, N, C)
    act = einops.rearrange(act, "... c b n -> ... b n c")
    act = self.output_projection(act)
    gate_out = self.gating_linear(input_act, post_op="sigmoid")
    act *= gate_out
    return original_act + act
  

class OuterProductMean(fnn.Module):
  """Computed mean outer product."""

  class Config(base_config.BaseConfig):
    chunk_size: int = 128
    num_outer_channel: int = 32

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      num_output_channel: int,
      distributed: bool = False,
      name: str,
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.num_channels = num_channels
    self.num_output_channel = num_output_channel
    self.chunk_size = config.chunk_size

    # ///// MODULE STRUCTURES ///// #
    self.layer_norm_input = fnn.LayerNorm(
      num_channels,
      name="layer_norm_input",
    )
    # ASC: initializer=linear
    self.left_projection = fnn.Linear(
      num_channels, config.num_outer_channel, name="left_projection"
    )
    # ASC: initializer=linear
    self.right_projection = fnn.Linear(
      num_channels, config.num_outer_channel, name="right_projection"
    )
    self._cache = ValueCache()

  def set_parameters(self, visitor, **kwargs):
    expected_bias = (self.num_output_channel,)
    outer = self.left_projection.output_shape
    expected_weight = (*outer, *outer, self.num_output_channel)
    for key, expected in (("output_b", expected_bias), ("output_w", expected_weight)):
      actual = tuple(visitor[key].shape)
      if actual != expected:
        raise ValueError(f"{self.name}/{key}: expected {expected}, got {actual}")
    super().set_parameters(visitor, **kwargs)
    # (self.num_output_channel,)
    self.output_b = torch.nn.Parameter(
      torch.as_tensor(visitor["output_b"], device=util.DEFAULT_DEVICE),
      requires_grad=False,
    )
    # (config.num_outer_channel,
    #  config.num_outer_channel,
    #  self.num_output_channel)
    self.output_w = torch.nn.Parameter(
      torch.as_tensor(visitor["output_w"], device=util.DEFAULT_DEVICE),
      requires_grad=False,
    )

  def forward(
      self, 
      act: torch.Tensor,  # (M, B, C)
      mask: torch.Tensor,  # (M, N), bool
  ):
    # (M, N, 1)
    mask = mask[..., None]
    # (M, B, 1)
    if self.distributed:
      mask_shard = dist.current_shard.shard(mask, dim=1)
    else:
      mask_shard = mask
    act = self.layer_norm_input(act)

    left_act = self.left_projection(act)
    left_act = left_act.masked_fill_(~mask_shard.bool(), 0.0)
    right_act = self.right_projection(act)
    right_act = right_act.masked_fill_(~mask_shard.bool(), 0.0)

    if self.distributed:
      right_act = dist.current_shard.all_gather_into_tensor(
          input_tensor=right_act, dim=-2
      )

    # if self.global_config.final_init == 'zeros':
    #   w_init = hk.initializers.Constant(0.0)
    # else:
    #   w_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in')

    output_w = self.output_w.to(act.dtype)
    output_b = self.output_b.to(act.dtype)

    def _compute_chunk(
        left_act: torch.Tensor,  # (M, b, D1)
        right_act: torch.Tensor,  # (M, N, D2)
    ):
      # (N, D1, D2, b)
      act = einops.einsum(left_act, right_act, "m b d1, m n d2 -> n d1 d2 b")
      # (b, N, C_2)
      act = einops.einsum(act, output_w, "n d1 d2 b, d1 d2 c -> b n c") + output_b
      return act

    # (B, N, C_2)
    act = mapping.inference_subbatch(
        _compute_chunk,
        self.chunk_size,
        batched_args=[left_act],
        nonbatched_args=[right_act],
        input_subbatch_dim=1,
        output_subbatch_dim=0,
        input_subbatch_dim_is_partitioned=False,
    )
    
    def _compute_norm():  # (B, N, 1)
      epsilon = 1e-3
      norm = einops.einsum(mask_shard.bfloat16(), mask.bfloat16(), "m b c, m n c -> b n c")
      return 1 / (epsilon + norm)
    
    norm = self._cache.get(mask, name="norm", factory=_compute_norm)
    return act * norm


class SingleActUpdate(fnn.Module):
  def __init__(
      self,
      global_config: model_config.GlobalConfig,
      num_channels: int,
      seq_channels: int,
      attn_config: diffusion_transformer.SelfAttention.Config,
      transition_config: "TransitionBlock.Config",
      distributed: bool = False,
      **kwargs,
  ):
    super().__init__()
    self.name = "<none>"
    self.distributed = distributed and dist.is_enabled()
    self.num_channels = num_channels
    self.seq_channels = seq_channels
    self.num_head = attn_config.num_head
    self.qkv_dim = seq_channels // self.num_head
    self.num_intermediate = int(
      seq_channels * transition_config.num_intermediate_factor
    )

    self.single_pair_logits_norm = fnn.LayerNorm(
      num_channels,
      name="single_pair_logits_norm",
    )
    self.single_pair_logits_projection = fnn.Linear(
      num_channels,
      attn_config.num_head,
      name="single_pair_logits_projection",
    )

    self.single_attention = diffusion_transformer.SelfAttention(
      attn_config,
      global_config,
      num_channels=seq_channels,
      distributed=self.distributed,
      name="single_attention_",
    )

    self.single_transition = TransitionBlock(
      transition_config,
      global_config,
      num_channels=seq_channels,
      name="single_transition",
    )

  def forward(
      self,
      pair_act: torch.Tensor,  # (*, B, N, C)
      single_act: torch.Tensor,  # (*, B, S)
      seq_mask: torch.Tensor,  # (*, N)
  ):
    pair_logits = self.single_pair_logits_norm(pair_act)
    # (*, B, N, H)
    pair_logits = self.single_pair_logits_projection(pair_logits)
    # (*, H, B, N)
    pair_logits = einops.rearrange(pair_logits, "... b n h -> ... h b n").contiguous()

    # single_act: (B, S)
    # q: (B, C_qk), k: (B, C_qk), (N, C_qk), s: (B, N)
    # SelfAttention owns its residual connection. Adding single_act here as
    # well doubles the residual once per Pairformer layer and makes the trunk
    # grow exponentially (48 layers are enough to reach ~1e18).
    single_act = self.single_attention(
        single_act, seq_mask, 
        pair_logits=pair_logits, 
    )
    single_act = self.single_transition(
        single_act, broadcast_dim=None, 
    )
    return single_act


class PairFormerIteration(fnn.Module):
  """Single Iteration of Pair Former."""

  __version__ = "1.0"

  class Config(base_config.BaseConfig):
    """Config for PairFormerIteration."""

    num_layer: int
    pair_attention: "GridSelfAttention.Config" = base_config.autocreate()
    pair_transition: "TransitionBlock.Config" = base_config.autocreate()
    single_attention: diffusion_transformer.SelfAttention.Config | None = None
    single_transition: "TransitionBlock.Config | None" = None
    triangle_multiplication_incoming: "TriangleMultiplication.Config" = (
      base_config.autocreate(equation="...kjc,...kic->...ijc")
    )
    triangle_multiplication_outgoing: "TriangleMultiplication.Config" = (
      base_config.autocreate(equation="...ikc,...jkc->...ijc")
    )
    shard_transition_blocks: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      with_single = False,
      *,
      num_channels: int,
      seq_channels: int | None = None,
      index_reuse_policy: IndexReusePolicy = default_reuse_policy,
      ibsa_params: IBSAParametersOrProvider = IBSAParameters.default,
      distributed: bool = False,
      name: str = "pairformer_stack",
  ):
    super().__init__()
    self.name = name
    self.with_single = with_single
    self.distributed = distributed and dist.is_enabled()
    if with_single:
      assert config.single_attention is not None
    self.shard_transition_blocks = config.shard_transition_blocks
    self.pair_transition_shard_spec = global_config.pair_transition_shard_spec

    # ///// MODULE STRUCTURES ///// #
    self.triangle_multiplication_outgoing = TriangleMultiplication(
      config.triangle_multiplication_outgoing,
      global_config,
      num_channels=num_channels,
      distributed=self.distributed,
      name="triangle_multiplication_outgoing",
    )
    self.triangle_multiplication_incoming = TriangleMultiplication(
      config.triangle_multiplication_incoming,
      global_config,
      num_channels=num_channels,
      distributed=self.distributed,
      name="triangle_multiplication_incoming",
    )
    self.pair_attention1 = GridSelfAttention(
      config.pair_attention,
      global_config,
      num_channels=num_channels,
      transpose=False,
      index_reuse_policy=index_reuse_policy,
      ibsa_params=ibsa_params,
      distributed=self.distributed,
      name="pair_attention1",
    )
    self.pair_attention2 = GridSelfAttention(
      config.pair_attention,
      global_config,
      num_channels=num_channels,
      transpose=True,
      index_reuse_policy=index_reuse_policy,
      ibsa_params=ibsa_params,
      distributed=self.distributed,
      name="pair_attention2",
    )
    self.pair_transition = TransitionBlock(
      config.pair_transition,
      global_config,
      num_channels=num_channels,
      name="pair_transition",
    )

    if with_single:
      assert seq_channels is not None
      self.single_act_update = SingleActUpdate(
        global_config,
        num_channels,
        seq_channels,
        config.single_attention,
        config.single_transition,
        distributed=self.distributed,
      )

  def forward(
      self,
      # may have batch dim
      act: torch.Tensor,  # (*, B, N, C)
      *,
      seq_mask: torch.Tensor,  # (*, N,), bool
      single_act: torch.Tensor | None = None,
  ):
    """Build a single iteration of the pair former.

    Args:
      act: [[batch], num_res, num_res, num_channel] Input pairwise activations.
      seq_mask: [num_res] Padding mask.
      single_act: [[batch], num_res, single_channel] Single Input activations, optional

    Returns:
      [num_res, num_res, num_channel] tensor of activations.
    """

    num_residues = act.shape[-3]

    with trace.into("trig_mul"):
      act = self.triangle_multiplication_outgoing(act, seq_mask)
      act = self.triangle_multiplication_incoming(act, seq_mask)

    with trace.into("pair_attn"):
      act = self.pair_attention1(act, seq_mask)
      act = self.pair_attention2(act, seq_mask)

    with trace.into("pair_transition"):
      transition_block = partial(self.pair_transition)
      if self.shard_transition_blocks:
        transition_block = mapping.sharded_apply(
            transition_block,
            get_shard_size(
                num_residues, self.pair_transition_shard_spec
            ),
            in_axes=-3, out_axes=-3,  # in case a batching dim present
        )
      act = transition_block(act)

    if self.with_single:
      with trace.into("single_act_update"):
        single_act = self.single_act_update(
            act, single_act, seq_mask
        )
      return act, single_act
    else:
      return act


class EvoformerIteration(fnn.Module):
  """Single Iteration of Evoformer Main Stack."""

  __version__ = "1.0"

  class Config(base_config.BaseConfig):
    """Configuration for EvoformerIteration."""

    num_layer: int = 4
    msa_attention: MSAAttention.Config = base_config.autocreate()
    outer_product_mean: OuterProductMean.Config = base_config.autocreate()
    msa_transition: "TransitionBlock.Config" = base_config.autocreate()
    pair_attention: "GridSelfAttention.Config" = base_config.autocreate()
    pair_transition: "TransitionBlock.Config" = base_config.autocreate()
    triangle_multiplication_incoming: "TriangleMultiplication.Config" = (
      base_config.autocreate(equation="...kjc,...kic->...ijc")
    )
    triangle_multiplication_outgoing: "TriangleMultiplication.Config" = (
      base_config.autocreate(equation="...ikc,...jkc->...ijc")
    )
    shard_transition_blocks: bool = True

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      num_output_channel: int,
      index_reuse_policy: IndexReusePolicy = default_reuse_policy,
      ibsa_params: IBSAParametersOrProvider = IBSAParameters.default,
      distributed: bool = False,
      name: str = "evoformer_iteration",
  ):
    super().__init__()
    self.name = name
    self.num_channels = num_channels
    self.distributed = distributed and dist.is_enabled()
    self.shard_transition_blocks = config.shard_transition_blocks
    self.pair_transition_shard_spec = global_config.pair_transition_shard_spec

    # ///// MODULE STRUCTURES ///// #
    self.outer_product_mean = OuterProductMean(
      config.outer_product_mean,
      global_config,
      num_channels=num_channels,
      num_output_channel=num_output_channel,
      distributed=self.distributed,
      name="outer_product_mean",
    )
    self.msa_attention1 = MSAAttention(
      config.msa_attention,
      global_config,
      num_channels=num_channels,
      num_output_channel=num_output_channel,
      distributed=self.distributed,
      name="msa_attention1",
    )
    self.msa_transition = TransitionBlock(
      config.msa_transition,
      global_config,
      num_channels=num_channels,
      name="msa_transition",
    )
    self.triangle_multiplication_outgoing = TriangleMultiplication(
      config.triangle_multiplication_outgoing,
      global_config,
      num_channels=num_output_channel,
      distributed=self.distributed,
      name="triangle_multiplication_outgoing",
    )
    self.triangle_multiplication_incoming = TriangleMultiplication(
      config.triangle_multiplication_incoming,
      global_config,
      num_channels=num_output_channel,
      distributed=self.distributed,
      name="triangle_multiplication_incoming",
    )
    self.pair_attention1 = GridSelfAttention(
      config.pair_attention,
      global_config,
      transpose=False,
      num_channels=num_output_channel,
      index_reuse_policy=index_reuse_policy,
      ibsa_params=ibsa_params,
      distributed=self.distributed,
      name="pair_attention1",
    )
    self.pair_attention2 = GridSelfAttention(
      config.pair_attention,
      global_config,
      transpose=True,
      num_channels=num_output_channel,
      index_reuse_policy=index_reuse_policy,
      ibsa_params=ibsa_params,
      distributed=self.distributed,
      name="pair_attention2",
    )

    self.pair_transition = TransitionBlock(
      config.pair_transition,
      global_config,
      num_channels=num_output_channel,
      name="pair_transition",
    )

    self.cache = ValueCache()

  def forward(
      self,
      activations: Mapping[str, torch.Tensor],
      masks: Mapping[str, torch.Tensor],
  ) -> Mapping[str, torch.Tensor]:
    msa_act, pair_act = activations["msa"], activations["pair"]
    msa_mask, seq_mask = masks["msa"], masks["seq"]
    
    num_residues = pair_act.shape[-3]

    with trace.into("outer_product_mean"):
      pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask)
    with trace.into("msa_attn"):
      msa_act = msa_act + self.msa_attention1(msa_act, msa_mask, pair_act=pair_act)
    with trace.into("msa_transition"):
      msa_act = self.msa_transition(msa_act)

    with trace.into("trig_mul"):
      pair_act = self.triangle_multiplication_outgoing(pair_act, seq_mask)
      pair_act = self.triangle_multiplication_incoming(pair_act, seq_mask)

    with trace.into("pair_attn"):
      pair_act = self.pair_attention1(pair_act, seq_mask)
      pair_act = self.pair_attention2(pair_act, seq_mask)

    with trace.into("pair_transition"):
      transition_block = partial(self.pair_transition)
      if self.shard_transition_blocks:
        transition_block = mapping.sharded_apply(
            transition_block,
            get_shard_size(
                num_residues, self.pair_transition_shard_spec
            ),
            in_axes=-3, out_axes=-3,  # in case a batching dim present
        )
      pair_act = transition_block(pair_act)

    return {"msa": msa_act, "pair": pair_act}
