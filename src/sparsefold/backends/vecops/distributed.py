# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Distributed orchestration around the native SparseFold kernels."""

import os
import resource
import time

import torch

from sparsefold.runtime import dist
from sparsefold.runtime import trace


_PROFILE_DISTRIBUTED = os.environ.get("SPARSEFOLD_PROFILE_DISTRIBUTED", os.environ.get("AF3_PROFILE_DISTRIBUTED")) == "1"


def _profile_enabled() -> bool:
  if not _PROFILE_DISTRIBUTED:
    return False
  import vecops

  return not vecops.is_precompiling()


def _profile_distributed_call(
    *, kind: str, transpose: bool, phases: dict[str, float], started
):
  """Emit one atomic per-rank record for a distributed native invocation."""
  if not _profile_enabled():
    return
  wall_start, cpu_start, usage_start = started
  usage_end = resource.getrusage(resource.RUSAGE_SELF)
  total_ms = (time.perf_counter_ns() - wall_start) / 1e6
  cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
  phase_text = " ".join(f"{name}={value:.3f}" for name, value in phases.items())
  # os.write keeps one diagnostic record intact when four rank log pumps share
  # a terminal.  A monotonic timestamp makes cross-rank arrival skew explicit.
  line = (
      f"[dist-profile rank={dist.get_rank()} t={time.monotonic_ns()} "
      f"path={trace.current_path()} "
      f"kind={kind} transpose={int(transpose)}] "
      f"total={total_ms:.3f} cpu={cpu_ms:.3f} "
      f"nvcsw={usage_end.ru_nvcsw - usage_start.ru_nvcsw} "
      f"nivcsw={usage_end.ru_nivcsw - usage_start.ru_nivcsw} {phase_text}\n"
  )
  os.write(2, line.encode())


def _timed_phase(phases: dict[str, float], name: str, fn):
  if not _profile_enabled():
    return fn()
  started = time.perf_counter_ns()
  result = fn()
  phases[name] = (time.perf_counter_ns() - started) / 1e6
  return result


def grid_self_attention_distributed(
    act: torch.Tensor,  # (*, B, N, C), TAct, last contiguous
    key_padding_mask: torch.Tensor,  # (*, N), bool, last contiguous
    packed_q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_output: torch.Tensor,
    packed_g: torch.Tensor,
    packed_bias: torch.Tensor,
    layernorm_scale: torch.Tensor,  # (C,), TNorm, last contiguous
    layernorm_offset: torch.Tensor,  # (C,), TNorm, last contiguous
    index_map: torch.Tensor | None,
    weight_map: torch.Tensor | None,
    num_samples: int,
    num_rand_blocks: int,
    t_prob: float,
    tr_prob: float,
    u_prob: float,
    transpose: bool,
    *,
    head_dim: int,
):
  from sparsefold.backends.vecops.kernels import grid_distributed_attention_packed
  from sparsefold.backends.vecops.kernels import grid_distributed_projection_packed

  shard = dist.current_shard
  if shard is None:
    raise RuntimeError("distributed grid attention requires a shard")
  local_rows = act.shape[-3]
  if local_rows != shard.size():
    raise ValueError(
        f"grid local rows {local_rows} != shard size {shard.size()}"
    )
  phases: dict[str, float] = {}
  profile_started = (
      (
          time.perf_counter_ns(),
          time.process_time_ns(),
          resource.getrusage(resource.RUSAGE_SELF),
      )
      if _profile_enabled()
      else None
  )
  normalized, bias_send = _timed_phase(
      phases,
      "projection",
      lambda: grid_distributed_projection_packed(
          act,
          layernorm_scale,
          layernorm_offset,
          packed_bias,
          heads=packed_q.shape[0],
      ),
  )
  gather_bias = _timed_phase(
      phases,
      "gather_launch",
      lambda: shard.all_gather_into_tensor(
          input_tensor=bias_send, dim=0, async_op=True
      ),
  )
  if transpose:
    exchange_norm = _timed_phase(
        phases,
        "reshard_launch",
        lambda: shard.reshard(
            input=normalized, dim1=-3, dim2=-2, async_op=True
        ),
    )
    global_bias = _timed_phase(phases, "gather_wait", gather_bias.wait)
    exchanged = _timed_phase(phases, "reshard_wait", exchange_norm.wait)
    oriented_norm = _timed_phase(
        phases,
        "orient",
        lambda: exchanged.transpose(-3, -2).contiguous(),
    )
  else:
    global_bias = _timed_phase(phases, "gather_wait", gather_bias.wait)
    oriented_norm = normalized

  projected, index_map, weight_map = _timed_phase(
      phases,
      "attention",
      lambda: grid_distributed_attention_packed(
          oriented_norm,
          key_padding_mask,
          global_bias,
          packed_q,
          packed_k,
          packed_v,
          packed_output,
          packed_g,
          index_map,
          weight_map,
          num_samples,
          num_rand_blocks,
          t_prob,
          tr_prob,
          u_prob,
          shard.offset(),
          head_dim=head_dim,
      ),
  )
  if transpose:
    projected = _timed_phase(
        phases,
        "output_reshard",
        lambda: shard.reshard(
            input=projected, dim1=-3, dim2=-2
        ).transpose(-3, -2).contiguous(),
    )
  result = _timed_phase(phases, "residual", lambda: act + projected)
  _profile_distributed_call(
      kind="grid", transpose=transpose, phases=phases,
      started=profile_started,
  )
  return result, index_map, weight_map


def triangle_multiplication_distributed(
    act: torch.Tensor,  # (*, B, N, C), TAct, last contiguous
    mask: torch.Tensor,  # (*, N), bool, last contiguous
    left_norm_scale: torch.Tensor,  # (C,), TNorm, last contiguous
    left_norm_offset: torch.Tensor,  # (C,), TNorm, last contiguous
    packed_projection: torch.Tensor,
    center_norm_scale: torch.Tensor,  # (C,), TNorm, last contiguous
    center_norm_offset: torch.Tensor,  # (C,), TNorm, last contiguous
    packed_output: torch.Tensor,
    packed_gate: torch.Tensor,
    transpose: bool = False,
):
  from sparsefold.backends.vecops.kernels import triangle_distributed_projection_packed
  from sparsefold.backends.vecops.kernels import triangle_distributed_update_packed

  shard = dist.current_shard
  if shard is None:
    raise RuntimeError("distributed triangle multiplication requires a shard")
  local_rows = act.shape[-3]
  if local_rows != shard.size():
    raise ValueError(
        f"triangle local rows {local_rows} != shard size {shard.size()}"
    )
  phases: dict[str, float] = {}
  profile_started = (
      (
          time.perf_counter_ns(),
          time.process_time_ns(),
          resource.getrusage(resource.RUSAGE_SELF),
      )
      if _profile_enabled()
      else None
  )
  row_mask = _timed_phase(
      phases, "row_mask", lambda: shard.shard(mask, dim=mask.ndim - 1)
  )
  normalized, a, b = _timed_phase(
      phases,
      "projection",
      lambda: triangle_distributed_projection_packed(
          act,
          row_mask,
          mask,
          left_norm_scale,
          left_norm_offset,
          packed_projection,
      ),
  )

  if transpose:
    # Incoming triangle multiplication needs A gathered over k and B
    # resharded from k to i. Launch both independent collectives together.
    gather_a = _timed_phase(
        phases,
        "gather_launch",
        lambda: shard.all_gather_into_tensor(
            input_tensor=a, dim=-2, async_op=True
        ),
    )
    reshard_b = _timed_phase(
        phases,
        "reshard_launch",
        lambda: shard.reshard(
            input=b, dim1=-2, dim2=-1, async_op=True
        ),
    )
    global_a = _timed_phase(phases, "gather_wait", gather_a.wait)
    column_b = _timed_phase(phases, "reshard_wait", reshard_b.wait)
    left, right = _timed_phase(
        phases,
        "orient",
        lambda: (
            column_b.transpose(-2, -1).contiguous(),
            global_a.transpose(-2, -1).contiguous(),
        ),
    )
  else:
    # Outgoing contraction keeps local i and gathers only projected B over j.
    left = a
    right = _timed_phase(
        phases,
        "gather_sync",
        lambda: shard.all_gather_into_tensor(input_tensor=b, dim=-2),
    )

  result = _timed_phase(
      phases,
      "update",
      lambda: triangle_distributed_update_packed(
          act,
          normalized,
          left,
          right,
          center_norm_scale,
          center_norm_offset,
          packed_output,
          packed_gate,
      ),
  )
  _profile_distributed_call(
      kind="triangle", transpose=transpose, phases=phases,
      started=profile_started,
  )
  return result


def msa_attention_distributed(
    act: torch.Tensor,
    mask: torch.Tensor,
    pair_act: torch.Tensor,
    act_norm_scale: torch.Tensor,
    act_norm_offset: torch.Tensor,
    pair_norm_scale: torch.Tensor,
    pair_norm_offset: torch.Tensor,
    packed_pair_logits: torch.Tensor,
    packed_value: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
  from sparsefold.backends.vecops.kernels import msa_distributed_projection_packed
  from sparsefold.backends.vecops.kernels import msa_distributed_update_packed

  shard = dist.current_shard
  if shard is None:
    raise RuntimeError("distributed MSA attention requires a shard")
  if act.shape[-2] != shard.size() or pair_act.shape[-3] != shard.size():
    raise ValueError("MSA and pair activations must use the current row shard")
  weights, value_send, gate = msa_distributed_projection_packed(
      act, mask, pair_act,
      act_norm_scale, act_norm_offset,
      pair_norm_scale, pair_norm_offset,
      packed_pair_logits, packed_value, packed_gate,
      heads=heads,
  )
  global_value = shard.all_gather_into_tensor(
      input_tensor=value_send, dim=0
  )
  return msa_distributed_update_packed(
      weights, global_value, gate, packed_output, head_dim=head_dim
  )


def outer_product_mean_distributed(
    act: torch.Tensor,
    mask: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
    packed_output: torch.Tensor,
    output_bias: torch.Tensor,
    *,
    outer_channels: int,
) -> torch.Tensor:
  from sparsefold.backends.vecops.kernels import outer_distributed_left_projection_packed
  from sparsefold.backends.vecops.kernels import outer_distributed_right_projection_packed
  from sparsefold.backends.vecops.kernels import outer_distributed_update_packed

  shard = dist.current_shard
  if shard is None:
    raise RuntimeError("distributed outer-product mean requires a shard")
  if act.shape[-2] != shard.size():
    raise ValueError("outer-product activation must use the current row shard")
  local_mask = shard.shard(mask, dim=mask.ndim - 1)
  normalized, right_send = outer_distributed_right_projection_packed(
      act, local_mask, norm_scale, norm_offset, packed_right,
      outer_channels=outer_channels,
  )
  gather_right = shard.all_gather_into_tensor(
      input_tensor=right_send, dim=0, async_op=True
  )
  left = outer_distributed_left_projection_packed(
      normalized, local_mask, packed_left, outer_channels=outer_channels
  )
  global_right = gather_right.wait()
  return outer_distributed_update_packed(
      left, global_right, mask, packed_output, output_bias, shard.offset(),
      outer_channels=outer_channels,
  )


def evoformer_single_act_update_distributed(
    pair_act: torch.Tensor,
    single_act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    ln_single_pair_scale: torch.Tensor,
    ln_single_pair_offset: torch.Tensor,
    packed_logits: torch.Tensor,
    ln_attn_scale: torch.Tensor,
    ln_attn_offset: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_g: torch.Tensor,
    packed_o: torch.Tensor,
    ln_input_scale: torch.Tensor,
    ln_input_offset: torch.Tensor,
    packed_transition1: torch.Tensor,
    packed_transition2: torch.Tensor,
    *,
    hidden: int,
):
  from sparsefold.backends.vecops.kernels import single_distributed_projection_packed
  from sparsefold.backends.vecops.kernels import single_distributed_update_packed

  shard = dist.current_shard
  if shard is None:
    raise RuntimeError("distributed single update requires a shard")
  local_rows = single_act.shape[-2]
  if local_rows != shard.size():
    raise ValueError(
        f"single local rows {local_rows} != shard size {shard.size()}"
    )
  pair_logits, q, k, v, gate = single_distributed_projection_packed(
      pair_act,
      single_act,
      ln_single_pair_scale,
      ln_single_pair_offset,
      packed_logits,
      ln_attn_scale,
      ln_attn_offset,
      packed_q,
      q_bias,
      packed_k,
      packed_v,
      packed_g,
  )
  gather_k = shard.all_gather_into_tensor(
      input_tensor=k, dim=-2, async_op=True
  )
  gather_v = shard.all_gather_into_tensor(
      input_tensor=v, dim=-2, async_op=True
  )
  global_k = gather_k.wait()
  global_v = gather_v.wait()
  return single_distributed_update_packed(
      single_act,
      key_padding_mask,
      pair_logits,
      q,
      global_k,
      global_v,
      gate,
      packed_o,
      ln_input_scale,
      ln_input_offset,
      packed_transition1,
      packed_transition2,
      hidden=hidden,
  )
