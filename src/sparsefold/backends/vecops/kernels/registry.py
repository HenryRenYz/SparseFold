# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Central registration and Python wrappers for SparseFold vecops kernels."""

from __future__ import annotations

import hashlib
import itertools
import math
import platform
from pathlib import Path

import torch
import vecops
import vecops.typing as vt


_STABLE_MATH_DIGEST = hashlib.sha256(
    (Path(__file__).parent / "resources/StableMath.h").read_bytes()
).hexdigest()[:16]


def _stable_math_recipe(name: str) -> str:
  # vecops fingerprints the .cpp recipe, not headers included by that recipe.
  return f"sparsefold-v1;{name};stablemath={_STABLE_MATH_DIGEST}"


@vecops.jit(
    source="resources/PackMatmulB.cpp",
    recipe_id="sparsefold-v1;pack-matmul-b-v2-dynamic-extents",
    symbols={
        "Batch": vt.Dynamic(lower=0),
        "N": vt.Dynamic(lower=0),
        "K": vt.Dynamic(lower=0),
        "P": vt.Dynamic(lower=0),
    },
)
def _pack_matmul_b_kernel(
    input: vt.In(vecops.bfloat16)[
        "Batch N K", "Dynamic<1,0,+inf> K 1"
    ],
    output: vt.Out(vecops.bfloat16)["Batch P", "P 1"],
) -> None: ...


_run_pack_matmul_b = vecops.ops.torch.register(
    "sparsefold_native::pack_matmul_b",
    _pack_matmul_b_kernel,
)


_FP32_MATMUL_PRECISION = {"medium": 0, "high": 1, "highest": 2}


def _normalize_fp32_matmul_precision(precision: str | None) -> tuple[str, int]:
  if precision is None:
    precision = "medium"
  precision = precision.lower()
  if precision not in _FP32_MATMUL_PRECISION:
    raise ValueError(
        "fp32_matmul_precision must be one of medium, high or highest; "
        f"got {precision!r}"
    )
  return precision, _FP32_MATMUL_PRECISION[precision]


def _canonical_last_contiguous(tensor: torch.Tensor) -> torch.Tensor:
  """Return a tensor whose innermost stride is exactly one.

  PyTorch considers arbitrary strides contiguous on size-one dimensions, but
  vecops benefits from retaining a compile-time unit-stride inner axis.
  """
  tensor = tensor.contiguous()
  if tensor.ndim and tensor.stride(-1) != 1:
    canonical = vecops.planner.empty(
        tensor.shape, dtype=tensor.dtype, device=tensor.device
    )
    canonical.copy_(tensor)
    return canonical
  return tensor


def pack_matmul_b(weight: torch.Tensor) -> torch.Tensor:
  """Pack one or more logical ``[N,K]`` BF16 B operands persistently."""
  if weight.ndim < 2:
    raise ValueError("pack_matmul_b expects at least a rank-two tensor")
  if weight.device.type not in {"cpu", "meta"}:
    raise ValueError(f"pack_matmul_b only supports CPU tensors, got {weight.device}")
  if weight.dtype != torch.bfloat16:
    weight = weight.to(torch.bfloat16)
  n, k = weight.shape[-2:]
  machine = platform.machine().lower()
  if machine in {"x86_64", "amd64", "x64"}:
    panel, k_tile = 16, 32
  elif machine in {"aarch64", "arm64"}:
    panel, k_tile = 32, 2
  else:
    raise RuntimeError(f"unsupported matmul packing target: {machine}")
  packed_elements = ((n + panel - 1) // panel * panel) * (
      (k + k_tile - 1) // k_tile * k_tile
  )
  batch_shape = weight.shape[:-2]
  flat_weight = weight.reshape(-1, n, k).contiguous()
  packed = vecops.planner.empty(
      (flat_weight.shape[0], packed_elements),
      dtype=torch.bfloat16,
      device=weight.device,
  )
  _run_pack_matmul_b(flat_weight, packed)
  return packed.reshape(batch_shape + (packed_elements,))


def _divisor_panel(size: int, limit: int = 128) -> int:
  """Return the widest panel no larger than ``limit`` dividing ``size``."""
  for panel in range(min(limit, size), 0, -1):
    if size % panel == 0:
      return panel
  raise AssertionError("positive dimensions always have a divisor")


@vecops.jit(
    source="resources/PackLinearB.cpp",
    recipe_id="sparsefold-v1;pack-linear-b-v2-unit-inner-stride",
    symbols={
        "Batch": vt.Dynamic(lower=0),
        "N": vt.Dynamic(lower=0),
        "K": vt.Dynamic(lower=0),
        "P": vt.Dynamic(lower=0),
    },
)
def _pack_linear_b_kernel(
    input: vt.In()[
        "Batch N K", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    output: vt.Out()["Batch P", "P 1"],
    *,
    FP32MatmulPrecision: vt.Const,
) -> None: ...


_run_pack_linear_b = vecops.ops.torch.register(
    "sparsefold_native::pack_linear_b",
    _pack_linear_b_kernel,
)


def pack_linear_b(
    weight: torch.Tensor,
    *,
    fp32_matmul_precision: str | None = None,
) -> torch.Tensor:
  """Persistently pack logical ``[N,K]`` B operands for generic Linear.

  FP32 weights use BF16 storage and BF16xBF16->FP32 computation at ``medium``.
  ``high`` and ``highest`` retain FP32 storage and select FP32xFP32->FP32.
  """
  if weight.ndim < 2:
    raise ValueError("pack_linear_b expects at least a rank-two tensor")
  if weight.device.type not in {"cpu", "meta"}:
    raise ValueError(f"pack_linear_b only supports CPU tensors, got {weight.device}")
  precision, precision_code = _normalize_fp32_matmul_precision(
      fp32_matmul_precision
  )
  if weight.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
    raise ValueError(f"unsupported matmul weight dtype: {weight.dtype}")
  n, k = weight.shape[-2:]
  machine = platform.machine().lower()
  packed_dtype = weight.dtype
  if weight.dtype == torch.float32 and precision == "medium":
    packed_dtype = torch.bfloat16
  if machine in {"x86_64", "amd64", "x64"}:
    if packed_dtype == torch.float32:
      raise RuntimeError(
          "fp32_matmul_precision high/highest requires an FP32 matmul atom, "
          "which AMX does not provide; use medium on x86"
      )
    if packed_dtype == torch.float16:
      raise RuntimeError(
          "FP16 Linear requires AMX-FP16, which this x86 target does not "
          "provide; use BF16 or the Torch fallback"
      )
    panel = 16
    k_tile = 32
  elif machine in {"aarch64", "arm64"}:
    # Registered ARM targets use VL=SVL=64B: two FP32 ZA tiles are 32 rows.
    panel = 32
    k_tile = 1 if packed_dtype == torch.float32 else 2
  else:
    raise RuntimeError(f"unsupported matmul packing target: {machine}")
  packed_elements = ((n + panel - 1) // panel * panel) * (
      (k + k_tile - 1) // k_tile * k_tile
  )
  batch_shape = weight.shape[:-2]
  flat_weight = _canonical_last_contiguous(weight.reshape(-1, n, k))
  packed = vecops.planner.empty(
      (flat_weight.shape[0], packed_elements),
      dtype=packed_dtype,
      device=weight.device,
  )
  _run_pack_linear_b(
      flat_weight,
      packed,
      FP32MatmulPrecision=precision_code,
  )
  return packed.reshape(batch_shape + (packed_elements,))


@vecops.jit(
    source="resources/Linear.cpp",
    recipe_id=_stable_math_recipe("linear-packed-v11-balanced-native-workers-exp-fast"),
    symbols={
        "M": vt.Dynamic(lower=0),
        "N": vt.Dynamic(lower=0),
        "K": vt.Dynamic(lower=0),
        "P": vt.Dynamic(lower=0),
    },
)
def _linear_kernel(
    input: vt.In()["M K", "Dynamic<1,0,+inf> 1"],
    packed_weight: vt.In()["P", "1"],
    bias: vt.In()["N", "Dynamic<1,0,+inf>"],
    residual: vt.In(optional=True)[
        "M N", "Dynamic<1,0,+inf> 1"
    ],
    output: vt.Out()["M N", "N 1"],
    post_op: vt.Scalar[vecops.int32],
    *,
    FP32MatmulPrecision: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_linear = vecops.ops.torch.register(
    "sparsefold_native::linear_packed",
    _linear_kernel,
)


def linear_packed(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    out_features: int,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    post_op: int = 0,
    fp32_matmul_precision: str | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Run a Linear projection from a persistent packed ``[N,K]`` weight."""
  if input.ndim < 1:
    raise ValueError("linear_packed expects an input with at least one axis")
  if input.device.type not in {"cpu", "meta"}:
    raise ValueError(f"linear_packed only supports CPU tensors, got {input.device}")
  if int(post_op) not in {0, 1, 2, 3}:
    raise ValueError(f"unsupported Linear post_op code: {post_op}")
  if fp32_matmul_precision is None:
    precision_code = _FP32_MATMUL_PRECISION["medium"]
  else:
    _, precision_code = _normalize_fp32_matmul_precision(fp32_matmul_precision)
  input_shape = input.shape
  k = input_shape[-1]
  # The hot path issues as few Torch dispatcher calls as possible: same-dtype
  # conversions, no-op contiguity and already-flat packed weights skip their
  # call entirely (each avoided call costs ~10us on slow hosts).
  compute_input = (
      input if input.dtype is packed_weight.dtype
      else input.to(packed_weight.dtype)
  )
  flat_input = compute_input.reshape(-1, k)
  if flat_input.stride(-1) != 1:
    flat_input = _canonical_last_contiguous(flat_input)
  m = flat_input.shape[0]
  output_shape = input_shape[:-1] + (out_features,)
  if out_dtype is None:
    out_dtype = input.dtype
  flat_residual = None
  if residual is not None:
    if residual.shape != output_shape:
      raise ValueError(
          "native Linear residual must have the exact output shape; "
          f"expected {output_shape}, got {residual.shape}"
      )
    flat_residual = _canonical_last_contiguous(
        residual.to(out_dtype).reshape(m, out_features)
    )
  if bias is None:
    bias = torch.zeros(
        (out_features,), dtype=torch.float32, device=input.device
    )
  elif bias.shape != (out_features,):
    raise ValueError(
        f"native Linear bias must have shape {(out_features,)}, got {bias.shape}"
    )
  elif bias.dtype is not torch.float32 or bias.stride(-1) != 1:
    bias = bias.to(torch.float32).contiguous()
  flat_weight = (
      packed_weight if packed_weight.ndim == 1
      else packed_weight.reshape(-1)
  )
  output = vecops.planner.empty(
      (m, out_features), dtype=out_dtype, device=input.device
  )
  _run_linear(
      flat_input,
      flat_weight,
      bias,
      flat_residual,
      output,
      int(post_op),
      FP32MatmulPrecision=precision_code,
      ComputeType=torch.float32,
  )
  return output.reshape(output_shape)


@vecops.jit(
    source="resources/Matmul.cpp",
    recipe_id="sparsefold-v1;raw-matmul-v2-aten-routing-fused-c-epilogue",
    symbols={
        "Batch": vt.Dynamic(lower=0),
        "WeightBatch": vt.Dynamic(lower=0),
        "M": vt.Dynamic(lower=0),
        "N": vt.Dynamic(lower=0),
        "K": vt.Dynamic(lower=0),
    },
)
def _matmul_kernel(
    input: vt.In()["Batch M K", "Dynamic<1,0,+inf> K 1"],
    weight: vt.In()[
        "WeightBatch N K", "Dynamic<1,0,+inf> K 1"
    ],
    c_input: vt.In(optional=True)[
        "Batch M N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Dynamic<1,0,+inf>"
    ],
    output: vt.Out()["Batch M N", "Dynamic<1,0,+inf> N 1"],
    alpha: vt.Scalar[vecops.float32],
    beta: vt.Scalar[vecops.float32],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_matmul = vecops.ops.torch.register(
    "sparsefold_native::matmul",
    _matmul_kernel,
)


def matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    c_input: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
  """Run raw ``[B,M,K] @ [B,N,K]^T`` with an optional fused C input."""
  if input.ndim != 3 or weight.ndim != 3:
    raise ValueError("matmul expects rank-three input and weight tensors")
  if input.device.type not in {"cpu", "meta"}:
    raise ValueError("matmul only supports CPU tensors")
  if input.dtype != weight.dtype:
    raise ValueError("matmul operands must have the same dtype")
  if weight.shape[0] not in {1, input.shape[0]}:
    raise ValueError("weight batch must be one or match input batch")
  batch, rows, _ = input.shape
  out_features = weight.shape[1]
  compute_input = _canonical_last_contiguous(input)
  compute_weight = _canonical_last_contiguous(weight)
  if out_dtype is None:
    out_dtype = input.dtype
  if c_input is not None:
    c_input = c_input.contiguous()
  output = vecops.planner.empty(
      (batch, rows, out_features), dtype=out_dtype, device=input.device
  )
  _run_matmul(
      compute_input,
      compute_weight,
      c_input,
      output,
      float(alpha),
      float(beta),
      ComputeType=torch.float32,
  )
  return output


@vecops.jit(
    source="resources/TransitionBlock.cpp",
    recipe_id=_stable_math_recipe("transition-block-v8-sme-polynomial-sigmoid"),
    symbols={"B": vt.Dynamic(lower=0)},
)
def _transition_block_kernel(
    act: vt.In(vecops.bfloat16)[
        "B N C", "Dynamic<1,0,+inf> C 1"
    ],
    ln_scale: vt.In(vecops.float32)["C", "1"],
    ln_offset: vt.In(vecops.float32)["C", "1"],
    transition1: vt.In(vecops.bfloat16)[
        "C 2 M", "Dynamic<1,0,+inf> M 1"
    ],
    transition2: vt.In(vecops.bfloat16)["M C", "C 1"],
    output: vt.Out(vecops.bfloat16)[
        "B N C", "Dynamic<1,0,+inf> C 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_transition_block = vecops.ops.torch.register(
    "sparsefold_native::transition_block",
    _transition_block_kernel,
)


@vecops.jit(
    source="resources/MSAAttention.cpp",
    recipe_id=_stable_math_recipe("msa-attention-packed-v3-softmax-estimate"),
    symbols={"Batch": vt.Dynamic(lower=0)},
)
def _msa_attention_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch M N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    mask: vt.In(vecops.bool_)["Batch M N", "Dynamic<1,0,+inf> N 1"],
    pair_act: vt.In(vecops.bfloat16)[
        "Batch N N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    act_norm_scale: vt.In(vecops.float32)["C", "1"],
    act_norm_offset: vt.In(vecops.float32)["C", "1"],
    pair_norm_scale: vt.In(vecops.float32)["Q", "1"],
    pair_norm_offset: vt.In(vecops.float32)["Q", "1"],
    pair_logits_projection: vt.In(vecops.bfloat16)["1 PPL", "PPL 1"],
    value_projection: vt.In(vecops.bfloat16)["1 PV", "PV 1"],
    gate_projection: vt.In(vecops.bfloat16)["1 PG", "PG 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch M N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    *,
    H: vt.Const,
    HeadDim: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_msa_attention = vecops.ops.torch.register(
    "sparsefold_native::msa_attention",
    _msa_attention_kernel,
)


@vecops.jit(
    source="resources/MSADistributedProjection.cpp",
    recipe_id="sparsefold-v1;msa-distributed-projection-packed-v2-softmax-estimate",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _msa_distributed_projection_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    mask: vt.In(vecops.bool_)["Batch M N", "Dynamic<1,0,+inf> N 1"],
    pair_act: vt.In(vecops.bfloat16)[
        "Batch L N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    act_norm_scale: vt.In(vecops.float32)["C", "1"],
    act_norm_offset: vt.In(vecops.float32)["C", "1"],
    pair_norm_scale: vt.In(vecops.float32)["Q", "1"],
    pair_norm_offset: vt.In(vecops.float32)["Q", "1"],
    pair_logits_projection: vt.In(vecops.bfloat16)["1 PPL", "PPL 1"],
    value_projection: vt.In(vecops.bfloat16)["1 PV", "PV 1"],
    gate_projection: vt.In(vecops.bfloat16)["1 PG", "PG 1"],
    weights: vt.Out(vecops.bfloat16)[
        "Batch H L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    value_send: vt.Out(vecops.bfloat16)[
        "L Batch M C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    gate: vt.Out(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_msa_distributed_projection = vecops.ops.torch.register(
    "sparsefold_native::msa_distributed_projection",
    _msa_distributed_projection_kernel,
)


@vecops.jit(
    source="resources/MSADistributedUpdate.cpp",
    recipe_id=_stable_math_recipe("msa-distributed-update-packed-v1-exp-fast"),
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _msa_distributed_update_kernel(
    weights: vt.In(vecops.bfloat16)[
        "Batch H L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    global_value: vt.In(vecops.bfloat16)[
        "N Batch M C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    gate: vt.In(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    *,
    HeadDim: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_msa_distributed_update = vecops.ops.torch.register(
    "sparsefold_native::msa_distributed_update",
    _msa_distributed_update_kernel,
)


@vecops.jit(
    source="resources/OuterProductMean.cpp",
    recipe_id="sparsefold-v1;outer-product-mean-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0)},
)
def _outer_product_mean_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch M N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    mask: vt.In(vecops.bool_)["Batch M N", "Dynamic<1,0,+inf> N 1"],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    left_projection: vt.In(vecops.bfloat16)["1 PL", "PL 1"],
    right_projection: vt.In(vecops.bfloat16)["1 PR", "PR 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output_bias: vt.In(vecops.bfloat16)["Q", "1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch N N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    *,
    OuterChannels: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_outer_product_mean = vecops.ops.torch.register(
    "sparsefold_native::outer_product_mean",
    _outer_product_mean_kernel,
)


@vecops.jit(
    source="resources/OuterDistributedProjection.cpp",
    recipe_id="sparsefold-v1;outer-distributed-projection-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _outer_distributed_projection_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    local_mask: vt.In(vecops.bool_)[
        "Batch M L", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    left_projection: vt.In(vecops.bfloat16)["1 PL", "PL 1"],
    right_projection: vt.In(vecops.bfloat16)["1 PR", "PR 1"],
    left: vt.Out(vecops.bfloat16)[
        "Batch M L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    right_send: vt.Out(vecops.bfloat16)[
        "L Batch M D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    *,
    OuterChannels: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_outer_distributed_projection = vecops.ops.torch.register(
    "sparsefold_native::outer_distributed_projection",
    _outer_distributed_projection_kernel,
)


@vecops.jit(
    source="resources/OuterDistributedRightProjection.cpp",
    recipe_id="sparsefold-v1;outer-distributed-right-projection-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _outer_distributed_right_projection_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    local_mask: vt.In(vecops.bool_)[
        "Batch M L", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    right_projection: vt.In(vecops.bfloat16)["1 PR", "PR 1"],
    normalized: vt.Out(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    right_send: vt.Out(vecops.bfloat16)[
        "L Batch M D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    *,
    OuterChannels: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_outer_distributed_right_projection = vecops.ops.torch.register(
    "sparsefold_native::outer_distributed_right_projection",
    _outer_distributed_right_projection_kernel,
)


@vecops.jit(
    source="resources/OuterDistributedLeftProjection.cpp",
    recipe_id="sparsefold-v1;outer-distributed-left-projection-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _outer_distributed_left_projection_kernel(
    normalized: vt.In(vecops.bfloat16)[
        "Batch M L C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    local_mask: vt.In(vecops.bool_)[
        "Batch M L", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    left_projection: vt.In(vecops.bfloat16)["1 PL", "PL 1"],
    left: vt.Out(vecops.bfloat16)[
        "Batch M L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    *,
    OuterChannels: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_outer_distributed_left_projection = vecops.ops.torch.register(
    "sparsefold_native::outer_distributed_left_projection",
    _outer_distributed_left_projection_kernel,
)


@vecops.jit(
    source="resources/OuterDistributedUpdate.cpp",
    recipe_id="sparsefold-v1;outer-distributed-update-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _outer_distributed_update_kernel(
    left: vt.In(vecops.bfloat16)[
        "Batch M L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    global_right: vt.In(vecops.bfloat16)[
        "N Batch M D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    mask: vt.In(vecops.bool_)["Batch M N", "Dynamic<1,0,+inf> N 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output_bias: vt.In(vecops.bfloat16)["Q", "1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch L N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    row_offset: vt.Scalar[vecops.int64],
    *,
    OuterChannels: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_outer_distributed_update = vecops.ops.torch.register(
    "sparsefold_native::outer_distributed_update",
    _outer_distributed_update_kernel,
)


@vecops.jit(
    source="resources/DiffusionConditioning.cpp",
    recipe_id="sparsefold-v1;diffusion-conditioning-packed-v1-superblock",
    symbols={"BCond": vt.Dynamic(lower=1)},
)
def _diffusion_conditioning_kernel(
    single_cond: vt.In(vecops.bfloat16)[
        "BCond N S", "Dynamic<1,0,+inf> S 1"
    ],
    cond_norm_scale: vt.In(vecops.float32)["L S", "S 1"],
    cond_scale_projection: vt.In(vecops.bfloat16)["L PCS", "PCS 1"],
    cond_scale_bias: vt.In(vecops.bfloat16)["L C", "C 1"],
    cond_bias_projection: vt.In(vecops.bfloat16)["L PCB", "PCB 1"],
    zero_cond_projection: vt.In(vecops.bfloat16)["L PZ", "PZ 1"],
    zero_cond_bias: vt.In(vecops.bfloat16)["L C", "C 1"],
    adaptive_scale: vt.Out(vecops.bfloat16)[
        "L BCond N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    adaptive_bias: vt.Out(vecops.bfloat16)[
        "L BCond N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    zero_gate: vt.Out(vecops.bfloat16)[
        "L BCond N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_diffusion_conditioning = vecops.ops.torch.register(
    "sparsefold_native::diffusion_conditioning",
    _diffusion_conditioning_kernel,
    return_outputs=True,
)


def diffusion_conditioning_packed(
    single_cond: torch.Tensor,
    cond_norm_scale: torch.Tensor,
    packed_cond_scale: torch.Tensor,
    cond_scale_bias: torch.Tensor,
    packed_cond_bias: torch.Tensor,
    packed_zero_cond: torch.Tensor,
    zero_cond_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Precompute AdaLN scale, bias and zero-gate tensors for one layer group."""
  if single_cond.ndim < 2:
    raise ValueError("single_cond must have at least two dimensions")
  n, s = single_cond.shape[-2:]
  layers = cond_norm_scale.shape[0]
  if cond_norm_scale.shape != (layers, s):
    raise ValueError("cond_norm_scale has an incompatible shape")
  if cond_scale_bias.ndim != 2:
    raise ValueError("cond_scale_bias must have shape (layers, channels)")
  channels = cond_scale_bias.shape[1]
  expected = (
      ("cond_scale_bias", cond_scale_bias, (layers, channels)),
      ("zero_cond_bias", zero_cond_bias, (layers, channels)),
  )
  for name, tensor, shape in expected:
    if tuple(tensor.shape) != shape:
      raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
  tensors = (
      cond_norm_scale, packed_cond_scale, cond_scale_bias, packed_cond_bias,
      packed_zero_cond, zero_cond_bias,
  )
  if any(tensor.device != single_cond.device for tensor in tensors):
    raise ValueError("all diffusion conditioning tensors must share a device")
  flat_cond = single_cond.reshape(-1, n, s).contiguous().to(torch.bfloat16)
  output_shape = (layers, flat_cond.shape[0], n, channels)
  adaptive_scale = vecops.planner.empty(
      output_shape, dtype=torch.bfloat16, device=single_cond.device
  )
  adaptive_bias = vecops.planner.empty_like(adaptive_scale)
  zero_gate = vecops.planner.empty_like(adaptive_scale)
  _run_diffusion_conditioning(
      flat_cond,
      cond_norm_scale.contiguous(),
      packed_cond_scale.contiguous(),
      cond_scale_bias.contiguous(),
      packed_cond_bias.contiguous(),
      packed_zero_cond.contiguous(),
      zero_cond_bias.contiguous(),
      adaptive_scale,
      adaptive_bias,
      zero_gate,
      ComputeType=torch.float32,
  )
  return adaptive_scale, adaptive_bias, zero_gate


@vecops.jit(
    source="resources/DiffusionTransitionBlock.cpp",
    recipe_id=_stable_math_recipe("diffusion-transition-block-packed-v24-sme-estimate-other-fast"),
    symbols={"B": vt.Dynamic(lower=0), "BCond": vt.Dynamic(lower=1)},
)
def _diffusion_transition_block_kernel(
    act: vt.In(vecops.bfloat16)["B N C", "Dynamic<1,0,+inf> C 1"],
    adaptive_scale: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    adaptive_bias: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    zero_gate: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    transition1: vt.In(vecops.bfloat16)[
        "2 UP PT", "Dynamic<1,0,+inf> PT 1"
    ],
    transition2: vt.In(vecops.bfloat16)["DOWN PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)["B N C", "Dynamic<1,0,+inf> C 1"],
    *,
    Hidden: vt.Const,
    HiddenPanel: vt.Const,
    OutputPanel: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_diffusion_transition_block = vecops.ops.torch.register(
    "sparsefold_native::diffusion_transition_block",
    _diffusion_transition_block_kernel,
    return_outputs=True,
)


@vecops.jit(
    source="resources/DiffusionSelfAttention.cpp",
    recipe_id=_stable_math_recipe("diffusion-self-attention-packed-v19-estimate-softmax"),
    symbols={
        "B": vt.Dynamic(lower=0),
        "BMask": vt.Dynamic(lower=1),
        "BPair": vt.Dynamic(lower=1),
        "BCond": vt.Dynamic(lower=1),
    },
)
def _diffusion_self_attention_kernel(
    act: vt.In(vecops.bfloat16)["B N C", "Dynamic<1,0,+inf> C 1"],
    key_mask: vt.In(vecops.bool_)["BMask N", "Dynamic<1,0,+inf> 1"],
    pair_logits: vt.In(vecops.bfloat16)[
        "BPair H N N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    adaptive_scale: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    adaptive_bias: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    zero_gate: vt.In(vecops.bfloat16)[
        "BCond N C", "Dynamic<1,0,+inf> C 1"
    ],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    q_bias: vt.In(vecops.bfloat16)["H D", "D 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    gate_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)["B N C", "Dynamic<1,0,+inf> C 1"],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_diffusion_self_attention = vecops.ops.torch.register(
    "sparsefold_native::diffusion_self_attention",
    _diffusion_self_attention_kernel,
    return_outputs=True,
)


@vecops.jit(
    source="resources/DiffusionCrossAttention.cpp",
    recipe_id=_stable_math_recipe("diffusion-cross-attention-packed-v9-softmax-estimate"),
    symbols={
        "B": vt.Dynamic(lower=0),
        "BQMask": vt.Dynamic(lower=1),
        "BKMask": vt.Dynamic(lower=1),
        "BPair": vt.Dynamic(lower=1),
        "BQCond": vt.Dynamic(lower=1),
        "BKCond": vt.Dynamic(lower=1),
    },
)
def _diffusion_cross_attention_kernel(
    x_q: vt.In(vecops.bfloat16)["B Q C", "Dynamic<1,0,+inf> C 1"],
    x_k: vt.In(vecops.bfloat16)["B K C", "Dynamic<1,0,+inf> C 1"],
    mask_q: vt.In(vecops.bool_)["BQMask Q", "Dynamic<1,0,+inf> 1"],
    mask_k: vt.In(vecops.bool_)["BKMask K", "Dynamic<1,0,+inf> 1"],
    pair_logits: vt.In(vecops.bfloat16)[
        "BPair H Q K", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> K 1"
    ],
    q_adaptive_scale: vt.In(vecops.bfloat16)[
        "BQCond Q C", "Dynamic<1,0,+inf> C 1"
    ],
    q_adaptive_bias: vt.In(vecops.bfloat16)[
        "BQCond Q C", "Dynamic<1,0,+inf> C 1"
    ],
    k_adaptive_scale: vt.In(vecops.bfloat16)[
        "BKCond K C", "Dynamic<1,0,+inf> C 1"
    ],
    k_adaptive_bias: vt.In(vecops.bfloat16)[
        "BKCond K C", "Dynamic<1,0,+inf> C 1"
    ],
    zero_gate: vt.In(vecops.bfloat16)[
        "BQCond Q C", "Dynamic<1,0,+inf> C 1"
    ],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    q_bias: vt.In(vecops.bfloat16)["H D", "D 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    gate_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)["B Q C", "Dynamic<1,0,+inf> C 1"],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_diffusion_cross_attention = vecops.ops.torch.register(
    "sparsefold_native::diffusion_cross_attention",
    _diffusion_cross_attention_kernel,
    return_outputs=True,
)


@vecops.jit(
    source="resources/DiffusionCrossAttentionIndexed.cpp",
    recipe_id=_stable_math_recipe("diffusion-cross-attention-indexed-packed-v15-estimate-softmax"),
    symbols={"B": vt.Dynamic(lower=0)},
)
def _diffusion_cross_attention_indexed_kernel(
    act: vt.In(vecops.bfloat16)["B A C", "Dynamic<1,0,+inf> C 1"],
    query_mask: vt.In(vecops.bool_)["U Q", "Q 1"],
    key_indices: vt.In(vecops.int32)["U K", "K 1"],
    key_mask: vt.In(vecops.bool_)["U K", "K 1"],
    pair_logits: vt.In(vecops.bfloat16)[
        "U H Q K", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> K 1"
    ],
    q_adaptive_scale: vt.In(vecops.bfloat16)["A C", "C 1"],
    q_adaptive_bias: vt.In(vecops.bfloat16)["A C", "C 1"],
    k_adaptive_scale: vt.In(vecops.bfloat16)["A C", "C 1"],
    k_adaptive_bias: vt.In(vecops.bfloat16)["A C", "C 1"],
    zero_gate: vt.In(vecops.bfloat16)["A C", "C 1"],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    q_bias: vt.In(vecops.bfloat16)["H D", "D 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    gate_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    output: vt.Out(vecops.bfloat16)["B A C", "Dynamic<1,0,+inf> C 1"],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_diffusion_cross_attention_indexed = vecops.ops.torch.register(
    "sparsefold_native::diffusion_cross_attention_indexed",
    _diffusion_cross_attention_indexed_kernel,
    return_outputs=True,
)


@vecops.jit(
    source="resources/TriangleDistributedProjection.cpp",
    recipe_id=_stable_math_recipe("triangle-distributed-projection-packed-v7-sme-polynomial-sigmoid"),
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _triangle_distributed_projection_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    row_mask: vt.In(vecops.bool_)["Batch L", "Dynamic<1,0,+inf> 1"],
    column_mask: vt.In(vecops.bool_)["Batch N", "Dynamic<1,0,+inf> 1"],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    packed_projection: vt.In(vecops.bfloat16)["4 P", "P 1"],
    normalized: vt.Out(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    a: vt.Out(vecops.bfloat16)[
        "Batch C L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    b: vt.Out(vecops.bfloat16)[
        "Batch C L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_triangle_distributed_projection = vecops.ops.torch.register(
    "sparsefold_native::triangle_distributed_projection",
    _triangle_distributed_projection_kernel,
)


@vecops.jit(
    source="resources/TriangleDistributedUpdate.cpp",
    recipe_id=_stable_math_recipe("triangle-distributed-update-packed-v2-sme-polynomial-sigmoid"),
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _triangle_distributed_update_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    normalized: vt.In(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    left: vt.In(vecops.bfloat16)[
        "Batch C L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    right: vt.In(vecops.bfloat16)[
        "Batch C N N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    center_norm_scale: vt.In(vecops.float32)["C", "1"],
    center_norm_offset: vt.In(vecops.float32)["C", "1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    gating_projection: vt.In(vecops.bfloat16)["1 PG", "PG 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_triangle_distributed_update = vecops.ops.torch.register(
    "sparsefold_native::triangle_distributed_update",
    _triangle_distributed_update_kernel,
)


@vecops.jit(
    source="resources/TriangleMultiplication.cpp",
    recipe_id=_stable_math_recipe("triangle-multiplication-packed-v8-sme-polynomial-sigmoid"),
    symbols={"Batch": vt.Dynamic(lower=0)},
)
def _triangle_multiplication_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch N N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    mask: vt.In(vecops.bool_)["Batch N", "Dynamic<1,0,+inf> 1"],
    left_norm_scale: vt.In(vecops.float32)["C", "1"],
    left_norm_offset: vt.In(vecops.float32)["C", "1"],
    gw_proj: vt.In(vecops.bfloat16)["2 G C", "Dynamic<1,0,+inf> C 1"],
    gw_projection_b: vt.In(vecops.bfloat16)["4 PG", "PG 1"],
    center_norm_scale: vt.In(vecops.float32)["C", "1"],
    center_norm_offset: vt.In(vecops.float32)["C", "1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    gating_linear: vt.In(vecops.bfloat16)["1 PGL", "PGL 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch N N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Dynamic<1,0,+inf> 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_triangle_multiplication = vecops.ops.torch.register(
    "sparsefold_native::triangle_multiplication",
    _triangle_multiplication_kernel,
)


@vecops.jit(
    source="resources/EvoformerSingleActUpdate.cpp",
    recipe_id=_stable_math_recipe("evoformer-single-act-update-packed-v4-softmax-estimate"),
    symbols={"Batch": vt.Dynamic(lower=0)},
)
def _evoformer_single_act_update_kernel(
    pair_act: vt.In(vecops.bfloat16)[
        "Batch N N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    single_act: vt.In(vecops.bfloat16)[
        "Batch N C", "Dynamic<1,0,+inf> C 1"
    ],
    key_padding_mask: vt.In(vecops.bool_)[
        "Batch N", "Dynamic<1,0,+inf> 1"
    ],
    pair_norm_scale: vt.In(vecops.float32)["Q", "1"],
    pair_norm_offset: vt.In(vecops.float32)["Q", "1"],
    logits_projection: vt.In(vecops.bfloat16)["1 PL", "PL 1"],
    attention_norm_scale: vt.In(vecops.float32)["C", "1"],
    attention_norm_offset: vt.In(vecops.float32)["C", "1"],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    q_bias: vt.In(vecops.bfloat16)["H D", "D 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    g_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    o_projection: vt.In(vecops.bfloat16)["H PO", "PO 1"],
    transition_norm_scale: vt.In(vecops.float32)["C", "1"],
    transition_norm_offset: vt.In(vecops.float32)["C", "1"],
    transition1: vt.In(vecops.bfloat16)["2 PT1", "PT1 1"],
    transition2: vt.In(vecops.bfloat16)["1 PT2", "PT2 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch N C", "Dynamic<1,0,+inf> C 1"
    ],
    *,
    Hidden: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_evoformer_single_act_update = vecops.ops.torch.register(
    "sparsefold_native::evoformer_single_act_update",
    _evoformer_single_act_update_kernel,
)


@vecops.jit(
    source="resources/SingleDistributedProjection.cpp",
    recipe_id="sparsefold-v1;single-distributed-projection-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _single_distributed_projection_kernel(
    pair_act: vt.In(vecops.bfloat16)[
        "Batch L N Q", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> Q 1"
    ],
    single_act: vt.In(vecops.bfloat16)[
        "Batch L C", "Dynamic<1,0,+inf> C 1"
    ],
    pair_norm_scale: vt.In(vecops.float32)["Q", "1"],
    pair_norm_offset: vt.In(vecops.float32)["Q", "1"],
    logits_projection: vt.In(vecops.bfloat16)["1 PL", "PL 1"],
    attention_norm_scale: vt.In(vecops.float32)["C", "1"],
    attention_norm_offset: vt.In(vecops.float32)["C", "1"],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    q_bias: vt.In(vecops.bfloat16)["H D", "D 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    g_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    pair_logits: vt.Out(vecops.bfloat16)[
        "Batch H L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    q: vt.Out(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    k: vt.Out(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    v: vt.Out(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    gate: vt.Out(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_single_distributed_projection = vecops.ops.torch.register(
    "sparsefold_native::single_distributed_projection",
    _single_distributed_projection_kernel,
)


@vecops.jit(
    source="resources/SingleDistributedUpdate.cpp",
    recipe_id=_stable_math_recipe("single-distributed-update-packed-v2-softmax-estimate"),
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _single_distributed_update_kernel(
    single_act: vt.In(vecops.bfloat16)[
        "Batch L C", "Dynamic<1,0,+inf> C 1"
    ],
    key_padding_mask: vt.In(vecops.bool_)["Batch N", "Dynamic<1,0,+inf> 1"],
    pair_logits: vt.In(vecops.bfloat16)[
        "Batch H L N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    q: vt.In(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    global_k: vt.In(vecops.bfloat16)[
        "Batch H N D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    global_v: vt.In(vecops.bfloat16)[
        "Batch H N D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    gate: vt.In(vecops.bfloat16)[
        "Batch H L D", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> D 1"
    ],
    o_projection: vt.In(vecops.bfloat16)["H PO", "PO 1"],
    transition_norm_scale: vt.In(vecops.float32)["C", "1"],
    transition_norm_offset: vt.In(vecops.float32)["C", "1"],
    transition1: vt.In(vecops.bfloat16)["2 PT1", "PT1 1"],
    transition2: vt.In(vecops.bfloat16)["1 PT2", "PT2 1"],
    output: vt.Out(vecops.bfloat16)[
        "Batch L C", "Dynamic<1,0,+inf> C 1"
    ],
    *,
    Hidden: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_single_distributed_update = vecops.ops.torch.register(
    "sparsefold_native::single_distributed_update",
    _single_distributed_update_kernel,
)


@vecops.jit(
    source="resources/GridSelfAttention.cpp",
    recipe_id=_stable_math_recipe("grid-self-attention-packed-v4-sme-polynomial-exp"),
    symbols={"Batch": vt.Dynamic(lower=0)},
)
def _grid_self_attention_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch N N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    key_padding_mask: vt.In(vecops.bool_)[
        "Batch N", "Dynamic<1,0,+inf> 1"
    ],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    g_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    bias_projection: vt.In(vecops.bfloat16)["1 PB", "PB 1"],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    index_map: vt.InOut(vecops.int32)[
        "Batch N H T S", "Dynamic Dynamic Dynamic S 1"
    ],
    weight_map: vt.InOut(vecops.float32)[
        "Batch N H T S", "Dynamic Dynamic Dynamic S 1"
    ],
    output: vt.Out(vecops.bfloat16)[
        "Batch N N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    random_blocks: vt.Scalar[vecops.int64],
    static_probability: vt.Scalar[vecops.float32],
    random_probability: vt.Scalar[vecops.float32],
    minimum_probability: vt.Scalar[vecops.float32],
    generate_maps: bool,
    seed: vt.Scalar[vecops.int64],
    *,
    Transpose: vt.Const = 0,
    HeadDim: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_grid_self_attention = vecops.ops.torch.register(
    "sparsefold_native::grid_self_attention",
    _grid_self_attention_kernel,
)


@vecops.jit(
    source="resources/GridDistributedProjection.cpp",
    recipe_id="sparsefold-v1;grid-distributed-projection-packed-v1",
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _grid_distributed_projection_kernel(
    act: vt.In(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    norm_scale: vt.In(vecops.float32)["C", "1"],
    norm_offset: vt.In(vecops.float32)["C", "1"],
    bias_projection: vt.In(vecops.bfloat16)["1 PB", "PB 1"],
    normalized: vt.Out(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    bias_send: vt.Out(vecops.bfloat16)[
        "L Batch H N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    *,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_grid_distributed_projection = vecops.ops.torch.register(
    "sparsefold_native::grid_distributed_projection",
    _grid_distributed_projection_kernel,
)


@vecops.jit(
    source="resources/GridDistributedAttention.cpp",
    recipe_id=_stable_math_recipe("grid-distributed-attention-packed-v2-softmax-estimate"),
    symbols={"Batch": vt.Dynamic(lower=0), "L": vt.Dynamic(lower=0)},
)
def _grid_distributed_attention_kernel(
    normalized: vt.In(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    key_padding_mask: vt.In(vecops.bool_)["Batch N", "Dynamic<1,0,+inf> 1"],
    global_bias: vt.In(vecops.bfloat16)[
        "N Batch H N", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> N 1"
    ],
    q_projection: vt.In(vecops.bfloat16)["H PQ", "PQ 1"],
    k_projection: vt.In(vecops.bfloat16)["H PK", "PK 1"],
    v_projection: vt.In(vecops.bfloat16)["H PV", "PV 1"],
    output_projection: vt.In(vecops.bfloat16)["1 PO", "PO 1"],
    g_projection: vt.In(vecops.bfloat16)["H PG", "PG 1"],
    index_map: vt.InOut(vecops.int32)[
        "Batch L H T S", "Dynamic Dynamic Dynamic S 1"
    ],
    weight_map: vt.InOut(vecops.float32)[
        "Batch L H T S", "Dynamic Dynamic Dynamic S 1"
    ],
    output: vt.Out(vecops.bfloat16)[
        "Batch L N C", "Dynamic<1,0,+inf> Dynamic<1,0,+inf> C 1"
    ],
    row_offset: vt.Scalar[vecops.int64],
    random_blocks: vt.Scalar[vecops.int64],
    static_probability: vt.Scalar[vecops.float32],
    random_probability: vt.Scalar[vecops.float32],
    minimum_probability: vt.Scalar[vecops.float32],
    generate_maps: bool,
    seed: vt.Scalar[vecops.int64],
    *,
    HeadDim: vt.Const,
    ComputeType: vt.DType = vecops.float32,
) -> None: ...


_run_grid_distributed_attention = vecops.ops.torch.register(
    "sparsefold_native::grid_distributed_attention",
    _grid_distributed_attention_kernel,
)


_grid_seed_counter = itertools.count()


def set_grid_attention_seed(seed: int) -> None:
  """Reset the deterministic per-call seed sequence used by IBS sampling."""
  global _grid_seed_counter
  _grid_seed_counter = itertools.count(int(seed))


@vt.check_tensors(symbols={"B": vt.Dynamic(lower=0)})
def transition_block(
    act: vt.Tensor(vecops.bfloat16)["*B N C"],
    ln_scale: vt.Tensor(vecops.float32)["C"],
    ln_offset: vt.Tensor(vecops.float32)["C"],
    transition1: vt.Tensor(vecops.bfloat16)["C 2*M"],
    transition2: vt.Tensor(vecops.bfloat16)["M C"],
) -> torch.Tensor:
  """Run the fused out-style transition block and return its owned output."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(f"transition_block only supports CPU tensors, got {act.device}")
  n, c = act.shape[-2:]
  m = transition2.shape[0]
  if any(tensor.device != act.device for tensor in (
      ln_scale, ln_offset, transition1, transition2
  )):
    raise ValueError("all transition_block tensors must be on the same device")

  flat_act = act.reshape(-1, n, c).contiguous()
  transition1_3d = transition1.reshape(c, 2, m).contiguous()
  transition2_2d = transition2.contiguous()
  output = vecops.planner.empty_like(flat_act)
  _run_transition_block(
      flat_act,
      ln_scale.contiguous(),
      ln_offset.contiguous(),
      transition1_3d,
      transition2_2d,
      output,
      ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def msa_attention(
    act: torch.Tensor,
    mask: torch.Tensor,
    pair_act: torch.Tensor,
    act_norm_scale: torch.Tensor,
    act_norm_offset: torch.Tensor,
    pair_norm_scale: torch.Tensor,
    pair_norm_offset: torch.Tensor,
    pair_logits_projection: torch.Tensor,
    value_projection: torch.Tensor,
    gate_projection: torch.Tensor,
    output_projection: torch.Tensor,
) -> torch.Tensor:
  """Run MSA attention, packing all learned B operands for this call."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(f"msa_attention only supports CPU tensors, got {act.device}")
  if act.ndim < 3:
    raise ValueError("act must have shape (..., M, N, C)")
  batch_shape = act.shape[:-3]
  m, n, c = act.shape[-3:]
  mask = torch.broadcast_to(mask, batch_shape + (m, n))
  q = pair_act.shape[-1]
  pair_act = torch.broadcast_to(pair_act, batch_shape + (n, n, q))
  if tuple(pair_logits_projection.shape) != (q, pair_logits_projection.shape[-1]):
    raise ValueError("pair_logits_projection must have shape (Q, H)")
  heads = pair_logits_projection.shape[-1]
  value_channels = value_projection.numel() // c
  if value_channels % heads or value_channels != c:
    raise ValueError("MSA attention requires H * HeadDim == C")
  head_dim = value_channels // heads
  expected = (
      ("act_norm_scale", act_norm_scale, (c,)),
      ("act_norm_offset", act_norm_offset, (c,)),
      ("pair_norm_scale", pair_norm_scale, (q,)),
      ("pair_norm_offset", pair_norm_offset, (q,)),
      ("value_projection", value_projection, (c, heads, head_dim)),
      ("gate_projection", gate_projection, (c, c)),
      ("output_projection", output_projection, (c, c)),
  )
  for name, tensor, shape in expected:
    if tuple(tensor.shape) != shape:
      raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
  tensors = (
      mask, pair_act, act_norm_scale, act_norm_offset, pair_norm_scale,
      pair_norm_offset, pair_logits_projection, value_projection,
      gate_projection, output_projection,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all msa_attention tensors must share a device")

  packed_pair = pack_matmul_b(
      pair_logits_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_value = pack_matmul_b(
      value_projection.reshape(c, c).transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_gate = pack_matmul_b(
      gate_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_output = pack_matmul_b(
      output_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return msa_attention_packed(
      act, mask, pair_act, act_norm_scale, act_norm_offset,
      pair_norm_scale, pair_norm_offset, packed_pair, packed_value,
      packed_gate, packed_output, heads=heads, head_dim=head_dim,
  )


def msa_attention_packed(
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
  """Run MSA attention with persistent architecture-specific packed weights."""
  batch_shape = act.shape[:-3]
  m, n, c = act.shape[-3:]
  q = pair_act.shape[-1]
  mask = torch.broadcast_to(mask, batch_shape + (m, n))
  pair_act = torch.broadcast_to(pair_act, batch_shape + (n, n, q))
  flat_act = act.reshape(-1, m, n, c).contiguous()
  flat_mask = mask.reshape(-1, m, n)
  flat_pair = pair_act.reshape(-1, n, n, q).contiguous()
  output = vecops.planner.empty_like(flat_act)
  _run_msa_attention(
      flat_act, flat_mask, flat_pair,
      act_norm_scale.contiguous(), act_norm_offset.contiguous(),
      pair_norm_scale.contiguous(), pair_norm_offset.contiguous(),
      packed_pair_logits, packed_value, packed_gate, packed_output, output,
      H=int(heads), HeadDim=int(head_dim), ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def msa_distributed_projection_packed(
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
    *,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project a local MSA sequence shard before exchanging values."""
  batch_shape = act.shape[:-3]
  msa_rows, local_rows, channels = act.shape[-3:]
  sequence, pair_channels = pair_act.shape[-2:]
  batch = math.prod(batch_shape) if batch_shape else 1
  mask = torch.broadcast_to(mask, batch_shape + (msa_rows, sequence))
  flat_act = act.reshape(batch, msa_rows, local_rows, channels).contiguous()
  flat_mask = mask.reshape(batch, msa_rows, sequence).contiguous()
  flat_pair = pair_act.reshape(
      batch, local_rows, sequence, pair_channels
  ).contiguous()
  weights = vecops.planner.empty(
      (batch, heads, local_rows, sequence),
      dtype=act.dtype,
      device=act.device,
  )
  value_send = vecops.planner.empty(
      (local_rows, batch, msa_rows, channels),
      dtype=act.dtype,
      device=act.device,
  )
  gate = vecops.planner.empty_like(flat_act)
  _run_msa_distributed_projection(
      flat_act, flat_mask, flat_pair,
      act_norm_scale.contiguous(), act_norm_offset.contiguous(),
      pair_norm_scale.contiguous(), pair_norm_offset.contiguous(),
      packed_pair_logits, packed_value, packed_gate,
      weights, value_send, gate, ComputeType=torch.float32,
  )
  return (
      weights.reshape(batch_shape + (heads, local_rows, sequence)),
      value_send,
      gate.reshape_as(act),
  )


def msa_distributed_update_packed(
    weights: torch.Tensor,
    global_value: torch.Tensor,
    gate: torch.Tensor,
    packed_output: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
  """Finish a local MSA sequence shard after the value AllGather."""
  batch_shape = gate.shape[:-3]
  msa_rows, local_rows, channels = gate.shape[-3:]
  batch = math.prod(batch_shape) if batch_shape else 1
  heads, sequence = weights.shape[-3], weights.shape[-1]
  flat_weights = weights.reshape(
      batch, heads, local_rows, sequence
  ).contiguous()
  expected_value_shape = (sequence, batch, msa_rows, channels)
  if global_value.shape != expected_value_shape:
    raise ValueError(
        f"global_value must have shape {expected_value_shape}, "
        f"got {tuple(global_value.shape)}"
    )
  flat_gate = gate.reshape(batch, msa_rows, local_rows, channels).contiguous()
  output = vecops.planner.empty_like(flat_gate)
  _run_msa_distributed_update(
      flat_weights, global_value.contiguous(), flat_gate, packed_output,
      output, HeadDim=head_dim, ComputeType=torch.float32,
  )
  return output.reshape_as(gate)


def outer_product_mean(
    act: torch.Tensor,
    mask: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    left_projection: torch.Tensor,
    right_projection: torch.Tensor,
    output_projection: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
  """Run outer-product mean, packing all learned B operands for this call."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"outer_product_mean only supports CPU tensors, got {act.device}"
    )
  if act.ndim < 3:
    raise ValueError("act must have shape (..., M, N, C)")
  batch_shape = act.shape[:-3]
  m, n, c = act.shape[-3:]
  mask = torch.broadcast_to(mask, batch_shape + (m, n))
  outer_channels = left_projection.shape[1]
  output_channels = output_bias.shape[0]
  expected = (
      ("norm_scale", norm_scale, (c,)),
      ("norm_offset", norm_offset, (c,)),
      ("left_projection", left_projection, (c, outer_channels)),
      ("right_projection", right_projection, (c, outer_channels)),
      ("output_projection", output_projection,
       (outer_channels, outer_channels, output_channels)),
  )
  for name, tensor, shape in expected:
    if tuple(tensor.shape) != shape:
      raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
  tensors = (
      mask, norm_scale, norm_offset, left_projection, right_projection,
      output_projection, output_bias,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all outer_product_mean tensors must share a device")

  packed_left = pack_matmul_b(
      left_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_right = pack_matmul_b(
      right_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  # The core produces (d2,d1) rows directly. Keep that order in the packed
  # learned projection so no D x D transpose sits in the hot tile loop.
  packed_output = pack_matmul_b(
      output_projection.permute(2, 1, 0).reshape(
          output_channels, outer_channels * outer_channels
      ).contiguous()
  ).unsqueeze(0)
  return outer_product_mean_packed(
      act, mask, norm_scale, norm_offset, packed_left, packed_right,
      packed_output, output_bias, outer_channels=outer_channels,
  )


def outer_product_mean_packed(
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
  """Run outer-product mean with persistent architecture-specific weights."""
  batch_shape = act.shape[:-3]
  m, n, c = act.shape[-3:]
  mask = torch.broadcast_to(mask, batch_shape + (m, n))
  flat_act = act.reshape(-1, m, n, c).contiguous()
  flat_mask = mask.reshape(-1, m, n)
  output_channels = output_bias.shape[0]
  output = vecops.planner.empty(
      (flat_act.shape[0], n, n, output_channels),
      dtype=act.dtype,
      device=act.device,
  )
  _run_outer_product_mean(
      flat_act, flat_mask, norm_scale.contiguous(), norm_offset.contiguous(),
      packed_left, packed_right, packed_output, output_bias.contiguous(),
      output, OuterChannels=int(outer_channels), ComputeType=torch.float32,
  )
  return output.reshape(batch_shape + (n, n, output_channels))


def outer_distributed_projection_packed(
    act: torch.Tensor,
    local_mask: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    packed_left: torch.Tensor,
    packed_right: torch.Tensor,
    *,
    outer_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Project local MSA activations before exchanging the right operand."""
  batch_shape = act.shape[:-3]
  msa_rows, local_rows, channels = act.shape[-3:]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_act = act.reshape(batch, msa_rows, local_rows, channels).contiguous()
  flat_mask = local_mask.reshape(batch, msa_rows, local_rows).contiguous()
  left = vecops.planner.empty(
      (batch, msa_rows, local_rows, outer_channels),
      dtype=act.dtype, device=act.device,
  )
  right_send = vecops.planner.empty(
      (local_rows, batch, msa_rows, outer_channels),
      dtype=act.dtype, device=act.device,
  )
  _run_outer_distributed_projection(
      flat_act, flat_mask, norm_scale.contiguous(), norm_offset.contiguous(),
      packed_left, packed_right, left, right_send,
      OuterChannels=outer_channels, ComputeType=torch.float32,
  )
  return left.reshape(batch_shape + (msa_rows, local_rows, outer_channels)), right_send


def outer_distributed_right_projection_packed(
    act: torch.Tensor,
    local_mask: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    packed_right: torch.Tensor,
    *,
    outer_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Normalize locally and produce the right operand in gather layout."""
  batch_shape = act.shape[:-3]
  msa_rows, local_rows, channels = act.shape[-3:]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_act = act.reshape(batch, msa_rows, local_rows, channels).contiguous()
  flat_mask = local_mask.reshape(batch, msa_rows, local_rows).contiguous()
  normalized = vecops.planner.empty_like(flat_act)
  right_send = vecops.planner.empty(
      (local_rows, batch, msa_rows, outer_channels),
      dtype=act.dtype,
      device=act.device,
  )
  _run_outer_distributed_right_projection(
      flat_act,
      flat_mask,
      norm_scale.contiguous(),
      norm_offset.contiguous(),
      packed_right,
      normalized,
      right_send,
      OuterChannels=outer_channels,
      ComputeType=torch.float32,
  )
  return (
      normalized.reshape(batch_shape + (msa_rows, local_rows, channels)),
      right_send,
  )


def outer_distributed_left_projection_packed(
    normalized: torch.Tensor,
    local_mask: torch.Tensor,
    packed_left: torch.Tensor,
    *,
    outer_channels: int,
) -> torch.Tensor:
  """Produce the local left operand while the right gather is in flight."""
  batch_shape = normalized.shape[:-3]
  msa_rows, local_rows, channels = normalized.shape[-3:]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_normalized = normalized.reshape(
      batch, msa_rows, local_rows, channels
  ).contiguous()
  flat_mask = local_mask.reshape(batch, msa_rows, local_rows).contiguous()
  left = vecops.planner.empty(
      (batch, msa_rows, local_rows, outer_channels),
      dtype=normalized.dtype,
      device=normalized.device,
  )
  _run_outer_distributed_left_projection(
      flat_normalized,
      flat_mask,
      packed_left,
      left,
      OuterChannels=outer_channels,
      ComputeType=torch.float32,
  )
  return left.reshape(
      batch_shape + (msa_rows, local_rows, outer_channels)
  )


def outer_distributed_update_packed(
    left: torch.Tensor,
    global_right: torch.Tensor,
    mask: torch.Tensor,
    packed_output: torch.Tensor,
    output_bias: torch.Tensor,
    row_offset: int,
    *,
    outer_channels: int,
) -> torch.Tensor:
  """Finish local outer-product rows using the globally gathered right side."""
  batch_shape = left.shape[:-3]
  msa_rows, local_rows = left.shape[-3:-1]
  sequence = mask.shape[-1]
  batch = math.prod(batch_shape) if batch_shape else 1
  output_channels = output_bias.shape[0]
  flat_left = left.reshape(
      batch, msa_rows, local_rows, outer_channels
  ).contiguous()
  flat_mask = mask.reshape(batch, msa_rows, sequence).contiguous()
  output = vecops.planner.empty(
      (batch, local_rows, sequence, output_channels),
      dtype=left.dtype, device=left.device,
  )
  _run_outer_distributed_update(
      flat_left, global_right.contiguous(), flat_mask, packed_output,
      output_bias.contiguous(), output, int(row_offset),
      OuterChannels=outer_channels, ComputeType=torch.float32,
  )
  return output.reshape(batch_shape + (local_rows, sequence, output_channels))


@vt.check_tensors(symbols={"B": vt.Dynamic(lower=0)})
def diffusion_transition_block(
    act: vt.Tensor(vecops.bfloat16)["*B N C"],
    single_cond: vt.Tensor(vecops.bfloat16)["*B N S"],
    cond_norm_scale: vt.Tensor(vecops.float32)["S"],
    cond_scale_projection: vt.Tensor(vecops.bfloat16)["S C"],
    cond_scale_bias: vt.Tensor(vecops.bfloat16)["C"],
    cond_bias_projection: vt.Tensor(vecops.bfloat16)["S C"],
    transition1: vt.Tensor(vecops.bfloat16)["C 2*M"],
    transition2: vt.Tensor(vecops.bfloat16)["M C"],
    zero_cond_projection: vt.Tensor(vecops.bfloat16)["S C"],
    zero_cond_bias: vt.Tensor(vecops.bfloat16)["C"],
) -> torch.Tensor:
  """Run the conditional diffusion SwiGLU transition block."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"diffusion_transition_block only supports CPU tensors, got {act.device}"
    )
  n, c = act.shape[-2:]
  s = single_cond.shape[-1]
  single_cond = torch.broadcast_to(
      single_cond, act.shape[:-2] + (n, s)
  )
  m = transition2.shape[0]
  expected = (
      ("cond_norm_scale", cond_norm_scale, (s,)),
      ("cond_scale_projection", cond_scale_projection, (s, c)),
      ("cond_scale_bias", cond_scale_bias, (c,)),
      ("cond_bias_projection", cond_bias_projection, (s, c)),
      ("transition1", transition1, (c, 2 * m)),
      ("transition2", transition2, (m, c)),
      ("zero_cond_projection", zero_cond_projection, (s, c)),
      ("zero_cond_bias", zero_cond_bias, (c,)),
  )
  for name, tensor, shape in expected:
    if tuple(tensor.shape) != shape:
      raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
  tensors = (
      single_cond, cond_norm_scale, cond_scale_projection, cond_scale_bias,
      cond_bias_projection, transition1, transition2, zero_cond_projection,
      zero_cond_bias,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all diffusion_transition_block tensors must share a device")

  flat_act = act.reshape(-1, n, c).contiguous()
  flat_cond = single_cond.reshape(-1, n, s)
  packed_cond_scale = pack_matmul_b(
      cond_scale_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_cond_bias = pack_matmul_b(
      cond_bias_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  hidden_panel = _divisor_panel(m)
  output_panel = _divisor_panel(c)
  packed_transition1 = pack_matmul_b(
      transition1.reshape(c, 2, m)
      .permute(1, 2, 0)
      .reshape(2, m // hidden_panel, hidden_panel, c)
  )
  packed_transition2 = pack_matmul_b(
      transition2.transpose(0, 1).reshape(c // output_panel, output_panel, m)
  )
  packed_zero_cond = pack_matmul_b(
      zero_cond_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return diffusion_transition_block_packed(
      flat_act,
      flat_cond,
      cond_norm_scale.contiguous(),
      packed_cond_scale,
      cond_scale_bias.contiguous(),
      packed_cond_bias,
      packed_transition1,
      packed_transition2,
      packed_zero_cond,
      zero_cond_bias.contiguous(),
      hidden=m,
      hidden_panel=hidden_panel,
      output_panel=output_panel,
  ).reshape_as(act)


def _flatten_shared_batch(
    tensor: torch.Tensor,
    batch_shape: tuple[int, ...],
    tail_shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
  """Flatten a possibly shared leading batch without materializing broadcast."""
  if tuple(tensor.shape[-len(tail_shape):]) != tail_shape:
    raise ValueError(
        f"{name} must end with {tail_shape}, got {tuple(tensor.shape)}"
    )
  flat = tensor.reshape((-1,) + tail_shape)
  target_batch = math.prod(batch_shape)
  if target_batch % flat.shape[0] != 0:
    raise ValueError(
        f"{name} batch {flat.shape[0]} does not broadcast to {target_batch}"
    )
  return flat


def diffusion_transition_block_packed(
    act: torch.Tensor,
    single_cond: torch.Tensor,
    cond_norm_scale: torch.Tensor,
    packed_cond_scale: torch.Tensor,
    cond_scale_bias: torch.Tensor,
    packed_cond_bias: torch.Tensor,
    packed_transition1: torch.Tensor,
    packed_transition2: torch.Tensor,
    packed_zero_cond: torch.Tensor,
    zero_cond_bias: torch.Tensor,
    *,
    hidden: int,
    hidden_panel: int,
    output_panel: int,
) -> torch.Tensor:
  """Run a diffusion transition and residual with packed B operands."""
  batch_shape = act.shape[:-2]
  n, c = act.shape[-2:]
  s = single_cond.shape[-1]
  flat_cond = _flatten_shared_batch(
      single_cond, batch_shape, (n, s), "single_cond"
  )
  adaptive_scale, adaptive_bias, zero_gate = diffusion_conditioning_packed(
      flat_cond,
      cond_norm_scale.unsqueeze(0),
      packed_cond_scale,
      cond_scale_bias.unsqueeze(0),
      packed_cond_bias,
      packed_zero_cond,
      zero_cond_bias.unsqueeze(0),
  )
  return diffusion_transition_block_prepared_packed(
      act,
      adaptive_scale[0],
      adaptive_bias[0],
      zero_gate[0],
      packed_transition1,
      packed_transition2,
      hidden=hidden,
      hidden_panel=hidden_panel,
      output_panel=output_panel,
  )


def diffusion_transition_block_prepared_packed(
    act: torch.Tensor,
    adaptive_scale: torch.Tensor,
    adaptive_bias: torch.Tensor,
    zero_gate: torch.Tensor,
    packed_transition1: torch.Tensor,
    packed_transition2: torch.Tensor,
    *,
    hidden: int,
    hidden_panel: int,
    output_panel: int,
) -> torch.Tensor:
  """Run a diffusion transition using precomputed shared conditioning."""
  batch_shape = act.shape[:-2]
  n, c = act.shape[-2:]
  flat_act = act.reshape(-1, n, c).contiguous()
  flat_scale = _flatten_shared_batch(
      adaptive_scale, batch_shape, (n, c), "adaptive_scale"
  )
  flat_bias = _flatten_shared_batch(
      adaptive_bias, batch_shape, (n, c), "adaptive_bias"
  )
  flat_zero = _flatten_shared_batch(
      zero_gate, batch_shape, (n, c), "zero_gate"
  )
  output = vecops.planner.empty_like(flat_act)
  _run_diffusion_transition_block(
      flat_act, flat_scale, flat_bias, flat_zero, packed_transition1,
      packed_transition2, output,
      Hidden=hidden, HiddenPanel=hidden_panel, OutputPanel=output_panel,
      ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def diffusion_self_attention(
    act: torch.Tensor,
    key_mask: torch.Tensor,
    pair_logits: torch.Tensor | None,
    single_cond: torch.Tensor,
    cond_norm_scale: torch.Tensor,
    cond_scale_projection: torch.Tensor,
    cond_scale_bias: torch.Tensor,
    cond_bias_projection: torch.Tensor,
    q_projection: torch.Tensor,
    q_bias: torch.Tensor,
    k_projection: torch.Tensor,
    v_projection: torch.Tensor,
    gate_projection: torch.Tensor,
    output_projection: torch.Tensor,
    zero_cond_projection: torch.Tensor,
    zero_cond_bias: torch.Tensor,
) -> torch.Tensor:
  """Run conditional pair-biased diffusion self-attention and residual."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"diffusion_self_attention only supports CPU tensors, got {act.device}"
    )
  batch_shape = act.shape[:-2]
  n, c = act.shape[-2:]
  s = single_cond.shape[-1]
  heads, projection_channels, head_dim = q_projection.shape
  if projection_channels != c or heads * head_dim != c:
    raise ValueError(
        f"self attention requires q_projection (H,C,D) with H*D=C; got "
        f"{tuple(q_projection.shape)} and C={c}"
    )
  if pair_logits is None:
    pair_logits = torch.zeros(
        (1, heads, n, n), dtype=act.dtype, device=act.device
    )
  tensors = (
      key_mask, pair_logits, single_cond, cond_norm_scale,
      cond_scale_projection, cond_scale_bias, cond_bias_projection,
      q_projection, q_bias, k_projection, v_projection, gate_projection,
      output_projection, zero_cond_projection, zero_cond_bias,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all diffusion_self_attention tensors must share a device")
  flat_act = act.reshape(-1, n, c).contiguous()
  # The native kernel accepts a dynamic leading stride.  Preserve broadcast
  # (stride-zero) conditioning, masks and pair bias instead of materializing
  # B identical copies on every invocation.
  flat_cond = _flatten_shared_batch(
      single_cond, batch_shape, (n, s), "single_cond"
  )
  flat_mask = _flatten_shared_batch(key_mask, batch_shape, (n,), "key_mask")
  flat_pair = _flatten_shared_batch(
      pair_logits, batch_shape, (heads, n, n), "pair_logits"
  )
  packed_cond_scale = pack_matmul_b(
      cond_scale_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_cond_bias = pack_matmul_b(
      cond_bias_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_q = pack_matmul_b(q_projection.permute(0, 2, 1).contiguous())
  packed_k = pack_matmul_b(k_projection.permute(0, 2, 1).contiguous())
  packed_v = pack_matmul_b(v_projection.permute(0, 2, 1).contiguous())
  packed_gate = pack_matmul_b(
      gate_projection.permute(0, 2, 1).contiguous()
  )
  packed_output = pack_matmul_b(
      output_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_zero = pack_matmul_b(
      zero_cond_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return diffusion_self_attention_packed(
      flat_act, flat_mask, flat_pair, flat_cond,
      cond_norm_scale.contiguous(), packed_cond_scale,
      cond_scale_bias.contiguous(), packed_cond_bias,
      packed_q, q_bias.contiguous(), packed_k, packed_v, packed_gate,
      packed_output, packed_zero, zero_cond_bias.contiguous(),
  ).reshape_as(act)


def diffusion_self_attention_packed(
    act: torch.Tensor,
    key_mask: torch.Tensor,
    pair_logits: torch.Tensor,
    single_cond: torch.Tensor,
    cond_norm_scale: torch.Tensor,
    packed_cond_scale: torch.Tensor,
    cond_scale_bias: torch.Tensor,
    packed_cond_bias: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
    packed_zero: torch.Tensor,
    zero_cond_bias: torch.Tensor,
) -> torch.Tensor:
  """Run diffusion self-attention and residual with packed B operands."""
  batch_shape = act.shape[:-2]
  n, c = act.shape[-2:]
  s = single_cond.shape[-1]
  flat_cond = _flatten_shared_batch(
      single_cond, batch_shape, (n, s), "single_cond"
  )
  adaptive_scale, adaptive_bias, zero_gate = diffusion_conditioning_packed(
      flat_cond,
      cond_norm_scale.unsqueeze(0),
      packed_cond_scale,
      cond_scale_bias.unsqueeze(0),
      packed_cond_bias,
      packed_zero,
      zero_cond_bias.unsqueeze(0),
  )
  return diffusion_self_attention_prepared_packed(
      act,
      key_mask,
      pair_logits,
      adaptive_scale[0],
      adaptive_bias[0],
      zero_gate[0],
      packed_q,
      q_bias,
      packed_k,
      packed_v,
      packed_gate,
      packed_output,
  )


def diffusion_self_attention_prepared_packed(
    act: torch.Tensor,
    key_mask: torch.Tensor,
    pair_logits: torch.Tensor,
    adaptive_scale: torch.Tensor,
    adaptive_bias: torch.Tensor,
    zero_gate: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
) -> torch.Tensor:
  """Run diffusion self-attention using precomputed shared conditioning."""
  batch_shape = act.shape[:-2]
  n, c = act.shape[-2:]
  heads = q_bias.shape[0]
  if pair_logits is None:
    pair_logits = torch.zeros(
        (1, heads, n, n), dtype=act.dtype, device=act.device
    )
  flat_act = act.reshape(-1, n, c).contiguous()
  flat_scale = _flatten_shared_batch(
      adaptive_scale, batch_shape, (n, c), "adaptive_scale"
  )
  flat_bias = _flatten_shared_batch(
      adaptive_bias, batch_shape, (n, c), "adaptive_bias"
  )
  flat_zero = _flatten_shared_batch(
      zero_gate, batch_shape, (n, c), "zero_gate"
  )
  flat_mask = _flatten_shared_batch(key_mask, batch_shape, (n,), "key_mask")
  flat_pair = _flatten_shared_batch(
      pair_logits, batch_shape, (heads, n, n), "pair_logits"
  )
  output = vecops.planner.empty_like(flat_act)
  _run_diffusion_self_attention(
      flat_act, flat_mask, flat_pair, flat_scale, flat_bias, flat_zero,
      packed_q, q_bias, packed_k, packed_v, packed_gate, packed_output, output,
      ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def diffusion_cross_attention(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    pair_logits: torch.Tensor | None,
    single_cond_q: torch.Tensor,
    single_cond_k: torch.Tensor,
    q_cond_norm_scale: torch.Tensor,
    q_cond_scale_projection: torch.Tensor,
    q_cond_scale_bias: torch.Tensor,
    q_cond_bias_projection: torch.Tensor,
    k_cond_norm_scale: torch.Tensor,
    k_cond_scale_projection: torch.Tensor,
    k_cond_scale_bias: torch.Tensor,
    k_cond_bias_projection: torch.Tensor,
    q_projection: torch.Tensor,
    q_bias: torch.Tensor,
    k_projection: torch.Tensor,
    v_projection: torch.Tensor,
    gate_projection: torch.Tensor,
    output_projection: torch.Tensor,
    zero_cond_projection: torch.Tensor,
    zero_cond_bias: torch.Tensor,
) -> torch.Tensor:
  """Run conditional pair-biased diffusion cross-attention and residual."""
  if x_q.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"diffusion_cross_attention only supports CPU tensors, got {x_q.device}"
    )
  batch_shape = x_q.shape[:-2]
  q_len, c = x_q.shape[-2:]
  k_len = x_k.shape[-2]
  if x_k.shape[:-2] != batch_shape or x_k.shape[-1] != c:
    raise ValueError(f"x_q and x_k shapes are incompatible: {x_q.shape}, {x_k.shape}")
  heads, projection_channels, head_dim = q_projection.shape
  if projection_channels != c or heads * head_dim != c:
    raise ValueError(
        f"cross attention requires q_projection (H,C,D) with H*D=C; got "
        f"{tuple(q_projection.shape)} and C={c}"
    )
  if pair_logits is None:
    pair_logits = torch.zeros(
        (1, heads, q_len, k_len), dtype=x_q.dtype, device=x_q.device
    )
  tensors = (
      x_k, mask_q, mask_k, pair_logits, single_cond_q, single_cond_k,
      q_cond_norm_scale, q_cond_scale_projection, q_cond_scale_bias,
      q_cond_bias_projection, k_cond_norm_scale, k_cond_scale_projection,
      k_cond_scale_bias, k_cond_bias_projection, q_projection, q_bias,
      k_projection, v_projection, gate_projection, output_projection,
      zero_cond_projection, zero_cond_bias,
  )
  if any(tensor.device != x_q.device for tensor in tensors):
    raise ValueError("all diffusion_cross_attention tensors must share a device")
  packed_q_cond_scale = pack_matmul_b(
      q_cond_scale_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_q_cond_bias = pack_matmul_b(
      q_cond_bias_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_k_cond_scale = pack_matmul_b(
      k_cond_scale_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_k_cond_bias = pack_matmul_b(
      k_cond_bias_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_q = pack_matmul_b(q_projection.permute(0, 2, 1).contiguous())
  packed_k = pack_matmul_b(k_projection.permute(0, 2, 1).contiguous())
  packed_v = pack_matmul_b(v_projection.permute(0, 2, 1).contiguous())
  packed_gate = pack_matmul_b(
      gate_projection.permute(0, 2, 1).contiguous()
  )
  packed_output = pack_matmul_b(
      output_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_zero = pack_matmul_b(
      zero_cond_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return diffusion_cross_attention_packed(
      x_q, x_k, mask_q, mask_k, pair_logits,
      single_cond_q, single_cond_k,
      q_cond_norm_scale.contiguous(), packed_q_cond_scale,
      q_cond_scale_bias.contiguous(), packed_q_cond_bias,
      k_cond_norm_scale.contiguous(), packed_k_cond_scale,
      k_cond_scale_bias.contiguous(), packed_k_cond_bias,
      packed_q, q_bias.contiguous(), packed_k, packed_v, packed_gate,
      packed_output, packed_zero, zero_cond_bias.contiguous(),
  ).reshape_as(x_q)


def diffusion_cross_attention_packed(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    pair_logits: torch.Tensor,
    single_cond_q: torch.Tensor,
    single_cond_k: torch.Tensor,
    q_cond_norm_scale: torch.Tensor,
    packed_q_cond_scale: torch.Tensor,
    q_cond_scale_bias: torch.Tensor,
    packed_q_cond_bias: torch.Tensor,
    k_cond_norm_scale: torch.Tensor,
    packed_k_cond_scale: torch.Tensor,
    k_cond_scale_bias: torch.Tensor,
    packed_k_cond_bias: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
    packed_zero: torch.Tensor,
    zero_cond_bias: torch.Tensor,
) -> torch.Tensor:
  """Run diffusion cross-attention and residual with packed B operands."""
  batch_shape = x_q.shape[:-2]
  q_len, c = x_q.shape[-2:]
  k_len = x_k.shape[-2]
  s = single_cond_q.shape[-1]
  flat_cond_q = _flatten_shared_batch(
      single_cond_q, batch_shape, (q_len, s), "single_cond_q"
  )
  flat_cond_k = _flatten_shared_batch(
      single_cond_k, batch_shape, (k_len, s), "single_cond_k"
  )
  q_scale, q_bias_state, q_zero = diffusion_conditioning_packed(
      flat_cond_q,
      q_cond_norm_scale.unsqueeze(0),
      packed_q_cond_scale,
      q_cond_scale_bias.unsqueeze(0),
      packed_q_cond_bias,
      packed_zero,
      zero_cond_bias.unsqueeze(0),
  )
  k_scale, k_bias_state, _ = diffusion_conditioning_packed(
      flat_cond_k,
      k_cond_norm_scale.unsqueeze(0),
      packed_k_cond_scale,
      k_cond_scale_bias.unsqueeze(0),
      packed_k_cond_bias,
      packed_zero,
      zero_cond_bias.unsqueeze(0),
  )
  return diffusion_cross_attention_prepared_packed(
      x_q,
      x_k,
      mask_q,
      mask_k,
      pair_logits,
      q_scale[0],
      q_bias_state[0],
      k_scale[0],
      k_bias_state[0],
      q_zero[0],
      packed_q,
      q_bias,
      packed_k,
      packed_v,
      packed_gate,
      packed_output,
  )


def diffusion_cross_attention_prepared_packed(
    x_q: torch.Tensor,
    x_k: torch.Tensor,
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    pair_logits: torch.Tensor,
    q_adaptive_scale: torch.Tensor,
    q_adaptive_bias: torch.Tensor,
    k_adaptive_scale: torch.Tensor,
    k_adaptive_bias: torch.Tensor,
    zero_gate: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
) -> torch.Tensor:
  """Run cross-attention using conditioning prepared by its parent stack."""
  batch_shape = x_q.shape[:-2]
  q_len, c = x_q.shape[-2:]
  k_len = x_k.shape[-2]
  heads = q_bias.shape[0]

  if pair_logits is None:
    pair_logits = torch.zeros(
        (1, heads, q_len, k_len), dtype=x_q.dtype, device=x_q.device
    )
  flat_q = x_q.reshape(-1, q_len, c).contiguous()
  flat_k = x_k.reshape(-1, k_len, c).contiguous()
  flat_mask_q = _flatten_shared_batch(
      mask_q, batch_shape, (q_len,), "mask_q"
  )
  flat_mask_k = _flatten_shared_batch(
      mask_k, batch_shape, (k_len,), "mask_k"
  )
  flat_pair = _flatten_shared_batch(
      pair_logits, batch_shape, (heads, q_len, k_len), "pair_logits"
  )
  flat_q_scale = _flatten_shared_batch(
      q_adaptive_scale, batch_shape, (q_len, c), "q_adaptive_scale"
  )
  flat_q_bias = _flatten_shared_batch(
      q_adaptive_bias, batch_shape, (q_len, c), "q_adaptive_bias"
  )
  flat_k_scale = _flatten_shared_batch(
      k_adaptive_scale, batch_shape, (k_len, c), "k_adaptive_scale"
  )
  flat_k_bias = _flatten_shared_batch(
      k_adaptive_bias, batch_shape, (k_len, c), "k_adaptive_bias"
  )
  flat_zero = _flatten_shared_batch(
      zero_gate, batch_shape, (q_len, c), "zero_gate"
  )
  output = vecops.planner.empty_like(flat_q)
  _run_diffusion_cross_attention(
      flat_q, flat_k, flat_mask_q, flat_mask_k, flat_pair,
      flat_q_scale, flat_q_bias, flat_k_scale, flat_k_bias, flat_zero,
      packed_q, q_bias, packed_k, packed_v, packed_gate, packed_output, output,
      ComputeType=torch.float32,
  )
  return output.reshape_as(x_q)


def diffusion_cross_attention_indexed_prepared_packed(
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
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_output: torch.Tensor,
) -> torch.Tensor:
  """Project unique atom queries once and gather K/V inside attention."""
  if queries_act.ndim < 3:
    raise ValueError("queries_act must have shape (..., subsets, queries, channels)")
  batch_shape = queries_act.shape[:-3]
  subsets, queries, channels = queries_act.shape[-3:]
  keys = key_indices.shape[-1]
  heads = q_bias.shape[0]
  atoms = subsets * queries
  expected = (
      ("query_mask", query_mask, (subsets, queries)),
      ("key_indices", key_indices, (subsets, keys)),
      ("key_mask", key_mask, (subsets, keys)),
      ("pair_logits", pair_logits, (subsets, heads, queries, keys)),
  )
  for name, tensor, shape in expected:
    if tuple(tensor.shape) != shape:
      raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
  state_shape = (subsets, queries, channels)
  for name, tensor in (
      ("q_adaptive_scale", q_adaptive_scale),
      ("q_adaptive_bias", q_adaptive_bias),
      ("k_adaptive_scale", k_adaptive_scale),
      ("k_adaptive_bias", k_adaptive_bias),
      ("zero_gate", zero_gate),
  ):
    if tuple(tensor.shape) != state_shape:
      raise ValueError(
          f"{name} must have shape {state_shape}, got {tuple(tensor.shape)}"
      )
  flat_act = queries_act.reshape(-1, atoms, channels).contiguous()
  output = vecops.planner.empty_like(flat_act)
  _run_diffusion_cross_attention_indexed(
      flat_act,
      query_mask.to(torch.bool).contiguous(),
      key_indices.to(torch.int32).contiguous(),
      key_mask.to(torch.bool).contiguous(),
      pair_logits.contiguous(),
      q_adaptive_scale.reshape(atoms, channels).contiguous(),
      q_adaptive_bias.reshape(atoms, channels).contiguous(),
      k_adaptive_scale.reshape(atoms, channels).contiguous(),
      k_adaptive_bias.reshape(atoms, channels).contiguous(),
      zero_gate.reshape(atoms, channels).contiguous(),
      packed_q,
      q_bias,
      packed_k,
      packed_v,
      packed_gate,
      packed_output,
      output,
      ComputeType=torch.float32,
  )
  return output.reshape(batch_shape + (subsets, queries, channels))


def triangle_multiplication(
    act: torch.Tensor,
    mask: torch.Tensor,
    left_norm_scale: torch.Tensor,
    left_norm_offset: torch.Tensor,
    gw_proj: torch.Tensor,
    center_norm_scale: torch.Tensor,
    center_norm_offset: torch.Tensor,
    output_projection: torch.Tensor,
    gating_linear: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
  """Run the fused out-style triangle multiplication kernel."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"triangle_multiplication only supports CPU tensors, got {act.device}"
    )
  n0, n1, channels = act.shape[-3:]
  if n0 != n1:
    raise ValueError(f"triangle_multiplication requires B == N, got {n0} and {n1}")
  batch_shape = act.shape[:-3]
  expected_mask_shape = batch_shape + (n1,)
  if mask.shape != expected_mask_shape:
    mask = torch.broadcast_to(mask, expected_mask_shape)
  if gw_proj.shape != (2, 2 * channels, channels):
    raise ValueError(
        "gw_proj must have shape "
        f"(2, {2 * channels}, {channels}), got {tuple(gw_proj.shape)}"
    )
  tensors = (
      mask, left_norm_scale, left_norm_offset, gw_proj,
      center_norm_scale, center_norm_offset,
      output_projection, gating_linear,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all triangle_multiplication tensors must be on the same device")

  packed_gw = pack_matmul_b(
      gw_proj.reshape(2, channels, 2, channels)
      .permute(0, 2, 1, 3)
      .reshape(4, channels, channels)
      .contiguous()
  )
  packed_output = pack_matmul_b(
      output_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_gate = pack_matmul_b(
      gating_linear.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return triangle_multiplication_packed(
      act, mask, left_norm_scale, left_norm_offset, gw_proj, packed_gw,
      center_norm_scale, center_norm_offset, packed_output, packed_gate,
      transpose,
  )


def triangle_multiplication_packed(
    act: torch.Tensor,
    mask: torch.Tensor,
    left_norm_scale: torch.Tensor,
    left_norm_offset: torch.Tensor,
    gw_proj: torch.Tensor,
    packed_gw: torch.Tensor,
    center_norm_scale: torch.Tensor,
    center_norm_offset: torch.Tensor,
    packed_output: torch.Tensor,
    packed_gate: torch.Tensor,
    transpose: bool = False,
) -> torch.Tensor:
  """Run triangle multiplication with persistent packed B operands."""
  n1 = act.shape[-2]
  channels = act.shape[-1]
  batch_shape = act.shape[:-3]
  expected_mask_shape = batch_shape + (n1,)
  if mask.shape != expected_mask_shape:
    mask = torch.broadcast_to(mask, expected_mask_shape)

  flat_act = act.reshape(-1, n1, n1, channels).contiguous()
  flat_mask = mask.reshape(-1, n1).contiguous()
  output = vecops.planner.empty_like(flat_act)
  kernel_act = flat_act.transpose(1, 2) if transpose else flat_act
  kernel_output = output.transpose(1, 2) if transpose else output
  _run_triangle_multiplication(
      kernel_act,
      flat_mask,
      left_norm_scale.contiguous(),
      left_norm_offset.contiguous(),
      gw_proj.contiguous(),
      packed_gw,
      center_norm_scale.contiguous(),
      center_norm_offset.contiguous(),
      packed_output,
      packed_gate,
      kernel_output,
      ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def triangle_distributed_projection_packed(
    act: torch.Tensor,
    row_mask: torch.Tensor,
    column_mask: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    packed_projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project the local triangle shard before its collective exchange."""
  local_rows, sequence, channels = act.shape[-3:]
  batch_shape = act.shape[:-3]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_act = act.reshape(batch, local_rows, sequence, channels).contiguous()
  row_mask = torch.broadcast_to(row_mask, batch_shape + (local_rows,))
  column_mask = torch.broadcast_to(column_mask, batch_shape + (sequence,))
  flat_row_mask = row_mask.reshape(batch, local_rows).contiguous()
  flat_column_mask = column_mask.reshape(batch, sequence).contiguous()
  normalized = vecops.planner.empty_like(flat_act)
  a = vecops.planner.empty(
      (batch, channels, local_rows, sequence),
      dtype=act.dtype,
      device=act.device,
  )
  b = vecops.planner.empty_like(a)
  _run_triangle_distributed_projection(
      flat_act,
      flat_row_mask,
      flat_column_mask,
      norm_scale.contiguous(),
      norm_offset.contiguous(),
      packed_projection,
      normalized,
      a,
      b,
      ComputeType=torch.float32,
  )
  projected_shape = batch_shape + (channels, local_rows, sequence)
  return (
      normalized.reshape_as(act),
      a.reshape(projected_shape),
      b.reshape(projected_shape),
  )


def triangle_distributed_update_packed(
    act: torch.Tensor,
    normalized: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    center_norm_scale: torch.Tensor,
    center_norm_offset: torch.Tensor,
    packed_output: torch.Tensor,
    packed_gate: torch.Tensor,
) -> torch.Tensor:
  """Finish a local triangle shard after projection exchange."""
  local_rows, sequence, channels = act.shape[-3:]
  batch_shape = act.shape[:-3]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_act = act.reshape(batch, local_rows, sequence, channels).contiguous()
  flat_normalized = normalized.reshape_as(flat_act).contiguous()
  flat_left = left.reshape(
      batch, channels, local_rows, sequence
  ).contiguous()
  flat_right = right.reshape(
      batch, channels, sequence, sequence
  ).contiguous()
  output = vecops.planner.empty_like(flat_act)
  _run_triangle_distributed_update(
      flat_act,
      flat_normalized,
      flat_left,
      flat_right,
      center_norm_scale.contiguous(),
      center_norm_offset.contiguous(),
      packed_output,
      packed_gate,
      output,
      ComputeType=torch.float32,
  )
  return output.reshape_as(act)


def evoformer_single_act_update(
    pair_act: torch.Tensor,
    single_act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    pair_norm_scale: torch.Tensor,
    pair_norm_offset: torch.Tensor,
    logits_projection: torch.Tensor,
    attention_norm_scale: torch.Tensor,
    attention_norm_offset: torch.Tensor,
    q_projection: torch.Tensor,
    q_bias: torch.Tensor,
    k_projection: torch.Tensor,
    v_projection: torch.Tensor,
    g_projection: torch.Tensor,
    o_projection: torch.Tensor,
    transition_norm_scale: torch.Tensor,
    transition_norm_offset: torch.Tensor,
    transition1: torch.Tensor,
    transition2: torch.Tensor,
) -> torch.Tensor:
  """Run pair-biased single attention followed by its transition block."""
  if pair_act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        "evoformer_single_act_update only supports CPU tensors, "
        f"got {pair_act.device}"
    )
  n0, n1, pair_channels = pair_act.shape[-3:]
  if n0 != n1:
    raise ValueError(
        f"evoformer_single_act_update requires B == N, got {n0} and {n1}"
    )
  if single_act.shape[:-2] != pair_act.shape[:-3] or single_act.shape[-2] != n1:
    raise ValueError(
        f"single_act shape {tuple(single_act.shape)} is incompatible with "
        f"pair_act shape {tuple(pair_act.shape)}"
    )
  channels = single_act.shape[-1]
  heads, projection_channels, head_dim = q_projection.shape
  if projection_channels != channels or channels != heads * head_dim:
    raise ValueError(
        "single attention requires C == H*D and projection shape (H,C,D), "
        f"got C={channels}, H={heads}, D={head_dim}"
    )
  hidden = transition2.shape[0]
  if transition1.shape != (channels, 2 * hidden):
    raise ValueError(
        f"transition1 must have shape ({channels}, {2 * hidden}), "
        f"got {tuple(transition1.shape)}"
    )
  batch_shape = pair_act.shape[:-3]
  expected_mask_shape = batch_shape + (n1,)
  if key_padding_mask.shape != expected_mask_shape:
    key_padding_mask = torch.broadcast_to(
        key_padding_mask, expected_mask_shape
    )
  tensors = (
      single_act, key_padding_mask,
      pair_norm_scale, pair_norm_offset, logits_projection,
      attention_norm_scale, attention_norm_offset,
      q_projection, q_bias, k_projection, v_projection,
      g_projection, o_projection,
      transition_norm_scale, transition_norm_offset,
      transition1, transition2,
  )
  if any(tensor.device != pair_act.device for tensor in tensors):
    raise ValueError(
        "all evoformer_single_act_update tensors must be on the same device"
    )

  packed_logits = pack_matmul_b(logits_projection).unsqueeze(0)
  packed_q = pack_matmul_b(q_projection.permute(0, 2, 1).contiguous())
  packed_k = pack_matmul_b(k_projection.permute(0, 2, 1).contiguous())
  packed_v = pack_matmul_b(v_projection.permute(0, 2, 1).contiguous())
  packed_g = pack_matmul_b(g_projection.permute(0, 2, 1).contiguous())
  packed_o = pack_matmul_b(o_projection)
  packed_transition1 = pack_matmul_b(
      transition1.reshape(channels, 2, hidden).permute(1, 2, 0).contiguous()
  )
  packed_transition2 = pack_matmul_b(
      transition2.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  return evoformer_single_act_update_packed(
      pair_act, single_act, key_padding_mask,
      pair_norm_scale, pair_norm_offset, packed_logits,
      attention_norm_scale, attention_norm_offset,
      packed_q, q_bias, packed_k, packed_v, packed_g, packed_o,
      transition_norm_scale, transition_norm_offset,
      packed_transition1, packed_transition2, hidden=hidden,
  )


def evoformer_single_act_update_packed(
    pair_act: torch.Tensor,
    single_act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    pair_norm_scale: torch.Tensor,
    pair_norm_offset: torch.Tensor,
    packed_logits: torch.Tensor,
    attention_norm_scale: torch.Tensor,
    attention_norm_offset: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_g: torch.Tensor,
    packed_o: torch.Tensor,
    transition_norm_scale: torch.Tensor,
    transition_norm_offset: torch.Tensor,
    packed_transition1: torch.Tensor,
    packed_transition2: torch.Tensor,
    *,
    hidden: int,
) -> torch.Tensor:
  """Run the single update with persistent packed B operands."""
  n1 = pair_act.shape[-2]
  pair_channels = pair_act.shape[-1]
  channels = single_act.shape[-1]
  batch_shape = pair_act.shape[:-3]
  expected_mask_shape = batch_shape + (n1,)
  if key_padding_mask.shape != expected_mask_shape:
    key_padding_mask = torch.broadcast_to(
        key_padding_mask, expected_mask_shape
    )

  flat_pair = pair_act.reshape(-1, n1, n1, pair_channels).contiguous()
  flat_single = single_act.reshape(-1, n1, channels).contiguous()
  flat_mask = key_padding_mask.reshape(-1, n1).contiguous()
  output = vecops.planner.empty_like(flat_single)
  _run_evoformer_single_act_update(
      flat_pair,
      flat_single,
      flat_mask,
      pair_norm_scale.contiguous(),
      pair_norm_offset.contiguous(),
      packed_logits,
      attention_norm_scale.contiguous(),
      attention_norm_offset.contiguous(),
      packed_q,
      q_bias.contiguous(),
      packed_k,
      packed_v,
      packed_g,
      packed_o,
      transition_norm_scale.contiguous(),
      transition_norm_offset.contiguous(),
      packed_transition1,
      packed_transition2,
      output,
      Hidden=hidden,
      ComputeType=torch.float32,
  )
  return output.reshape_as(single_act)


def single_distributed_projection_packed(
    pair_act: torch.Tensor,
    single_act: torch.Tensor,
    pair_norm_scale: torch.Tensor,
    pair_norm_offset: torch.Tensor,
    packed_logits: torch.Tensor,
    attention_norm_scale: torch.Tensor,
    attention_norm_offset: torch.Tensor,
    packed_q: torch.Tensor,
    q_bias: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project a local single/pair row shard before gathering K and V."""
  local_rows, sequence, pair_channels = pair_act.shape[-3:]
  channels = single_act.shape[-1]
  batch_shape = pair_act.shape[:-3]
  batch = math.prod(batch_shape) if batch_shape else 1
  if single_act.shape != batch_shape + (local_rows, channels):
    raise ValueError(
        f"single_act shape {tuple(single_act.shape)} is incompatible with "
        f"pair_act shape {tuple(pair_act.shape)}"
    )
  heads = packed_q.shape[0]
  head_dim = q_bias.shape[-1]
  flat_pair = pair_act.reshape(
      batch, local_rows, sequence, pair_channels
  ).contiguous()
  flat_single = single_act.reshape(batch, local_rows, channels).contiguous()
  pair_logits = vecops.planner.empty(
      (batch, heads, local_rows, sequence),
      dtype=pair_act.dtype,
      device=pair_act.device,
  )
  projection_shape = (batch, heads, local_rows, head_dim)
  q = vecops.planner.empty(
      projection_shape, dtype=single_act.dtype, device=single_act.device
  )
  k = vecops.planner.empty_like(q)
  v = vecops.planner.empty_like(q)
  gate = vecops.planner.empty_like(q)
  _run_single_distributed_projection(
      flat_pair,
      flat_single,
      pair_norm_scale.contiguous(),
      pair_norm_offset.contiguous(),
      packed_logits,
      attention_norm_scale.contiguous(),
      attention_norm_offset.contiguous(),
      packed_q,
      q_bias.contiguous(),
      packed_k,
      packed_v,
      packed_g,
      pair_logits,
      q,
      k,
      v,
      gate,
      ComputeType=torch.float32,
  )
  logits_shape = batch_shape + (heads, local_rows, sequence)
  external_projection_shape = batch_shape + (heads, local_rows, head_dim)
  return (
      pair_logits.reshape(logits_shape),
      q.reshape(external_projection_shape),
      k.reshape(external_projection_shape),
      v.reshape(external_projection_shape),
      gate.reshape(external_projection_shape),
  )


def single_distributed_update_packed(
    single_act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    pair_logits: torch.Tensor,
    q: torch.Tensor,
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    gate: torch.Tensor,
    packed_o: torch.Tensor,
    transition_norm_scale: torch.Tensor,
    transition_norm_offset: torch.Tensor,
    packed_transition1: torch.Tensor,
    packed_transition2: torch.Tensor,
    *,
    hidden: int,
) -> torch.Tensor:
  """Finish local single attention after K/V collectives."""
  local_rows, channels = single_act.shape[-2:]
  sequence = global_k.shape[-2]
  batch_shape = single_act.shape[:-2]
  batch = math.prod(batch_shape) if batch_shape else 1
  heads, head_dim = q.shape[-3], q.shape[-1]
  expected_mask_shape = batch_shape + (sequence,)
  if key_padding_mask.shape != expected_mask_shape:
    key_padding_mask = torch.broadcast_to(key_padding_mask, expected_mask_shape)
  flat_single = single_act.reshape(batch, local_rows, channels).contiguous()
  flat_mask = key_padding_mask.reshape(batch, sequence).contiguous()
  flat_logits = pair_logits.reshape(
      batch, heads, local_rows, sequence
  ).contiguous()
  local_projection_shape = (batch, heads, local_rows, head_dim)
  global_projection_shape = (batch, heads, sequence, head_dim)
  output = vecops.planner.empty_like(flat_single)
  _run_single_distributed_update(
      flat_single,
      flat_mask,
      flat_logits,
      q.reshape(local_projection_shape).contiguous(),
      global_k.reshape(global_projection_shape).contiguous(),
      global_v.reshape(global_projection_shape).contiguous(),
      gate.reshape(local_projection_shape).contiguous(),
      packed_o,
      transition_norm_scale.contiguous(),
      transition_norm_offset.contiguous(),
      packed_transition1,
      packed_transition2,
      output,
      Hidden=hidden,
      ComputeType=torch.float32,
  )
  return output.reshape_as(single_act)


def grid_self_attention(
    act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    q_projection: torch.Tensor,
    k_projection: torch.Tensor,
    v_projection: torch.Tensor,
    output_projection: torch.Tensor,
    g_projection: torch.Tensor,
    bias_projection: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    index_map: torch.Tensor | None,
    weight_map: torch.Tensor | None,
    num_samples: int,
    num_random_blocks: int,
    static_probability: float,
    random_probability: float,
    minimum_probability: float,
    transpose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run IBS grid self-attention and create or reuse its base maps."""
  if act.device.type not in {"cpu", "meta"}:
    raise ValueError(
        f"grid_self_attention only supports CPU tensors, got {act.device}"
    )
  n0, n1, channels = act.shape[-3:]
  if n0 != n1:
    raise ValueError(f"grid_self_attention requires B == N, got {n0} and {n1}")
  heads, projection_channels, head_dim = q_projection.shape
  if projection_channels != channels:
    raise ValueError(
        f"q_projection has C={projection_channels}, expected {channels}"
    )
  if output_projection.shape != (heads * head_dim, channels):
    raise ValueError(
        "output_projection must have shape "
        f"({heads * head_dim}, {channels}), got {tuple(output_projection.shape)}"
    )
  if num_samples < 1:
    raise ValueError("num_samples must be positive")
  if not 0 <= num_random_blocks <= num_samples:
    raise ValueError("num_random_blocks must be in [0, num_samples]")
  packed_q = pack_matmul_b(q_projection.permute(0, 2, 1).contiguous())
  packed_k = pack_matmul_b(k_projection.permute(0, 2, 1).contiguous())
  packed_v = pack_matmul_b(v_projection.permute(0, 2, 1).contiguous())
  packed_g = pack_matmul_b(g_projection.permute(0, 2, 1).contiguous())
  packed_output = pack_matmul_b(
      output_projection.transpose(0, 1).contiguous()
  ).unsqueeze(0)
  packed_bias = pack_matmul_b(bias_projection).unsqueeze(0)
  return grid_self_attention_packed(
      act, key_padding_mask, packed_q, packed_k, packed_v, packed_output,
      packed_g, packed_bias, norm_scale, norm_offset, index_map, weight_map,
      num_samples, num_random_blocks, static_probability,
      random_probability, minimum_probability, transpose,
      head_dim=head_dim,
  )


def grid_self_attention_packed(
    act: torch.Tensor,
    key_padding_mask: torch.Tensor,
    packed_q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_output: torch.Tensor,
    packed_g: torch.Tensor,
    packed_bias: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    index_map: torch.Tensor | None,
    weight_map: torch.Tensor | None,
    num_samples: int,
    num_random_blocks: int,
    static_probability: float,
    random_probability: float,
    minimum_probability: float,
    transpose: bool,
    *,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Run grid self-attention with persistent packed B operands."""
  n1 = act.shape[-2]
  channels = act.shape[-1]
  heads = packed_q.shape[0]
  batch_shape = act.shape[:-3]
  expected_mask_shape = batch_shape + (n1,)
  if key_padding_mask.shape != expected_mask_shape:
    key_padding_mask = torch.broadcast_to(
        key_padding_mask, expected_mask_shape
    )
  batch = 1
  for extent in batch_shape:
    batch *= extent
  query_tiles = (n1 + 31) // 32
  map_shape = (batch, n1, heads, query_tiles, num_samples)
  generate_maps = index_map is None
  if generate_maps != (weight_map is None):
    raise ValueError("index_map and weight_map must be both present or both omitted")
  if generate_maps:
    index_map = vecops.planner.empty(
        map_shape, dtype=torch.int32, device=act.device
    )
    weight_map = vecops.planner.empty(
        map_shape, dtype=torch.float32, device=act.device
    )
  else:
    assert index_map is not None and weight_map is not None
    expected_external_shape = batch_shape + (
        n1, heads, query_tiles, num_samples
    )
    index_map = torch.broadcast_to(index_map, expected_external_shape)
    weight_map = torch.broadcast_to(weight_map, expected_external_shape)
    index_map = index_map.reshape(map_shape).contiguous()
    weight_map = weight_map.reshape(map_shape).contiguous()

  tensors = (
      key_padding_mask,
      packed_q, packed_k, packed_v,
      packed_output, packed_g, packed_bias,
      norm_scale, norm_offset, index_map, weight_map,
  )
  if any(tensor.device != act.device for tensor in tensors):
    raise ValueError("all grid_self_attention tensors must be on the same device")
  flat_act = act.reshape(batch, n1, n1, channels).contiguous()
  flat_mask = key_padding_mask.reshape(batch, n1).contiguous()
  output = vecops.planner.empty_like(flat_act)
  _run_grid_self_attention(
      flat_act,
      flat_mask,
      packed_q,
      packed_k,
      packed_v,
      packed_output,
      packed_g,
      packed_bias,
      norm_scale.contiguous(),
      norm_offset.contiguous(),
      index_map,
      weight_map,
      output,
      int(num_random_blocks),
      float(static_probability),
      float(random_probability),
      float(minimum_probability),
      generate_maps,
      next(_grid_seed_counter),
      Transpose=int(transpose),
      HeadDim=head_dim,
      ComputeType=torch.float32,
  )
  external_map_shape = batch_shape + (
      n1, heads, query_tiles, num_samples
  )
  return (
      output.reshape_as(act),
      index_map.reshape(external_map_shape),
      weight_map.reshape(external_map_shape),
  )


def grid_distributed_projection_packed(
    act: torch.Tensor,
    norm_scale: torch.Tensor,
    norm_offset: torch.Tensor,
    packed_bias: torch.Tensor,
    *,
    heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Normalize a local pair shard and make a gather-friendly bias shard."""
  local_rows, sequence, channels = act.shape[-3:]
  batch_shape = act.shape[:-3]
  batch = math.prod(batch_shape) if batch_shape else 1
  flat_act = act.reshape(batch, local_rows, sequence, channels).contiguous()
  normalized = vecops.planner.empty_like(flat_act)
  bias_send = vecops.planner.empty(
      (local_rows, batch, heads, sequence),
      dtype=act.dtype,
      device=act.device,
  )
  _run_grid_distributed_projection(
      flat_act,
      norm_scale.contiguous(),
      norm_offset.contiguous(),
      packed_bias,
      normalized,
      bias_send,
      ComputeType=torch.float32,
  )
  return normalized.reshape_as(act), bias_send


def grid_distributed_attention_packed(
    normalized: torch.Tensor,
    key_padding_mask: torch.Tensor,
    global_bias: torch.Tensor,
    packed_q: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    packed_output: torch.Tensor,
    packed_g: torch.Tensor,
    index_map: torch.Tensor | None,
    weight_map: torch.Tensor | None,
    num_samples: int,
    num_random_blocks: int,
    static_probability: float,
    random_probability: float,
    minimum_probability: float,
    row_offset: int,
    *,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Apply local grid attention using the globally gathered pair bias."""
  local_rows, sequence, channels = normalized.shape[-3:]
  batch_shape = normalized.shape[:-3]
  batch = math.prod(batch_shape) if batch_shape else 1
  heads = packed_q.shape[0]
  flat_normalized = normalized.reshape(
      batch, local_rows, sequence, channels
  ).contiguous()
  expected_mask_shape = batch_shape + (sequence,)
  if key_padding_mask.shape != expected_mask_shape:
    key_padding_mask = torch.broadcast_to(key_padding_mask, expected_mask_shape)
  flat_mask = key_padding_mask.reshape(batch, sequence).contiguous()
  expected_bias_shape = (sequence, batch, heads, sequence)
  if global_bias.shape != expected_bias_shape:
    raise ValueError(
        f"global bias must have shape {expected_bias_shape}, "
        f"got {tuple(global_bias.shape)}"
    )
  query_tiles = (sequence + 31) // 32
  map_shape = (batch, local_rows, heads, query_tiles, num_samples)
  generate_maps = index_map is None
  if generate_maps != (weight_map is None):
    raise ValueError("index_map and weight_map must be both present or both omitted")
  external_map_shape = batch_shape + (
      local_rows, heads, query_tiles, num_samples
  )
  if generate_maps:
    index_map = vecops.planner.empty(
        map_shape, dtype=torch.int32, device=normalized.device
    )
    weight_map = vecops.planner.empty(
        map_shape, dtype=torch.float32, device=normalized.device
    )
  else:
    assert index_map is not None and weight_map is not None
    index_map = torch.broadcast_to(index_map, external_map_shape)
    weight_map = torch.broadcast_to(weight_map, external_map_shape)
    index_map = index_map.reshape(map_shape).contiguous()
    weight_map = weight_map.reshape(map_shape).contiguous()
  output = vecops.planner.empty_like(flat_normalized)
  _run_grid_distributed_attention(
      flat_normalized,
      flat_mask,
      global_bias.contiguous(),
      packed_q,
      packed_k,
      packed_v,
      packed_output,
      packed_g,
      index_map,
      weight_map,
      output,
      int(row_offset),
      int(num_random_blocks),
      float(static_probability),
      float(random_probability),
      float(minimum_probability),
      generate_maps,
      next(_grid_seed_counter),
      HeadDim=head_dim,
      ComputeType=torch.float32,
  )
  return (
      output.reshape_as(normalized),
      index_map.reshape(external_map_shape),
      weight_map.reshape(external_map_shape),
  )


__all__ = [
    "pack_matmul_b",
    "transition_block",
    "msa_attention",
    "msa_attention_packed",
    "msa_distributed_projection_packed",
    "msa_distributed_update_packed",
    "outer_product_mean",
    "outer_product_mean_packed",
    "outer_distributed_projection_packed",
    "outer_distributed_right_projection_packed",
    "outer_distributed_left_projection_packed",
    "outer_distributed_update_packed",
    "diffusion_conditioning_packed",
    "diffusion_transition_block",
    "diffusion_transition_block_packed",
    "diffusion_transition_block_prepared_packed",
    "diffusion_self_attention",
    "diffusion_self_attention_packed",
    "diffusion_self_attention_prepared_packed",
    "diffusion_cross_attention",
    "diffusion_cross_attention_packed",
    "diffusion_cross_attention_prepared_packed",
    "diffusion_cross_attention_indexed_prepared_packed",
    "triangle_multiplication",
    "triangle_multiplication_packed",
    "triangle_distributed_projection_packed",
    "triangle_distributed_update_packed",
    "evoformer_single_act_update",
    "evoformer_single_act_update_packed",
    "single_distributed_projection_packed",
    "single_distributed_update_packed",
    "grid_self_attention",
    "grid_self_attention_packed",
    "grid_distributed_projection_packed",
    "grid_distributed_attention_packed",
    "set_grid_attention_seed",
]
