# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Explicit opt-in for CPU native kernels beyond the tested integration coverage."""
import os
from collections.abc import Sequence

AUDITED_NATIVE_OPS = frozenset({"linear"})
FUSED_NATIVE_OPS = frozenset({
    "evoformer_transition_block", "msa_attention", "outer_product_mean",
    "grid_self_attn", "triangle_multiplication", "evoformer_single_act_update",
    "diffusion_transition_block", "diffusion_self_attention", "diffusion_cross_attention",
})


def validate_native_device(device) -> None:
  if str(device).split(":", 1)[0] not in {"cpu", "meta"}:
    raise RuntimeError("vecops kernels support CPU only; choose backend='torch' for CUDA")


def validate_native_ops(selected: Sequence[str]) -> None:
  """Retain fused implementation behind explicit experimental opt-in.

  Strict checkpoint assignments use reference Parameters; all native casting,
  transpose and packing happens afterward in nonpersistent buffers. Full x86 integration checks cover three input sizes; they do not establish
  every native shape, precision or hardware platform.
  """
  chosen = set(selected)
  if chosen & {"*", "all"}:
    chosen = set(AUDITED_NATIVE_OPS | FUSED_NATIVE_OPS)
  unknown = chosen - AUDITED_NATIVE_OPS - FUSED_NATIVE_OPS
  if unknown:
    raise ValueError(f"Unknown native operators: {sorted(unknown)}")
  fused = chosen & FUSED_NATIVE_OPS
  if fused and os.environ.get("SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION") != "1":
    raise RuntimeError(
        "Fused native kernels require explicit experimental opt-in: "
        + ", ".join(sorted(fused))
        + ". Set SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION=1 explicitly to test them "
        "with the compatible pinned vecops dependency. Tested coverage is documented in docs/validation.md; the Torch backend remains available. "
        "Derived parameters are now nonpersistent prepared buffers; strict loading is unchanged."
    )
