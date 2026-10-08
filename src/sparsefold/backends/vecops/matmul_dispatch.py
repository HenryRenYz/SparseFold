# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Route CPU ATen matrix products through vecops on targets with unusable BLAS."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import functools
import math
import platform

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._python_dispatch import _pop_mode_temporarily

from sparsefold.runtime import config


_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


@functools.cache
def _kernels():
  from sparsefold.backends.vecops import kernels

  return kernels


def _parse_einsum_equation(
    equation: str, operands: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> tuple[list[list[str]], list[str]]:
  """Parse einsum labels without depending on the optional opt_einsum package."""
  equation = "".join(equation.split())
  if equation.count("->") > 1:
    raise ValueError(f"invalid einsum equation: {equation!r}")
  if "->" in equation:
    input_equation, output_equation = equation.split("->")
  else:
    input_equation, output_equation = equation, None
  input_terms = input_equation.split(",")
  if len(input_terms) != len(operands):
    raise ValueError("einsum operand count does not match the equation")

  ellipsis_ranks = []
  for term, operand in zip(input_terms, operands, strict=True):
    if term.count("...") > 1 or "." in term.replace("...", ""):
      raise ValueError(f"invalid einsum ellipsis in {term!r}")
    explicit_rank = len(term.replace("...", ""))
    ellipsis_rank = operand.ndim - explicit_rank
    if ellipsis_rank < 0 or ("..." not in term and ellipsis_rank != 0):
      raise ValueError(
          f"einsum term {term!r} does not match rank {operand.ndim}"
      )
    ellipsis_ranks.append(ellipsis_rank)

  max_ellipsis_rank = max(ellipsis_ranks, default=0)
  ellipsis_labels = [f"@ellipsis{index}" for index in range(max_ellipsis_rank)]
  parsed_inputs = []
  for term, ellipsis_rank in zip(input_terms, ellipsis_ranks, strict=True):
    if "..." in term:
      # Einsum broadcasts ellipses by aligning their dimensions from the right.
      replacement = ellipsis_labels[max_ellipsis_rank - ellipsis_rank:]
      prefix, suffix = term.split("...")
      labels = list(prefix) + replacement + list(suffix)
    else:
      labels = list(term)
    parsed_inputs.append(labels)

  if output_equation is not None:
    if output_equation.count("...") > 1 \
        or "." in output_equation.replace("...", ""):
      raise ValueError(f"invalid output ellipsis in {output_equation!r}")
    if "..." in output_equation:
      prefix, suffix = output_equation.split("...")
      output_labels = list(prefix) + ellipsis_labels + list(suffix)
    else:
      output_labels = list(output_equation)
  else:
    counts: dict[str, int] = {}
    for labels in parsed_inputs:
      for label in labels:
        counts[label] = counts.get(label, 0) + 1
    output_labels = ellipsis_labels + sorted(
        label for label, count in counts.items()
        if count == 1 and label not in ellipsis_labels
    )
  known_labels = {label for labels in parsed_inputs for label in labels}
  if any(label not in known_labels for label in output_labels):
    raise ValueError("einsum output contains a label absent from all inputs")
  return parsed_inputs, output_labels


def _native_einsum(
    equation: str, operands: list[torch.Tensor] | tuple[torch.Tensor, ...]
) -> torch.Tensor | None:
  """Lower a two-operand contraction to one raw vecops matmul."""
  if len(operands) != 2:
    return None
  lhs, rhs = operands
  if (
      lhs.device.type != "cpu"
      or rhs.device.type != "cpu"
      or lhs.dtype != rhs.dtype
      or lhs.dtype not in _SUPPORTED_DTYPES
  ):
    return None
  try:
    parsed_inputs, output_labels = _parse_einsum_equation(
        equation, operands
    )
  except (TypeError, ValueError):
    return None
  lhs_labels, rhs_labels = parsed_inputs
  if (
      len(set(lhs_labels)) != len(lhs_labels)
      or len(set(rhs_labels)) != len(rhs_labels)
      or len(set(output_labels)) != len(output_labels)
  ):
    return None

  # Axes occurring in only one input and not in the output are independent
  # reductions. Remove them before forming the matrix contraction.
  lhs_reduce = [
      label for label in lhs_labels
      if label not in rhs_labels and label not in output_labels
  ]
  rhs_reduce = [
      label for label in rhs_labels
      if label not in lhs_labels and label not in output_labels
  ]
  if lhs_reduce:
    lhs = lhs.sum(tuple(lhs_labels.index(x) for x in lhs_reduce))
    lhs_labels = [x for x in lhs_labels if x not in lhs_reduce]
  if rhs_reduce:
    rhs = rhs.sum(tuple(rhs_labels.index(x) for x in rhs_reduce))
    rhs_labels = [x for x in rhs_labels if x not in rhs_reduce]

  batch_labels = [
      label for label in output_labels
      if label in lhs_labels and label in rhs_labels
  ]
  contract_labels = [
      label for label in lhs_labels
      if label in rhs_labels and label not in output_labels
  ]
  lhs_output_labels = [
      label for label in lhs_labels
      if label in output_labels and label not in rhs_labels
  ]
  rhs_output_labels = [
      label for label in rhs_labels
      if label in output_labels and label not in lhs_labels
  ]
  current_labels = batch_labels + lhs_output_labels + rhs_output_labels
  if set(current_labels) != set(output_labels):
    return None

  lhs_order = batch_labels + lhs_output_labels + contract_labels
  rhs_order = batch_labels + rhs_output_labels + contract_labels
  try:
    lhs = lhs.permute(*(lhs_labels.index(x) for x in lhs_order))
    rhs = rhs.permute(*(rhs_labels.index(x) for x in rhs_order))
    lhs_batch_shape = tuple(
        lhs.shape[index] for index in range(len(batch_labels))
    )
    rhs_batch_shape = tuple(
        rhs.shape[index] for index in range(len(batch_labels))
    )
    batch_shape = torch.broadcast_shapes(lhs_batch_shape, rhs_batch_shape)
  except RuntimeError:
    return None

  lhs_output_shape = tuple(
      lhs.shape[len(batch_labels) + index]
      for index in range(len(lhs_output_labels))
  )
  rhs_output_shape = tuple(
      rhs.shape[len(batch_labels) + index]
      for index in range(len(rhs_output_labels))
  )
  lhs_contract_shape = tuple(lhs.shape[-len(contract_labels):]) \
      if contract_labels else ()
  rhs_contract_shape = tuple(rhs.shape[-len(contract_labels):]) \
      if contract_labels else ()
  if lhs_contract_shape != rhs_contract_shape:
    return None
  m = math.prod(lhs_output_shape) if lhs_output_shape else 1
  n = math.prod(rhs_output_shape) if rhs_output_shape else 1
  k = math.prod(lhs_contract_shape) if lhs_contract_shape else 1
  if m == 0 or n == 0 or k == 0:
    return None
  batch = math.prod(batch_shape) if batch_shape else 1
  lhs = lhs.expand(batch_shape + lhs_output_shape + lhs_contract_shape)
  rhs = rhs.expand(batch_shape + rhs_output_shape + rhs_contract_shape)
  lhs_matrix = lhs.reshape(batch, m, k).contiguous()
  rhs_weight = rhs.reshape(batch, n, k).contiguous()
  result = _kernels().matmul(
      lhs_matrix, rhs_weight, alpha=1.0, beta=0.0, out_dtype=lhs.dtype
  ).reshape(batch_shape + lhs_output_shape + rhs_output_shape)
  if current_labels != output_labels:
    result = result.permute(*(current_labels.index(x) for x in output_labels))
  return result


def _native_matmul(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    c_input: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor | None:
  if (
      lhs.layout != torch.strided
      or rhs.layout != torch.strided
      or lhs.device.type != "cpu"
      or rhs.device.type != "cpu"
      or lhs.dtype != rhs.dtype
      or lhs.dtype not in _SUPPORTED_DTYPES
      or lhs.ndim == 0
      or rhs.ndim == 0
  ):
    return None

  lhs_was_vector = lhs.ndim == 1
  rhs_was_vector = rhs.ndim == 1
  if lhs_was_vector:
    lhs = lhs.unsqueeze(0)
  if rhs_was_vector:
    rhs = rhs.unsqueeze(-1)
  if lhs.shape[-1] != rhs.shape[-2]:
    return None
  m, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[-1]
  if m == 0 or n == 0 or k == 0:
    return None

  try:
    batch_shape = torch.broadcast_shapes(lhs.shape[:-2], rhs.shape[:-2])
  except RuntimeError:
    return None
  lhs_expanded = lhs.expand(batch_shape + (m, k))
  batch = math.prod(batch_shape) if batch_shape else 1
  if rhs.ndim == 2:
    # Preserve one physical RHS copy. The native kernel broadcasts it across
    # the flattened batch and lets vecops choose its on-the-fly packing plan.
    weight = rhs.transpose(-2, -1).reshape(1, n, k).contiguous()
  else:
    rhs_batched = rhs.expand(batch_shape + (k, n))
    weight = (
        rhs_batched.transpose(-2, -1).reshape(batch, n, k).contiguous()
    )

  lhs_batched = lhs_expanded.reshape(batch, m, k).contiguous()
  native_c_input = None
  if c_input is not None:
    try:
      native_c_input = c_input.expand(batch_shape + (m, n)).reshape(
          batch, m, n
      )
    except RuntimeError:
      return None
  result = _kernels().matmul(
      lhs_batched,
      weight,
      c_input=native_c_input,
      alpha=alpha,
      beta=beta,
      out_dtype=lhs.dtype,
  ).reshape(batch_shape + (m, n))

  if lhs_was_vector:
    result = result.squeeze(-2)
  if rhs_was_vector:
    result = result.squeeze(-1)
  return result


class NativeMatmulDispatchMode(TorchDispatchMode):
  """Replace inference matrix products while leaving unsupported cases intact."""

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    kwargs = kwargs or {}
    if func == torch.ops.aten.einsum.default:
      result = _native_einsum(args[0], args[1])
      if result is not None:
        return result
      operands = args[1]
      if (
          len(operands) == 2
          and all(isinstance(x, torch.Tensor) for x in operands)
          and all(x.device.type == "cpu" for x in operands)
          and all(x.dtype in _SUPPORTED_DTYPES for x in operands)
      ):
        metadata = [
            {
                "shape": tuple(x.shape),
                "stride": tuple(x.stride()),
                "dtype": str(x.dtype),
                "requires_grad": x.requires_grad,
            }
            for x in operands
        ]
        raise RuntimeError(
            f"native matmul dispatch could not lower einsum {args[0]!r}: "
            f"{metadata}"
        )
    elif func in {
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
    }:
      result = _native_matmul(args[0], args[1])
      if result is not None:
        return result
    elif func == torch.ops.aten.addmm.default:
      beta = kwargs.get("beta", 1)
      alpha = kwargs.get("alpha", 1)
      result = _native_matmul(
          args[1], args[2], c_input=args[0], alpha=alpha, beta=beta
      )
      if result is not None:
        return result
    elif func == torch.ops.aten.baddbmm.default:
      beta = kwargs.get("beta", 1)
      alpha = kwargs.get("alpha", 1)
      result = _native_matmul(
          args[1], args[2], c_input=args[0], alpha=alpha, beta=beta
      )
      if result is not None:
        return result
    return func(*args, **kwargs)


@contextmanager
def suspend_native_matmul_dispatch():
  """Temporarily remove this router while an all-explicit region executes.

  The diffusion hot loop calls vecops matrix kernels explicitly.  Leaving the
  model-wide mode installed there would send every unrelated ATen operation
  through Python merely to reject it.  During graph capture the native router
  is deliberately the innermost mode, so popping it exposes (rather than
  disables) the outer capture mode.
  """
  current = _get_current_dispatch_mode()
  if isinstance(current, NativeMatmulDispatchMode):
    with _pop_mode_temporarily() as popped:
      if popped is not current:
        raise RuntimeError("unexpected TorchDispatchMode stack mutation")
      yield
    return
  yield


def routed_einsum(
    equation: str,
    *operands: torch.Tensor,
) -> torch.Tensor:
  """Run one known contraction explicitly through vecops when configured."""
  use_native = (
      config.native_matmul_dispatch
      and platform.machine().lower() in {"aarch64", "arm64"}
  )
  if not use_native:
    return torch.einsum(equation, *operands)
  with suspend_native_matmul_dispatch():
    result = _native_einsum(equation, operands)
    if result is None:
      return torch.einsum(equation, *operands)
    return result


def execution_context():
  """Return the configured model-wide ATen matmul routing context."""
  if (
      config.native_matmul_dispatch
      and platform.machine().lower() in {"aarch64", "arm64"}
  ):
    return NativeMatmulDispatchMode()
  return nullcontext()


__all__ = [
    "NativeMatmulDispatchMode",
    "execution_context",
    "routed_einsum",
    "suspend_native_matmul_dispatch",
]
