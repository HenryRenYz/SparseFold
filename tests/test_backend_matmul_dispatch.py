# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Platform-independent checks for the model-wide matmul lowering."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sparsefold.backends.vecops import matmul_dispatch


def _reference_native_matmul(
    lhs: torch.Tensor,
    weight: torch.Tensor,
    *,
    c_input: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    out_dtype: torch.dtype,
) -> torch.Tensor:
  del out_dtype
  if weight.shape[0] == 1 and lhs.shape[0] != 1:
    weight = weight.expand(lhs.shape[0], -1, -1)
  output = torch.bmm(lhs, weight.transpose(-2, -1)) * alpha
  if c_input is not None:
    output = output + c_input * beta
  return output


def test_native_einsum_expands_ellipsis_without_opt_einsum(monkeypatch) -> None:
  monkeypatch.setattr(
      matmul_dispatch,
      "_kernels",
      lambda: SimpleNamespace(matmul=_reference_native_matmul),
  )
  generator = torch.Generator().manual_seed(20260911)
  lhs = torch.randn(2, 3, 4, 5, generator=generator)
  rhs = torch.randn(5, 6, 7, generator=generator)

  actual = matmul_dispatch._native_einsum(
      "...a, ahi -> ...hi", (lhs, rhs)
  )
  expected = torch.einsum("...a, ahi -> ...hi", lhs, rhs)

  assert actual is not None
  torch.testing.assert_close(actual, expected)


def test_native_einsum_handles_broadcast_batch_and_output_permutation(
    monkeypatch,
) -> None:
  monkeypatch.setattr(
      matmul_dispatch,
      "_kernels",
      lambda: SimpleNamespace(matmul=_reference_native_matmul),
  )
  generator = torch.Generator().manual_seed(20260912)
  lhs = torch.randn(2, 3, 4, 5, generator=generator)
  rhs = torch.randn(1, 6, 5, 3, 7, generator=generator)

  actual = matmul_dispatch._native_einsum(
      "...abc,...dcae->...dbae", (lhs, rhs)
  )
  expected = torch.einsum("...abc,...dcae->...dbae", lhs, rhs)

  assert actual is not None
  torch.testing.assert_close(actual, expected)


def test_native_matmul_fuses_addmm_epilogue(monkeypatch) -> None:
  monkeypatch.setattr(
      matmul_dispatch,
      "_kernels",
      lambda: SimpleNamespace(matmul=_reference_native_matmul),
  )
  generator = torch.Generator().manual_seed(20260913)
  lhs = torch.randn(7, 5, generator=generator)
  rhs = torch.randn(5, 11, generator=generator)
  bias = torch.randn(11, generator=generator)

  actual = matmul_dispatch._native_matmul(
      lhs, rhs, c_input=bias, alpha=1.5, beta=0.25
  )
  expected = torch.addmm(bias, lhs, rhs, alpha=1.5, beta=0.25)

  assert actual is not None
  torch.testing.assert_close(actual, expected)


def test_native_matmul_fuses_baddbmm_epilogue(monkeypatch) -> None:
  monkeypatch.setattr(
      matmul_dispatch,
      "_kernels",
      lambda: SimpleNamespace(matmul=_reference_native_matmul),
  )
  generator = torch.Generator().manual_seed(20260914)
  lhs = torch.randn(3, 7, 5, generator=generator)
  rhs = torch.randn(3, 5, 11, generator=generator)
  c_input = torch.randn(3, 7, 11, generator=generator)

  actual = matmul_dispatch._native_matmul(
      lhs, rhs, c_input=c_input, alpha=0.75, beta=0.5
  )
  expected = torch.baddbmm(c_input, lhs, rhs, alpha=0.75, beta=0.5)

  assert actual is not None
  torch.testing.assert_close(actual, expected)


def test_suspend_native_matmul_dispatch_skips_unrelated_ops() -> None:
  class CountingMode(matmul_dispatch.NativeMatmulDispatchMode):
    def __init__(self):
      super().__init__()
      self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      self.calls += 1
      return super().__torch_dispatch__(func, types, args, kwargs)

  lhs = torch.ones(4)
  rhs = torch.ones(4)
  mode = CountingMode()
  with mode:
    torch.add(lhs, rhs)
    with matmul_dispatch.suspend_native_matmul_dispatch():
      torch.add(lhs, rhs)
    torch.add(lhs, rhs)

  assert mode.calls == 2
