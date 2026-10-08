# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Utils for geometry library."""

import numbers
from collections.abc import Iterable

import torch


def unstack(value: torch.Tensor, dim: int = -1) -> list[torch.Tensor]:
  return [
    torch.squeeze(v, dim=dim)
    for v in torch.chunk(value, value.shape[dim], dim=dim)
  ]


def angdiff(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
  """Compute absolute difference between two angles."""
  d = alpha - beta
  d = (d + torch.pi) % (2 * torch.pi) - torch.pi
  return d


def weighted_mean(
    weights: torch.Tensor,
    value: torch.Tensor,
    dim: int | Iterable[int] | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
  """Computes weighted mean in a safe way that avoids NaNs.

  This is equivalent to jnp.average for the case eps=0.0, but adds a small
  constant to the denominator of the weighted average to avoid NaNs.
  'weights' should be broadcastable to the shape of value.

  Args:
    weights: Weights to weight value by.
    value: Values to average
    dim: Dimensions to average over.
    eps: Epsilon to add to the denominator.

  Returns:
    Weighted average.
  """

  weights = torch.as_tensor(weights, dtype=value.dtype, device=value.device)
  weights = torch.broadcast_to(weights, value.shape)

  weights_shape = weights.shape

  if isinstance(dim, numbers.Integral):
    dim = [dim]
  elif dim is None:
    dim = list(range(len(weights_shape)))

  return torch.sum(weights * value, dim=dim) / (
      torch.sum(weights, dim=dim) + eps
  )
