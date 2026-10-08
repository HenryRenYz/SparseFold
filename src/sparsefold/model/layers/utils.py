# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for training AlphaFold and similar models."""

import numbers
from collections import abc

import torch


def mask_mean(mask: torch.Tensor, value: torch.Tensor, axis=None, keepdims=False, eps=1e-10) -> torch.Tensor:
  """Masked mean."""

  mask_shape = mask.shape
  value_shape = value.shape

  has_batch = False
  if len(mask_shape) == len(value_shape) - 1:
    has_batch = True
    # mask = mask[None]  # Align with batch dim
  elif len(mask_shape) != len(value_shape):
    raise ValueError('Shapes are not compatible: {}, {}'.format(mask_shape, value_shape))

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(
    axis, abc.Iterable
  ), 'axis needs to be either an iterable, integer or "None"'

  broadcast_factor = 1.0
  for axis_ in axis:
    if has_batch and axis_ >= 0:
      raise ValueError('Axis needs to be negative when a batch dim presents')
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
      assert mask_size == value_size, error

  return torch.sum(mask * value, keepdims=keepdims, axis=axis) / (
    torch.clamp(
      torch.sum(mask, keepdims=keepdims, axis=axis) * broadcast_factor, min=eps
    )
  )
