# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: Apache-2.0

"""Tensor gather operations only; chemistry and layout construction are upstream."""
from collections.abc import Mapping
import math
from typing import Self, TypeAlias
import torch
from sparsefold.runtime import tree
TensorLike: TypeAlias = torch.Tensor

@tree.dataclass(mappable_dataclass=False, frozen=True)
class GatherInfo:
    """Gather indices to translate from one atom layout to another.

  All members are tensor (usually 1-dim or 2-dim) with the same
  shape, e.g.
  - [num_atoms]
  - [num_residues, max_atoms_per_residue]
  - [num_fragments, max_fragments_per_residue]

  Attributes:
    gather_idxs: tensor of int: gather indices into a flattened array
    gather_mask: tensor of bool: mask for resulting array
    input_shape: list of int: the shape of the unflattened input
      array
    shape: output shape. Just returns gather_idxs.shape
  """
    gather_idxs: torch.IntTensor
    gather_mask: torch.BoolTensor
    input_shape: torch.IntTensor

    def __getitem__(self, key: torch.Tensor) -> Self:
        key = key.long() if isinstance(key, torch.Tensor) else key
        return GatherInfo(gather_idxs=self.gather_idxs[key], gather_mask=self.gather_mask[key], input_shape=self.input_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.gather_idxs.shape

    def as_dict(self, key_prefix: str | None=None) -> dict[str, torch.Tensor]:
        prefix = f'{key_prefix}:' if key_prefix else ''
        return {prefix + 'gather_idxs': self.gather_idxs, prefix + 'gather_mask': self.gather_mask, prefix + 'input_shape': self.input_shape}

    @classmethod
    def from_dict(cls, d: Mapping[str, torch.Tensor], key_prefix: str | None=None) -> Self:
        """Creates GatherInfo from a given dictionary."""
        prefix = f'{key_prefix}:' if key_prefix else ''
        return cls(gather_idxs=torch.as_tensor(d[prefix + 'gather_idxs'], dtype=torch.int32), gather_mask=torch.as_tensor(d[prefix + 'gather_mask'], dtype=torch.bool), input_shape=torch.as_tensor(d[prefix + 'input_shape'], dtype=torch.int32))

def convert(gather_info: GatherInfo, arr: TensorLike, *, layout_axes: tuple[int, ...]=(0,)) -> torch.Tensor:
    """Convert an array from one atom layout to another."""
    if not isinstance(arr, torch.Tensor):
        arr = torch.tensor(arr)
    layout_axes = tuple((i if i >= 0 else i + arr.ndim for i in layout_axes))
    if not layout_axes or any(i < 0 or i >= arr.ndim for i in layout_axes):
        raise ValueError('layout_axes must contain valid axes')
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1
    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f'layout_axes must be continuous. Got {layout_axes}.')
    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]
    expected = tuple(int(x) for x in gather_info.input_shape)
    # Indices address the flattened source, so an equal-size view is valid.
    # Fixed main and official AF3 use a [tokens,24] view with queries_to_keys
    # whose logical source is [subsets,32] for reference-space identifiers.
    # Preserve that source ordering; changing the caller would change results.
    # Additional padding is permitted only along the original leading axis,
    # because changing a trailing dimension also changes linear index strides.
    same_volume = math.prod(layout_shape) == math.prod(expected)
    leading_padding = len(layout_shape) == len(expected) and bool(expected) and layout_shape[0] >= expected[0] and tuple(layout_shape[1:]) == expected[1:]
    if not expected or any(size < 0 for size in expected) or not (same_volume or leading_padding):
        raise ValueError(f'Input layout {tuple(layout_shape)} incompatible with gather shape {expected}')
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    arr_flattened_shape = batch_shape + (-1,) + features_shape
    arr_flattened = arr.reshape(arr_flattened_shape)
    indices = (slice(None),) * layout_axes_begin + (gather_info.gather_idxs, ...)
    out_arr = arr_flattened[indices]
    broadcasted_mask_shape = (1,) * len(batch_shape) + gather_info.gather_mask.shape + (1,) * len(features_shape)
    out_arr.masked_fill_(~gather_info.gather_mask.reshape(broadcasted_mask_shape), 0.0)
    return out_arr
