# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Global config for the model."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from sparsefold.model import legacy_config as base_config

_Shape2DType: TypeAlias = tuple[int | None, int | None]


class GlobalConfig(base_config.BaseConfig):
  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
    (2048, None),
    (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation = 'triton'
