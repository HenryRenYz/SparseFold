# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Distogram head."""

from typing import Final

import torch

from sparsefold.model import legacy_config as base_config
from sparsefold.schema import batch as feat_batch
from sparsefold.model import config as model_config
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import util


_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


class DistogramHead(fnn.Module):
  """Distogram head."""
  __version__ = '1.0'

  class Config(base_config.BaseConfig):
    first_break: float = 2.3125
    last_break: float = 21.6875
    num_bins: int = 64

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      pair_channels: int,
      name='distogram_head',
  ):
    super().__init__()
    self.name = name
    self.pair_channels = pair_channels
    self.first_break = config.first_break
    self.last_break = config.last_break
    self.num_bins = config.num_bins

    # ASC: initializer=self.global_config.final_init
    self.half_logits = fnn.Linear(
        pair_channels, config.num_bins, bias=False,
        name='half_logits'
    )

  def forward(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, torch.Tensor],
      return_distogram: bool = False,
  ) -> dict[str, torch.Tensor]:
    pair_act = embeddings["pair"].to(torch.float32)
    seq_mask = batch.token_features.mask.to(torch.bool)
    pair_mask = seq_mask[:, None] * seq_mask[None, :]

    left_half_logits = self.half_logits(pair_act)

    right_half_logits = left_half_logits
    logits = left_half_logits + torch.swapaxes(right_half_logits, -2, -3)
    probs = torch.softmax(logits, dim=-1)
    breaks = torch.linspace(
        self.first_break,
        self.last_break,
        self.num_bins - 1,
        device=util.DEFAULT_DEVICE,
    )

    bin_tops = torch.cat([breaks.flatten(), (breaks[-1] + (breaks[-1] - breaks[-2])).flatten()])
    threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
    is_contact_bin = 1.0 * (bin_tops <= threshold)
    with util.fp32_matmul_precision("highest"):
      contact_probs = torch.einsum(
          'ijk,k->ij', probs, is_contact_bin
      )
    contact_probs = pair_mask * contact_probs

    return_dict = {"bin_edges": breaks, "contact_probs": contact_probs}
    if return_distogram:
      return_dict["distogram"] = logits
    
    return return_dict
