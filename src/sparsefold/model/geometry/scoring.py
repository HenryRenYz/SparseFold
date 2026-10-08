# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Library of scoring methods of the model outputs."""

import torch


def pseudo_beta_fn_torch(
    aatype: torch.Tensor,
    dense_atom_positions: torch.Tensor,
    dense_atom_masks: torch.Tensor,
    indices: torch.Tensor,
    is_ligand: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Create pseudo beta atom positions and optionally mask.

  Args:
    aatype: [num_res] amino acid types.
    dense_atom_positions: [num_res, NUM_DENSE, 3] vector of all atom positions.
    dense_atom_masks: [num_res, NUM_DENSE] mask.
    is_ligand: [num_res] flag if something is a ligand.

  Returns:
    Pseudo beta dense atom positions and the corresponding mask.
  """
  if is_ligand is None:
    # TODO ASC: dabian, weird constant selection
    is_ligand = torch.zeros_like(aatype, dtype=torch.bool)

  pseudobeta_index_polymer = torch.index_select(
    indices.to(device=aatype.device),
    dim=0,
    index=aatype.long()
  ).to(torch.int32)

  pseudobeta_index = torch.where(
    is_ligand,
    torch.zeros_like(pseudobeta_index_polymer),
    pseudobeta_index_polymer,
  )

  pseudo_beta = torch.take_along_dim(
    dense_atom_positions,
    indices=pseudobeta_index[..., None, None].long(),
    dim=-2
  )
  pseudo_beta = torch.squeeze(pseudo_beta, dim=-2)

  pseudo_beta_mask = torch.take_along_dim(
    dense_atom_masks,
    indices=pseudobeta_index[..., None].long(),
    dim=-1
  ).to(torch.float32)
  pseudo_beta_mask = torch.squeeze(pseudo_beta_mask, dim=-1)

  return pseudo_beta, pseudo_beta_mask
