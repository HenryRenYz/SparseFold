# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Confidence Head."""

import torch

from sparsefold.model import legacy_config as base_config
from sparsefold.model import config as model_config
from sparsefold.schema import atom_layout
from sparsefold.model.trunk import blocks as modules
from sparsefold.model.embeddings import template as template_modules
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import util
from sparsefold.runtime import compiler
from sparsefold.runtime import trace


def _safe_norm(x, keepdim, dim, eps=1e-8):
  return torch.sqrt(eps + torch.sum(torch.square(x), dim=dim, keepdim=keepdim))


def _bounded_softmax(logits: torch.Tensor) -> torch.Tensor:
  """Stable CPU softmax without the data-dependent SLEEF exp slow path.

  exp(x / 256) is approximated by a degree-five polynomial and squared eight
  times. Clipping at -60 drops probabilities too small to affect float32 sums.
  """
  shifted = (logits - torch.amax(logits, dim=-1, keepdim=True)).clamp_min(-60.0)
  x = shifted / 256.0
  exp_values = 1.0 + x * (
    1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
  )
  for _ in range(8):
    exp_values = exp_values * exp_values
  return exp_values / torch.sum(exp_values, dim=-1, keepdim=True)


# ASC: Vmapped version of original
class ConfidenceHead(fnn.Module):
  """Head to predict the distance errors in a prediction."""
  __version__ = '1.0'

  class PAEConfig(base_config.BaseConfig):
    max_error_bin: float = 31.0
    num_bins: int = 64

  class Config(base_config.BaseConfig):
    """Configuration for ConfidenceHead."""

    pairformer: modules.PairFormerIteration.Config = base_config.autocreate(
      single_attention=base_config.autocreate(),
      single_transition=base_config.autocreate(),
      num_layer=4,
    )
    max_error_bin: float = 31.0
    num_plddt_bins: int = 50
    num_bins: int = 64
    no_embedding_prob: float = 0.2
    pae: 'ConfidenceHead.PAEConfig' = base_config.autocreate()
    dgram_features: template_modules.DistogramFeaturesConfig = (
      base_config.autocreate()
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      target_feat_size: int,
      pair_channels: int,
      seq_channels: int,
      num_atoms: int,
      name='confidence_head',
  ):
    super().__init__()
    self.name = name
    self.target_feat_size = target_feat_size
    self.pair_channels = pair_channels
    self.seq_channels = seq_channels
    self.dgram_features = config.dgram_features.as_tuple()
    self.use_bfloat16 = global_config.bfloat16
    self.max_error_bin = config.max_error_bin
    self.num_bins = config.num_bins
    self.pae_max_error_bin = config.pae.max_error_bin
    self.pae_num_bins = config.pae.num_bins
    self.num_plddt_bins = config.num_plddt_bins

    # ///// MODULE STRUCTURES ///// #
    self.left_target_feat_project = fnn.Linear(
      target_feat_size, pair_channels, bias=False,
      name='_embed_features/left_target_feat_project'
    )
    self.right_target_feat_project = fnn.Linear(
      target_feat_size, pair_channels, bias=False,
      name='_embed_features/right_target_feat_project'
    )
    self.distogram_feat_project = fnn.Linear(
      config.dgram_features.num_bins, pair_channels, bias=False,
      name='_embed_features/distogram_feat_project'
    )

    self.confidence_pairformer = fnn.LayerStack(
      [modules.PairFormerIteration(
        config.pairformer, global_config,
        with_single=True,
        num_channels=pair_channels,
        seq_channels=seq_channels,
        name='confidence_pairformer'
      ) for _ in range(config.pairformer.num_layer)]
    )

    self.logits_ln = fnn.LayerNorm(
      pair_channels, bias=True,
      name='logits_ln'
    )
    # ASC: initializer=self.global_config.final_init
    self.left_half_distance_logits = fnn.Linear(
      pair_channels, config.num_bins, bias=False,
      name='left_half_distance_logits'
    )

    self.pae_logits_ln = fnn.LayerNorm(
      pair_channels, bias=True,
      name='pae_logits_ln'
    )
    self.pae_logits = fnn.Linear(
      pair_channels, config.pae.num_bins, bias=False,
      name='pae_logits'
    )

    self.plddt_logits_ln = fnn.LayerNorm(
      seq_channels, bias=True,
      name='plddt_logits_ln'
    )
    # ASC: initializer=self.global_config.final_init
    self.plddt_logits = fnn.Linear(
      seq_channels, (num_atoms, config.num_plddt_bins),
      name='plddt_logits'
    )

    self.experimentally_resolved_ln = fnn.LayerNorm(
      seq_channels, bias=True,
      name='experimentally_resolved_ln'
    )
    # ASC: initializer=self.global_config.final_init
    self.experimentally_resolved_logits = fnn.Linear(
      seq_channels, (num_atoms, 2), bias=False,
      name='experimentally_resolved_logits'
    )

  def _embed_features(
      self,
      dense_atom_positions: torch.Tensor,
      token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
      pair_mask: torch.Tensor,
      pair_act: torch.Tensor,
      target_feat: torch.Tensor,
  ):
    assert target_feat.shape[-1] == self.target_feat_size
    assert pair_act.shape[-1] == self.pair_channels

    out = self.left_target_feat_project(
      target_feat
    ).to(pair_act.dtype)
    out = out + self.right_target_feat_project(
      target_feat
    ).to(pair_act.dtype)[:, None]
    positions = atom_layout.convert(
      token_atoms_to_pseudo_beta,
      dense_atom_positions,
      layout_axes=(-3, -2),
    )
    min_bin, max_bin, num_bins = self.dgram_features
    dgram = template_modules.dgram_from_positions(
      positions, min_bin, max_bin, num_bins
    )
    dgram = dgram * pair_mask[..., None]

    out = self.distogram_feat_project(
      dgram.to(pair_act.dtype), residual=out
    )
    return out

  def forward(
      self,
      # May contains batch dimension
      dense_atom_positions: torch.Tensor,
      embeddings: dict[str, torch.Tensor],
      seq_mask: torch.Tensor,
      token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
      asym_id: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Builds ConfidenceHead module.

    Arguments:
      dense_atom_positions: ([batch], N_res, N_atom, 3) array of positions.
      embeddings: Dictionary of representations.
      seq_mask: Sequence mask.
      token_atoms_to_pseudo_beta: Pseudo beta info for atom tokens.
      asym_id: Asym ID token features.

    Returns:
      Dictionary of results.
    """
    dtype = (
      torch.bfloat16 if self.use_bfloat16 == 'all' else torch.float32
    )
    has_batch = dense_atom_positions.ndim == 4

    # Begin 'BFloat16' Context
    assert len(seq_mask.shape) == 1 and seq_mask.dtype == torch.bool
    pair_mask = seq_mask[:, None] * seq_mask[None, :]
    pair_mask = pair_mask.to(dtype)

    pair_act = embeddings["pair"].to(dtype)
    single_act = embeddings["single"].to(dtype)
    target_feat = embeddings["target_feat"].to(dtype)

    num_residues = seq_mask.shape[0]
    num_pair_channels = pair_act.shape[2]

    assert num_pair_channels == self.pair_channels

    pair_act = pair_act + self._embed_features(
      dense_atom_positions,
      token_atoms_to_pseudo_beta,
      pair_mask,
      pair_act,
      target_feat,
    )

    if has_batch:
      batch_size = len(pair_act)
      single_act = torch.expand_copy(single_act[None], (batch_size,) + single_act.shape)
      assert single_act.shape[:-2] == pair_act.shape[:-3], f"Batch mismstach: {single_act.shape}, {pair_act.shape}"
    
    @compiler.mark_compile(fullgraph=False)
    def pairformer_fn(mod, carry: tuple[torch.Tensor, torch.Tensor]):
      pair_act, single_act = carry
      return mod(
        act=pair_act,
        single_act=single_act,
        seq_mask=seq_mask,
      )
    
    pairformer_fn = fnn.layer_stack(pairformer_fn, self.confidence_pairformer)

    pair_act, single_act = pairformer_fn((pair_act, single_act))
    pair_act = pair_act.to(torch.float32)
    assert pair_act.shape[-3:] == (num_residues, num_residues, self.pair_channels)

    # Produce logits to predict a distogram of pairwise distance errors
    # between the input prediction and the ground truth.

    # Shape (num_res, num_res, num_bins)
    left_distance_logits = self.left_half_distance_logits(
      self.logits_ln(pair_act)
    )
    right_distance_logits = left_distance_logits
    distance_logits = left_distance_logits + torch.swapaxes(  # Symmetrize.
      right_distance_logits, -2, -3
    )
    # Shape (num_bins,)
    distance_breaks = torch.linspace(
      0.0, self.max_error_bin, self.num_bins - 1,
      device=util.DEFAULT_DEVICE
    )

    step = distance_breaks[1] - distance_breaks[0]

    # Add half-step to get the center
    bin_centers = distance_breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = torch.cat(
        [bin_centers, bin_centers[-1:] + step], dim=0
    )

    with trace.into('distance_softmax'):
      distance_probs = _bounded_softmax(distance_logits)

    pred_distance_error = (
        torch.sum(distance_probs * bin_centers, dim=-1) * pair_mask
    )
    average_pred_distance_error = torch.sum(
      pred_distance_error, dim=(-2, -1)
    ) / torch.sum(pair_mask, dim=(-2, -1))

    pae_outputs = {}
    # Shape (num_res, num_res, num_bins)
    pae_logits = self.pae_logits(
      self.pae_logits_ln(pair_act)
    )
    # Shape (num_bins,)
    pae_breaks = torch.linspace(
      0.0, self.pae_max_error_bin, self.pae_num_bins - 1,
      device=util.DEFAULT_DEVICE
    )
    step = pae_breaks[1] - pae_breaks[0]
    # Add half-step to get the center
    bin_centers = pae_breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = torch.cat(
      [bin_centers, bin_centers[-1:] + step], dim=0
    )
    with trace.into('pae_softmax'):
      pae_probs = _bounded_softmax(pae_logits)

    seq_mask_bool = seq_mask.to(torch.bool)
    pair_mask_bool = seq_mask_bool[:, None] * seq_mask_bool[None, :]
    pae = torch.sum(pae_probs * bin_centers, dim=-1) * pair_mask_bool
    pae_outputs.update({
        "full_pae": pae,
    })
    # End 'BFloat16' Context

    # The pTM is computed outside of bfloat16 context.
    tmscore_adjusted_pae_global, tmscore_adjusted_pae_interface = (
      self._get_tmscore_adjusted_pae(
        asym_id=asym_id,
        seq_mask=seq_mask,
        pair_mask=pair_mask_bool,
        bin_centers=bin_centers,
        pae_probs=pae_probs,
      )
    )
    pae_outputs.update({
        "tmscore_adjusted_pae_global": tmscore_adjusted_pae_global,
        "tmscore_adjusted_pae_interface": tmscore_adjusted_pae_interface,
    })
    single_act = single_act.to(torch.float32)

    # pLDDT
    # Shape (num_res, num_atom, num_bins)
    plddt_logits = self.plddt_logits(
      self.plddt_logits_ln(single_act)
    )

    bin_width = 1.0 / self.num_plddt_bins
    bin_centers = torch.arange(
      0.5 * bin_width, 1.0, bin_width, device=util.DEFAULT_DEVICE
    )
    with trace.into('plddt_softmax'):
      plddt_probs = _bounded_softmax(plddt_logits)
    predicted_lddt = torch.sum(plddt_probs * bin_centers, dim=-1)
    predicted_lddt *= 100.0

    # Experimentally resolved
    # Shape (num_res, num_atom, 2)
    experimentally_resolved_logits = self.experimentally_resolved_logits(
      self.experimentally_resolved_ln(single_act)
    )

    with trace.into('experimentally_resolved_softmax'):
      predicted_experimentally_resolved = _bounded_softmax(
        experimentally_resolved_logits
      )[..., 1]

    return {
        'predicted_lddt': predicted_lddt,
        'predicted_experimentally_resolved': predicted_experimentally_resolved,
        'full_pde': pred_distance_error,
        'average_pde': average_pred_distance_error,
        **pae_outputs,
    }

  def _get_tmscore_adjusted_pae(
      self,
      asym_id: torch.Tensor,
      seq_mask: torch.Tensor,
      pair_mask: torch.Tensor,
      bin_centers: torch.Tensor,
      pae_probs: torch.Tensor,
  ):
    def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):
      # Clip to avoid negative/undefined d0.
      clipped_num_res = torch.clamp(num_interface_tokens, min=19)

      # Compute d_0(num_res) as defined by TM-score, eqn. (5) in
      # http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
      # Yang & Skolnick "Scoring function for automated
      # assessment of protein structure template quality" 2004.
      d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

      # Make compatible with ([batch], num_tokens, num_tokens, num_bins)
      d0 = d0[..., :, :, None]
      bin_centers = bin_centers[..., None, None, :]

      # TM-Score term for every bin.
      tm_per_bin = 1.0 / (1 + torch.square(bin_centers) / torch.square(d0))
      # E_distances tm(distance).
      predicted_tm_term = torch.sum(pae_probs * tm_per_bin, dim=-1)
      return predicted_tm_term

    # Interface version
    x = asym_id[None, :] == asym_id[:, None]
    num_chain_tokens = torch.sum(x * pair_mask, dim=-1, dtype=torch.int32)
    num_interface_tokens = num_chain_tokens[None, :] + num_chain_tokens[:, None]
    # Don't double-count within a single chain
    num_interface_tokens -= x * (num_interface_tokens // 2)
    num_interface_tokens *= pair_mask

    num_global_tokens = torch.ones(
      pair_mask.shape,
      dtype=torch.int32,
      device=num_interface_tokens.device,
    ) * seq_mask.sum(dtype=torch.int32)

    assert num_global_tokens.dtype == torch.int32, f'bad type: {num_global_tokens.dtype}'
    assert num_interface_tokens.dtype == torch.int32, f'bad type: {num_interface_tokens.dtype}'
    global_apae = get_tmscore_adjusted_pae(
      num_global_tokens, bin_centers, pae_probs
    )
    interface_apae = get_tmscore_adjusted_pae(
      num_interface_tokens, bin_centers, pae_probs
    )
    return global_apae, interface_apae
