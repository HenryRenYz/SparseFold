# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Evoformer network."""

import torch

from sparsefold.model import legacy_config as base_config
from sparsefold.model.vocabulary import residue_names
from sparsefold.schema import batch as feat_batch
from sparsefold.schema import features
from sparsefold.model import config as model_config
from sparsefold.model.embeddings import atom as atom_cross_attention
from sparsefold.model.embeddings import features as featurization
from sparsefold.model.trunk import blocks as modules
from sparsefold.model.embeddings import template as template_modules
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import util
from sparsefold.runtime import trace
from sparsefold.runtime import compiler
from sparsefold.runtime.cache import ValueCache
from sparsefold.runtime import dist


class Evoformer(fnn.Module):
  """Creates 'single' and 'pair' embeddings."""

  __version__ = "1.0"

  class PairformerConfig(
    modules.PairFormerIteration.Config
  ):  # pytype: disable=invalid-function-definition
    block_remat: bool = False
    remat_block_size: int = 8

  class Config(base_config.BaseConfig):
    """Configuration for Evoformer."""

    max_relative_chain: int = 2
    msa_channel: int = 64
    seq_channel: int = 384
    max_relative_idx: int = 32
    num_msa: int = 1024
    pair_channel: int = 128
    pairformer: "Evoformer.PairformerConfig" = base_config.autocreate(
      single_transition=base_config.autocreate(),
      single_attention=base_config.autocreate(),
      num_layer=48,
    )
    per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoder.Config = (
      base_config.autocreate(
        per_token_channels=384,
        per_atom_channels=128,
        atom_transformer=base_config.autocreate(
          num_intermediate_factor=2,
          num_blocks=3,
        ),
        per_atom_pair_channels=16,
      )
    )
    template: template_modules.TemplateEmbedding.Config = base_config.autocreate()
    msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      target_feat_size: int,
      distributed: bool = False,
      name="evoformer",
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.target_feat_size = target_feat_size
    self.max_relative_idx = config.max_relative_idx
    self.max_relative_chain = config.max_relative_chain
    self.pair_channel = config.pair_channel
    self.seq_channel = config.seq_channel
    self.num_msa = config.num_msa
    self.use_bfloat16 = global_config.bfloat16

    # ///// MODULE STRUCTURES ///// #
    self.left_single = fnn.Linear(
      target_feat_size, config.pair_channel, bias=False, name="left_single"
    )
    self.right_single = fnn.Linear(
      target_feat_size, config.pair_channel, bias=False, name="right_single"
    )

    self.prev_embedding_layer_norm = fnn.LayerNorm(
      config.pair_channel,
      name="prev_embedding_layer_norm",
    )
    # ASC: initializer=self.global_config.final_init
    self.prev_embedding = fnn.Linear(
      config.pair_channel, config.pair_channel, bias=False, name="prev_embedding"
    )

    # 447
    rel_feat_size = 4 * config.max_relative_idx + 2 * config.max_relative_chain + 7
    self.position_activations = fnn.Linear(
      rel_feat_size,
      config.pair_channel,
      bias=False,
      name="_relative_encoding/position_activations",
    )

    self.bond_embedding = fnn.Linear(
      1, config.pair_channel, bias=False, name="bond_embedding"
    )

    self.template_embedding = template_modules.TemplateEmbedding(
      config.template,
      global_config,
      query_num_channels=config.pair_channel,
      distributed=self.distributed,
      name="template_embedding",
    )

    msa_feat_size = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 3

    self.msa_activations = fnn.Linear(
      msa_feat_size, config.msa_channel, name="msa_activations"
    )
    self.extra_msa_target_feat = fnn.Linear(
      target_feat_size, config.msa_channel, name="extra_msa_target_feat"
    )

    # MSA embedding, 4 layers
    self.msa_stack_iteration = fnn.LayerStack(
      [modules.EvoformerIteration(
        config.msa_stack,
        global_config,
        num_channels=config.msa_channel,
        num_output_channel=config.pair_channel,
        distributed=self.distributed,
        name="msa_stack",
      ) for _ in range(config.msa_stack.num_layer)]
    )

    self.single_activations = fnn.Linear(
      target_feat_size, config.seq_channel, bias=False, name="single_activations"
    )
    self.prev_single_embedding_layer_norm = fnn.LayerNorm(
      config.seq_channel,
      bias=True,
      name="prev_single_embedding_layer_norm",
    )
    # initializer=self.global_config.final_init
    self.prev_single_embedding = fnn.Linear(
      config.seq_channel,
      config.seq_channel,
      bias=False,
      name="prev_single_embedding",
    )

    # 48 Layers
    def layerwise_reuse_policy(slice: util.Slice):
      return slice
    
    self.trunk_pairformer = fnn.LayerStack(
      [modules.PairFormerIteration(
        config.pairformer,
        global_config,
        with_single=True,
        num_channels=config.pair_channel,
        seq_channels=config.seq_channel,
        index_reuse_policy=layerwise_reuse_policy,  # Only first run computes indices
        distributed=self.distributed,
        name="trunk_pairformer",
      ) for _ in range(config.pairformer.num_layer)],
      name="_1",
    )
    
    self._cache = ValueCache()

  def _get_relative_feat(self, batch: feat_batch.Batch) -> torch.Tensor:
    """Add relative position encodings."""
    rel_feat = featurization.create_relative_encoding(
      batch.token_features,
      self.max_relative_idx,
      self.max_relative_chain,
    )
    return rel_feat

  def _get_position_feat(
      self, rel_features: torch.Tensor, dtype: torch.dtype
  ) -> torch.Tensor:
    """Add relative position encodings."""
    return self.position_activations(rel_features.to(dtype))

  def _get_seq_pair_embedding(
      self,
      token_features: features.TokenFeatures,
      target_feat: torch.Tensor,  # (N, C_T)
      target_feat_shard: torch.Tensor,  # (B, C_T)
  ) -> torch.Tensor:
    """Generated Pair embedding from sequence."""
    left_single: torch.Tensor = self.left_single(target_feat_shard)[:, None] # (B, 1, C_T)
    right_single: torch.Tensor = self.right_single(target_feat)[None, :]  # (1, N, C_T)

    pair_activations = left_single + right_single  # (B, N, C_T)
    shard_size, num_res, _ = pair_activations.shape
    assert pair_activations.shape == (
      shard_size,
      num_res,
      self.pair_channel,
    )
    return pair_activations

  def _get_bonds_act(
      self, batch: feat_batch.Batch, dtype: torch.dtype
  ) -> torch.Tensor:
    """Embeds bond features and merges into pair activations."""
    # Construct contact matrix.
    num_tokens = batch.token_features.token_index.shape[0]
    contact_matrix = torch.zeros(
      (num_tokens, num_tokens), device=util.DEFAULT_DEVICE
    )

    tokens_to_polymer_ligand_bonds = (
      batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    )
    gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
    gather_mask_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_mask.prod(
      dim=1
    ).to(gather_idxs_polymer_ligand.dtype)[:, None]
    # If valid mask then it will be all 1's, so idxs should be unchanged.
    gather_idxs_polymer_ligand = (
        gather_idxs_polymer_ligand * gather_mask_polymer_ligand
    )

    tokens_to_ligand_ligand_bonds = (
      batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    )
    gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
    gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
      dim=1
    ).to(gather_idxs_ligand_ligand.dtype)[:, None]
    gather_idxs_ligand_ligand = (
        gather_idxs_ligand_ligand * gather_mask_ligand_ligand
    )

    gather_idxs = torch.cat([gather_idxs_polymer_ligand, gather_idxs_ligand_ligand])
    contact_matrix[gather_idxs[:, 0], gather_idxs[:, 1]] = 1.0

    # Because all the padded index's are 0's.
    contact_matrix[0, 0] = 0.0

    bonds_act = self.bond_embedding(contact_matrix[:, :, None].to(dtype))
    return bonds_act

  @trace.mark(name="template")
  def _embed_template_pair(
      self,
      batch: feat_batch.Batch,
      pair_activations: torch.Tensor,
      seq_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Embeds Templates and merges into pair activations."""
    def _compute_cache():
      dtype = pair_activations.dtype
      asym_id = batch.token_features.asym_id
      # Construct a mask such that only intra-chain template features are
      # computed, since all templates are for each chain individually.
      multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(dtype)

      templates = self.template_embedding.get_template_slack(
        batch.templates, multichain_mask, dtype
      )
      if dist.is_enabled():
        templates = dist.current_shard.shard(templates, dim=1)
      return templates
      
    templates = self._cache.get(batch, factory=_compute_cache)
      
    template_act = self.template_embedding(
      query_embedding=pair_activations,
      templates=templates,
      seq_mask=seq_mask,
    )
    return pair_activations + template_act

  @trace.mark(name="msa")
  def _embed_process_msa(
      self,
      msa_batch: features.MSA,
      pair_activations: torch.Tensor,  # (B, N, C)
      seq_mask: torch.Tensor,  # (N,), bool
      target_feat_shard: torch.Tensor,  # (B, C_T)
  ) -> torch.Tensor:
    """Processes MSA and returns updated pair activations."""
    def _compute_cache():
      dtype = pair_activations.dtype
      msa_batch_ = featurization.shuffle_msa(msa_batch)
      msa_batch_ = featurization.truncate_msa_batch(msa_batch_, self.num_msa)
      msa_feat = featurization.create_msa_feat(msa_batch_).to(dtype)  # (M, N, d)

      if dist.is_enabled():
        msa_feat = dist.current_shard.shard(msa_feat, dim=1)

      msa_activations = self.msa_activations(msa_feat)  # (M, N, C)
      msa_activations += self.extra_msa_target_feat(target_feat_shard)[None]
      
      msa_mask = msa_batch_.mask.to(torch.bool)  # (M, N), bool
      return msa_activations, msa_mask
    
    msa_activations, msa_mask = self._cache.get(
        msa_batch, target_feat_shard, factory=_compute_cache
    )
    
    evoformer_input = {"msa": msa_activations, "pair": pair_activations}
    masks = {"msa": msa_mask, "seq": seq_mask}

    @compiler.mark_compile(fullgraph=False)
    def evoformer_fn(mod, x: torch.Tensor):
      return mod(
          activations=x, 
          masks=masks, 
      )

    evoformer_output = fnn.layer_stack(
        evoformer_fn, self.msa_stack_iteration
    )(evoformer_input)

    return evoformer_output["pair"]

  def forward(
      self,
      batch: feat_batch.Batch,
      prev: dict[str, torch.Tensor],
      target_feat: torch.Tensor,
      relative_feat: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    
    assert self.use_bfloat16 in ["all", "none"]

    num_residues = target_feat.shape[0]
    # (B, C_T), in non distributed env, this is identical to `target_feat`
    target_feat_shard = prev["target_feat"]  
    shard_size = target_feat_shard.shape[0]
    # TODO bad constant
    dtype = torch.bfloat16 if self.use_bfloat16 == "all" else torch.float32

    def _compute_cache():
      # pair_activations, (B, N, C)
      pair_activations = self._get_seq_pair_embedding(
          batch.token_features, target_feat, target_feat_shard
      )
      position_feat = self._get_position_feat(relative_feat, dtype)
      bonds_act = self._get_bonds_act(batch, dtype)
      # TODO move interior
      if dist.is_enabled():
        position_feat = dist.current_shard.shard(position_feat, dim=0)
        bonds_act = dist.current_shard.shard(bonds_act, dim=0)
      ## R.P.E.
      pair_activations += position_feat
      pair_activations += bonds_act
      
      # single_activations, (B, S)
      single_activations = self.single_activations(target_feat_shard)
      
      return pair_activations, single_activations
    
    pair_activations, single_activations = self._cache.get(
        batch, target_feat, relative_feat,
        name="cache", factory=_compute_cache
    )
    seq_mask = batch.token_features.mask
    
    pair_activations = self.prev_embedding(
        self.prev_embedding_layer_norm(prev["pair"].to(pair_activations.dtype)),
        residual=pair_activations,
    )

    # Template module (4 PairFormerIteration)
    pair_activations = self._embed_template_pair(
        batch=batch,
        pair_activations=pair_activations,
        seq_mask=seq_mask
    )
    # MSA module (4 EvoformerIteration)
    pair_activations = self._embed_process_msa(
        msa_batch=batch.msa,
        pair_activations=pair_activations,
        seq_mask=seq_mask,
        target_feat_shard=target_feat_shard
    )

    single_activations = self.prev_single_embedding(
        self.prev_single_embedding_layer_norm(
            prev["single"].to(single_activations.dtype)
        ),
        residual=single_activations,
    )

    @compiler.mark_compile(fullgraph=False)
    def pairformer_fn(mod, x):
      pair_act, single_act = x
      return mod(
          act=pair_act, 
          single_act=single_act, 
          seq_mask=seq_mask, 
      )

    # Pairformer Module (48 PairFormerIteration)
    with trace.into("pairformer_stack"):
      pair_activations, single_activations = fnn.layer_stack(
          pairformer_fn, self.trunk_pairformer
      )((pair_activations, single_activations))

    assert pair_activations.shape == (
      shard_size,
      num_residues,
      self.pair_channel,
    )
    assert single_activations.shape == (shard_size, self.seq_channel)
    assert len(target_feat.shape) == 2
    assert target_feat.shape[0] == num_residues
    output = {
        "single": single_activations,
        "pair": pair_activations,
        "target_feat": target_feat_shard,
    }

    return output
