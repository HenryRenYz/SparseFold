# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Per-atom cross attention."""

import torch
import einops

from sparsefold.weights import legacy_scopes as pv
from sparsefold.model import legacy_config as base_config
from sparsefold.schema import batch as feat_batch
from sparsefold.model import config as model_config
from sparsefold.schema import atom_layout
from sparsefold.model.layers import utils
from sparsefold.model.diffusion import transformer as diffusion_transformer
from sparsefold.runtime import trace
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import tree
from sparsefold.runtime import cache
from sparsefold.runtime import compiler


class AtomCrossAttEncoder(fnn.Module):
  __version__ = '1.0'

  class Config(base_config.BaseConfig):
    per_token_channels: int = 768
    per_atom_channels: int = 128
    atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
    )
    per_atom_pair_channels: int = 16

  @tree.dataclass(mappable_dataclass=False, frozen=True)
  class Output:
    token_act: torch.Tensor  # (num_tokens, ch)
    skip_connection: torch.Tensor  # (num_subsets, num_queries, ch)
    queries_mask: torch.Tensor  # (num_subsets, num_queries)
    queries_single_cond: torch.Tensor  # (num_subsets, num_queries, ch)
    keys_mask: torch.Tensor  # (num_subsets, num_keys)
    keys_single_cond: torch.Tensor  # (num_subsets, num_keys, ch)
    pair_cond: torch.Tensor  # (num_subsets, num_queries, num_keys, ch)

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      has_token_atoms_act: bool,
      seq_channels: int | None = None,
      pair_channels: int | None = None,
      dtype: torch.dtype | None = None,
      name: str = 'encoder'
  ):
    super().__init__()
    self.name = name
    self.has_token_atoms_act = has_token_atoms_act
    self.seq_channels = seq_channels
    self.pair_channels = pair_channels
    self.dtype = dtype

    # ///// MODULE STRUCTURE ///// #
    self.embed_ref_pos = fnn.Linear(
      3, config.per_atom_channels, bias=False,
      precision="highest",
      name='embed_ref_pos'
    )
    self.embed_ref_mask = fnn.Linear(
      1, config.per_atom_channels, bias=False,
      name='embed_ref_mask'
    )
    self.embed_ref_element = fnn.Linear(
      128, config.per_atom_channels, bias=False,
      name='embed_ref_element'
    )
    self.embed_ref_charge = fnn.Linear(
      1, config.per_atom_channels, bias=False,
      name='embed_ref_charge'
    )
    self.embed_ref_atom_name = fnn.Linear(
      4 * 64, config.per_atom_channels, bias=False,
      name='embed_ref_atom_name'
    )

    self.single_to_pair_cond_row = fnn.Linear(
      config.per_atom_channels, config.per_atom_pair_channels, bias=False,
      name='single_to_pair_cond_row'
    )
    self.single_to_pair_cond_col = fnn.Linear(
      config.per_atom_channels, config.per_atom_pair_channels, bias=False,
      name='single_to_pair_cond_col'
    )
    self.embed_pair_offsets = fnn.Linear(
      3, config.per_atom_pair_channels, bias=False,
      precision="highest",
      name='embed_pair_offsets'
    )
    self.embed_pair_distances = fnn.Linear(
      1, config.per_atom_pair_channels, bias=False,
      name='embed_pair_distances'
    )

    # TODO ASC: does this size connected to the other in diffusor?
    self.single_to_pair_cond_row_1 = fnn.Linear(
      config.per_atom_channels, config.per_atom_pair_channels, bias=False,
      name='single_to_pair_cond_row_1'
    )
    self.single_to_pair_cond_col_1 = fnn.Linear(
      config.per_atom_channels, config.per_atom_pair_channels, bias=False,
      name='single_to_pair_cond_col_1'
    )
    self.embed_pair_offsets_1 = fnn.Linear(
      3, config.per_atom_pair_channels, bias=False,
      precision="highest",
      name='embed_pair_offsets_1'
    )
    self.embed_pair_distances_1 = fnn.Linear(
      1, config.per_atom_pair_channels, bias=False,
      name='embed_pair_distances_1'
    )
    self.embed_pair_offsets_valid = fnn.Linear(
      1, config.per_atom_pair_channels, bias=False,
      name='embed_pair_offsets_valid'
    )

    # ASC: relu initialization
    self.pair_mlps = fnn.ModuleList([
      fnn.Linear(
        config.per_atom_pair_channels, config.per_atom_pair_channels, bias=False,
        name=f'pair_mlp_{i}'
      ) for i in range(1, 4)
    ])

    self.atom_transformer_encoder = diffusion_transformer.CrossAttTransformer(
      config.atom_transformer, global_config,
      pair_cond_size=config.per_atom_pair_channels,
      num_channels=config.per_atom_channels,
      dtype=dtype,
      super_name=name,
      name='atom_transformer_encoder'
    )

    self.project_atom_features_for_aggr = fnn.Linear(
      config.per_atom_channels, config.per_token_channels, bias=False,
      dtype=dtype,
      name='project_atom_features_for_aggr'
    )

    # Has trunk
    if seq_channels is not None:
      self.lnorm_trunk_single_cond = fnn.LayerNorm(
        seq_channels, bias=False,
        name='lnorm_trunk_single_cond'
      )
      # ASC: initializer=self.global_config.final_init
      self.embed_trunk_single_cond = fnn.Linear(
        seq_channels, config.per_atom_channels, bias=False,
        precision="highest",
        name='embed_trunk_single_cond'
      )

    if has_token_atoms_act:
      self.atom_positions_to_features = fnn.Linear(
        3, config.per_atom_channels, bias=False,
        precision="highest",
        name='atom_positions_to_features'
      )

    if pair_channels is not None:
      self.lnorm_trunk_pair_cond = fnn.LayerNorm(
        pair_channels, bias=False,
        name='lnorm_trunk_pair_cond'
      )
      # ASC: initializer=self.global_config.final_init
      self.embed_trunk_pair_cond = fnn.Linear(
        pair_channels, config.per_atom_pair_channels, bias=False,
        precision="highest",
        name='embed_trunk_pair_cond'
      )
    
    self._cache = cache.ValueCache()

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='_', **kwargs)

  def _per_atom_conditioning(
      self, batch: feat_batch.Batch
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """computes single and pair conditioning for all atoms in each token."""

    # Compute per-atom single conditioning
    # Shape (num_tokens, num_dense, channels)
    act = self.embed_ref_pos(batch.ref_structure.positions)
    act = self.embed_ref_mask(
      batch.ref_structure.mask.to(torch.float32)[:, :, None], residual=act
    )
    # Element is encoded as atomic number if the periodic table, so
    # 128 should be fine.
    act = self.embed_ref_element(
      fnn.one_hot(batch.ref_structure.element, 128), residual=act
    )
    act = self.embed_ref_charge(
      torch.arcsinh(batch.ref_structure.charge)[:, :, None], residual=act
    )
    # Characters are encoded as ASCII code minus 32, so we need 64 classes,
    # to encode all standard ASCII characters between 32 and 96.
    atom_name_chars_1hot = fnn.one_hot(batch.ref_structure.atom_name_chars, 64)
    num_token, num_dense, _ = act.shape
    act = self.embed_ref_atom_name(
        atom_name_chars_1hot.reshape(num_token, num_dense, -1), residual=act
    )
    # TODO ASC: redundant conversion
    act = act * batch.ref_structure.mask.to(torch.float32)[:, :, None]

    # Compute pair conditioning
    # shape (num_tokens, num_dense, num_dense, channels)
    # Embed single features
    act_relu = torch.relu(act)
    row_act = self.single_to_pair_cond_row(act_relu)
    col_act = self.single_to_pair_cond_col(act_relu)
    pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
    # Embed pairwise offsets
    pair_act = self.embed_pair_offsets(
        batch.ref_structure.positions[:, :, None, :]
        - batch.ref_structure.positions[:, None, :, :],
        residual=pair_act,
    )
    # Embed pairwise inverse squared distances
    sq_dists = torch.sum(
        torch.square(
            batch.ref_structure.positions[:, :, None, :]
            - batch.ref_structure.positions[:, None, :, :]
        ),
        dim=-1,
    )

    pair_act = self.embed_pair_distances(
        1.0 / (1 + sq_dists[:, :, :, None]), residual=pair_act
    )

    return act, pair_act

  def forward(
      self,
      token_atoms_act: torch.Tensor | None,  # ([batch], num_tokens, max_atoms_per_token, 3)
      trunk_single_cond: torch.Tensor | None,  # (num_tokens, ch)
      trunk_pair_cond: torch.Tensor | None,  # (num_tokens, num_tokens, ch)
      batch: feat_batch.Batch,
  ) -> Output:
    """Cross-attention on flat atom subsets and mapping to per-token features."""

    def _compute_cache():
      token_atoms_single_cond, _ = self._per_atom_conditioning(batch)
      token_atoms_mask = batch.predicted_structure_info.atom_mask
      queries_single_cond = atom_layout.convert(
          batch.atom_cross_att.token_atoms_to_queries,
          token_atoms_single_cond,
          layout_axes=(-3, -2),
      )
      queries_mask = atom_layout.convert(
          batch.atom_cross_att.token_atoms_to_queries,
          token_atoms_mask,
          layout_axes=(-2, -1),
      )
    
      # If provided, broadcast single conditioning from trunk to all queries
      if trunk_single_cond is not None:
        trunk_single_cond_ = self.embed_trunk_single_cond(
            self.lnorm_trunk_single_cond(trunk_single_cond)
        )
        queries_single_cond += atom_layout.convert(
            batch.atom_cross_att.tokens_to_queries,
            trunk_single_cond_,
            layout_axes=(-2,),
        )
      
      # Gather the keys from the queries.
      keys_single_cond = atom_layout.convert(
          batch.atom_cross_att.queries_to_keys,
          queries_single_cond,
          layout_axes=(-3, -2),
      )
      keys_mask = atom_layout.convert(
          batch.atom_cross_att.queries_to_keys, 
          queries_mask, 
          layout_axes=(-2, -1)
      )
      
      # Embed single features into the pair conditioning.
      # shape (num_subsets, num_queries, num_keys, ch)
      queries_relu = torch.relu(queries_single_cond)
      row_act = self.single_to_pair_cond_row_1(queries_relu)
      pair_cond_keys_input = atom_layout.convert(
          batch.atom_cross_att.queries_to_keys,
          queries_single_cond,
          layout_axes=(-3, -2),
      )
      keys_relu = torch.relu(pair_cond_keys_input)
      col_act = self.single_to_pair_cond_col_1(keys_relu)
      pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

      if trunk_pair_cond is not None:
        # If provided, broadcast the pair conditioning for the trunk (evoformer
        # pairs) to the atom pair activations. This should boost ligands, but also
        # help for cross attention within proteins, because we always have atoms
        # from multiple residues in a subset.
        # Map trunk pair conditioning to per_atom_pair_channels
        # (num_tokens, num_tokens, per_atom_pair_channels)
        trunk_pair_cond_ = self.embed_trunk_pair_cond(
            self.lnorm_trunk_pair_cond(trunk_pair_cond)
        )

        # Create the GatherInfo into a flattened trunk_pair_cond from the
        # queries and keys gather infos.
        num_tokens = trunk_pair_cond_.shape[0]
        # (num_subsets, num_queries)
        tokens_to_queries = batch.atom_cross_att.tokens_to_queries
        # (num_subsets, num_keys)
        tokens_to_keys = batch.atom_cross_att.tokens_to_keys
        # (num_subsets, num_queries, num_keys)
        trunk_pair_to_atom_pair = atom_layout.GatherInfo(
            gather_idxs=(
                num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                + tokens_to_keys.gather_idxs[:, None, :]
            ),
            gather_mask=(
                tokens_to_queries.gather_mask[:, :, None]
                & tokens_to_keys.gather_mask[:, None, :]
            ),
            input_shape=torch.tensor([num_tokens, num_tokens], dtype=torch.int32),
        )
        # Gather the conditioning and add it to the atom-pair activations.
        pair_act += atom_layout.convert(
            trunk_pair_to_atom_pair, 
            trunk_pair_cond_, 
            layout_axes=(-3, -2)
        )
        
      # Embed pairwise offsets
      queries_ref_pos = atom_layout.convert(
          batch.atom_cross_att.token_atoms_to_queries,
          batch.ref_structure.positions,
          layout_axes=(-3, -2),
      )
      queries_ref_space_uid = atom_layout.convert(
          batch.atom_cross_att.token_atoms_to_queries,
          batch.ref_structure.ref_space_uid,
          layout_axes=(-2, -1),
      )
      keys_ref_pos = atom_layout.convert(
          batch.atom_cross_att.queries_to_keys,
          queries_ref_pos,
          layout_axes=(-3, -2),
      )
      keys_ref_space_uid = atom_layout.convert(
          batch.atom_cross_att.queries_to_keys,
          batch.ref_structure.ref_space_uid,
          layout_axes=(-2, -1),
      )

      offsets_valid = (
          queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
      )
      offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
      pair_act += (
          self.embed_pair_offsets_1(offsets)
          * offsets_valid[:, :, :, None]
      )

      # Embed pairwise inverse squared distances
      sq_dists = torch.sum(torch.square(offsets), dim=-1)
      pair_act += (
          self.embed_pair_distances_1(
              1.0 / (1 + sq_dists[:, :, :, None])
          )
          * offsets_valid[:, :, :, None]
      )
      # Embed offsets valid mask
      pair_act = self.embed_pair_offsets_valid(
          offsets_valid[:, :, :, None].to(torch.float32), residual=pair_act
      )

      # Run a small MLP on the pair acitvations
      pair_act2 = pair_act
      for mlp in self.pair_mlps:
        pair_act2 = mlp(torch.relu(pair_act2))
      pair_act += pair_act2
      
      return (
          token_atoms_mask.bool(),
          queries_single_cond,
          queries_mask.bool(),
          keys_single_cond,
          keys_mask.bool(),
          pair_act
      )

    token_atoms_mask, queries_single_cond, queries_mask, keys_single_cond, keys_mask, pair_act = (
        self._cache.get(trunk_single_cond, trunk_pair_cond, batch, name="cache", factory=_compute_cache)
    )
    
    @compiler.mark_compile
    def _pre_compute():
      if not self.has_token_atoms_act:
        # if no token_atoms_act is given (e.g. begin of evoformer), we use the
        # static conditioning only
        queries_act = queries_single_cond.to(self.dtype)
      else:
        # Convert token_atoms_act to queries layout and map to per_atom_channels
        # (num_subsets, num_queries, channels)
        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_act.to(self.dtype),
            layout_axes=(-3, -2),
        )
        queries_act = self.atom_positions_to_features(queries_act)
        queries_act = queries_act.masked_fill_(~queries_mask[..., None], 0.0)
        queries_act += queries_single_cond
      # queries_act = queries_act.to(self.dtype)
      return queries_act
  
    queries_act = _pre_compute()
    # Run the atom cross attention transformer.
    with trace.into("cross_attn_transformer"):
      queries_act = self.atom_transformer_encoder(
          queries_act=queries_act,
          queries_mask=queries_mask,
          queries_to_keys=batch.atom_cross_att.queries_to_keys,
          keys_mask=keys_mask,
          queries_single_cond=queries_single_cond,
          keys_single_cond=keys_single_cond,
          pair_cond=pair_act,
      )
    
    @compiler.mark_compile
    def _post_compute(queries_act: torch.Tensor):
      # old_dtype = queries_act.dtype
      # The transformer result has no remaining consumer. Mask in-place to
      # avoid cloning the full atom activation before projection/skip capture.
      queries_act.masked_fill_(~queries_mask[..., None], 0.0)
      skip_connection = queries_act

      # Convert back to token-atom layout and aggregate to tokens
      queries_act = self.project_atom_features_for_aggr(queries_act)
      token_atoms_act = atom_layout.convert(
          batch.atom_cross_att.queries_to_token_atoms,
          queries_act,
          layout_axes=(-3, -2),
      )
      token_act = utils.mask_mean(
          token_atoms_mask[..., None], torch.relu(token_atoms_act), axis=-2
      )
      return token_act, skip_connection
    
    token_act, skip_connection = _post_compute(queries_act)
    return self.Output(
        token_act=token_act,  # 1 of 2 variables in denoising loop
        skip_connection=skip_connection,  # 2 of 2 variables in denoising loop
        queries_mask=queries_mask,
        queries_single_cond=queries_single_cond,
        keys_mask=keys_mask,
        keys_single_cond=keys_single_cond,
        pair_cond=pair_act,
    )


class AtomCrossAttDecoder(fnn.Module):
  class Config(base_config.BaseConfig):
    per_atom_channels: int = 128
    atom_transformer: diffusion_transformer.CrossAttTransformer.Config = (
      base_config.autocreate(num_intermediate_factor=2, num_blocks=3)
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      num_channels: int,
      pair_cond_size: int,
      dtype: torch.dtype | None = None,
      name: str = 'decoder'
  ):
    super().__init__()
    self.name = name
    self.per_atom_channels = config.per_atom_channels
    self.dtype = dtype

    # ///// MODULE STRUCTURE ///// #
    self.project_token_features_for_broadcast = fnn.Linear(
      num_channels, config.per_atom_channels, bias=False,
      dtype=dtype,
      name='project_token_features_for_broadcast'
    )

    self.atom_transformer_decoder = diffusion_transformer.CrossAttTransformer(
      config.atom_transformer, global_config,
      pair_cond_size=pair_cond_size,
      num_channels=config.per_atom_channels,
      dtype=dtype,
      super_name=name,
      name='atom_transformer_decoder'
    )

    self.atom_features_layer_norm = fnn.LayerNorm(
      config.per_atom_channels, bias=False,  # TODO untested
      name='atom_features_layer_norm'
    )
    # ASC: initializer=self.global_config.final_init
    self.atom_features_to_position_update = fnn.Linear(
      config.per_atom_channels, 3, bias=False,
      precision="highest",
      name='atom_features_to_position_update'
    )

  def set_parameters(self, visitor: pv.ParamVisitor, **kwargs):
    super().set_parameters(visitor, sep='_', **kwargs)

  def forward(
      self,
      token_act: torch.Tensor,  # ([batch], num_tokens, ch)
      enc: AtomCrossAttEncoder.Output,
      batch: feat_batch.Batch
  ):  # ([batch], num_tokens, max_atoms_per_token, 3)
    """Mapping to per-atom features and self-attention on subsets."""
    # map per-token act down to per_atom channels
    @compiler.mark_compile
    def _pre_compute(token_act):
      token_act = self.project_token_features_for_broadcast(token_act.to(self.dtype))
      # Broadcast to token-atoms layout and convert to queries layout.
      num_token, max_atoms_per_token = (
        batch.atom_cross_att.queries_to_token_atoms.shape
      )
      token_atom_act = einops.repeat(token_act, "... c -> ... a c", a=max_atoms_per_token)
      queries_act = atom_layout.convert(
        batch.atom_cross_att.token_atoms_to_queries,
        token_atom_act,
        layout_axes=(-3, -2),
      )
      queries_act += enc.skip_connection
      queries_act = queries_act.masked_fill_(~enc.queries_mask[..., None], 0.0)
      return queries_act
    
    queries_act = _pre_compute(token_act)
    # Run the atom cross attention transformer.
    with trace.into("cross_attn_transformer"):
      queries_act = self.atom_transformer_decoder(
        queries_act=queries_act,
        queries_mask=enc.queries_mask,
        queries_to_keys=batch.atom_cross_att.queries_to_keys,
        keys_mask=enc.keys_mask,
        queries_single_cond=enc.queries_single_cond,
        keys_single_cond=enc.keys_single_cond,
        pair_cond=enc.pair_cond,
      )
    
    @compiler.mark_compile
    def _post_compute(queries_act):
      # Decoder output is consumed only below; avoid an activation-sized clone.
      queries_act.masked_fill_(~enc.queries_mask[..., None], 0.0)
      queries_act = self.atom_features_layer_norm(queries_act)
      queries_position_update = self.atom_features_to_position_update(queries_act)
      position_update = atom_layout.convert(
        batch.atom_cross_att.queries_to_token_atoms,
        queries_position_update,
        layout_axes=(-3, -2),
      )
      return position_update

    position_update = _post_compute(queries_act)
    return position_update
