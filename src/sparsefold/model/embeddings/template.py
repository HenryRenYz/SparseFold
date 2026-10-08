# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Modules for embedding templates."""

from typing import Final

import torch

from sparsefold.model import legacy_config as base_config
from sparsefold.model.vocabulary import residue_names
from sparsefold.schema import features
from sparsefold.model import config as model_config
from sparsefold.model.geometry import residue_frames as protein_data_processing
from sparsefold.model.trunk import blocks as modules
from sparsefold.model.geometry import scoring
from sparsefold.runtime import util
from sparsefold.model import geometry
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import compiler
from sparsefold.runtime import ops
from sparsefold.runtime import dist


class DistogramFeaturesConfig(base_config.BaseConfig):
  # The left edge of the first bin.
  min_bin: float = 3.25
  # The left edge of the final bin. The final bin catches everything larger than
  # `max_bin`.
  max_bin: float = 50.75
  # The number of bins in the distogram.
  num_bins: int = 39


def dgram_from_positions(
    positions: torch.Tensor,
    min_bin: float,
    max_bin: float,
    num_bins: int
):
  """Compute distogram from amino acid positions.

  Args:
    positions: ([batch], num_res, 3) Position coordinates.
    config: Distogram bin configuration.

  Returns:
    Distogram with the specified number of bins.
    Shapes (..., config.num_bins)
  """
  device = positions.device
  lower_breaks = torch.linspace(min_bin, max_bin, num_bins, dtype=torch.float32, device=device)
  lower_breaks = torch.square(lower_breaks)
  upper_breaks = torch.cat(
    (lower_breaks[1:], torch.tensor([1e8], dtype=torch.float32, device=device)), dim=-1
  )
  dist2 = torch.sum(
    torch.square(
      torch.unsqueeze(positions, dim=-2)
      - torch.unsqueeze(positions, dim=-3)
    ),
    dim=-1,
    keepdim=True,
  )

  dgram = (dist2 > lower_breaks).to(torch.float32) * (
      dist2 < upper_breaks
  ).to(torch.float32)
  return dgram


def make_backbone_rigid(
    positions: geometry.Vec3Array,
    mask: torch.Tensor,
    group_indices: torch.IntTensor,
) -> tuple[geometry.Rigid3Array, torch.Tensor]:
  """Make backbone Rigid3Array and mask.

  Args:
    positions: (num_res, num_atoms) of atom positions as Vec3Array.
    mask: (num_res, num_atoms) for atom mask.
    group_indices: (num_res, num_group, 3) for atom indices forming groups.

  Returns:
    tuple of backbone Rigid3Array and mask (num_res,).
  """
  backbone_indices = group_indices[:, 0].long()

  # main backbone frames differ in sidechain frame convention.
  # for sidechain it's (C, CA, N), for backbone it's (N, CA, C)
  # Hence using c, b, a, each of shape (num_res,).
  c, b, a = [backbone_indices[..., i] for i in range(3)]

  slice_index = util.vmap_slice
  rigid_mask = (
      slice_index(mask, a) * slice_index(mask, b) * slice_index(mask, c)
  ).to(torch.float32)

  frame_positions = []
  for indices in [a, b, c]:
    frame_positions.append(
      geometry.Vec3Array(
        slice_index(positions.x, indices),
        slice_index(positions.y, indices),
        slice_index(positions.z, indices)
      )
    )

  rotation = geometry.Rot3Array.from_two_vectors(
    frame_positions[2] - frame_positions[1],
    frame_positions[0] - frame_positions[1],
  )
  rigid = geometry.Rigid3Array(rotation, frame_positions[1])

  return rigid, rigid_mask


class TemplateEmbedding(fnn.Module):
  """Embed a set of templates."""

  class Config(base_config.BaseConfig):
    num_channels: int = 64
    template_stack: modules.PairFormerIteration.Config = base_config.autocreate(
      num_layer=2,
      pair_transition=base_config.autocreate(num_intermediate_factor=2),
    )
    dgram_features: DistogramFeaturesConfig = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      query_num_channels: int,
      distributed: bool = False,
      name='template_embedding',
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.num_channels = config.num_channels
    self.query_num_channels = query_num_channels

    # ///// MODULE STRUCTURE ///// #
    self.template_embedder = SingleTemplateEmbedding(
      config, global_config,
      query_num_channels=query_num_channels,
      distributed=self.distributed,
      name='single_template_embedding'
    )
    # ASC: initializer=relu
    self.output_linear = fnn.Linear(
      config.num_channels, query_num_channels,
      name='output_linear'
    )

  def get_template_slack(
      self,
      templates: features.Templates,
      multichain_mask_2d: torch.Tensor,
      dtype: torch.dtype
  ) -> torch.Tensor:
    result = []
    for i in range(len(templates.aatype)):
      aatype = templates.aatype[i]
      atom_positions = templates.atom_positions[i]
      atom_mask = templates.atom_mask[i]
      lane = self.template_embedder.get_template_slack(
        features.Templates(aatype, atom_positions, atom_mask),
        multichain_mask_2d=multichain_mask_2d,
        dtype=dtype
      )
      result.append(lane)
    return torch.stack(result)

  def forward(
      self,
      query_embedding: torch.Tensor,
      seq_mask: torch.Tensor,
      templates: torch.Tensor,
  ) -> torch.Tensor:
    """Generate an embedding for a set of templates.

    Args:
      query_embedding: [num_res, num_res, num_channel] a query tensor that will
        be used to attend over the templates to remove the num_templates
        dimension.
      templates: A 'Templates' object.
      seq_mask: [num_res] Padding mask for attention operations.
      multichain_mask_2d: [num_res, num_res] Pair mask for multichain.

    Returns:
      An embedding of size [num_res, num_res, num_channels]
    """
    # N_tokens
    shard_size, num_res, query_num_channels = query_embedding.shape
    # num_templates = templates.aatype.shape[0]
    num_atoms = 24
    assert query_embedding.shape == (
        shard_size,
        num_res,
        self.query_num_channels,
    )
    # assert templates.aatype.shape == (
    #     num_templates,
    #     num_res
    # )
    # assert templates.atom_positions.shape == (
    #     num_templates,
    #     num_res,
    #     num_atoms,
    #     3,
    # )
    # assert templates.atom_mask.shape == (
    #     num_templates,
    #     num_res,
    #     num_atoms
    # )
    assert seq_mask.shape == (
        num_res,
    )

    # num_templates = templates.aatype.shape[0]
    num_templates = len(templates)

    # Embed each template separately.
    # TODO ASC: possible for parallel
    def scan_fn(carry, templates):
      embedding = self.template_embedder(
          query_embedding,
          templates,
          seq_mask,
      )
      return carry + embedding
    
    scan_init = torch.zeros(
        (shard_size, num_res, self.num_channels), dtype=query_embedding.dtype,
        device=util.DEFAULT_DEVICE
    )
    summed_template_embeddings = ops.stack_scan(
        scan_fn, scan_init, templates
    )

    embedding = summed_template_embeddings / (1e-7 + num_templates)
    embedding = torch.relu(embedding)
    embedding = self.output_linear(embedding)

    assert embedding.shape == (shard_size, num_res, query_num_channels)
    return embedding

_DENSE_ATOM_IDX: Final[torch.Tensor] = torch.as_tensor(protein_data_processing.RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX)
_RESTYPE_PSEUDOBETA_INDEX: Final[torch.Tensor] = torch.as_tensor(protein_data_processing.RESTYPE_PSEUDOBETA_INDEX)


class SingleTemplateEmbedding(fnn.Module):
  """Embed a single template."""

  def __init__(
      self,
      config: TemplateEmbedding.Config,
      global_config: model_config.GlobalConfig,
      *,
      query_num_channels: int,
      distributed: bool = False,
      name='single_template_embedding',
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    self.dgram_features = config.dgram_features.as_tuple()
    num_channels = config.num_channels

    # ///// MODULE STRUCTURES ///// #
    self.query_embedding_norm = fnn.LayerNorm(
      query_num_channels,
      name='query_embedding_norm'
    )

    # shapes
    poly_types_num = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
    template_pair_embeddings_input_shapes = [
      (config.dgram_features.num_bins),  # dgram
      (),  # pseudo_beta_mask_2d
      (poly_types_num,),  # aatype[None, :, :]
      (poly_types_num,),  # aatype[:, None, :]
      (),  # unit_vector[0]
      (),  # unit_vector[1]
      (),  # unit_vector[2]
      (),  # backbone_mask_2d
      # TODO ASC: wrong shape
      (query_num_channels,),  # query_embedding (TODO Cannot extract)
    ]

    # ASC: ReLU initialized
    self.template_pair_embeddings = fnn.ModuleList([
      fnn.Linear(
        x, num_channels, bias=False,
        name=f'template_pair_embedding_{i}'
      )
      for i, x in enumerate(template_pair_embeddings_input_shapes)
    ])

    self.has_template_stack = config.template_stack.num_layer > 0
    if self.has_template_stack:
      self.template_embedding_iteration = fnn.LayerStack(
        [modules.PairFormerIteration(
          config.template_stack, global_config,
          num_channels=num_channels,
          distributed=distributed,
          name='template_embedding_iteration'
        ) for _ in range(config.template_stack.num_layer)]
      )

    self.output_layer_norm = fnn.LayerNorm(
      num_channels, bias=True,
      name='output_layer_norm'
    )

  def get_template_slack(
      self,
      templates: features.Templates,
      multichain_mask_2d: torch.Tensor,
      dtype: torch.dtype
  ) -> torch.Tensor:
    # Compute distogram feature for the template.
    aatype = templates.aatype
    dense_atom_mask = templates.atom_mask

    dense_atom_positions = templates.atom_positions
    dense_atom_positions = dense_atom_positions * dense_atom_mask[..., None]

    pseudo_beta_positions, pseudo_beta_mask = scoring.pseudo_beta_fn_torch(
      templates.aatype, dense_atom_positions, dense_atom_mask, _RESTYPE_PSEUDOBETA_INDEX
    )
    pseudo_beta_mask_2d = (
        pseudo_beta_mask[:, None] * pseudo_beta_mask[None, :]
    )
    pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask_2d
    min_bin, max_bin, num_bins = self.dgram_features
    dgram = dgram_from_positions(
      pseudo_beta_positions, min_bin, max_bin, num_bins
    )
    dgram = (dgram * pseudo_beta_mask_2d[..., None])
    dgram = dgram.to(dtype)
    pseudo_beta_mask_2d = pseudo_beta_mask_2d.to(dtype)
    # ASC: init len = 2
    to_concat = [dgram, pseudo_beta_mask_2d]

    aatype = fnn.one_hot(
      aatype,
      residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
      value=1.0,
      dtype=dtype
    )
    to_concat.append(aatype[None, :, :])
    to_concat.append(aatype[:, None, :])

    # Compute a feature representing the normalized vector between each
    # backbone affine - i.e. in each residues local frame, what direction are
    # each of the other residues.

    template_group_indices = torch.index_select(
      _DENSE_ATOM_IDX.to(device=templates.aatype.device),
      dim=0,
      index=templates.aatype.long()
    )
    rigid, backbone_mask = make_backbone_rigid(
      geometry.Vec3Array.from_array(dense_atom_positions),
      dense_atom_mask,
      template_group_indices.to(torch.int32),
    )
    points = rigid.translation
    rigid_vec = rigid[:, None].inverse().apply_to_point(points)
    unit_vector = rigid_vec.normalized()
    unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

    unit_vector = [x.to(dtype) for x in unit_vector]
    backbone_mask = backbone_mask.to(dtype)

    backbone_mask_2d = backbone_mask[:, None] * backbone_mask[None, :]
    backbone_mask_2d = backbone_mask_2d * multichain_mask_2d
    unit_vector = [x * backbone_mask_2d for x in unit_vector]

    # Note that the backbone_mask takes into account C, CA and N (unlike
    # pseudo beta mask which just needs CB) so we add both masks as features.
    # ASC: len + 3
    to_concat.extend(unit_vector)
    to_concat.append(backbone_mask_2d)

    # query_embedding = self.query_embedding_norm(query_embedding)
    # # Allow the template embedder to see the query embedding.  Note this
    # # contains the position relative feature, so this is how the network knows
    # # which residues are next to each other.
    # # ASC: total len = 9
    # to_concat.append(query_embedding)

    act = 0
    assert len(to_concat) == len(self.template_pair_embeddings) - 1
    for i, layer in enumerate(self.template_pair_embeddings[:-1]):
      act += layer(to_concat[i])

    return act

  def construct_input(
      self,
      query_embedding: torch.Tensor,
      # The one computed by get_template_slack
      template_slack: torch.Tensor
  ):
    query_embedding = self.query_embedding_norm(query_embedding)
    return template_slack + self.template_pair_embeddings[-1](query_embedding)

  def forward(
      self,
      query_embedding: torch.Tensor,
      template_slack: torch.Tensor,
      padding_mask: torch.Tensor,
  ) -> torch.Tensor:
    """Build the single template embedding graph.

    Args:
      query_embedding: (num_res, num_res, num_channels) - embedding of the query
        sequence/msa.
      templates: 'Templates' object containing single Template.
      padding_mask: 1D Padding mask (Note: this doesn't care if a template
        exists, unlike the template_pseudo_beta_mask).
      multichain_mask_2d: A mask indicating intra-chain residue pairs, used to
        mask out between chain distances/features when templates are for single
        chains.

    Returns:
      A template embedding (num_res, num_res, num_channels).
    """
    # assert padding_mask_2d.dtype == query_embedding.dtype

    act = self.construct_input(query_embedding, template_slack)

    if self.has_template_stack:
      @compiler.mark_compile(fullgraph=False)
      def template_iteration_fn(mod, x: torch.Tensor):
        return mod(
            act=x, 
            seq_mask=padding_mask, 
        )

      act = fnn.layer_stack(
        template_iteration_fn, self.template_embedding_iteration
      )(act)

    act = self.output_layer_norm(act)
    return act
