# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""SparseFold numerical model; external AF3 host objects stop at the adapter."""
import functools
import time
import torch
from sparsefold.model import legacy_config as base_config, config as model_config
from sparsefold.model.vocabulary import residue_names
from sparsefold.schema import batch as feat_batch, features, layout
from sparsefold.model.embeddings import atom as atom_cross_attn, features as featurization
from sparsefold.model.heads import confidence as confidence_head, distogram as distogram_head
from sparsefold.model.diffusion import generator as diffusion_head
from sparsefold.model.trunk import network as evoformer_network
from sparsefold.model.layers import mapping, primitives as fnn
from sparsefold.runtime import config as _config, log, trace, dist, util, ops

class SparseFoldModel(fnn.Module):
  """Full model. Takes in data batch and returns model outputs."""

  __version__ = 1

  class HeadsConfig(base_config.BaseConfig):
    diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
    confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
    distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

  class Config(base_config.BaseConfig):
    evoformer: evoformer_network.Evoformer.Config = base_config.autocreate()
    global_config: model_config.GlobalConfig = base_config.autocreate()
    heads: "SparseFoldModel.HeadsConfig" = base_config.autocreate()
    num_recycles: int = 10
    return_embeddings: bool = False
    return_distogram: bool = False

  def __init__(
      self,
      config: Config,
      *,
      distributed: bool=False,
      name: str = "diffuser",
  ):
    super().__init__()
    self.name = name
    self.distributed = distributed and dist.is_enabled()
    global_config = config.global_config
    self.pair_channels = config.evoformer.pair_channel
    self.seq_channels = config.evoformer.seq_channel
    self.num_recycles = config.num_recycles
    self.diffusion_config = config.heads.diffusion.eval
    self.return_embeddings = config.return_embeddings
    self.return_distogram = config.return_distogram
    self.use_bfloat16 = global_config.bfloat16

    target_feat_size = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP * 2 + 1
    target_feat_size += config.evoformer.per_atom_conditioning.per_token_channels
    # target_feat: (N_tokens, target_feat_size)
    # where N_tokens is a parameter-independent runtime value
    self.target_feat_size = target_feat_size

    # ///// MODEL STRUCTURES ///// #
    self.trunk = evoformer_network.Evoformer(
      config.evoformer,
      global_config,
      target_feat_size=target_feat_size,
      distributed=self.distributed,
      name="evoformer",
    )

    self.atom_conditioner = atom_cross_attn.AtomCrossAttEncoder(
      config=config.evoformer.per_atom_conditioning,
      global_config=global_config,
      has_token_atoms_act=False,
      name="evoformer_conditioning",
    )
    
    self.diffusion_generator = diffusion_head.DiffusionHead(
      config.heads.diffusion,
      global_config,
      pair_channels=config.evoformer.pair_channel,
      seq_channels=config.evoformer.seq_channel,
      target_feat_size=target_feat_size,
      dtype=util.get_dtype(_config.diffusion_precision),
      name="diffusion_head",
    )
    self.confidence_head = confidence_head.ConfidenceHead(
      config.heads.confidence,
      global_config,
      pair_channels=config.evoformer.pair_channel,
      seq_channels=config.evoformer.seq_channel,
      target_feat_size=target_feat_size,
      # Fixed trained numerical layout contract.
      num_atoms=layout.DENSE_ATOMS_PER_TOKEN,
      name="confidence_head",
    )

    self.distogram_head = distogram_head.DistogramHead(
      config.heads.distogram,
      global_config,
      pair_channels=config.evoformer.pair_channel,
      name="distogram_head",
    )

  def forward(self, batch: feat_batch.ModelBatch | features.TorchBatchDict):
    # A raw numerical dictionary is accepted only without host metadata.
    if not isinstance(batch, feat_batch.ModelBatch):
      from sparsefold.schema.validation import REQUIRED_KEYS
      extra = set(batch) - REQUIRED_KEYS
      if extra:
        raise ValueError(f"SparseFoldModel accepts only numerical fields; separate output context before entry: {sorted(extra)}")
    # Reset all caches
    self.reset()

    torch.set_float32_matmul_precision(util.DEFAULT_FP32_MATMUL_PRECISION)
    Dtype = torch.bfloat16  # TODO

    with torch.amp.autocast(torch.device(util.DEFAULT_DEVICE).type, dtype=torch.bfloat16, enabled=False):
      batch = batch if isinstance(batch, feat_batch.ModelBatch) else feat_batch.ModelBatch.from_data_dict(batch)

      with trace.into("feat_embedding"):
        target_feat = self._create_target_feat_embedding(batch=batch)
      
      relative_feat = self.trunk._get_relative_feat(batch)

      def recycle_body(i, prev):
        embeddings = self.trunk(
          batch=batch,
          prev=prev,
          target_feat=target_feat,
          relative_feat=relative_feat
        )
        embeddings["pair"] = embeddings["pair"].to(Dtype)
        embeddings["single"] = embeddings["single"].to(Dtype)
        return embeddings
      
      num_res = batch.num_res
      if dist.is_enabled():
        # World sizes such as six do not divide the standard 512-token bucket.
        # Use backend-specific AllGatherV only for those ragged partitions and
        # retain the uniform collective fast path for exact divisors.
        evenly_partitioned = num_res % dist.get_world_size() == 0
        shard = dist.current_shard = dist.Shard(
            num_res, even=evenly_partitioned
        )
        shard_size = shard.size()
        target_feat_shard = shard.shard(target_feat, dim=0)
      else:
        shard_size = num_res
        target_feat_shard = target_feat

      embeddings = {
          "pair": torch.zeros(
              [shard_size, num_res, self.pair_channels],
              dtype=Dtype,
              device=util.DEFAULT_DEVICE,
          ),
          "single": torch.zeros(
              [shard_size, self.seq_channels],
              dtype=Dtype,
              device=util.DEFAULT_DEVICE,
          ),
          "target_feat": target_feat_shard
      }

      with trace.into("evoformer"):
        # Number of recycles is number of additional forward trunk passes.
        num_iter = self.num_recycles + 1
        embeddings = ops.fori_loop(0, num_iter, recycle_body, (embeddings))

        util.check_tensor_sanity(embeddings["single"], "embeddings.single")
        util.check_tensor_sanity(embeddings["pair"], "embeddings.pair")

      if dist.is_enabled():
        # embeddings["pair"]: (B, N, C) -> (N, N, C)
        embeddings["pair"] = shard.all_gather_into_tensor(
            input_tensor=embeddings["pair"]
        )
        # embeddings["single"]: (B, C2) -> (N, C2)
        embeddings["single"] = shard.all_gather_into_tensor(
            input_tensor=embeddings["single"]
        )
        embeddings["target_feat"] = target_feat

      total_samples = self.diffusion_config.num_samples
      if dist.is_enabled():
        # Keep one gather slot per rank. Ranks beyond the requested sample
        # count supply placeholders instead of running an unused diffusion
        # trajectory; atom-layout conversion cannot handle a zero-sized batch.
        distributed_samples = max(total_samples, dist.get_world_size())
        diffusion_shard = dist.Shard(distributed_samples, even=False)
        num_samples = diffusion_shard.size()
        padded_sample_rank = dist.get_rank() >= total_samples
      else:
        num_samples = total_samples
        padded_sample_rank = False

      with trace.into("diffusion"):
        if padded_sample_rank:
          atom_mask = batch.predicted_structure_info.atom_mask
          samples = {
              "atom_positions": torch.zeros(
                  (num_samples, *atom_mask.shape, 3),
                  dtype=torch.float32,
                  device=atom_mask.device,
              ),
              "mask": atom_mask.unsqueeze(0),
          }
          log.info(f"Skipping unused diffusion sample on rank {dist.get_rank()}.")
        else:
          # Give each non-primary rank a distinct Torch random stream; leave
          # rank 0's stream intact so its first sample remains stable.
          if dist.is_enabled() and dist.get_rank() != 0:
            torch.manual_seed(
                (torch.initial_seed() + dist.get_rank()) % (1 << 64)
            )
          samples = self.diffusion_generator(
              batch=batch,
              rel_features=relative_feat,
              embeddings=embeddings,
              use_conditioning=True,
              num_samples=num_samples,
          )

        util.check_tensor_sanity(samples["atom_positions"], "samples.atom_positions")

      # Compute dist_error_fn over all samples for distance error logging.
      with trace.into("confidence"):
        if padded_sample_rank:
          num_tokens, num_atoms = samples["mask"].shape[-2:]
          atom_shape = (num_samples, num_tokens, num_atoms)
          pair_shape = (num_samples, num_tokens, num_tokens)
          confidence_kwargs = {
              "dtype": torch.float32,
              "device": samples["atom_positions"].device,
          }
          confidence_output = {
              "predicted_lddt": torch.zeros(atom_shape, **confidence_kwargs),
              "predicted_experimentally_resolved": torch.zeros(
                  atom_shape, **confidence_kwargs
              ),
              "full_pde": torch.zeros(pair_shape, **confidence_kwargs),
              "average_pde": torch.zeros((num_samples,), **confidence_kwargs),
              "full_pae": torch.zeros(pair_shape, **confidence_kwargs),
              "tmscore_adjusted_pae_global": torch.zeros(
                  pair_shape, **confidence_kwargs
              ),
              "tmscore_adjusted_pae_interface": torch.zeros(
                  pair_shape, **confidence_kwargs
              ),
          }
        else:
          confidence_output = mapping.sharded_apply(
              functools.partial(
                  self.confidence_head,
                  seq_mask=batch.token_features.mask,
                  embeddings=embeddings,
                  token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                  asym_id=batch.token_features.asym_id,
              ),
              in_axes=0,
              enabled=True
          )(samples["atom_positions"])
      
      if dist.is_enabled():
        def gather_result_tree(
            group: str, values: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
          gathered = {}
          for key, value in values.items():
            log.info(
                f'Gathering {group}.{key}: local_shape={tuple(value.shape)}, '
                f'dtype={value.dtype}, shard_sizes={diffusion_shard.sizes()}.'
            )
            gather_start = time.perf_counter()
            gathered[key] = diffusion_shard.all_gather_padded_into_tensor(
                input_tensor=value
            )[:total_samples]
            log.info(
                f'Gathered {group}.{key}: '
                f'global_shape={tuple(gathered[key].shape)}, '
                f'elapsed={time.perf_counter() - gather_start:.6f}s.'
            )
          return gathered

        samples = gather_result_tree('samples', samples)
        confidence_output = gather_result_tree(
            'confidence_output', confidence_output
        )

      with trace.into("distogram"):
        distogram = self.distogram_head(batch, embeddings, return_distogram=self.return_distogram)

      output = {
        "diffusion_samples": samples,
        "distogram": distogram,
        **confidence_output,
      }
      if self.return_embeddings:
        output["single_embeddings"] = embeddings["single"]
        output["pair_embeddings"] = embeddings["pair"]
      return output

  def _create_target_feat_embedding(
      self, batch: feat_batch.Batch
  ) -> torch.Tensor:
    """Create target feature embedding."""

    dtype = torch.bfloat16 if self.use_bfloat16 == "all" else torch.float32

    # (N_tokens, poly_types_num * 2 + 1)
    # default poly_types_num * 2 + 1 = 31 * 2 + 1 = 63
    target_feat = featurization.create_target_feat(
      batch,
      append_per_atom_features=False,
    ).to(dtype)
    ## agress atom to token
    enc = self.atom_conditioner(
      token_atoms_act=None,
      trunk_single_cond=None,
      trunk_pair_cond=None,
      batch=batch,
    )
    # (N_tokens, poly_types_num * 2 + 1 + C)
    # where C = self.atom_conditioner.config.per_token_channels
    # default C = 63 + 384 = 447
    target_feat = torch.cat((target_feat, enc.token_act), dim=-1).to(dtype)

    return target_feat


