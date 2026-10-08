# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Head."""

import os
from numbers import Number
from typing import Final

import torch

from sparsefold.model import legacy_config as base_config
from sparsefold.schema import batch as feat_batch
from sparsefold.model import config as model_config
from sparsefold.model.layers import utils
from sparsefold.model.embeddings import atom as atom_cross_attention
from sparsefold.model.diffusion import transformer as diffusion_transformer
from sparsefold.model.diffusion import noise as noise_level_embeddings
from sparsefold.runtime import config as torch_config
from sparsefold.model.layers import primitives as fnn
from sparsefold.runtime import ops
from sparsefold.runtime import tree
from sparsefold.runtime import util
from sparsefold.runtime import trace

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


def _routed_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
  """Use the optional ARM vecops route without coupling other backends to it."""
  if not torch_config.native_matmul_dispatch:
    return torch.einsum(equation, *operands)
  from sparsefold.backends.vecops.matmul_dispatch import routed_einsum

  return routed_einsum(equation, *operands)


def random_rotation(batch: int | None = None):
  # Create a random rotation (Gram-Schmidt orthogonalization of two
  # random normal vectors)
  with util.fp32_matmul_precision("highest"):
    if batch is not None:
      v0, v1 = torch.randn((2, batch, 3), device=util.DEFAULT_DEVICE)
      e0 = v0 / torch.clamp(torch.norm(v0, p=2, dim=-1), min=1e-10)[..., None]
      v1 = v1 - e0 * _routed_einsum(
          "...n,...n->...", v1, e0
      )[..., None]
      e1 = v1 / torch.clamp(torch.norm(v1, p=2, dim=-1), min=1e-10)[..., None]
      e2 = torch.cross(e0, e1, dim=-1)
      return torch.stack([e0, e1, e2], dim=1)
    else:
      v0, v1 = torch.randn((2, 3), device=util.DEFAULT_DEVICE)
      e0 = v0 / torch.clamp(torch.norm(v0, p=2), min=1e-10)
      v1 = v1 - e0 * torch.dot(v1, e0)
      e1 = v1 / torch.clamp(torch.norm(v1, p=2), min=1e-10)
      e2 = torch.cross(e0, e1, dim=-1)
      return torch.stack([e0, e1, e2])


def random_augmentation(
    positions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
  """Apply random rigid augmentation.

  Args:
    positions: atom positions of shape ([batch], <common_axes>, 3)
      The batch dim is optional
    mask: per-atom mask of shape (<common_axes>,)

  Returns:
    Transformed positions with the same shape as input positions.
  """

  assert positions.ndim == mask.ndim + 2
  batch_size = positions.shape[0]

  center = utils.mask_mean(
    mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )

  # (batch_size, 3, 3)
  rot = random_rotation(batch_size)
  translation = torch.randn((batch_size, 1, 1, 3), device=util.DEFAULT_DEVICE)

  with util.fp32_matmul_precision("highest"):
    # This contraction is one of the few matrix products left outside the
    # fused diffusion kernels. Route it explicitly so the hot loop can suspend
    # the model-wide Python TorchDispatchMode.
    augmented_positions = (
        _routed_einsum(
          # "...i,...ij->...j",
          "b...i,b...ij->b...j",  # batched version
          positions - center,
          rot,
        )
        + translation
    )
  return augmented_positions * mask[..., None]


def noise_schedule(
    t: torch.Tensor, smin: Number = 0.0004, smax: Number = 160.0, p: Number = 7
) -> torch.Tensor:
  return SIGMA_DATA * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p


class ConditioningConfig(base_config.BaseConfig):
  pair_channel: int
  seq_channel: int
  prob: float


class SampleConfig(base_config.BaseConfig):
  steps: int
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  num_samples: int = 1


class DiffusionHead(fnn.Module):
  """Denoising Diffusion Head."""

  __version__ = "1.0"
  FOURIER_EMBEDDING_DIM: Final[int] = 256

  class Config(
    atom_cross_attention.AtomCrossAttEncoder.Config,
    atom_cross_attention.AtomCrossAttDecoder.Config,
  ):
    """Configuration for DiffusionHead."""

    eval_batch_size: int = 5
    eval_batch_dim_shard_size: int = 5
    conditioning: ConditioningConfig = base_config.autocreate(
        prob=0.8, pair_channel=128, seq_channel=384
    )
    eval: SampleConfig = base_config.autocreate(
        num_samples=5,
        steps=200,
    )
    transformer: diffusion_transformer.Transformer.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      *,
      pair_channels: int,
      seq_channels: int,
      target_feat_size: int,
      dtype: torch.dtype | None = None,
      name="diffusion_head",
  ):
    super().__init__()
    self.name = name
    self.per_token_channels = config.per_token_channels
    self.config = config
    self.dtype = dtype
    self.num_samples = config.eval.num_samples

    # ///// MODULE STRUCTURE ///// #
    # 4 * max_relative_idx + 2 * max_relative_chain + 7
    #   where max_relative_idx=32, max_relative_chain=2
    features_2d_size = 139 + pair_channels
    self.pair_cond_initial_norm = fnn.LayerNorm(
        features_2d_size, bias=False, name="pair_cond_initial_norm"
    )
    self.pair_cond_initial_projection = fnn.Linear(
        features_2d_size,
        config.conditioning.pair_channel,
        bias=False,
        precision="highest",
        name="pair_cond_initial_projection",
    )

    self.pair_transitions = fnn.ModuleList([
        diffusion_transformer.TransitionBlock(
            global_config,
            num_channels=config.conditioning.pair_channel,
            num_intermediate_factor=2,
            name=f"pair_transition_{i}",
        ) for i in range(2)
    ])

    features_1d_size = target_feat_size + seq_channels
    self.single_cond_initial_norm = fnn.LayerNorm(
        features_1d_size, bias=False, name="single_cond_initial_norm"
    )
    self.single_cond_initial_projection = fnn.Linear(
        features_1d_size,
        config.conditioning.seq_channel,
        bias=False,
        precision="highest",
        name="single_cond_initial_projection",
    )

    self.noise_embedding_initial_norm = fnn.LayerNorm(
        256, bias=False, name="noise_embedding_initial_norm"
    )
    self.noise_embedding_initial_projection = fnn.Linear(
        256,
        config.conditioning.seq_channel,
        bias=False,
        precision="highest",
        name="noise_embedding_initial_projection",
    )

    self.single_transitions = fnn.ModuleList([
        diffusion_transformer.TransitionBlock(
            global_config,
            num_channels=config.conditioning.seq_channel,
            num_intermediate_factor=2,
            name=f"single_transition_{i}",
        ) for i in range(2)
    ])

    self.diffusion_encoder = atom_cross_attention.AtomCrossAttEncoder(
        config,
        global_config,
        has_token_atoms_act=True,
        seq_channels=seq_channels,
        pair_channels=pair_channels,
        dtype=dtype,
        name="diffusion",
    )

    self.single_cond_embedding_norm = fnn.LayerNorm(
        seq_channels, bias=False, name="single_cond_embedding_norm"
    )
    # ASC: initializer=self.global_config.final_init
    self.single_cond_embedding_projection = fnn.Linear(
        seq_channels,
        config.per_token_channels,
        bias=False,
        precision="highest",
        name="single_cond_embedding_projection",
    )

    self.transformer = diffusion_transformer.Transformer(
        config.transformer,
        global_config,
        num_channels=config.per_token_channels,
        single_cond_size=seq_channels,
        pair_cond_size=config.conditioning.pair_channel,
        dtype=dtype,
        name="transformer",
    )

    self.output_norm = fnn.LayerNorm(
        config.per_token_channels, bias=False, name="output_norm"
    )

    self.diffusion_decoder = atom_cross_attention.AtomCrossAttDecoder(
        config,
        global_config,
        num_channels=config.per_token_channels,
        pair_cond_size=config.per_atom_pair_channels,
        dtype=dtype,
        name="diffusion",
    )

  def _conditioning(
      self,
      base_single_cond: torch.Tensor,
      noise_level: torch.Tensor,
  ) -> torch.Tensor:
    # (N, S=384)
    single_cond = base_single_cond
    # (num_steps, 1)
    noise_embedding = noise_level_embeddings.noise_embeddings(
      sigma_scaled_noise_level=noise_level / SIGMA_DATA,
    )

    noise_embedding_normed = self.noise_embedding_initial_norm(noise_embedding)
    # (num_steps, 1)
    single_cond = self.noise_embedding_initial_projection(
      noise_embedding_normed, residual=single_cond
    )

    for single_transition in self.single_transitions:
      single_cond = single_transition(single_cond)

    return single_cond

  # ASC: original 'forward' of diffusion head, the vmapped version
  def _denoising(
      self,
      # positions_noisy.shape: ([batch], num_token, max_atoms_per_token, 3)
      positions_noisy: torch.Tensor,
      noise_level: torch.Tensor,
      batch: feat_batch.Batch,  # Constant across denoising loop
      base_single_cond: torch.Tensor,  # No batching dim, constant inside one iteration
      base_pair_cond: torch.Tensor,  # No batching dim, constant across denoising loop
      embeddings: dict[str, torch.Tensor],  # Constant across denoising loop
  ) -> torch.Tensor:

    # BFloat16 context
    # Get conditioning
    trunk_single_cond = self._conditioning(base_single_cond, noise_level)
    trunk_pair_cond = base_pair_cond

    # Extract features
    sequence_mask = batch.token_features.mask
    atom_mask = batch.predicted_structure_info.atom_mask

    # Position features
    act = positions_noisy * atom_mask[..., None]
    act /= torch.sqrt(noise_level ** 2 + SIGMA_DATA ** 2)

    with trace.into("encoder"):
      enc = self.diffusion_encoder(
          token_atoms_act=act,
          trunk_single_cond=embeddings["single"],
          trunk_pair_cond=trunk_pair_cond,
          batch=batch,
      )
      act = enc.token_act
    
    act = torch.clone(act).to(dtype=self.dtype)

    # Token-token attention
    # if has_batch:
    #   assert act.shape[2] == self.per_token_channels
    # else:
    #   assert act.shape[1] == self.per_token_channels
    # act = act.to(torch.float32)

    act = self.single_cond_embedding_projection(
        self.single_cond_embedding_norm(trunk_single_cond), residual=act
    )
    
    act = torch.asarray(act, dtype=self.dtype)
    trunk_single_cond = torch.asarray(trunk_single_cond, dtype=self.dtype)
    trunk_pair_cond = torch.asarray(trunk_pair_cond, dtype=self.dtype)

    with trace.into("transformer"):
      act = self.transformer(
          act=act,
          mask=sequence_mask,
          single_cond=trunk_single_cond,
          pair_cond=trunk_pair_cond,
      )
      act = self.output_norm(act)

    # (Possibly) atom-granularity decoder
    with trace.into("decoder"):
      position_update = self.diffusion_decoder(
          token_act=act,
          enc=enc,
          batch=batch,
      )

    skip_scaling = SIGMA_DATA ** 2 / (noise_level ** 2 + SIGMA_DATA ** 2)
    out_scaling = (
        noise_level * SIGMA_DATA / torch.sqrt(noise_level ** 2 + SIGMA_DATA ** 2)
    )
    # End `BFloat16 context`.

    return (
        skip_scaling * positions_noisy + out_scaling * position_update
    ) * atom_mask[..., None]

  def _get_base_cond(
      self,
      rel_features: torch.Tensor,  # (..., 139)
      embeddings: dict[str, torch.Tensor],
      use_conditioning: bool,
  ):
    single_embedding = use_conditioning * embeddings["single"]
    pair_embedding = use_conditioning * embeddings["pair"]

    # (..., 139)
    # rel_features = rel_features.to(pair_embedding.dtype)
    # (..., rel_features.shape[-1] + pair_channels)
    # =(..., 139+128=267)
    features_2d = torch.cat((pair_embedding, rel_features), dim=-1)
    pair_cond = self.pair_cond_initial_norm(features_2d)
    pair_cond = self.pair_cond_initial_projection(pair_cond)

    for pair_transition in self.pair_transitions:
      pair_cond = pair_transition(pair_cond)

    target_feat = embeddings["target_feat"]
    features_1d = torch.cat((single_embedding, target_feat), dim=-1)
    single_cond = self.single_cond_initial_norm(features_1d)
    single_cond = self.single_cond_initial_projection(single_cond)
    return single_cond, pair_cond

  def forward(
      self,
      batch: feat_batch.Batch,
      rel_features: torch.Tensor,
      embeddings: dict[str, torch.Tensor],
      use_conditioning: bool,
      *,
      num_samples: int | None = None,
  ) -> dict[str, torch.Tensor]:
    """Sample using denoiser on batch.

    Args:
      denoising_step: the denoising function.
      batch: the batch
      config: config for the sampling process (e.g. number of denoising steps,
        etc.)

    Returns:
      a dict
        {
           'atom_positions': torch.Tensor(...)       # shape (<common_axes>, 3)
           'mask': torch.Tensor(...)                 # shape (<common_axes>,)
        }
      where the <common_axes> are
      (num_samples, num_tokens, max_atoms_per_token)
    """

    mask = batch.predicted_structure_info.atom_mask

    config = self.config.eval
    if num_samples is None:
      num_samples = self.num_samples

    base_single_cond, base_pair_cond = self._get_base_cond(
        rel_features, embeddings, use_conditioning
    )

    def _apply_denoising_step(
        carry: tuple[torch.Tensor, torch.Tensor], noise_level: torch.Tensor
    ):
      """
      Note: to support for manual vmap action, the first dimension of `positions`
        might be the batch dim.
      """
      positions, noise_level_prev = carry

      positions = random_augmentation(
          positions=positions, mask=mask
      )

      gamma = config.gamma_0 * (noise_level > config.gamma_min)
      t_hat = noise_level_prev * (1 + gamma)

      noise_scale = config.noise_scale * torch.sqrt(
          t_hat ** 2 - noise_level_prev ** 2
      )
      noise = noise_scale * torch.randn(
          positions.shape, device=noise_level.device
      )
      positions_noisy = positions + noise

      positions_denoised = self._denoising(
          positions_noisy,
          t_hat,
          batch=batch,
          base_single_cond=base_single_cond,
          base_pair_cond=base_pair_cond,
          embeddings=embeddings,
      )

      grad = (positions_noisy - positions_denoised) / t_hat

      d_t = noise_level - t_hat
      positions_out = positions_noisy + config.step_scale * d_t * grad

      return positions_out, noise_level

    def apply_denoising_step(
        carry: tuple[torch.Tensor, torch.Tensor], noise_level: torch.Tensor
    ):
      # Most matrix products in this loop are explicit native calls. Removing
      # the global router avoids a Python callback for every other ATen op
      # (more than 1,200 callbacks per step on 107aa). A Torch transition
      # fallback reinstalls the router around its own matrix products.
      if not torch_config.native_matmul_dispatch:
        return _apply_denoising_step(carry, noise_level)
      from sparsefold.backends.vecops.matmul_dispatch import suspend_native_matmul_dispatch

      with suspend_native_matmul_dispatch():
        return _apply_denoising_step(carry, noise_level)

    noise_levels = noise_schedule(
        torch.linspace(0, 1, config.steps + 1, device=util.DEFAULT_DEVICE)
    )

    positions = torch.randn(
        (num_samples,) + mask.shape + (3,), device=noise_levels.device
    )
    positions *= noise_levels[0]
    
    positions_out, _ = ops.stack_scan(
        apply_denoising_step, 
        (positions, noise_levels[0]), 
        noise_levels[1:],
    )

    final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))

    return {"atom_positions": positions_out, "mask": final_dense_atom_mask}
