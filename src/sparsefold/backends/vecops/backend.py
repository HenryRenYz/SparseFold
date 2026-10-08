# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Lifecycle hooks for the SparseFold vecops acceleration backend."""

import platform
import contextlib

import torch

import sparsefold.model.diffusion.transformer as _diffusion_transformer
import sparsefold.model.trunk.blocks as _modules

from sparsefold.runtime import config
from sparsefold.runtime import dist
from sparsefold.runtime import log
from sparsefold.model.layers import primitives as fnn
from sparsefold.backends.vecops.matmul_dispatch import execution_context as _matmul_execution_context
from sparsefold.backends.vecops.modules import CrossAttTransformerVecops
from sparsefold.backends.vecops.modules import DiffusionCrossAttentionVecops
from sparsefold.backends.vecops.modules import DiffusionSelfAttentionVecops
from sparsefold.backends.vecops.modules import DiffusionTransitionBlockVecops
from sparsefold.backends.vecops.modules import GridSelfAttentionVecops
from sparsefold.backends.vecops.modules import LinearVecops
from sparsefold.backends.vecops.modules import MSAAttentionVecops
from sparsefold.backends.vecops.modules import OuterProductMeanVecops
from sparsefold.backends.vecops.modules import SingleActUpdateVecops
from sparsefold.backends.vecops.modules import TransformerVecops
from sparsefold.backends.vecops.modules import TransitionBlockVecops
from sparsefold.backends.vecops.modules import TriangleMultiplicationVecops


@contextlib.contextmanager
def execution_context():
  """Use optional matmul routing and metadata-only eager native-call observations."""
  from sparsefold.backends.vecops.audit import observe_native_calls
  with _matmul_execution_context(), observe_native_calls():
    yield


def prepare(module: torch.nn.Module) -> dict[str, int]:
  """Prepare native buffers after strict checkpoint loading; never create Parameters.

  Simple fusions own child projections, avoiding duplicate generic packing.
  Transformer superblocks require child attention/transition buffers first.
  """
  fused_types = (
      TransitionBlockVecops, MSAAttentionVecops, OuterProductMeanVecops,
      GridSelfAttentionVecops, TriangleMultiplicationVecops, SingleActUpdateVecops,
      DiffusionTransitionBlockVecops, DiffusionSelfAttentionVecops,
      DiffusionCrossAttentionVecops, TransformerVecops, CrossAttTransformerVecops,
  )
  composite_types = (TransformerVecops, CrossAttTransformerVecops)
  counts = {"packed": 0, "fallback": 0, "opaque": 0}
  visited: set[int] = set()

  def visit(current: torch.nn.Module, covered: bool = False) -> None:
    if id(current) in visited:
      return
    visited.add(id(current))
    if isinstance(current, fused_types):
      if isinstance(current, composite_types):
        for child in current.children():
          visit(child, covered=True)
      current.prepare_native()
      counts["opaque"] += 1
      return
    if isinstance(current, LinearVecops):
      if not covered:
        counts["packed" if current.prepare_native() else "fallback"] += 1
      return
    for child in current.children():
      visit(child, covered=covered)

  visit(module)
  log.info(f"Prepared native buffers: {counts['packed']} generic Linear, "
           f"{counts['fallback']} fallback, {counts['opaque']} fused modules")
  return counts


def is_precompiling() -> bool:
  """Return whether vecops is collecting native compilation requests."""
  import vecops

  return vecops.is_precompiling()


def materialize(*, parallelism: int | None = None) -> int:
  """Build host-global bridges and parameter-pack kernels before model load."""
  import vecops

  owner = not dist.is_enabled() or dist.is_primary()
  if dist.is_enabled() and not owner:
    dist.barrier()
  count = vecops.ops.torch.compile_pending(parallelism=parallelism)
  if owner:
    from sparsefold.backends.vecops.kernels import pack_linear_b
    from sparsefold.backends.vecops.kernels import pack_matmul_b

    # Parameter loading invokes these four dynamic pack specializations before
    # model-level precompilation starts. Collecting them here lets one CMake
    # project compile them once on rank 0; all other ranks subsequently load
    # the published artifacts instead of racing on identical cache misses.
    with vecops.collect_compile_requests() as collector:
      pack_matmul_b(torch.empty((1, 1), dtype=torch.bfloat16))
      pack_linear_b(torch.empty((1, 1), dtype=torch.bfloat16))
      pack_linear_b(
          torch.empty((1, 1), dtype=torch.float32),
          fp32_matmul_precision="medium",
      )
      if platform.machine().lower() in {"aarch64", "arm64"}:
        pack_linear_b(
            torch.empty((1, 1), dtype=torch.float32),
            fp32_matmul_precision="highest",
        )
    prepared = vecops.compile_batch(
        collector.requests, parallelism=parallelism
    ).prepared
    log.info(
        f"Prepared {prepared} parameter-pack kernel specialization(s) "
        "before model construction"
    )
    count += prepared
  if dist.is_enabled() and owner:
    dist.barrier()
  return count


def precompile(
    module: torch.nn.Module,
    *args,
    use_real_tensors: bool = False,
    parallelism: int | None = None,
    capture_graph: bool = False,
    graph_name: str | None = None,
    **kwargs,
):
  """Trace native calls through vecops and compile the unique requests."""
  import vecops

  def routed_model(*model_args, **model_kwargs):
    # Enter after vecops has installed its collector and force-eager scopes.
    # PyTorch 2.7 otherwise executes CompositeImplicitAutograd einsum with the
    # outer dispatch mode temporarily removed, bypassing the matmul router.
    with execution_context():
      return module(*model_args, **model_kwargs)

  try:
    return vecops.precompile(
        routed_model,
        *args,
        use_real_tensors=use_real_tensors,
        parallelism=parallelism,
        capture_graph=capture_graph,
        graph_name="sparsefold-v1:" + (graph_name or module.__class__.__qualname__),
        _graph_state_owner=module,
        prepare_owner=not dist.is_enabled() or dist.is_primary(),
        prepare_barrier=dist.barrier if dist.is_enabled() else None,
        prepare_rank=dist.get_rank() if dist.is_enabled() else None,
        prepare_world_size=dist.get_world_size() if dist.is_enabled() else None,
        **kwargs,
    )
  finally:
    # The tracing pass deliberately returns uninitialized native outputs.
    # Never retain ValueCache entries derived from fake or incomplete values.
    reset = getattr(module, "reset", None)
    if callable(reset):
      reset()


_PATCHED_FACTORIES: list[tuple[object, str, object]] = []


def _replace_factory(module, name: str, implementation) -> None:
  if not any(owner is module and attr == name for owner, attr, _ in _PATCHED_FACTORIES):
    _PATCHED_FACTORIES.append((module, name, getattr(module, name)))
  setattr(module, name, implementation)


def unapply() -> None:
  """Restore model factories before constructing reference models."""
  for module, name, original in reversed(_PATCHED_FACTORIES):
    setattr(module, name, original)
  _PATCHED_FACTORIES.clear()


def apply() -> None:
  from sparsefold.backends.vecops.policy import validate_native_ops
  validate_native_ops(config.native_ops)
  replace_map = {
      "linear": (fnn, "Linear", LinearVecops),
      "evoformer_transition_block": (
          _modules, "TransitionBlock", TransitionBlockVecops
      ),
      "msa_attention": (_modules, "MSAAttention", MSAAttentionVecops),
      "outer_product_mean": (
          _modules, "OuterProductMean", OuterProductMeanVecops
      ),
      "grid_self_attn": (
          _modules, "GridSelfAttention", GridSelfAttentionVecops
      ),
      "triangle_multiplication": (
          _modules, "TriangleMultiplication", TriangleMultiplicationVecops
      ),
      "evoformer_single_act_update": (
          _modules, "SingleActUpdate", SingleActUpdateVecops
      ),
      "diffusion_transition_block": (
          _diffusion_transformer,
          "TransitionBlock",
          DiffusionTransitionBlockVecops,
      ),
      "diffusion_self_attention": (
          _diffusion_transformer,
          "SelfAttention",
          DiffusionSelfAttentionVecops,
      ),
      "diffusion_cross_attention": (
          _diffusion_transformer,
          "CrossAttention",
          DiffusionCrossAttentionVecops,
      ),
  }

  selected_native_ops = list(config.native_ops)
  if "*" in selected_native_ops:
    selected_native_ops = list(replace_map)

  if not any(name in replace_map for name in selected_native_ops):
    log.info("No valid native op match in `native_ops`, skipped native kernels")
    return

  # Importing the facade declares every Torch bridge. Materialization happens
  # after process-group initialization to avoid duplicate compiler batches.
  from sparsefold.backends.vecops import kernels

  for key, (module, name, implementation) in replace_map.items():
    if key in selected_native_ops:
      _replace_factory(module, name, implementation)

  if {
      "diffusion_transition_block", "diffusion_self_attention"
  }.issubset(selected_native_ops):
    _replace_factory(_diffusion_transformer, "Transformer", TransformerVecops)
  if {
      "diffusion_transition_block", "diffusion_cross_attention"
  }.issubset(selected_native_ops):
    _replace_factory(_diffusion_transformer, "CrossAttTransformer", CrossAttTransformerVecops)


def manual_seed(seed: int) -> None:
  torch.manual_seed(seed)
  from sparsefold.backends.vecops.kernels import set_grid_attention_seed

  set_grid_attention_seed(seed & 0x7fffffff)
