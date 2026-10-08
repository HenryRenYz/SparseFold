# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""SparseFold prediction execution from fixed-main, with explicit external AF3 boundaries."""
from __future__ import annotations
from collections.abc import Sequence
import contextlib
import datetime
import functools
import dataclasses
import json
import os
import pathlib
import string
import textwrap
import time
import numpy as np
import torch
from absl import app, flags
from sparsefold.cli.flags import _AF3_REVISION, _AF3_SOURCE_ROOT, _BUCKETS, _CONFORMER_MAX_ITERATIONS, _DIST, _DIST_BACKEND, _FORCE_OUTPUT_DIR, _GPU_DEVICE, _HMMALIGN_BINARY_PATH, _HMMBUILD_BINARY_PATH, _HMMSEARCH_BINARY_PATH, _INPUT_DIR, _JACKHMMER_BINARY_PATH, _JACKHMMER_MAX_PARALLEL_SHARDS, _JACKHMMER_N_CPU, _JSON_PATH, _MAX_TEMPLATE_DATE, _MEMORY_PLAN_ALLOW_FAST_FALLBACK, _MEMORY_PLAN_FACTORY_REPLAY, _MEMORY_PLAN_FAST_BANDWIDTH_MIB_S, _MEMORY_PLAN_FAST_MIB, _MEMORY_PLAN_LARGE_PAGES, _MEMORY_PLAN_MIN_FAST_MIB, _MEMORY_PLAN_MIN_REUSE, _MEMORY_PLAN_PATH, _MEMORY_PLAN_SLOW_BANDWIDTH_MIB_S, _MEMORY_PLAN_SLOW_MIB, _MEMORY_PLAN_STATE_REPLAY, _MEMORY_PLAN_TRANSFER_BANDWIDTH_MIB_S, _MGNIFY_DATABASE_PATH, _MGNIFY_Z_VALUE, _NHMMER_BINARY_PATH, _NHMMER_MAX_PARALLEL_SHARDS, _NHMMER_N_CPU, _NTRNA_DATABASE_PATH, _NTRNA_Z_VALUE, _NUM_DIFFUSION_SAMPLES, _NUM_RECYCLES, _NUM_SEEDS, _OUTPUT_DIR, _PDB_DATABASE_PATH, _PRECOMPILE, _PRECOMPILE_GRAPH_PATH, _PRECOMPILE_ONLY, _PRECOMPILE_PARALLELISM, _PRECOMPILE_USE_REAL_TENSORS, _RESOLVE_MSA_OVERLAPS, _RFAM_DATABASE_PATH, _RFAM_Z_VALUE, _RNA_CENTRAL_DATABASE_PATH, _RNA_CENTRAL_Z_VALUE, _RUN_DATA_PIPELINE, _RUN_INFERENCE, _SAVE_DISTOGRAM, _SAVE_EMBEDDINGS, _SEQRES_DATABASE_PATH, _SMALL_BFD_DATABASE_PATH, _SMALL_BFD_Z_VALUE, _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH, _UNIPROT_CLUSTER_ANNOT_Z_VALUE, _USE_CPU, _VECOPS_LARGE_PAGES, _VECOPS_WORKSPACE_DOMAIN, _VECOPS_WORKSPACE_FAST_FALLBACK, _VECOPS_WORKSPACE_FAST_MIB, _VECOPS_WORKSPACE_SLOW_MIB
from sparsefold.cli.flags import MODEL_DIR, DB_DIR, set_output_dir_overrides
from sparsefold.cli import flags as prediction_flags
from sparsefold.schema import features
from sparsefold.model import network as model
from sparsefold.runtime import util, tree, compiler, dist, log
from sparsefold import backends as acceleration
from sparsefold.output import decode_predictions, ResultsForSeed, write_outputs
from sparsefold.output.decode import to_numpy
from sparsefold.schema.batch import ModelBatch

@functools.cache
def _provider():
    from sparsefold.integrations.af3.provider import AF3Provider
    return AF3Provider(revision=_AF3_REVISION.value, source_root=_AF3_SOURCE_ROOT.value)

def _output_terms():
    fingerprint = _provider().verify()
    root = pathlib.Path(fingerprint["source_root"])
    path = root / "OUTPUT_TERMS_OF_USE.md"
    if not path.is_file():
        raise FileNotFoundError(f"Verified AF3 checkout lacks output terms: {path}")
    return path.read_text()


def make_model_config(
    *,
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
) -> model.SparseFoldModel.Config:
    """Returns a model config with some defaults overridden."""
    config = model.SparseFoldModel.Config()
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    diffusion_steps = os.environ.get("SPARSEFOLD_DIFFUSION_STEPS")
    if diffusion_steps is None and os.environ.get("AF3_DIFFUSION_STEPS"):
        import warnings
        warnings.warn("AF3_DIFFUSION_STEPS is deprecated; use SPARSEFOLD_DIFFUSION_STEPS", DeprecationWarning)
        diffusion_steps = os.environ["AF3_DIFFUSION_STEPS"]
    if diffusion_steps:
        steps = int(diffusion_steps)
        if steps < 1:
            raise ValueError("SPARSEFOLD_DIFFUSION_STEPS must be >= 1")
        config.heads.diffusion.eval.steps = steps
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    return config


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
            self,
            config: model.SparseFoldModel.Config,
            device: str,
            model_dir: pathlib.Path,
    ):
        self._model_config = config
        self._device = device
        # Runtime tensors and local checkpoint weights share the selected device.
        util.DEFAULT_DEVICE = device
        self._model_dir = model_dir
        self._memory_plan_binding = None

    def configure_memory_plan(self, graph, plan, system) -> None:
        from vecops.planner import validate_plan

        validate_plan(graph, plan)
        self._memory_plan_binding = (graph, plan, system)

    @functools.cached_property
    def checkpoint(self):
        from sparsefold.weights import read_checkpoint
        if dist.is_primary():
            checkpoint = read_checkpoint(pathlib.Path(self._model_dir))
            checkpoint = dataclasses.replace(
                checkpoint, params=tree.map(lambda tensor: tensor.to(device=self._device), checkpoint.params))
        else:
            checkpoint = None
        if dist.is_enabled():
            checkpoint, = dist.broadcast_object_list([checkpoint], src=0)
        return checkpoint

    @property
    def model_params(self):
        return self.checkpoint.params

    @functools.cached_property
    def _model(self):
        from sparsefold.weights import load_checkpoint
        # Checkpoint transforms must remain visible to the strict dispatch audit.
        with torch.inference_mode(False), torch.no_grad():
            model_ = model.SparseFoldModel(self._model_config, distributed=dist.is_enabled()).eval()
            self.weight_mapping_report = load_checkpoint(model_, self.checkpoint, strict=True)
            acceleration.prepare(model_)
        return model_

    def prepare_model(self) -> None:
        """Construct and prepare the model before any rank enters inference.

        Model construction is intentionally cached, but leaving the first
        construction inside ``run_inference`` lets the fastest rank reach the
        first tensor collective while other ranks are still packing weights.
        That turns rank-local initialization skew into an apparent operator
        tail.  Eagerly touching the cached property gives every rank the same
        initialization boundary; the caller performs the distributed barrier.
        """
        _ = self._model

    def run_inference(
            self, featurised_example: ModelBatch
    ) -> dict:
        """Run only typed numerical tensors, never official host metadata."""
        featurised_example = ModelBatch.from_data_dict(featurised_example).to_torch(device=self._device)
        # Lazy initialization happens before the forward-only inference context.
        _ = self._model
        if self._memory_plan_binding is None:
            with torch.inference_mode():
                result = self._model(featurised_example)
        else:
            from vecops.planner import PlanSession

            graph, plan, system = self._memory_plan_binding
            replay_start = time.perf_counter()
            with torch.inference_mode():
                with PlanSession(
                    graph,
                    plan,
                    system=system,
                    allow_fast_fallback=_MEMORY_PLAN_ALLOW_FAST_FALLBACK.value,
                    large_pages=_MEMORY_PLAN_LARGE_PAGES.value,
                    strict_factory_replay=_MEMORY_PLAN_FACTORY_REPLAY.value,
                    enable_factory_replay=_MEMORY_PLAN_FACTORY_REPLAY.value,
                    validate=False,
                ) as session:
                    arena_seconds = time.perf_counter() - replay_start
                    binding_start = time.perf_counter()
                    binding = session.bind_module_state(self._model)
                    binding_seconds = time.perf_counter() - binding_start
                    log.info(
                        'Bound planned model state: '
                        f'{binding["copied_storages"]} storage(s), '
                        f'{binding["copied_bytes"] >> 20} MiB copied.'
                    )
                    model_start = time.perf_counter()
                    result = self._model(featurised_example)
                    model_seconds = time.perf_counter() - model_start
                    runtime_stats = session.stats()
                    log.info(
                        'Memory plan replay: '
                        f'factory={runtime_stats["planned_factory_allocations"]}, '
                        f'fallback={runtime_stats["factory_fallbacks"]}, '
                        f'factory_bytes={runtime_stats["factory_allocated_bytes"] >> 20} MiB, '
                        f'factory_hugetlb={runtime_stats["factory_large_page_bytes"] >> 20} MiB, '
                        f'factory_reserved={runtime_stats["factory_reserved_bytes"] >> 20} MiB, '
                        f'factory_owners={runtime_stats["factory_owner_allocations"]}, '
                        f'fast_target={runtime_stats["fast_target_id"]}, '
                        f'fast_target_verified={runtime_stats["fast_target_verified"]}, '
                        f'fast_hugetlb='
                        f'{runtime_stats["fast_large_page_bytes"] >> 20} MiB.'
                    )
                    if runtime_stats['factory_fallbacks']:
                        log.info(
                            'Memory plan replay diagnostics: '
                            f'{session.diagnostics()}'
                        )
            replay_seconds = time.perf_counter() - replay_start
            log.info(
                'Memory plan timing: '
                f'arena={arena_seconds:.6f}s, '
                f'state_copy={binding_seconds:.6f}s, '
                f'model={model_seconds:.6f}s, '
                f'close={replay_seconds - arena_seconds - binding_seconds - model_seconds:.6f}s.'
            )

        result = to_numpy(result)
        result = dict(result)

        identifier = self.checkpoint.identifier
        result['__identifier__'] = identifier
        return result

    def extract_inference_results(self, batch, result, target_name, context):
        return decode_predictions(batch=batch, result=result, context=context, target_name=target_name, provider=_provider())

    def extract_embeddings(
            self, result: dict, num_tokens: int
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if 'single_embeddings' in result:
            embeddings['single_embeddings'] = result['single_embeddings'][
                :num_tokens
            ].astype(np.float16)
        if 'pair_embeddings' in result:
            embeddings['pair_embeddings'] = result['pair_embeddings'][
                :num_tokens, :num_tokens
            ].astype(np.float16)
        return embeddings or None

    def extract_distogram(
            self, result: dict, num_tokens: int
    ) -> np.ndarray | None:
        """Extracts distogram from model outputs."""
        if 'distogram' not in result['distogram']:
            return None
        distogram = result['distogram']['distogram'][:num_tokens, :num_tokens, :]
        return distogram


@contextlib.contextmanager
def _vecops_workspace_session():
    """Own one bounded vecops workspace lifetime around synchronous inference."""
    if _VECOPS_WORKSPACE_FAST_MIB.value == 0:
        yield
        return

    import vecops

    if not hasattr(vecops, 'memory'):
        raise RuntimeError(
            'vecops workspace memory was requested, but this editable vecops '
            'build has no memory component'
        )
    memory_system = vecops.memory.System()
    mib = 1 << 20
    session = memory_system.workspace_session(
        fast_capacity=_VECOPS_WORKSPACE_FAST_MIB.value * mib,
        slow_capacity=(
            _VECOPS_WORKSPACE_SLOW_MIB.value
            or _VECOPS_WORKSPACE_FAST_MIB.value
        )
        * mib,
        domain=(
            None
            if _VECOPS_WORKSPACE_DOMAIN.value < 0
            else _VECOPS_WORKSPACE_DOMAIN.value
        ),
        allow_fast_fallback=_VECOPS_WORKSPACE_FAST_FALLBACK.value,
        large_pages=_VECOPS_LARGE_PAGES.value,
    )
    log.info(
        'Using vecops workspace session: '
        f'domain={session.domain}, fast={session.fast_capacity >> 20} MiB, '
        f'slow={session.slow_capacity >> 20} MiB'
    )
    with session:
        try:
            yield
        finally:
            live = [
                f"node {item['os_numa_id']}="
                f"{item['managed_bytes'] >> 20} MiB "
                f"(large={item['managed_large_page_bytes'] >> 20} MiB)"
                for item in memory_system.stats()
                if item['managed_bytes']
            ]
            log.info('Vecops workspace arenas before release: ' + ', '.join(live))
    if any(item['managed_bytes'] for item in memory_system.stats()):
        raise RuntimeError('vecops workspace session did not release all arenas')



def predict_structure(
        fold_input: object,
        model_runner: ModelRunner,
        buckets: Sequence[int] | None = None,
        ref_max_modified_date: datetime.date | None = None,
        conformer_max_iterations: int | None = None,
        resolve_msa_overlaps: bool = True,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    if dist.is_primary():
        log.info(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
        featurisation_start_time = time.time()
        prepared_examples = _provider().featurise(
            fold_input, buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            resolve_msa_overlaps=resolve_msa_overlaps,
        )
        featurised_examples = [example.model_batch for example in prepared_examples]
        log.info(
            f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
            f' {time.time() - featurisation_start_time:.2f} seconds.'
        )

    if _PRECOMPILE.value:
        if dist.is_primary():
            seed, example = fold_input.rng_seeds[0], featurised_examples[0]
            example_torch = example.to_torch(device=model_runner._device)
        else:
            example_torch, seed = None, None
        if dist.is_enabled():
            example_torch, seed = dist.broadcast_object_list([example_torch, seed], src=0)

        util.manual_seed(seed)
        log.info(f'Precompiling model using seed {seed}...')
        precompile_start_time = time.time()
        model_to_precompile = model_runner._model
        # Recycling reuses the same modules with the same tensor shapes. One
        # trunk pass therefore discovers the complete native request set; a
        # real-tensor precompile over every inference recycle only repeats the
        # expensive backend-independent fallback work. Restore the configured
        # value before the measured inference so prediction semantics do not
        # change.
        configured_num_recycles = model_to_precompile.num_recycles
        # Request-only precompilation needs one trunk pass. Graph capture keeps
        # the configured count so indexed_stack_scan can record the complete
        # loop trip count while still sampling only the first two iterations.
        capture_memory_graph = (
            _PRECOMPILE_GRAPH_PATH.value is not None
            or _MEMORY_PLAN_FAST_MIB.value > 0
        )
        memory_graph_name = None
        if capture_memory_graph:
            memory_graph_name = (
                f'sparsefold-v1:{fold_input.name}'
                if fold_input is not None
                else f'sparsefold-v1:distributed-rank-{dist.get_rank()}'
            )
        if not capture_memory_graph:
            model_to_precompile.num_recycles = 0
        try:
            result = compiler.precompile(
                model_to_precompile,
                example_torch,
                use_real_tensors=_PRECOMPILE_USE_REAL_TENSORS.value,
                parallelism=_PRECOMPILE_PARALLELISM.value,
                capture_graph=capture_memory_graph,
                graph_name=memory_graph_name,
            )
        finally:
            model_to_precompile.num_recycles = configured_num_recycles
        if result is not None:
            log.info(
                f'Collected and prepared {result.prepared} unique native '
                f'operator request(s) with parallelism {result.parallelism}.'
            )
            if _PRECOMPILE_GRAPH_PATH.value is not None:
                if result.graph is None:
                    raise RuntimeError('Precompile graph capture returned no graph.')
                graph_path = pathlib.Path(_PRECOMPILE_GRAPH_PATH.value)
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                result.graph.validate()
                result.graph.write(graph_path)
                log.info(f'Wrote precompile execution graph to {graph_path}.')
            if _MEMORY_PLAN_FAST_MIB.value > 0:
                if result.graph is None:
                    raise RuntimeError('Memory planning requires a captured graph.')
                import vecops

                memory_system = vecops.memory.System()
                memory_profile = vecops.planner.profile_from_system(
                    memory_system,
                    fast_capacity=_MEMORY_PLAN_FAST_MIB.value << 20,
                    slow_capacity=_MEMORY_PLAN_SLOW_MIB.value << 20,
                    fast_bandwidth_mib_s=(
                        _MEMORY_PLAN_FAST_BANDWIDTH_MIB_S.value or None
                    ),
                    slow_bandwidth_mib_s=(
                        _MEMORY_PLAN_SLOW_BANDWIDTH_MIB_S.value or None
                    ),
                    transfer_bandwidth_mib_s=(
                        _MEMORY_PLAN_TRANSFER_BANDWIDTH_MIB_S.value or None
                    ),
                )
                planner_config = vecops.planner.PlannerConfig(
                    minimum_fast_bytes=max(
                        4096,
                        _MEMORY_PLAN_MIN_FAST_MIB.value << 20,
                    ),
                    allow_temporary_fast=_MEMORY_PLAN_FACTORY_REPLAY.value,
                    allow_input_fast=False,
                    allow_persistent_fast=_MEMORY_PLAN_STATE_REPLAY.value,
                    require_replayable_temporaries=(
                        _MEMORY_PLAN_FACTORY_REPLAY.value
                    ),
                    minimum_weighted_accesses=_MEMORY_PLAN_MIN_REUSE.value,
                )
                if _MEMORY_PLAN_PATH.value is not None:
                    plan_path = pathlib.Path(_MEMORY_PLAN_PATH.value)
                    memory_plan, cache_hit = vecops.planner.load_or_plan(
                        result.graph,
                        memory_profile,
                        plan_path,
                        config=planner_config,
                    )
                    log.info(
                        f'{"Loaded" if cache_hit else "Wrote"} tensor memory '
                        f'plan at {plan_path}.'
                    )
                else:
                    memory_plan = vecops.planner.plan_memory(
                        result.graph,
                        memory_profile,
                        config=planner_config,
                    )
                model_runner.configure_memory_plan(
                    result.graph,
                    memory_plan,
                    memory_system,
                )
                log.info(
                    'Planned tensor arenas: '
                    f'fast={memory_plan.fast_arena_bytes >> 20} MiB, '
                    f'slow={memory_plan.slow_arena_bytes >> 20} MiB.'
                )
        log.info(
            f'Precompiling model using seed {seed} took'
            f' {time.time() - precompile_start_time:.2f} seconds.'
        )
        if _PRECOMPILE_ONLY.value:
            log.info('Precompile-only mode requested; skipping model inference.')
            return [] if dist.is_primary() else None
    
    if dist.is_primary():
        num_seed = len(fold_input.rng_seeds)
        log.info(
            'Running model inference and extracting output structure samples with'
            f' {num_seed} seed(s)...'
        )
    else:
        num_seed = None
    if dist.is_enabled():
        num_seed, = dist.broadcast_object_list([num_seed], src=0)

    all_inference_start_time = time.time()
    all_inference_results = []
    for i in range(num_seed):
        if dist.is_primary():
            seed, example = fold_input.rng_seeds[i], featurised_examples[i]
            log.info(f'Running model inference for seed {seed}...')
            example_torch = example.to_torch(device=model_runner._device)
        else:
            example_torch, seed = None, None
        if dist.is_enabled():
            example_torch, seed = dist.broadcast_object_list([example_torch, seed], src=0)

        inference_start_time = time.time()
        util.manual_seed(seed)
        with _vecops_workspace_session():
            result = model_runner.run_inference(example_torch)
        if dist.is_primary():
            log.info(
                f'Running model inference for seed {seed} took'
                f' {time.time() - inference_start_time:.2f} seconds.'
            )
            log.info(f'Extracting inference results with seed {seed}...')
            extract_structures = time.time()
            inference_results = model_runner.extract_inference_results(
                batch=example, result=result, target_name=fold_input.name,
                context=prepared_examples[i].output_context
            )
            num_tokens = len(inference_results[0].metadata['token_chain_ids'])
            embeddings = model_runner.extract_embeddings(
                result=result, num_tokens=num_tokens
            )
            distogram = model_runner.extract_distogram(
                result=result, num_tokens=num_tokens
            )
            log.info(
                f'Extracting {len(inference_results)} inference samples with'
                f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
            )

            all_inference_results.append(
                ResultsForSeed(
                    seed=seed,
                    inference_results=inference_results,
                    embeddings=embeddings,
                    distogram=distogram,
                )
            )
    if dist.is_primary():
        log.info(
            'Running model inference and extracting output structures with'
            f' {len(fold_input.rng_seeds)} seed(s) took'
            f' {time.time() - all_inference_start_time:.2f} seconds.'
        )
        return all_inference_results
    else:
        return None


def write_fold_input_json(
      fold_input: object,
      output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
    log.info(f'Writing model input JSON to {path}')
    with open(path, 'wt') as f:
        f.write(fold_input.to_json())


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
    """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
    template = string.Template(path_with_db_dir)
    if 'DB_DIR' in template.get_identifiers():
        for db_dir in db_dirs:
            path = template.substitute(DB_DIR=db_dir)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
        )
    if not os.path.exists(path_with_db_dir):
        raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
    return path_with_db_dir


def process_fold_input(
    fold_input: object,
    data_pipeline_config: dict | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    force_output_dir: bool = False,
) -> object | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.
      ref_max_modified_date: Optional maximum date that controls whether to allow
        use of model coordinates for a chemical component from the CCD if RDKit
        conformer generation fails and the component does not have ideal
        coordinates set. Only for components that have been released before this
        date the model coordinates can be used as a fallback.
      conformer_max_iterations: Optional override for maximum number of iterations
        to run for RDKit conformer search.
      resolve_msa_overlaps: Whether to deduplicate unpaired MSA against paired
        MSA. The default behaviour matches the method described in the AlphaFold 3
        paper. Set this to false if providing custom paired MSA using the unpaired
        MSA field to keep it exactly as is as deduplication against the paired MSA
        could break the manually crafted pairing between MSA sequences.
      force_output_dir: If True, do not create a new output directory even if the
        existing one is non-empty. Instead use the existing output directory and
        potentially overwrite existing files. If False, create a new timestamped
        output directory instead if the existing one is non-empty.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """

    if dist.is_primary():
        log.info(f'\nRunning fold job {fold_input.name}...')

        if not fold_input.chains:
            raise ValueError('Fold input has no chains.')

        if (
            not force_output_dir
            and os.path.exists(output_dir)
            and os.listdir(output_dir)
        ):
            new_output_dir = (
                f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            log.info(
                f'Output will be written in {new_output_dir} since {output_dir} is'
                ' non-empty.'
            )
            output_dir = new_output_dir
        else:
            log.info(f'Output will be written in {output_dir}')
        
        if data_pipeline_config is None:
            log.info('Skipping data pipeline...')
        else:
            log.info('Running data pipeline...')
            fold_input = _provider().prepare(fold_input, data_pipeline_config)
            
        write_fold_input_json(fold_input, output_dir)
        fingerprint = _provider().verify()
        pathlib.Path(output_dir, 'af3_provider.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
        if model_runner is not None:
            pathlib.Path(output_dir, 'weight_mapping.json').write_text(
                json.dumps(model_runner.weight_mapping_report.to_dict(), indent=2) + '\n')

    if model_runner is None:
        if dist.is_primary():
            log.info('Skipping model inference...')
        output = fold_input
    else:
        if dist.is_primary():
            log.info(
                f'Predicting 3D structure for {fold_input.name} with'
                f' {len(fold_input.rng_seeds)} seed(s)...'
            )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            resolve_msa_overlaps=resolve_msa_overlaps,
        )
        if dist.is_primary():
            log.info(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
            write_outputs(
                all_results=all_inference_results,
                output_dir=output_dir,
                job_name=fold_input.sanitised_name(),
                terms_of_use=_output_terms(),
            )
            output = all_inference_results

    if dist.is_primary():
        log.info(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
        return output
    else:
        return None



def main(_):
    if _PRECOMPILE_ONLY.value and not _PRECOMPILE.value:
        raise ValueError('--precompile_only requires --precompile.')
    if _MEMORY_PLAN_FAST_MIB.value > 0 and not _PRECOMPILE.value:
        raise ValueError('--memory_plan_fast_mib requires --precompile.')
    if (
        _MEMORY_PLAN_FACTORY_REPLAY.value or _MEMORY_PLAN_STATE_REPLAY.value
    ) and _MEMORY_PLAN_FAST_MIB.value <= 0:
        raise ValueError(
            'memory-plan replay requires --memory_plan_fast_mib > 0.'
        )
    if _MEMORY_PLAN_FACTORY_REPLAY.value and compiler.config.enable_compile:
        raise ValueError(
            '--memory_plan_factory_replay currently requires '
            'runtime.compile=false.'
        )

    if _DIST.value == True:
        backend = _DIST_BACKEND.value
        if backend == 'default':
            if _USE_CPU.value == True:
                backend = 'gloo'
            else:
                backend = 'nccl'
        dist.init(backend=backend)
    
    if dist.is_primary() and _USE_CPU.value == True:
        log.info("Using CPU backend.")
    
    if not compiler.config.uses_vecops() and (
        _MEMORY_PLAN_FAST_MIB.value > 0 or _PRECOMPILE_GRAPH_PATH.value is not None
        or _VECOPS_WORKSPACE_FAST_MIB.value > 0
    ):
        raise ValueError('Native graph, memory planning and workspace options require the compatible vecops backend.')
    if _PRECOMPILE_ONLY.value and not compiler.config.uses_vecops():
        raise ValueError('--precompile_only requires a backend with native precompilation; Torch initializes compiled regions during inference.')

    # Init acceleration library
    acceleration.init()
    acceleration.materialize(parallelism=_PRECOMPILE_PARALLELISM.value)

    if (_JSON_PATH.value is None) == (_INPUT_DIR.value is None):
        raise ValueError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            'At least one of --run_inference or --run_data_pipeline must be'
            ' set to true.'
        )

    if dist.is_primary():
        fold_inputs = _provider().read_inputs(pathlib.Path(_INPUT_DIR.value or _JSON_PATH.value))
        fold_inputs = list(fold_inputs)
        
        # Make sure we can create the output directory before running anything.
        try:
            os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
        except OSError as e:
            log.error(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
            raise

        notice = textwrap.wrap(
            'Running SparseFold. Please note that standard AlphaFold 3 model'
            ' parameters are only available under terms of use provided at'
            ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
            ' If you do not agree to these terms and are using AlphaFold 3 derived'
            ' model parameters, cancel execution of AlphaFold 3 inference with'
            ' CTRL-C, and do not use the model parameters.',
            break_long_words=False,
            break_on_hyphens=False,
            width=80,
        )
        log.info('\n'.join(notice))

        max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
        if _RUN_DATA_PIPELINE.value:
            expand_path = lambda x: replace_db_dir(x, DB_DIR.value)
            data_pipeline_config = dict(
                jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
                nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
                hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
                hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
                hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
                small_bfd_database_path=expand_path(_SMALL_BFD_DATABASE_PATH.value),
                small_bfd_z_value=_SMALL_BFD_Z_VALUE.value,
                mgnify_database_path=expand_path(_MGNIFY_DATABASE_PATH.value),
                mgnify_z_value=_MGNIFY_Z_VALUE.value,
                uniprot_cluster_annot_database_path=expand_path(
                    _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
                ),
                uniprot_cluster_annot_z_value=_UNIPROT_CLUSTER_ANNOT_Z_VALUE.value,
                uniref90_database_path=expand_path(_UNIREF90_DATABASE_PATH.value),
                uniref90_z_value=_UNIREF90_Z_VALUE.value,
                ntrna_database_path=expand_path(_NTRNA_DATABASE_PATH.value),
                ntrna_z_value=_NTRNA_Z_VALUE.value,
                rfam_database_path=expand_path(_RFAM_DATABASE_PATH.value),
                rfam_z_value=_RFAM_Z_VALUE.value,
                rna_central_database_path=expand_path(_RNA_CENTRAL_DATABASE_PATH.value),
                rna_central_z_value=_RNA_CENTRAL_Z_VALUE.value,
                pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
                seqres_database_path=expand_path(_SEQRES_DATABASE_PATH.value),
                jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
                jackhmmer_max_parallel_shards=_JACKHMMER_MAX_PARALLEL_SHARDS.value,
                nhmmer_n_cpu=_NHMMER_N_CPU.value,
                nhmmer_max_parallel_shards=_NHMMER_MAX_PARALLEL_SHARDS.value,
                max_template_date=max_template_date,
            )
        else:
            data_pipeline_config = None
    else:
        fold_inputs = None
        max_template_date = None
        data_pipeline_config = None

    if _RUN_INFERENCE.value:
        if _USE_CPU.value == True:
            devices = [torch.device('cpu')]
            log.info(
                f'Found local devices: {devices}, using device "cpu".'
            )
        else:
            if not torch.cuda.is_available():
                raise ValueError('CUDA not available!')
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
            log.info(
                f'Found local devices: {devices}, using device {_GPU_DEVICE.value}:'
                f' {devices[_GPU_DEVICE.value]}'
            )

        if dist.is_primary():
            log.info('Building model from scratch...')
        model_runner = ModelRunner(
            config=make_model_config(
                num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
                num_recycles=_NUM_RECYCLES.value,
                return_embeddings=_SAVE_EMBEDDINGS.value,
                return_distogram=_SAVE_DISTOGRAM.value,
            ),
            device=str(devices[0] if _USE_CPU.value else devices[_GPU_DEVICE.value]),
            model_dir=pathlib.Path(MODEL_DIR.value),
        )
        # Check we can load the model parameters before launching anything.
        if dist.is_primary():
            log.info('Checking that model parameters can be loaded...')
        _ = model_runner.model_params
        model_runner.prepare_model()
        if dist.is_enabled():
            dist.barrier()
    else:
        model_runner = None

    if dist.is_primary():
        num_fold_inputs = len(fold_inputs)
        num_fold_inputs_l = [num_fold_inputs]
    else:
        num_fold_inputs_l = [None]
    
    if dist.is_enabled():
        dist.broadcast_object_list(num_fold_inputs_l, src=0)
        num_fold_inputs, = num_fold_inputs_l

    for i in range(num_fold_inputs):
        if dist.is_primary():
            fold_input = fold_inputs[i]
            if prediction_flags._OUTPUT_DIR_OVERRIDES:
                if len(prediction_flags._OUTPUT_DIR_OVERRIDES) != num_fold_inputs:
                    raise ValueError(
                        'Launcher resolved '
                        f'{len(prediction_flags._OUTPUT_DIR_OVERRIDES)} output directories for '
                        f'{num_fold_inputs} fold inputs.'
                    )
                output_dir = prediction_flags._OUTPUT_DIR_OVERRIDES[i]
            else:
                output_dir = os.path.join(
                    _OUTPUT_DIR.value, fold_input.sanitised_name()
                )
            if _NUM_SEEDS.value is not None:
                log.info(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
                fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
        else:
            fold_input = None
            output_dir = None
        
        with acceleration.execution_context():
            process_fold_input(
                fold_input=fold_input,
                data_pipeline_config=data_pipeline_config,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
                ref_max_modified_date=max_template_date,
                conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
                resolve_msa_overlaps=_RESOLVE_MSA_OVERLAPS.value,
                force_output_dir=(
                    True if prediction_flags._OUTPUT_DIR_OVERRIDES else _FORCE_OUTPUT_DIR.value
                )
            )
    
    if dist.is_primary():
        log.info(f'Done running {num_fold_inputs} fold jobs.')



def entrypoint():
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)

if __name__ == '__main__':
    entrypoint()
