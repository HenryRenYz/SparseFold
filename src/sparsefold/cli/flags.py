# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz (SparseFold modifications)
# SPDX-License-Identifier: Apache-2.0

"""Prediction flags; this module does not import Torch or AlphaFold 3."""
from __future__ import annotations
from collections.abc import Sequence
import os
import pathlib
import shutil
import sys
from absl import flags
from sparsefold.integrations.af3.compatibility import DEFAULT_AF3_REVISION

def _find_binary(name: str) -> str | None:
    """Prefer tools in this interpreter's environment, then search PATH."""
    candidate = pathlib.Path(sys.prefix) / 'bin' / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(name)


_HOME_DIR = pathlib.Path.home()
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'

# Set only by sparsefold.cli.worker. The launcher reserves these directories
# before any rank starts so console logs and model artifacts share one exact,
# collision-free run directory. Direct invocation keeps the upstream path
# selection below.
_OUTPUT_DIR_OVERRIDES: tuple[pathlib.Path, ...] = ()


def set_output_dir_overrides(paths: Sequence[os.PathLike[str] | str]) -> None:
    """Install launcher-resolved output directories in fold-input order."""
    global _OUTPUT_DIR_OVERRIDES
    _OUTPUT_DIR_OVERRIDES = tuple(pathlib.Path(path) for path in paths)

_USE_CPU = flags.DEFINE_bool(
    'use_cpu',
    False,
    'Use cpu backend.'
)
_PRECOMPILE = flags.DEFINE_bool(
    'precompile',
    False,
    'Collect and batch-compile native operators before inference.',
)
_PRECOMPILE_USE_REAL_TENSORS = flags.DEFINE_bool(
    'precompile_use_real_tensors',
    False,
    'Use the supplied tensors instead of FakeTensorMode while tracing native operators.',
)
_PRECOMPILE_PARALLELISM = flags.DEFINE_integer(
    'precompile_parallelism',
    8,
    'Maximum number of native operator compilation requests to run concurrently.',
    lower_bound=1,
)
_PRECOMPILE_GRAPH_PATH = flags.DEFINE_string(
    'precompile_graph_path',
    None,
    'Write the framework-neutral precompile execution graph to this JSON path.',
)
_PRECOMPILE_ONLY = flags.DEFINE_bool(
    'precompile_only',
    False,
    'Stop after precompilation instead of running model inference.',
)
_MEMORY_PLAN_FAST_MIB = flags.DEFINE_integer(
    'memory_plan_fast_mib',
    0,
    'Enable tensor memory planning with this per-process HBM budget.',
    lower_bound=0,
)
_MEMORY_PLAN_SLOW_MIB = flags.DEFINE_integer(
    'memory_plan_slow_mib',
    65536,
    'Per-process managed DDR arena limit used by the tensor planner.',
    lower_bound=1,
)
_MEMORY_PLAN_PATH = flags.DEFINE_string(
    'memory_plan_path',
    None,
    'Optional path for the generated HBM/DDR memory plan JSON.',
)
_MEMORY_PLAN_ALLOW_FAST_FALLBACK = flags.DEFINE_bool(
    'memory_plan_allow_fast_fallback',
    False,
    'Allow the planned HBM arena to fall back to default memory.',
)
_MEMORY_PLAN_LARGE_PAGES = flags.DEFINE_bool(
    'memory_plan_large_pages',
    True,
    'Prefer mixed HugeTLB/THP backing for planned tensor arenas.',
)
_MEMORY_PLAN_FACTORY_REPLAY = flags.DEFINE_bool(
    'memory_plan_factory_replay',
    False,
    'Replay eligible empty/empty_like allocations from one planned peak arena.',
)
_MEMORY_PLAN_STATE_REPLAY = flags.DEFINE_bool(
    'memory_plan_state_replay',
    False,
    'Place selected read-only model state in HBM.',
)
_MEMORY_PLAN_MIN_FAST_MIB = flags.DEFINE_integer(
    'memory_plan_min_fast_mib',
    0,
    'Minimum storage size in MiB eligible for production HBM placement.',
    lower_bound=0,
)
_MEMORY_PLAN_MIN_REUSE = flags.DEFINE_float(
    'memory_plan_min_reuse',
    1.0,
    'Minimum loop-weighted access count eligible for production HBM placement.',
    lower_bound=0.0,
)
_MEMORY_PLAN_FAST_BANDWIDTH_MIB_S = flags.DEFINE_float(
    'memory_plan_fast_bandwidth_mib_s',
    0.0,
    'Measured HBM bandwidth for planner scoring; 0 uses topology metadata.',
    lower_bound=0.0,
)
_MEMORY_PLAN_SLOW_BANDWIDTH_MIB_S = flags.DEFINE_float(
    'memory_plan_slow_bandwidth_mib_s',
    0.0,
    'Measured DDR bandwidth for planner scoring; 0 uses topology metadata.',
    lower_bound=0.0,
)
_MEMORY_PLAN_TRANSFER_BANDWIDTH_MIB_S = flags.DEFINE_float(
    'memory_plan_transfer_bandwidth_mib_s',
    0.0,
    'Measured DDR/HBM transfer bandwidth; 0 uses discovered DDR bandwidth.',
    lower_bound=0.0,
)
_VECOPS_WORKSPACE_FAST_MIB = flags.DEFINE_integer(
    'vecops_workspace_fast_mib',
    0,
    'Total high-bandwidth workspace arena per process. Zero disables the '
    'vecops memory session.',
    lower_bound=0,
)
_VECOPS_WORKSPACE_SLOW_MIB = flags.DEFINE_integer(
    'vecops_workspace_slow_mib',
    0,
    'Total default-memory workspace arena per process. Zero uses the fast '
    'arena size.',
    lower_bound=0,
)
_VECOPS_WORKSPACE_DOMAIN = flags.DEFINE_integer(
    'vecops_workspace_domain',
    -1,
    'Dense vecops CPU domain for workspace placement; -1 freezes the current '
    'affinity domain when the session is created.',
    lower_bound=-1,
)
_VECOPS_WORKSPACE_FAST_FALLBACK = flags.DEFINE_bool(
    'vecops_workspace_fast_fallback',
    True,
    'Continue with a zero-sized fast arena when HBM allocation is unavailable.',
)
_VECOPS_LARGE_PAGES = flags.DEFINE_bool(
    'vecops_large_pages',
    True,
    'Prefer large pages for vecops workspace arenas. Set false to force '
    'ordinary pages; THP and HugeTLB are selected internally.',
)
_DIST = flags.DEFINE_bool(
    'dist',
    False,
    'Run in distributed mode (requires envvars be properly set).',
)
_DIST_BACKEND = flags.DEFINE_string(
    'dist_backend',
    'default',
    'Specify backend for distributed mode.'
)

# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    _find_binary('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    _find_binary('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    _find_binary('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    _find_binary('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    _find_binary('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_SMALL_BFD_Z_VALUE = flags.DEFINE_integer(
    'small_bfd_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_MGNIFY_Z_VALUE = flags.DEFINE_integer(
    'mgnify_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_Z_VALUE = flags.DEFINE_integer(
    'uniprot_cluster_annot_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_UNIREF90_Z_VALUE = flags.DEFINE_integer(
    'uniref90_z_value',
    None,
    'The Z-value representing the database size in number of sequences for'
    ' E-value calculation. Must be set for sharded databases.',
    lower_bound=0,
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_NTRNA_Z_VALUE = flags.DEFINE_float(
    'ntrna_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RFAM_Z_VALUE = flags.DEFINE_float(
    'rfam_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_RNA_CENTRAL_Z_VALUE = flags.DEFINE_float(
    'rna_central_z_value',
    None,
    'The Z-value representing the database size in megabases for E-value'
    ' calculation. Must be set for sharded databases.',
    lower_bound=0.0,
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/mmcif_files',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Jackhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_JACKHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'jackhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Jackhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Nhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
_NHMMER_MAX_PARALLEL_SHARDS = flags.DEFINE_integer(
    'nhmmer_max_parallel_shards',
    None,
    'Maximum number of shards to search against in parallel. If unset, one'
    ' Nhmmer instance will be run per shard. Only applicable if the'
    ' database is sharded.',
    lower_bound=1,
)

# Data pipeline configuration.
_RESOLVE_MSA_OVERLAPS = flags.DEFINE_bool(
    'resolve_msa_overlaps',
    True,
    'Whether to deduplicate unpaired MSA against paired MSA. The default'
    ' behaviour matches the method described in the AlphaFold 3 paper. Set this'
    ' to false if providing custom paired MSA using the unpaired MSA field to'
    ' keep it exactly as is as deduplication against the paired MSA could break'
    ' the manually crafted pairing between MSA sequences.',
)
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    'max_template_date',
    '2021-09-30',  # By default, use the date from the AlphaFold 3 paper.
    'Maximum template release date to consider. Format: YYYY-MM-DD. All'
    ' templates released after this date will be ignored. Controls also whether'
    ' to allow use of model coordinates for a chemical component from the CCD'
    ' if RDKit conformer generation fails and the component does not have ideal'
    ' coordinates set. Only for components that have been released before this'
    ' date the model coordinates can be used as a fallback.',
)
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
    lower_bound=0,
)

_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference, uses zero-based'
    ' indexing. Defaults to the 0th GPU on the system. Useful on multi-GPU'
    ' systems to pin each run to a specific GPU. Note that if GPUs are already'
    ' pre-filtered by the environment (e.g. by using CUDA_VISIBLE_DEVICES),'
    ' this flag refers to the GPU index after the filtering has been done.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    10,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    False,
    'Whether to save the final trunk single and pair embeddings in the output.'
    ' Note that the embeddings are large float16 arrays: num_tokens * 384'
    ' + num_tokens * num_tokens * 128.',
)
_SAVE_DISTOGRAM = flags.DEFINE_bool(
    'save_distogram',
    False,
    'Whether to save the final distogram in the output. Note that the distogram'
    ' is a large float16 array: num_tokens * num_tokens * 64.',
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    'force_output_dir',
    False,
    'Whether to force the output directory to be used even if it already exists'
    ' and is non-empty. Useful to set this to True to run the data pipeline and'
    ' the inference separately, but use the same output directory.',
)


_AF3_REVISION = flags.DEFINE_string('af3_revision', DEFAULT_AF3_REVISION, 'Explicit supported official AlphaFold 3 source revision.')
_AF3_SOURCE_ROOT = flags.DEFINE_string('af3_source_root', None, 'Verified official AlphaFold 3 checkout used by the external adapter.')
