# Configuration and execution

See the [README](../README.md) for installation and a first CPU prediction. A configuration describes one run. Start with `configs/cpu.toml`, then keep machine-specific settings and data paths in a local TOML file.

## Configuration files

All example files listed below are in `configs/`.

| File | Purpose |
| --- | --- |
| `cpu.toml` | Single-process Torch CPU execution with math attention and the default allocator/OpenMP runtime. |
| `native_cpu.toml` | Extends the CPU profile with vecops fusion and precompilation. |
| `x86.toml` | An x86 entry point extending the native CPU profile; requires AMX/BF16. |
| `x86_comm.toml` | Experimental two-process Gloo example. Requires two visible CPU NUMA nodes or explicit local core assignments. Full distributed inference is unvalidated. |
| `920f.toml` | Experimental ARM/SME and HBM planning profile. The compiler and Kutacc CPU communication backend require local setup. Full integration is unvalidated. |
| `template.toml` | An annotated reference for all configuration fields. |

`extends` is a top-level key. A relative parent path is resolved from the child configuration's directory. Tables are merged recursively by key; lists and scalar values replace the inherited value. From highest to lowest, precedence is: dedicated CLI options, `--set`, the current TOML, the parent TOML, and defaults.

For example, create `configs/local.toml` with:

```toml
extends = "native_cpu.toml"
[paths]
input = "/path/to/inputs"
model = "/path/to/model-parameters"
output = "/path/to/results"
[model]
af3_source_root = "/path/to/official-alphafold3"
```

```bash
sparsefold print-config -c configs/local.toml
sparsefold -c configs/local.toml --input-name example
```

Path fields are interpreted relative to the runtime working directory, with `~` expanded. Only a relative `extends` path is based on the configuration file's directory. Installing a wheel does not install the root-level configuration examples; obtain them from the source or create a local TOML using this guide.

## Common options

`--input-name NAME` or the positional argument `NAME` selects `paths.input/NAME.json`; use one form per call. `--set KEY=VALUE` is repeatable, for example `--set runtime.threads=8`. Model options require the full `model.` prefix, as in `--set model.num_recycles=10`. `print-config` prints the configuration without starting inference. The inference worker also checks `[model]` for unknown prediction flags.

The launcher manages input, output, parameter paths, and process-group settings through `[paths]` and `[launch]`. Keep `json_path`, `input_dir`, `output_dir`, `model_dir`, `db_dir`, `dist`, and `dist_backend` out of `[model]`. Prediction flags without a launcher option can be forwarded after `--`, though TOML is preferable for recording routine runs.

## CPU affinity and threads

`affinity.mode="numa"` selects one CPU NUMA node per local rank and excludes SMT siblings. `runtime.threads=0` derives the thread count from the binding plan. Explicit mode requires physical-core lists for each rank; the IDs depend on the actual topology.

If memory binding is prohibited in a container, use `explicit` mode with `compute_cpu_lists` and leave `memory_node_ids=[]` to constrain CPUs alone. Assign disjoint CPU lists to different ranks. In automatic NUMA mode, the local process count should not exceed the available CPU NUMA nodes.

`launch.nproc=1` does not initialize a distributed process group or require MPI. With `nproc>1`, the selected launcher/backend is used. HBM, Kutacc, MPI, and nondefault allocator/OpenMP choices are platform-specific options, not prerequisites for ordinary CPU installation.

## Precision, precompilation, and resources

The regular CPU example uses math attention and BF16 diffusion. `runtime.sharding=true` splits existing tensor operations to reduce some temporary tensors while retaining the sample count, diffusion steps, and recycles. The native example disables this splitting, matching the tested path.

Native precompilation collects the required kernel specializations, then compiles and caches them. `model.precompile_parallelism` controls compilation parallelism, while `runtime.threads` controls model computation threads. Kernel request collection can traverse a single trunk pass; the runtime restores the configured additional recycles before inference. `precompile_only=true` exits after precompilation and is not evidence of a successful full prediction.

To configure the toolchain, set `CC`, `CXX`, `CMAKE`, and `VECOPS_CACHE_DIR` in `[runtime.env]`. Validation tools can use `SPARSEFOLD_NATIVE_AUDIT_DIR` to record successful native bridge calls outside precompilation. Ordinary runs do not require this audit.

The supported official AF3 revision comes from [compatibility.py](../src/sparsefold/integrations/af3/compatibility.py), and the vecops installation revision from [pyproject.toml](../pyproject.toml). Normally omit `model.af3_revision`. An upgrade requires review of adapter interfaces and the schema, followed by the relevant tests; changing the checkout alone is insufficient.
