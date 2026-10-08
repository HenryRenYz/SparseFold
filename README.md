# SparseFold

SparseFold performs molecular structure inference in PyTorch, with optional vecops native CPU kernels. A separately installed official AlphaFold 3 (AF3) package provides input parsing, database searches, featurization, and chemical resources. Model computation uses SparseFold's own tensor interfaces.

This guide covers single-process CPU installation and a first prediction, followed by native CPU acceleration. See [configuration](docs/configuration.md) for configuration details and [validation](docs/validation.md) for test results and their scope.

## 1. System and hardware

Use Linux and Python 3.12. Full CPU inference has been tested on an Intel Xeon w9-3495X running Ubuntu 24.04; CUDA inference has also been tested on an RTX 4090. The regular Torch CPU path does not require AMX. The current x86 native kernels require AMX/BF16 support, and ARM/SME configurations still need separate validation.

The official AF3 extension and vecops require a C++20 compiler, CMake, and Ninja. The following system packages are an example for Ubuntu 24.04. On other distributions, use the corresponding packages and check that GCC is at least version 13 and CMake at least version 3.28.

```bash
sudo apt-get update
sudo apt-get install git build-essential cmake ninja-build zlib1g-dev zstd libnuma1 \
  python3.12 python3.12-venv python3.12-dev
g++ --version
cmake --version
```

CPU inference does not require a GPU or driver. However, the supported official AF3 revision declares `jax[cuda12]` on Linux, so its standard installation includes CUDA-related Python packages. This guide follows that dependency declaration; the validated CPU environment is not a minimal CPU-only installation.

## 2. Create an environment and install SparseFold

From the SparseFold source or delivery directory, create a new environment. Keep other `alphafold3` forks out of `PYTHONPATH` so they do not shadow the official package.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
python -m pip install 'setuptools>=77' setuptools-scm wheel scikit-build-core pybind11 ninja \
  'cmake>=3.28' 'numpy>=2,<3'
python -m pip install 'torch==2.9.0'
python -m pip install -e .
```

Alternatively, create and activate a `conda create -n sparsefold python=3.12` environment, then run the same pip commands. Torch 2.9.0 was used for the full inference tests. The wider installable range in [pyproject.toml](pyproject.toml) does not imply that every version in that range has been tested.

SparseFold is packaged with setuptools. The official AF3 extension and vecops are built in the following steps.

## 3. Install official AF3 and build chemical resources

Official AF3 must be installed in editable mode from a clean, supported Git checkout. The adapter verifies the source location and the installed extension's SHA256 checksum. A regular wheel, a source copy without Git metadata, or another PyTorch fork does not satisfy these checks.

The supported revision is defined in [compatibility.py](src/sparsefold/integrations/af3/compatibility.py). The commands below read that definition directly so this guide does not maintain a second commit hash. Keep the official checkout outside the SparseFold release source tree.

```bash
AF3_SRC="$HOME/sparsefold-deps/alphafold3"
mkdir -p "$(dirname "$AF3_SRC")"
AF3_REVISION="$(python -c \
  'from sparsefold.integrations.af3.compatibility import DEFAULT_AF3_REVISION; print(DEFAULT_AF3_REVISION)')"
git clone https://github.com/google-deepmind/alphafold3.git "$AF3_SRC"
git -C "$AF3_SRC" checkout --detach "$AF3_REVISION"
CMAKE_BUILD_PARALLEL_LEVEL=8 python -m pip install --no-build-isolation \
  --config-settings cmake.define.BUILD_TESTING=OFF -e "$AF3_SRC"
build_data
python -m pip check
python -c "from sparsefold.integrations.af3.provider import AF3Provider; print(AF3Provider(source_root='$AF3_SRC').verify())"
```

`build_data` converts `components.cif` into a chemical component dictionary (CCD) pickle and generates chemical component sets. Both resources are required even when the input already has MSAs and templates and database searches are skipped. If `components.cif` cannot be found, set `LIBCIFPP_DATA_DIR` to the environment's `share/libcifpp` directory containing that file, then run `build_data` again. Keep the official tracked source unchanged and retain its Git metadata.

## 4. Prepare model parameters and input

Obtain model parameters under the official [parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md). Place `af3.bin.zst` in the directory configured by `paths.model`. Installation does not download parameters or change their license.

Inputs use the official AF3 JSON format. `input_name = "example"` selects `paths.input/example.json`. When skipping searches, a protein input must provide `unpairedMsa`, `pairedMsa`, and `templates`. Empty strings or an empty list explicitly indicate that the corresponding information is unused; missing fields are not equivalent.

Save the following synthetic short sequence as `inputs/example.json` for a first run. It has no MSAs or templates and demonstrates the workflow without assessing prediction quality.

```json
{
  "name": "example",
  "modelSeeds": [1],
  "sequences": [{
    "protein": {
      "id": "A",
      "sequence": "ACDEFGHIKLMNPQRSTVWY",
      "unpairedMsa": "",
      "pairedMsa": "",
      "templates": []
    }
  }],
  "dialect": "alphafold3",
  "version": 4
}
```

For actual tasks, supply your own sequences and the corresponding features. Private inputs, weights, and databases are outside the release source; the example data directories are excluded by `.gitignore`.

## 5. Run a first CPU prediction

`configs/cpu.toml` selects Torch CPU execution, math attention, a single process, and the default allocator and OpenMP runtime. Print the configuration before running. The commands below set the official source directory through the CLI; other paths can be stored in a local configuration.

```bash
sparsefold --help
sparsefold print-config -c configs/cpu.toml \
  --set model.af3_source_root="$AF3_SRC"
sparsefold -c configs/cpu.toml --input-name example \
  --set model.af3_source_root="$AF3_SRC" \
  --set paths.model=/path/to/model-parameters
```

Use `run_data_pipeline=false` for inputs with existing MSAs and templates. Official featurization still runs. Each seed produces five samples by default, with 200 diffusion steps and 10 additional recycles, for 11 trunk passes. These are the full settings, rather than a reduced-step demonstration.

Automatic NUMA binding selects one hardware thread per physical core. If a container prohibits memory binding, use `affinity.mode="explicit"` in a local configuration, supply available physical-core IDs, and leave `memory_node_ids=[]`. CPU IDs depend on the machine's topology. See [configuration](docs/configuration.md) for details.

## 6. Enable native CPU kernels

After confirming that the hardware is suitable, install the vecops extra. Its pinned revision comes from `pyproject.toml`.

```bash
python -m pip install -e '.[vecops]'
sparsefold print-config -c configs/native_cpu.toml \
  --set model.af3_source_root="$AF3_SRC"
sparsefold -c configs/native_cpu.toml --input-name example \
  --set model.af3_source_root="$AF3_SRC" \
  --set paths.model=/path/to/model-parameters
```

This configuration explicitly sets `SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION=1`, enabling all native kernel families and precompilation. The experimental gate remains because the x86 tests cover only three input lengths, not every shape, precision, or platform. Unsupported Linear cases continue through Torch, so enabling every family does not make every tensor operation native.

The first run compiles and caches C++ bridges and kernels. `model.precompile_parallelism` controls compilation parallelism, while `runtime.threads` controls Torch computation threads. Set them independently. If necessary, supply working `CC`, `CXX`, and `CMAKE` executables in `[runtime.env]` and check that they can run.

## 7. Run full database searches

For raw sequences, enable the official data pipeline and prepare HMMER and the databases. Follow the official [installation documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md), using the filenames and tool requirements of the supported checkout. In SparseFold, set `paths.database` and `model.run_data_pipeline=true`.

Keep the full databases outside the source tree and plan their disk space separately. The pinned revision's documentation records about 252 GB to download and 630 GB after extraction. HMMER 3.4 requires the official `jackhmmer_seq_limit.patch`; a distribution's unpatched `hmmer` package does not provide that capability. Search tools can be found through PATH or specified by their full paths in `[model]`.

Searches and inference can run in separate calls. First set `run_data_pipeline=true, run_inference=false` to write an enriched JSON input, then use `run_data_pipeline=false, run_inference=true` for inference. Some Z-value and shard options from the historical data pipeline have no official equivalent, and the adapter rejects unsupported custom values. The full inference tests used existing features and did not validate live database searches.

## 8. Outputs, comparison, and troubleshooting

Outputs are written under the configured root, with a subdirectory named after the job in the input JSON. When that directory already contains results, the launcher reserves a timestamped directory for the new attempt. Outputs include `console.log`, the enriched input JSON, `af3_provider.json`, CIF and confidence files in `seed-…_sample-…` subdirectories, a ranking CSV, and the selected best model. Embeddings and distograms require explicit options and can be large.

The structure comparison tool is optional:

```bash
python -m pip install -e '.[compare]'
python -m sparsefold.cli.structure_compare compare \
  /path/to/predicted.cif /path/to/reference.cif --format json
```

The comparator's default RMSD procedure can remove outliers iteratively. Use `--cycles 0` for a fit without outlier removal. The three inputs in the [validation summary](docs/validation.md) were compared using all matched Cα atoms without removing outliers, so use the same protocol when comparing results.

| Issue | Check or remedy |
| --- | --- |
| AF3 namespace shadowed / version mismatch | Remove paths to other forks and check the interpreter, official editable installation, and supported revision. |
| Dirty official checkout or mismatched native RECORD | Use a clean, independent checkout and rebuild its official extension in the target environment. |
| Missing CCD/components.cif | Run `build_data` in the official package's environment; set `LIBCIFPP_DATA_DIR` if needed. |
| cmake file exists but cannot run | Check its shebang and interpreter. Configure a working CMake executable with version ≥3.28. |
| Missing `GLIBCXX` version | Check that the compiler and runtime libstdc++ match. Conda users can install a compatible `libstdcxx-ng` in the new environment. |
| Incompatible vecops API, unsupported hardware, or disabled fusion | Use the declared pinned dependency and a supported CPU. Check the native configuration or select the Torch CPU profile. |
| Intel OpenMP / tcmalloc not found | The basic profiles use defaults. Change `runtime.omp` or `runtime.malloc` only after installing the corresponding optional libraries. |
| Missing input MSAs/templates | Supply existing features, or prepare the search resources and enable the data pipeline. |
