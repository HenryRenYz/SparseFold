# Validation scope and results

The full inference results below were obtained in an independent validation environment on 2026-10-07/08. Documentation and configuration cleanup did not rerun the full model. Historical versions and measurements are retained unchanged, with existing official JAX GPU results used as the structural comparison baseline.

## Full inference

The tested environment used Python 3.12.11, Torch 2.9.0+cu128, JAX 0.10.2, and NumPy 2.5.3. CPU runs used physical cores 0–55 of an Intel Xeon w9-3495X, excluding SMT siblings 56–111. GPU runs used an RTX 4090 with 24 GB. The tested official AF3 revision was `c121cd877ae3d9ac62922f0f8bedde7179bb39b1`, and the native runs used vecops `f8500b9c07dec26342ac980693c53f48c2b31906`. These identify historical tests; code and package declarations maintain the active compatibility revisions.

Each input used seed=1, five samples, 200 diffusion steps, and 10 additional recycles, for 11 trunk passes. GPU runs used Torch math attention and BF16 diffusion. The input of length 740 completed with tensor splitting enabled. Native CPU runs enabled every fused kernel family, retained Torch for unsupported Linear cases, and disabled both the matmul dispatcher and tensor splitting.

| Input length | Torch GPU inference (s) | Native CPU inference (s) | GPU vs. historical baseline: mean TM-score / Cα RMSD (Å) | Native CPU vs. historical baseline: mean TM-score / Cα RMSD (Å) |
| --- | ---: | ---: | --- | --- |
| 107 | 17.29 | 64.79 | 0.974993 / 0.980993 | 0.975902 / 0.792447 |
| 436 | 78.35 | 237.73 | 0.996960 / 0.408117 | 0.998044 / 0.327855 |
| 740 | 220.83 | 444.59 | 0.997319 / 0.561926 | 0.998055 / 0.443377 |

For each platform, all 15 corresponding target/seed/sample combinations were compared without selecting the best sample. Cα RMSD used Kabsch alignment over all matched Cα atoms, with no outlier removal. Heavy atoms used the same alignment. TM-align used the actual amino-acid sequences and recorded scores normalized by the prediction length and by the baseline length. All compared atom identifiers matched, and confidence JSON values were finite.

Each complete native input recorded 18,255 successful bridge calls outside precompilation, covering 11 kernel families. Model preparation produced 428 fused modules, 53 general native Linear modules, and 19 explicit Linear fallbacks. Affinity checks found no execution outside the specified physical-core range.

The timings include logging and audit overhead and are not isolated performance benchmarks. The historical official JAX GPU baseline lacks complete version and configuration records, and CPU, CUDA, and JAX use different random streams. The comparisons therefore demonstrate structural consistency, not elementwise numerical equivalence. No undeclared acceptance threshold was applied.

## Installation and lightweight checks

Earlier validation covered the official AF3 editable build, chemical-resource generation, strict checkpoint mapping, real schema/output interfaces, small native JIT numerical checks, and CPU/CUDA tests. The full checkpoint covered 3,761 standard parameter keys; this does not establish correctness for every native shape.

On 2026-10-08, a new Python 3.12.11 venv that did not inherit third-party site-packages completed SparseFold installation, an official AF3 editable extension rebuild, public chemical-resource regeneration, and installation of the declared vecops extra. `pip check` found no dependency conflicts. All six configurations loaded, and CLI help worked. Official featurization succeeded for a synthetic 20-residue input. A small BF16 native Linear check produced finite outputs with a maximum absolute difference of 0 from the Torch reference. No model parameters were loaded and no full prediction ran in this installation check.

The installation used uv and public download caches. The official Git checkout and some C++ dependencies came from source caches verified against fixed revisions; Eigen used a shallow clone of the upstream-specified version. The public CCD cache was reused after SHA256 verification. Extensions and chemical resources were rebuilt in the new directory, without copying installed packages or extensions from an existing environment. GCC/CMake were already available. This check did not cover a cold network download, every original pip command, system-package installation, or a minimal CPU-only dependency set.

Documentation and configuration cleanup also passed 77 lightweight tests, including the explicit experimental fusion gate. Configuration fields, documentation links, and example syntax were checked. The system-package commands and the synthetic example's full prediction command were not executed.

## Remaining coverage

Full inference for the three validation inputs used existing MSAs and templates without rerunning database searches. Live searches, full distributed inference, compiled models and memory graphs, other precisions and sizes, ARM/SME, and full native regressions for multichain and ligand inputs still require separate validation. A minimal CPU-only installation has not been established by the environments containing CUDA packages.

Detailed environment records, installation commands and cache sources, aborted or failed attempts, migration inventories, and reproduction scripts are retained outside the release source tree. Private inputs, parameters, CIFs, logs, and kernel caches are excluded from the release. This page provides a summary of the tests and their limits.
