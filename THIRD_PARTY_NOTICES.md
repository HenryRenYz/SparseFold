# Third-party notices and source provenance

SparseFold original contributions are copyright 2026 HenryRenYz and licensed under
MIT (`LICENSE`). Files carrying an Apache-2.0 SPDX identifier retain that license;
MIT does not replace their original attribution or license.

## AlphaFold 3

Copyright 2024 DeepMind Technologies Limited. AlphaFold 3 source code is now
published under Apache License 2.0 by Google DeepMind. The upstream code-license transition is recorded at commit `7b197fe859790fc3e04d03ea70dd0b9ba48881c9`.
The official source reviewed on 2026-10-07 was
`c121cd877ae3d9ac62922f0f8bedde7179bb39b1` at
https://github.com/google-deepmind/alphafold3.
The complete Apache license is included at `LICENSES/Apache-2.0.txt`.
This dated provenance is distinct from the active adapter compatibility definition
in `src/sparsefold/integrations/af3/compatibility.py`.
Legacy local Creative Commons (CC) headers were stale according to the source owner; the derived
code below follows the upstream Apache code release while preserving DeepMind
copyright. This statement covers source code, not model parameters or datasets.

The fixed local source tree was `af3-torchneo` main
`13e9995b170c0a0da2216ca996f86b966bb50400`. Its history is not included.
Modifications include the PyTorch translation and runtime, splitting numerical
model input from host metadata, reorganizing module responsibilities, strict
checkpoint mapping, externalizing the official pipeline, and output integration.
A translated model remains derived code; renaming does not change that status.

The tensor primitives also retain Apache-2.0 because their translated Linear,
normalization and gating semantics include upstream-derived material; an absent
legacy header is not treated as proof of original MIT authorship.

### Included Apache-derived file mapping

For rows without a specific entry in the Changes column, the changes are the
PyTorch port or adaptations to SparseFold module boundaries, imports and
validation. Original attribution is retained throughout the table.

| SparseFold file | Fixed-main source | Changes |
| --- | --- | --- |
| `src/sparsefold/model/layers/primitives.py` | `src/alphafold3/torch/nn.py`; upstream `model/components/haiku_modules.py` | PyTorch translation, runtime device resolution and explicit parameter mapping |
| `src/sparsefold/cli/flags.py` | `scripts/run_alphafold.py` | Extracted prediction flags; original attribution retained |
| `src/sparsefold/cli/predict.py` | `scripts/run_alphafold.py` |  |
| `src/sparsefold/model/config.py` | `src/alphafold3/model/model_config.py` |  |
| `src/sparsefold/model/diffusion/generator.py` | `src/alphafold3/model/network/diffusion_head.py` |  |
| `src/sparsefold/model/diffusion/noise.py` | `src/alphafold3/model/network/noise_level_embeddings.py` |  |
| `src/sparsefold/model/diffusion/transformer.py` | `src/alphafold3/model/network/diffusion_transformer.py` |  |
| `src/sparsefold/model/embeddings/atom.py` | `src/alphafold3/model/network/atom_cross_attention.py` |  |
| `src/sparsefold/model/embeddings/features.py` | `src/alphafold3/model/network/featurization.py` |  |
| `src/sparsefold/model/embeddings/template.py` | `src/alphafold3/model/network/template_modules.py` |  |
| `src/sparsefold/model/geometry/__init__.py` | `src/alphafold3/torch/geometry/__init__.py` |  |
| `src/sparsefold/model/geometry/residue_frames.py` | `src/alphafold3/model/protein_data_processing.py` |  |
| `src/sparsefold/model/geometry/rigid_matrix_vector.py` | `src/alphafold3/torch/geometry/rigid_matrix_vector.py` |  |
| `src/sparsefold/model/geometry/rotation_matrix.py` | `src/alphafold3/torch/geometry/rotation_matrix.py` |  |
| `src/sparsefold/model/geometry/scoring.py` | `src/alphafold3/model/scoring/scoring.py` |  |
| `src/sparsefold/model/geometry/struct_of_array.py` | `src/alphafold3/torch/geometry/struct_of_array.py` |  |
| `src/sparsefold/model/geometry/utils.py` | `src/alphafold3/torch/geometry/utils.py` |  |
| `src/sparsefold/model/geometry/vector.py` | `src/alphafold3/torch/geometry/vector.py` |  |
| `src/sparsefold/model/heads/confidence.py` | `src/alphafold3/model/network/confidence_head.py` |  |
| `src/sparsefold/model/heads/distogram.py` | `src/alphafold3/model/network/distogram_head.py` |  |
| `src/sparsefold/model/layers/mapping.py` | `src/alphafold3/model/components/mapping.py` |  |
| `src/sparsefold/model/layers/utils.py` | `src/alphafold3/model/components/utils.py` |  |
| `src/sparsefold/model/legacy_config.py` | `src/alphafold3/common/base_config.py` |  |
| `src/sparsefold/model/network.py` | `src/alphafold3/model/model.py` |  |
| `src/sparsefold/model/trunk/blocks.py` | `src/alphafold3/model/network/modules.py` |  |
| `src/sparsefold/model/trunk/network.py` | `src/alphafold3/model/network/evoformer.py` |  |
| `src/sparsefold/model/vocabulary/atom_types.py` | `src/alphafold3/constants/atom_types.py` |  |
| `src/sparsefold/model/vocabulary/residue_names.py` | `src/alphafold3/constants/residue_names.py` |  |
| `src/sparsefold/model/vocabulary/side_chains.py` | `src/alphafold3/constants/side_chains.py` |  |
| `src/sparsefold/output/__init__.py` | `Exports of the derived output modules listed below` |  |
| `src/sparsefold/output/confidence.py` | `src/alphafold3/model/confidences.py` |  |
| `src/sparsefold/output/confidence_types.py` | `src/alphafold3/model/confidence_types.py` |  |
| `src/sparsefold/output/decode.py` | `src/alphafold3/model/model.py` |  |
| `src/sparsefold/output/mmcif_metadata.py` | `src/alphafold3/model/mmcif_metadata.py` |  |
| `src/sparsefold/output/types.py` | `src/alphafold3/model/model.py` |  |
| `src/sparsefold/output/writers.py` | `src/alphafold3/model/post_processing.py` |  |
| `src/sparsefold/schema/atom_layout.py` | `src/alphafold3/model/atom_layout/atom_layout.py` |  |
| `src/sparsefold/schema/batch.py` | `src/alphafold3/model/feat_batch.py` |  |
| `src/sparsefold/schema/features.py` | `src/alphafold3/model/features.py` |  |
| `src/sparsefold/weights/records.py` | `src/alphafold3/model/params.py` |  |

## External components

Official AF3 parsing, chemical components, data search, featurization and structure
objects are provided by a separately installed, provenance-checked AF3 package;
its native extensions and third-party implementations are not vendored here.
Torch (BSD-3-Clause), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), einops (MIT),
absl-py (Apache-2.0), ml-dtypes (Apache-2.0), zstandard (BSD-3-Clause) and optional
Biopython and tmtools retain their respective package licenses. Distribution of a
binary environment requires its own dependency-license inventory.
Vecops is an optional external MIT component maintained by HenryRenYz. The model
kernel recipes (source definitions used to build native kernels) included here
have their own per-file SPDX headers. The component source is https://github.com/HenryRenYz/vecops. Its active
installation revision is declared in `pyproject.toml`; tested identities and
coverage are recorded separately in `docs/validation.md`. The runtime checks the
required API before use.

## Parameters, databases, inputs and outputs

No model parameters or private validation data are included. Model parameters and
converted checkpoints remain subject to their separate terms:
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.
Each user must supply legitimately obtained local parameters and applicable data.
Generated outputs retain applicable upstream metadata and output notices:
https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md.
Source licenses do not grant rights to publish weights, databases or third-party
inputs. Validation files stay outside the release source tree.
