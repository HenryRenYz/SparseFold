# Model and interface notes

The numerical model accepts a `ModelBatch`. Host-side objects needed to produce output structures are kept separately in `OutputContext`. The adapter provides official AF3 parsing, searches, featurization, and chemical resources. The numerical model, tensor geometry, layers, and checkpoint reader do not depend on the `alphafold3` Python namespace.

## Parameter loading

Internal module names are maintained separately from checkpoint Haiku scopes. The strict loader maps those scopes and stack slices to standard SparseFold parameter keys. Missing parameters, shape mismatches, and unused checkpoint slices are errors. The validated parameter file covered all 3,761 standard keys, without discarding surplus parameters to obtain a successful load.

Casting, transposition, and packing for native execution happen after strict loading. The resulting tensors are `nonpersistent buffers`, not new `Parameters`, and do not change checkpoint keys. Renaming a module requires updating its explicit mapping rather than replacing scope strings in the parameter file.

## Learned normalization scales

The official checkpoint supplies three learned cross-attention pair-normalization scales in the atom encoder, atom decoder, and trunk atom-conditioning module. Their LayerNorms use an affine scale without bias. The model retains and strictly loads these values instead of treating the learned scaling as a constant.

## Features, devices, and output

Atom gather indices address the flattened logical source layout. Some call-site views differ from the declared shape but have the same flattened volume. Schema validation checks this equivalence; additional padding is permitted only when the relevant trailing strides remain unchanged. Gathers from real features have been compared with an independent flat-index reference.

Template pseudo-beta and rigid-group lookup tables retain their original values. Lookups select a tensor on the input device. Allocations resolve their device after runtime configuration, and one-hot tensors follow the input device. Autocast uses the device type while allocations retain the selected device index. Before serialization, CUDA/BF16 results are detached and converted to CPU. Padding is excluded from the final chain-pair confidence statistics.

These interfaces and correctness constraints are separate from licensing. AF3-derived files retain their original copyrights and Apache-2.0. Parameters and databases remain subject to their own terms; see [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).
