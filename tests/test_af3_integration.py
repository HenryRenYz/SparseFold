# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Opt-in real enriched-input schema test; performs no database search or inference."""
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from sparsefold.integrations.af3.provider import AF3Provider, APACHE_REVISION
from sparsefold.integrations.af3.artifact import save_prepared, load_prepared, validate_context
from sparsefold.schema.validation import REQUIRED_KEYS


INPUTS = [Path(value) for value in os.environ.get('SPARSEFOLD_TEST_AF3_INPUTS', os.environ.get('SPARSEFOLD_TEST_AF3_INPUT', '')).split(os.pathsep) if value]


@pytest.mark.parametrize('input_path', INPUTS or [None], ids=lambda value: value.stem if value is not None else 'not-configured')
@pytest.mark.skipif(not INPUTS,reason='Explicit local enriched AF3 input required')
def test_fixed_official_real_features_and_context_roundtrip(tmp_path,input_path):
    source=os.environ.get('SPARSEFOLD_TEST_AF3_SOURCE')
    provider=AF3Provider(source_root=source)
    fingerprint=provider.verify()
    assert fingerprint['revision']==APACHE_REVISION
    assert fingerprint['native_artifacts']
    inputs=provider.read_inputs(input_path)
    assert inputs
    examples=provider.featurise(inputs[0],buckets=(256,512,768,1024))
    assert len(examples)==len(inputs[0].rng_seeds)
    for example in examples:
        numeric=example.model_batch.as_data_dict()
        assert set(numeric)==set(REQUIRED_KEYS)
        assert all(isinstance(value,torch.Tensor) for value in numeric.values())
        _assert_all_numerical_gathers(example.model_batch)
        assert 'ccd_sha256' in example.provider_fingerprint
        print({'input': input_path.name, 'seed': example.seed, 'padded_tokens': example.model_batch.num_res, 'active_tokens': int(example.model_batch.token_features.mask.sum().item()), 'dense_atoms_shape': list(example.model_batch.predicted_structure_info.atom_mask.shape), 'numeric_fields': len(numeric)})
        validate_context(example.output_context,example.model_batch)
        stem=tmp_path/f'seed_{example.seed}'
        npz,meta=save_prepared(example,stem)
        with np.load(npz,allow_pickle=False) as archive:
            assert all(not archive[key].dtype.hasobject for key in archive.files)
        restored=load_prepared(stem,provider)
        for key,value in numeric.items():
            torch.testing.assert_close(value,restored.model_batch.as_data_dict()[key],rtol=0,atol=0)
        for name in ['token_atoms_layout','flat_output_layout','polymer_ligand_bonds','ligand_ligand_bonds']:
            before=getattr(example.output_context,name);after=getattr(restored.output_context,name)
            if before is None:
                assert after is None
                continue
            for field in ['atom_name','res_id','chain_id','atom_element','res_name','chain_type']:
                v_before=getattr(before,field);v_after=getattr(after,field)
                if v_before is None: assert v_after is None
                else: np.testing.assert_array_equal(v_before,v_after)
        for name in ['cleaned_struc','empty_output_struc']:
            before=getattr(example.output_context,name);after=getattr(restored.output_context,name)
            if before is None: assert after is None;continue
            for field in ['chain_id','res_id','res_name','atom_name','atom_element']:
                np.testing.assert_array_equal(getattr(before,field),getattr(after,field))


def _assert_all_numerical_gathers(batch):
    """Exercise every numerical gather using real source values and flat oracle."""
    from sparsefold.schema.atom_layout import convert
    from sparsefold.runtime import tree
    from sparsefold.schema.atom_layout import GatherInfo
    for info in tree.leaves(batch, is_leaf=lambda value:isinstance(value,GatherInfo)):
        if not isinstance(info,GatherInfo):continue
        shape=tuple(int(value) for value in info.input_shape)
        source=torch.arange(int(np.prod(shape)),dtype=torch.int64).reshape(shape)
        actual=convert(info,source,layout_axes=tuple(range(len(shape))))
        expected=source.flatten()[info.gather_idxs.long()]
        expected=expected.masked_fill(~info.gather_mask,0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # This caller is present in both fixed main and current official AF3:
    # queries_to_keys indexes flattened dense reference-space identifiers.
    info=batch.atom_cross_att.queries_to_keys
    source=batch.ref_structure.ref_space_uid
    actual=convert(info,source,layout_axes=(-2,-1))
    expected=source.flatten()[info.gather_idxs.long()].masked_fill(~info.gather_mask,0)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Standard dense -> queries -> keys position path with feature axes.
    queries=convert(batch.atom_cross_att.token_atoms_to_queries,batch.ref_structure.positions,layout_axes=(-3,-2))
    keys=convert(info,queries,layout_axes=(-3,-2))
    expected=queries.reshape(-1,3)[info.gather_idxs.long()].masked_fill(~info.gather_mask[...,None],0)
    torch.testing.assert_close(keys,expected,rtol=0,atol=0)
