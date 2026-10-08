# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
import json

import pytest
import torch

from sparsefold.model.layers import primitives as nn
from sparsefold.weights import Checkpoint, WeightLoadError, load_checkpoint


class Tree(nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.name = 'diffuser'
        self.token_projection = nn.Linear(2, 3, bias=True, device='cpu', dtype=torch.float32, name='legacy_embedding')
        self.normalization = nn.LayerNorm(3, device='cpu', name='legacy_norm')
        self.semantic_layers = nn.LayerStack([nn.Linear(3, 3, device='cpu', name='legacy_block') for _ in range(count)], name='legacy_stack')


def params(stack_count=2):
    return {
        'diffuser/legacy_embedding': {'weights':torch.arange(6).reshape(2,3).to(torch.bfloat16), 'bias':torch.zeros(3)},
        'diffuser/legacy_norm': {'scale':torch.ones(3), 'offset':torch.zeros(3)},
        'diffuser/__layer_stack_no_per_layerlegacy_stack/legacy_block': {'weights':torch.arange(stack_count*9).reshape(stack_count,3,3).float()},
    }


def test_scope_mapping_survives_semantic_renames_and_slices():
    model = Tree()
    checkpoint = Checkpoint(params(), b'model-id', {'diffuser/legacy_embedding':'diffuser/~/legacy_embedding'})
    report = load_checkpoint(model, checkpoint)
    assert report.complete and report.identifier == b'model-id'
    assert model.token_projection.weight.shape == (2,3)  # JAX layout; never transpose blindly.
    assert model.token_projection.weight.dtype == torch.float32
    entries = {entry.target_key:entry for entry in report.entries}
    assert entries['token_projection.weight'].source_scope == 'diffuser/~/legacy_embedding'
    assert entries['token_projection.weight'].source_name == 'weights'
    assert entries['token_projection.weight'].transforms == ('dtype:torch.bfloat16->torch.float32',)
    assert entries['normalization.weight'].source_name == 'scale'
    assert entries['normalization.bias'].source_name == 'offset'
    for index in range(2):
        entry = entries[f'semantic_layers.{index}.weight']
        assert entry.stack_indices == (index,) and entry.source_shape == (2,3,3)
        assert torch.equal(model.semantic_layers[index].weight, params()['diffuser/__layer_stack_no_per_layerlegacy_stack/legacy_block']['weights'][index])
    assert report.to_dict()['identifier'] == b'model-id'.hex()


@pytest.mark.parametrize('change,match', [
    (lambda p:p['diffuser/legacy_embedding'].pop('bias'), 'No bias'),
    (lambda p:p.update({'diffuser/unused':{'weights':torch.zeros(2)}}), 'unexpected'),
    (lambda p:p['diffuser/legacy_embedding'].update(weights=torch.zeros(3,2)), 'shape mismatch'),
])
def test_strict_failure_rolls_back(change, match):
    model = Tree()
    original = dict(model.named_parameters())
    source = params()
    change(source)
    with pytest.raises(WeightLoadError, match=match):
        load_checkpoint(model, source)
    assert all(dict(model.named_parameters())[key] is value for key,value in original.items())
    # Hook must have been removed; ordinary registration still works.
    torch.nn.Linear(2,3)


def test_uncovered_stack_rejected():
    model = Tree(count=1)
    with pytest.raises(WeightLoadError, match='stack slices') as exc:
        load_checkpoint(model, params())
    assert exc.value.report and not exc.value.report.complete


def test_non_strict_reports_unused_keys():
    source = params()
    source['diffuser/unused'] = {'weights':torch.zeros(2)}
    report = load_checkpoint(Tree(), source, strict=False)
    assert report.unexpected_source_keys == ('diffuser/unused/weights',)
    assert not report.complete


class SpecialScopes(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'diffuser'
        self.renamed_atom = Underscore()
        self.output_b = torch.nn.Parameter(torch.empty(3))
        self.output_w = torch.nn.Parameter(torch.empty(2,2,3))

    def set_parameters(self, visitor, **kwargs):
        super().set_parameters(visitor, **kwargs)
        self.output_b = torch.nn.Parameter(torch.as_tensor(visitor['output_b'], device='cpu'), requires_grad=False)
        self.output_w = torch.nn.Parameter(torch.as_tensor(visitor['output_w'], device='cpu'), requires_grad=False)


class Underscore(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'atom'
        self.renamed_projection = nn.Linear(2,3,device='cpu',name='projection')

    def set_parameters(self, visitor, **kwargs):
        super().set_parameters(visitor, sep='_', **kwargs)


def test_atom_separator_and_outer_product_parameters():
    source = {'diffuser/atom_projection':{'weight':torch.ones(2,3)},
              'diffuser':{'output_b':torch.zeros(3), 'output_w':torch.ones(2,2,3)}}
    report = load_checkpoint(SpecialScopes(), source)
    keys = {entry.target_key:entry for entry in report.entries}
    assert keys['renamed_atom.renamed_projection.weight'].source_scope == 'diffuser/atom_projection'
    assert keys['output_w'].source_name == 'output_w'


class Duplicate(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'diffuser'
        self.a = torch.nn.Parameter(torch.empty(2))
        self.b = torch.nn.Parameter(torch.empty(2))

    def set_parameters(self, visitor, **kwargs):
        self.a = torch.nn.Parameter(visitor['weights'])
        self.b = torch.nn.Parameter(visitor['weights'])


def test_duplicate_source_rejected():
    with pytest.raises(WeightLoadError, match='Duplicate'):
        load_checkpoint(Duplicate(), {'diffuser':{'weights':torch.ones(2)}})


class BadTransform(Duplicate):
    def set_parameters(self, visitor, **kwargs):
        self.a = torch.nn.Parameter(visitor['weights'] * 2)


def test_unlisted_transform_rejected():
    with pytest.raises(WeightLoadError, match='unsupported transform'):
        load_checkpoint(BadTransform(), {'diffuser':{'weights':torch.ones(2)}})


class BackendCast(Tree):
    def set_parameters(self, visitor, **kwargs):
        super().set_parameters(visitor, **kwargs)
        self.token_projection.weight = torch.nn.Parameter(self.token_projection.weight.to(torch.bfloat16).contiguous())


def test_loaded_parameter_dtype_conversion_is_not_duplicate_source_read():
    source = params()
    source['diffuser/legacy_embedding']['weights'] = source['diffuser/legacy_embedding']['weights'].float()
    model = BackendCast()
    report = load_checkpoint(model, source)
    assert report.complete and model.token_projection.weight.dtype == torch.bfloat16
    [entry] = [e for e in report.entries if e.target_key == 'token_projection.weight']
    assert entry.transforms == ('dtype:torch.float32->torch.bfloat16',)


class PrevalidatedScopes(SpecialScopes):
    def set_parameters(self, visitor, **kwargs):
        # Architecture shape checks may read a tensor before the actual assignment.
        assert tuple(visitor['output_b'].shape) == (3,)
        assert tuple(visitor['output_w'].shape) == (2,2,3)
        super().set_parameters(visitor, **kwargs)


def test_shape_validation_reads_preserve_strict_assignment_coverage():
    source = {'diffuser/atom_projection': {'weight':torch.ones(2,3)},
              'diffuser': {'output_b':torch.zeros(3), 'output_w':torch.ones(2,2,3)}}
    report = load_checkpoint(PrevalidatedScopes(),source)
    assert report.complete and len(report.entries) == 3


class UnassignedRead(Duplicate):
    def set_parameters(self, visitor, **kwargs):
        visitor['validation_only']
        self.a = torch.nn.Parameter(visitor['weights'])


def test_distinct_read_without_assignment_is_rejected_even_nonstrict():
    with pytest.raises(WeightLoadError,match='without a mapped'):
        load_checkpoint(UnassignedRead(), {'diffuser': {'weights':torch.ones(2),
                         'validation_only':torch.zeros(2)}}, strict=False)


@pytest.mark.parametrize('unused_source', [False, True])
def test_weight_map_json_version_and_roundtrip_preserve_mapping_semantics(unused_source):
    source = params()
    if unused_source:
        source['diffuser/unused'] = {'weights':torch.zeros(2)}
    checkpoint = Checkpoint(source, b'\x00\xffmodel-id',
                            {'diffuser/legacy_embedding':'diffuser/~/legacy_embedding'})
    report = load_checkpoint(Tree(), checkpoint, strict=not unused_source)
    decoded = json.loads(json.dumps(report.to_dict()))
    assert decoded['schema_id'] == 'sparsefold.weight-map.v1'
    assert decoded['source_format'] == 'af3-record-stream.v1'
    assert decoded['tensor_storage'] == 'jax-logical-layout'
    assert decoded['complete'] is (not unused_source)
    assert bytes.fromhex(decoded['identifier']) == checkpoint.identifier
    assert decoded['unexpected_source_keys'] == (['diffuser/unused/weights'] if unused_source else [])
    entries = {entry['target_key']:entry for entry in decoded['entries']}
    assert entries['token_projection.weight']['source_scope'] == 'diffuser/~/legacy_embedding'
    assert entries['token_projection.weight']['source_shape'] == [2,3]
    assert entries['token_projection.weight']['target_shape'] == [2,3]
    assert entries['token_projection.weight']['transforms'] == ['dtype:torch.bfloat16->torch.float32']
    for index in range(2):
        entry = entries[f'semantic_layers.{index}.weight']
        assert entry['stack_indices'] == [index]
        assert entry['source_shape'] == [2,3,3]
        assert entry['target_shape'] == [3,3]
    # Encoding the decoded public schema must not discard optional status fields.
    assert json.loads(json.dumps(decoded)) == decoded


@pytest.mark.parametrize('inference_source',[False,True])
def test_inference_mode_call_preserves_dtype_provenance_and_strict_coverage(inference_source):
    # Source float32 -> registered bfloat16 exercises the failing GPU direction.
    if inference_source:
        with torch.inference_mode():
            source = params()
            source['diffuser/legacy_embedding']['weights'] = torch.arange(6).reshape(2,3).float()
    else:
        source = params()
        source['diffuser/legacy_embedding']['weights'] = torch.arange(6).reshape(2,3).float()
    with torch.inference_mode():
        model = Tree()
        model.token_projection.dtype = torch.bfloat16
        report = load_checkpoint(model,source)
        assert torch.is_inference_mode_enabled()  # caller state restored
    assert report.complete
    assert model.token_projection.weight.dtype == torch.bfloat16
    assert not any(parameter.is_inference() for parameter in model.parameters())
    entry = next(e for e in report.entries if e.target_key == 'token_projection.weight')
    assert 'dtype:torch.float32->torch.bfloat16' in entry.transforms
    assert ('inference_tensor_copy' in entry.transforms) is inference_source
    assert torch.equal(model.token_projection.weight,source['diffuser/legacy_embedding']['weights'].to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
@pytest.mark.parametrize('inference_source',[False,True])
def test_cuda_inference_mode_float32_to_bfloat16_mapping(inference_source):
    with torch.inference_mode(inference_source):
        source = {scope:{name:tensor.float().to('cuda') for name,tensor in tensors.items()}
                  for scope,tensors in params().items()}
    with torch.inference_mode():
        model = Tree().to('cuda')
        model.token_projection.dtype = torch.bfloat16
        report = load_checkpoint(model,source)
        assert torch.is_inference_mode_enabled()
    assert report.complete and model.token_projection.weight.device.type == 'cuda'
    entry = next(e for e in report.entries if e.target_key == 'token_projection.weight')
    assert 'dtype:torch.float32->torch.bfloat16' in entry.transforms
    assert ('inference_tensor_copy' in entry.transforms) is inference_source
    assert torch.equal(model.token_projection.weight,source['diffuser/legacy_embedding']['weights'].to(torch.bfloat16))
