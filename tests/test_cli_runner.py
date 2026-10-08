# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""CPU tests for the actual CLI-to-model and official-provider boundaries."""
import contextlib
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from absl import flags
from absl.testing import flagsaver
from sparsefold.cli import predict
from sparsefold.integrations.af3.provider import APACHE_REVISION, PreparedExample
from sparsefold.model.layers import primitives
from sparsefold.schema.batch import ModelBatch
from sparsefold.schema.context import OutputContext
from sparsefold.weights import Checkpoint
from test_schema import numeric_fixture

@pytest.fixture(autouse=True)
def inference_flags():
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(['test'], known_only=True)
    with flagsaver.flagsaver(precompile=False, precompile_only=False,
                              memory_plan_fast_mib=0,
                              vecops_workspace_fast_mib=0,
                              precompile_graph_path=None):
        yield

class TinyModel(primitives.Module):
    constructions = 0
    def __init__(self, config, distributed=False):
        super().__init__()
        assert not torch.is_inference_mode_enabled()
        type(self).constructions += 1
        self.name = 'diffuser'
        self.token_projection = primitives.Linear(1,1,bias=False,device='cpu',name='legacy_projection')
    def forward(self, batch):
        assert torch.is_inference_mode_enabled()
        assert isinstance(batch,ModelBatch)
        assert all(isinstance(v,torch.Tensor) for v in batch.as_data_dict().values())
        assert not hasattr(batch,'empty_output_struc')
        output=self.token_projection(torch.ones((2,1))).to(torch.bfloat16)
        return {'value':output, 'nested':{'value':output}, 'distogram':{}}


def test_runner_real_strict_loader_caches_model_and_normalizes_bfloat16(monkeypatch,tmp_path):
    import sparsefold.weights as weights
    cp=Checkpoint({'diffuser/legacy_projection':{'weights':torch.tensor([[2.]],dtype=torch.bfloat16)}},b'synthetic-id',{})
    monkeypatch.setattr(weights,'read_checkpoint',lambda path:cp)
    monkeypatch.setattr(predict.model,'SparseFoldModel',TinyModel)
    monkeypatch.setattr(predict.dist,'is_enabled',lambda:False)
    monkeypatch.setattr(predict.dist,'is_primary',lambda:True)
    prepared=[]
    monkeypatch.setattr(predict.acceleration,'prepare',lambda model:prepared.append(model))
    TinyModel.constructions=0
    runner=predict.ModelRunner(object(),'cpu',tmp_path)
    batch=ModelBatch.from_data_dict(numeric_fixture())
    first=runner.run_inference(batch)
    second=runner.run_inference(batch)
    np.testing.assert_array_equal(first['value'],np.full((2,1),2,dtype=np.float32))
    assert first['nested']['value'].dtype==np.float32
    assert first['__identifier__']==b'synthetic-id'
    assert second['__identifier__']==b'synthetic-id'
    assert TinyModel.constructions==1 and len(prepared)==1
    assert runner.weight_mapping_report.complete
    assert runner.weight_mapping_report.entries[0].target_key=='token_projection.weight'
    assert runner.checkpoint.params['diffuser/legacy_projection']['weights'].device.type=='cpu'


def test_prediction_keeps_host_context_out_of_model_and_dispatches_every_seed(monkeypatch):
    batch=ModelBatch.from_data_dict(numeric_fixture())
    contexts=[OutputContext(empty_output_struc=object()),OutputContext(empty_output_struc=object())]
    fold=SimpleNamespace(name='synthetic',rng_seeds=(11,19))
    examples=[PreparedExample(seed,batch,ctx,{}) for seed,ctx in zip(fold.rng_seeds,contexts)]
    provider=SimpleNamespace(featurise=lambda *args,**kwargs:examples)
    monkeypatch.setattr(predict,'_provider',lambda:provider)
    monkeypatch.setattr(predict.dist,'is_enabled',lambda:False)
    monkeypatch.setattr(predict.dist,'is_primary',lambda:True)
    seeds=[];calls=[]
    monkeypatch.setattr(predict.util,'manual_seed',seeds.append)
    def decode(**kwargs):
        calls.append(kwargs)
        assert kwargs['batch'] is batch
        assert kwargs['provider'] is provider
        return [SimpleNamespace(metadata={'token_chain_ids':['A']*3})]
    monkeypatch.setattr(predict,'decode_predictions',decode)
    runner=predict.ModelRunner(object(),'cpu','unused')
    model_inputs=[]
    def infer(example):
        assert isinstance(example,ModelBatch)
        model_inputs.append(example)
        return {'single_embeddings':np.ones((3,2)), 'distogram':{'distogram':np.ones((3,3,2))}}
    monkeypatch.setattr(runner,'run_inference',infer)
    results=predict.predict_structure(fold,runner,buckets=(256,))
    assert seeds==[11,19] and len(model_inputs)==2
    assert [result.seed for result in results]==[11,19]
    assert [call['context'] for call in calls]==contexts
    assert all(result.embeddings['single_embeddings'].dtype==np.float16 for result in results)
    assert all(result.distogram.shape==(3,3,2) for result in results)


def test_terms_come_from_verified_checkout_root(monkeypatch,tmp_path):
    text='Synthetic terms; no model parameters.'
    (tmp_path/'OUTPUT_TERMS_OF_USE.md').write_text(text)
    monkeypatch.setattr(predict,'_provider',lambda:SimpleNamespace(verify=lambda:{'source_root':str(tmp_path)}))
    assert predict._output_terms()==text
    (tmp_path/'OUTPUT_TERMS_OF_USE.md').unlink()
    with pytest.raises(FileNotFoundError,match='checkout lacks output terms'):
        predict._output_terms()


def test_default_pin_matches_reviewed_provider():
    assert flags.FLAGS['af3_revision'].default==APACHE_REVISION


def test_reference_backend_does_not_silently_ignore_native_plan(monkeypatch):
    monkeypatch.setattr(predict.compiler.config,'uses_vecops',lambda:False)
    with flagsaver.flagsaver(precompile=True,memory_plan_fast_mib=1):
        with pytest.raises(ValueError,match='require.*vecops'):
            predict.main([])
    with flagsaver.flagsaver(precompile=True,precompile_only=True):
        with pytest.raises(ValueError,match='requires a backend with native precompilation'):
            predict.main([])


def test_tool_discovery_prefers_interpreter_environment(monkeypatch,tmp_path):
    from sparsefold.cli import flags as cli_flags
    binary=tmp_path/'bin'/'jackhmmer'
    binary.parent.mkdir();binary.write_text('synthetic executable marker')
    binary.chmod(0o755)
    monkeypatch.setattr(cli_flags.sys,'prefix',str(tmp_path))
    monkeypatch.setattr(cli_flags.shutil,'which',lambda name:'/system/'+name)
    assert cli_flags._find_binary('jackhmmer')==str(binary)
    assert cli_flags._find_binary('nhmmer')=='/system/nhmmer'
    binary.chmod(0o644)
    assert cli_flags._find_binary('jackhmmer')=='/system/jackhmmer'


def test_explicit_tool_path_overrides_discovery_default():
    with flagsaver.flagsaver(jackhmmer_binary_path='/explicit/patched-jackhmmer'):
        assert flags.FLAGS['jackhmmer_binary_path'].value=='/explicit/patched-jackhmmer'
