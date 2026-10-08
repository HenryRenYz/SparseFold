# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Public API runtime/device consistency, with a tiny optional CUDA boundary."""
import numpy as np
import pytest
import torch
from sparsefold import api, backends
from sparsefold.runtime import config, util
from sparsefold.cli import predict
from sparsefold.model.layers import primitives
from sparsefold.schema.batch import ModelBatch
from sparsefold.weights import Checkpoint
from test_schema import numeric_fixture

@pytest.fixture(autouse=True)
def restore_runtime():
    before=config.get_runtime_options()
    yield
    config.configure_runtime(**before)


def test_api_preserves_runtime_backend_and_precision_before_runner(monkeypatch,tmp_path):
    config.configure_runtime(backend='synthetic',device='cpu',diffusion_precision='bf16',
                             enable_cache=False,native_ops=['linear'])
    calls=[]
    monkeypatch.setattr(backends,'init',lambda:calls.append(config.get_runtime_options()))
    runner=api.build_model_runner(model_dir=tmp_path,device=torch.device('cpu:0'),num_recycles=3)
    assert runner._device==config.device==util.DEFAULT_DEVICE=='cpu'
    assert config.acclib_implementation=='synthetic'
    assert config.diffusion_precision=='bf16'
    assert config.enable_cache is False and config.native_ops==['linear']
    assert calls[0]['acclib_implementation']=='synthetic'
    assert calls[0]['device']=='cpu'
    assert runner._model_config.num_recycles==3
    assert '_model' not in vars(runner) and 'checkpoint' not in vars(runner)


def test_api_explicit_runtime_override_selects_reference_backend(monkeypatch,tmp_path):
    config.configure_runtime(backend='synthetic',diffusion_precision='bf16')
    monkeypatch.setattr(backends,'init',lambda:None)
    api.build_model_runner(model_dir=tmp_path,runtime_options={'backend':'torch','diffusion_precision':'fp32'})
    assert config.acclib_implementation=='torch'
    assert config.diffusion_precision=='fp32'


@pytest.mark.parametrize('device',['meta','cpu:2','not-a-device'])
def test_api_rejects_invalid_device_before_mutating_runtime(tmp_path,device):
    before=config.get_runtime_options()
    with pytest.raises(ValueError):api.build_model_runner(model_dir=tmp_path,device=device)
    assert config.get_runtime_options()==before


def test_api_rejects_unavailable_or_out_of_range_cuda(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA.*unavailable'):
        api.build_model_runner(model_dir=tmp_path,device='cuda:0')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(ValueError,match='visible device range'):
        api.build_model_runner(model_dir=tmp_path,device='cuda:1')


def test_api_rejects_ambiguous_device_and_backend_settings(tmp_path):
    with pytest.raises(ValueError,match='device argument'):
        api.build_model_runner(model_dir=tmp_path,runtime_options={'device':'cpu'})
    with pytest.raises(ValueError,match='either backend'):
        api.build_model_runner(model_dir=tmp_path,runtime_options={'backend':'torch','acclib_implementation':'torch'})
    with pytest.raises(TypeError,match='mapping'):
        api.build_model_runner(model_dir=tmp_path,runtime_options=['backend','torch'])


def test_predict_batch_requires_typed_numeric_input():
    with pytest.raises(TypeError,match='requires ModelBatch'):
        api.predict_batch(object(),{},seed=1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_api_cuda_runner_weights_inputs_and_core_allocations_share_device(monkeypatch,tmp_path):
    import sparsefold.weights as weights
    original_config=predict.model.SparseFoldModel.Config
    original_heads=predict.model.SparseFoldModel.HeadsConfig
    class TinyCudaModel(primitives.Module):
        Config=original_config
        HeadsConfig=original_heads
        def __init__(self,model_config,distributed=False):
            super().__init__()
            self.name='diffuser'
            self.token_projection=primitives.Linear(1,1,bias=False,name='legacy_projection',dtype=torch.bfloat16)
            self.scratch=torch.zeros(1,device=config.device)
        def forward(self,batch):
            assert isinstance(batch,ModelBatch)
            assert all(value.device==self.scratch.device for value in batch.as_data_dict().values())
            assert self.token_projection.weight.device==self.scratch.device
            result=self.token_projection(torch.ones((2,1),device=config.device,dtype=torch.bfloat16))
            return {'value':result.to(torch.bfloat16),'distogram':{}}
    checkpoint=Checkpoint({'diffuser/legacy_projection':{'weights':torch.tensor([[2.]])}},b'gpu-synthetic',{})
    monkeypatch.setattr(weights,'read_checkpoint',lambda path:checkpoint)
    monkeypatch.setattr(predict.model,'SparseFoldModel',TinyCudaModel)
    runner=api.build_model_runner(model_dir=tmp_path,device='cuda:0',runtime_options={'backend':'torch'},
                                  num_recycles=1,num_diffusion_samples=1)
    assert runner._device==config.device==util.DEFAULT_DEVICE=='cuda:0'
    assert torch.cuda.current_device()==0
    result=api.predict_batch(runner,ModelBatch.from_data_dict(numeric_fixture()),seed=610)
    np.testing.assert_array_equal(result['value'],np.full((2,1),2,dtype=np.float32))
    assert result['__identifier__']==b'gpu-synthetic'
    assert runner.weight_mapping_report.complete
    assert 'dtype:torch.float32->torch.bfloat16' in runner.weight_mapping_report.entries[0].transforms
