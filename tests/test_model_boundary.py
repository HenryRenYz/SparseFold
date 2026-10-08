# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
"""Private numerical core and dynamic allocation contract."""
import os
import subprocess
import sys
import torch
from sparsefold.model.layers.primitives import Linear, LayerNorm, one_hot
from sparsefold.model.geometry import Vec3Array
from sparsefold.runtime import config

def test_model_import_and_numeric_geometry_with_official_namespace_blocked():
    script = """
import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'alphafold3' or fullname.startswith('alphafold3.'):
   raise ImportError('External AF3 forbidden in model closure: ' + fullname)
sys.meta_path.insert(0, Block())
from sparsefold.model import SparseFoldModel
from sparsefold.model.geometry import Vec3Array, Rot3Array
from sparsefold.schema.batch import ModelBatch
assert SparseFoldModel.Config().num_recycles == 10
assert Vec3Array.zeros((2,)).to_array().shape == (2,3)
assert not any(n == 'alphafold3' or n.startswith('alphafold3.') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr

def test_core_allocation_resolves_device_after_import():
    try:
        config.configure_runtime(device='meta')
        assert Linear(2,3).weight.device.type == 'meta'
        assert LayerNorm(3).weight.device.type == 'meta'
        # Input-local one-hot never inherits a different configured model device.
        assert one_hot(torch.tensor([0,1]), 3).device.type == 'cpu'
    finally:
        config.reset()

def test_geometry_zero_dtype():
    value = Vec3Array.zeros((2,), dtype=torch.float64).to_array()
    assert value.dtype == torch.float64
    assert torch.equal(value, torch.zeros((2,3), dtype=torch.float64))

def test_core_rejects_host_metadata_before_any_computation():
    import pytest
    from sparsefold.model import SparseFoldModel
    class Unstarted:
        def reset(self):
            raise AssertionError('Core should not start processing host metadata')
    with pytest.raises(ValueError, match='separate output context'):
        SparseFoldModel.forward(Unstarted(), {'cleaned_struc': object()})


def test_pseudo_beta_cpu_vocabulary_with_cuda_input():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    from sparsefold.model.geometry.scoring import pseudo_beta_fn_torch
    positions = torch.arange(18, dtype=torch.float32, device='cuda').reshape(2,3,3)
    masks = torch.tensor([[1,0,1],[1,1,0]], device='cuda', dtype=torch.bool)
    beta, beta_mask = pseudo_beta_fn_torch(torch.tensor([0,1],device='cuda'), positions, masks, torch.tensor([2,1]))
    torch.testing.assert_close(beta, positions[torch.arange(2,device='cuda'), torch.tensor([2,1],device='cuda')])
    torch.testing.assert_close(beta_mask, torch.ones(2,device='cuda'))
