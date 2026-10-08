# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Bounded CPU/CUDA regression checks for inference chunking."""
import pytest
import torch

from sparsefold.model.layers import mapping
from sparsefold.runtime import config


@pytest.fixture(autouse=True)
def restore_runtime():
  previous = config.get_runtime_options()
  yield
  config.configure_runtime(**previous)


def on_device(device):
  if device == "cuda" and not torch.cuda.is_available():
    pytest.skip("CUDA unavailable")
  config.configure_runtime(device=device, backend="torch", enable_compile=False,
                           enable_sharding=True, logs=set())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shard_size", [3, 20])
@pytest.mark.parametrize("axis", [0, -2])
def test_sharded_apply_nested_output_matches_direct_with_remainder(device, shard_size, axis):
  on_device(device)
  values = torch.arange(35, device=device, dtype=torch.float32).reshape(7, 5) / 35
  offsets = torch.ones_like(values) / 7
  seen = []
  def transform(x, y, scale=1.):
    seen.append(x.shape[axis])
    return {"linear": (x + y) * scale, "stats": (x.square(), x.sin())}
  actual = mapping.sharded_apply(transform, shard_size=shard_size, in_axes=axis,
                                 out_axes=axis)(values, offsets, scale=0.25)
  assert seen == ([3, 3, 1] if shard_size == 3 else [7])
  expected = transform(values, offsets, scale=0.25)
  torch.testing.assert_close(actual, expected)
  assert actual["linear"].device == values.device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sharding_configuration_and_explicit_override_bound_calls(device):
  on_device(device)
  values = torch.arange(10, device=device).reshape(5, 2)
  seen = []
  def transform(x):
    seen.append(x.shape[0])
    return x + 1
  config.configure_runtime(device=device, enable_sharding=False)
  direct = mapping.sharded_apply(transform, shard_size=2)(values)
  assert seen == [5]
  seen.clear()
  forced = mapping.sharded_apply(transform, shard_size=2, enabled=True)(values)
  assert seen == [2, 2, 1]
  torch.testing.assert_close(forced, direct)
  seen.clear()
  config.configure_runtime(device=device, enable_sharding=True)
  disabled = mapping.sharded_apply(transform, shard_size=2, enabled=False)(values)
  assert seen == [5]
  torch.testing.assert_close(disabled, direct)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("partitioned", [False, True])
def test_inference_subbatch_preserves_shared_args_and_output_axes(device, partitioned):
  on_device(device)
  count = 12 if partitioned else 7
  values = torch.arange(2 * count * 4, device=device, dtype=torch.float32).reshape(2, count, 4) / 100
  residual = torch.ones_like(values) / 10
  weight = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3) / 12
  seen = []
  def module(x, skip, shared_weight):
    seen.append(x.shape[1])
    assert shared_weight is weight
    # Deliberately move output batch dimension from 1 to 0.
    return ((x + skip) @ shared_weight).transpose(0, 1)
  actual = mapping.inference_subbatch(
      module, 3, [values, residual], [weight], input_subbatch_dim=1,
      output_subbatch_dim=0, input_subbatch_dim_is_partitioned=partitioned)
  assert seen == ([3] * 4 if partitioned else [3, 3, 1])
  expected = ((values + residual) @ weight).transpose(0, 1)
  torch.testing.assert_close(actual, expected)
  assert actual.device == values.device
