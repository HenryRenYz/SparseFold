# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Small CUDA correctness tests; no weights, compilation, native JIT or inference."""
from types import SimpleNamespace

import pytest
import torch

from sparsefold import backends
from sparsefold.backends.torch.attention import dot_product_attention
from sparsefold.runtime import cache, config, ops, trace, util
from sparsefold.model.layers import primitives

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture(autouse=True)
def gpu_reference():
  config.configure_runtime(device="cuda", backend="torch", sdpa_implementation="math",
                           enable_compile=False, enable_sharding=False, logs=set())
  backends.init()
  yield
  config.reset()
  backends.init()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_attention_math_matches_cpu(dtype):
  gen = torch.Generator().manual_seed(610)
  q = torch.randn(2, 7, 3, 16, generator=gen).to(dtype)
  k = torch.randn(2, 9, 3, 16, generator=gen).to(dtype)
  v = torch.randn(2, 9, 3, 12, generator=gen).to(dtype)
  bias = torch.randn(2, 3, 7, 9, generator=gen).to(dtype)
  mask_k = torch.tensor([[True, False, True, True, True, False, True, False, True]]).expand(2, 9)
  expected = dot_product_attention(q, k, v, mask_k=mask_k, bias=bias, scale=0.3)
  actual = dot_product_attention(q.cuda(), k.cuda(), v.cuda(), mask_k=mask_k.cuda(), bias=bias.cuda(), scale=0.3)
  assert actual.device.type == "cuda"
  torch.testing.assert_close(actual.cpu(), expected, rtol=0.02 if dtype == torch.bfloat16 else 2e-5,
                             atol=0.02 if dtype == torch.bfloat16 else 2e-6)


def test_cuda_bf16_flash_matches_math():
  gen = torch.Generator(device="cuda").manual_seed(611)
  q = torch.randn(2, 16, 4, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
  k = torch.randn(2, 16, 4, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
  v = torch.randn(2, 16, 4, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
  expected = dot_product_attention(q, k, v)
  config.configure_runtime(device="cuda", sdpa_implementation="flash")
  actual = dot_product_attention(q, k, v)
  torch.testing.assert_close(actual.float(), expected.float(), rtol=0.025, atol=0.01)


def test_configure_after_import_updates_gpu_device_and_precision(tmp_path):
  config.configure_runtime(device="cpu")
  assert util.DEFAULT_DEVICE == "cpu"
  opts = SimpleNamespace(
      model={"use_cpu": False}, log=SimpleNamespace(trace_output=None, channels=[]),
      paths=SimpleNamespace(output=str(tmp_path)),
      runtime=SimpleNamespace(cache=True, compile=False, sharding=False),
      native=SimpleNamespace(impl="none", ops=["all"], matmul_dispatch=False,
                            sparse_attention={}, gsa_index_reuse=True),
      precision=SimpleNamespace(sdpa="flash", diffusion="bf16"))
  config.configure(opts)
  assert config.device == util.DEFAULT_DEVICE == "cuda"
  assert config.sdpa_implementation == "flash"
  assert config.diffusion_precision == "bf16"
  assert config.acclib_implementation == "torch"
  seen = []
  def body(index, carry):
    seen.append(index.device.type)
    return carry + index
  actual = ops.indexed_stack_scan(body, torch.tensor(0, device="cuda"), length=4)
  assert actual.item() == 6
  assert seen == ["cuda"] * 4
  config.configure_runtime(device="cpu")
  assert config.device == util.DEFAULT_DEVICE == "cpu"


def test_cuda_cache_and_execution_context_use_tensor_device():
  values = cache.ValueCache()
  source = torch.ones(3, device="cuda")
  with backends.execution_context():
    result = values.get(source, name="square", factory=lambda: source.square())
    assert result.device.type == "cuda"
    assert values.get(source, name="square", factory=lambda: None) is result
  values.reset()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_layer_defaults_follow_configuration_after_import(dtype):
  linear = primitives.Linear(5, 7, bias=True, dtype=dtype)
  norm = primitives.LayerNorm(7, dtype=torch.float32)
  assert linear.weight.device.type == norm.weight.device.type == "cuda"
  with torch.no_grad():
    linear.weight.copy_(torch.arange(35, device="cuda").reshape(5, 7) / 35)
    linear.bias.copy_(torch.arange(7, device="cuda") / 7)
    norm.weight.fill_(1)
    norm.bias.zero_()
  value = torch.arange(10, device="cuda").reshape(2, 5).to(dtype) / 10
  residual = torch.ones(2, 7, device="cuda", dtype=dtype)
  actual = linear(value, residual=residual, post_op="silu")
  expected = torch.nn.functional.silu(value @ linear.weight + linear.bias + residual)
  torch.testing.assert_close(actual, expected)
  norm_output = norm(actual)
  expected_norm = torch.nn.functional.layer_norm(actual.float(), (7,), eps=norm.eps).to(dtype)
  torch.testing.assert_close(norm_output, expected_norm)
  index = torch.tensor([0, 2, 1], device="cuda")
  torch.testing.assert_close(primitives.one_hot(index, 3), torch.nn.functional.one_hot(index, 3).float())
  config.configure_runtime(device="cpu")
  assert primitives.Linear(1, 1).weight.device.type == "cpu"
  assert primitives.LayerNorm(1).weight.device.type == "cpu"
