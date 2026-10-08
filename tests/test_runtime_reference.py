# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Synthetic CPU checks for backend isolation and preserved runtime semantics."""
from __future__ import annotations

import ast
import importlib.abc
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sparsefold import backends
from sparsefold.runtime import cache, compiler, config, ops, trace, util
from sparsefold.backends.torch.attention import dot_product_attention


@pytest.fixture(autouse=True)
def reference_runtime():
  config.configure_runtime(backend="torch", device="cpu", enable_compile=False,
                           enable_sharding=False, sdpa_implementation="math", logs=set())
  backends.init()
  yield
  config.reset()
  backends.init()


def test_reference_isolation_in_fresh_process():
  script = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vecops', 'alphafold3'}:
            raise AssertionError('Unexpected dependency import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import torch
from sparsefold import backends
from sparsefold.runtime import cache, compiler, config, dist, ops, trace
from sparsefold.backends.torch.attention import dot_product_attention
backends.init(); backends.init()
backends.manual_seed(18)
assert backends.materialize() == 0
assert backends.precompile(lambda: None) is None
assert not backends.is_precompiling()
with backends.execution_context():
    q=torch.ones(2,1,3)
    assert dot_product_attention(q,q,q).shape == q.shape
    assert cache.ValueCache().get(q, name='x', factory=lambda: 5) == 5
    f=compiler.mark_compile(lambda x: x + 1)
    assert f(torch.tensor(2)).item() == 3
    result=ops.indexed_stack_scan(lambda i,c: c+i, torch.tensor(0), length=4)
    assert result.item() == 6
assert not any(n.split('.')[0] in {'vecops','alphafold3'} for n in sys.modules)
"""
  subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_legacy_options_and_environment_remain_supported(monkeypatch, tmp_path):
  options = SimpleNamespace(
      model={"use_cpu": True}, log=SimpleNamespace(trace_output=None, channels=["time"]),
      paths=SimpleNamespace(output=str(tmp_path)),
      runtime=SimpleNamespace(cache=True, compile=False, sharding=False),
      native=SimpleNamespace(impl="none", ops=["all"], matmul_dispatch=False,
                             sparse_attention={}, gsa_index_reuse=True),
      precision=SimpleNamespace(sdpa="math", diffusion="fp32"))
  config.configure(options)
  assert config.acclib_implementation == "torch"
  assert util.DEFAULT_DEVICE == "cpu"
  assert trace.LOG_TIME
  assert config.native_ops == ["*"]
  config.reset()
  monkeypatch.setenv("AF3_DEVICE", "cpu")
  with pytest.warns(DeprecationWarning, match="AF3_DEVICE"):
    assert config.device == "cpu"
  monkeypatch.setenv("SPARSEFOLD_DEVICE", "cuda")
  assert config.device == "cuda"


def test_reference_seed_and_lifecycle_are_idempotent():
  backends.init(); backends.init()
  backends.manual_seed(701)
  first = torch.rand(5)
  backends.manual_seed(701)
  torch.testing.assert_close(first, torch.rand(5), rtol=0, atol=0)
  module = torch.nn.Linear(2, 3)
  assert backends.prepare(module) == {"packed": 0, "fallback": 0, "opaque": 0}
  assert backends.materialize() == 0
  assert not backends.is_precompiling()


def test_cache_key_identity_clear_reset_and_disabled_factory():
  values = cache.ValueCache()
  a = torch.ones(2); b = torch.ones(2)
  count = 0
  def factory():
    nonlocal count
    count += 1
    return torch.tensor(count)
  first = values.get(a, name="value", factory=factory)
  assert values.get(a, name="value", factory=factory) is first
  second = values.get(b, name="value", factory=factory)
  assert second is not first
  values.clear(a)
  assert values.get(b, name="value", factory=factory) is second
  assert values.get(a, name="value", factory=factory).item() == 3
  values.reset()
  assert values.get(a, name="value", factory=factory).item() == 4
  config.configure_runtime(enable_cache=False)
  assert values.get(a, name="value", factory=factory).item() == 5
  assert values.get(a, name="value", factory=factory).item() == 6


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("with_masks", [False, True])
def test_attention_preserves_mask_bias_and_batch_semantics(batch_shape, with_masks):
  gen = torch.Generator().manual_seed(808)
  q = torch.randn(*batch_shape, 3, 2, 4, generator=gen)
  k = torch.randn(*batch_shape, 5, 2, 4, generator=gen)
  v = torch.randn(*batch_shape, 5, 2, 6, generator=gen)
  bias = torch.randn(*batch_shape, 2, 3, 5, generator=gen)
  mask_q = torch.tensor([True, False, True]).expand(*batch_shape, 3) if with_masks else None
  mask_k = torch.tensor([True, False, True, True, False]).expand(*batch_shape, 5) if with_masks else None
  actual = dot_product_attention(q, k, v, scale=0.4, mask_q=mask_q, mask_k=mask_k, bias=bias)
  attn_bias = torch.zeros(*batch_shape, 1, 3, 5)
  if with_masks:
    min_val = -float(torch.finfo(q.dtype).max)
    attn_bias.masked_fill_(~mask_q[..., None, :, None], min_val)
    attn_bias.masked_fill_(~mask_k[..., None, None, :], min_val)
  attn_bias = attn_bias + bias
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2),
        attn_mask=attn_bias, scale=0.4).transpose(-3, -2)
  torch.testing.assert_close(actual, expected)


def test_attention_tensor_trace_without_bias(monkeypatch):
  monkeypatch.setattr(trace, "log_tensor", lambda **kw: kw["probs"](()))
  q = torch.ones(2, 1, 3)
  torch.testing.assert_close(dot_product_attention(q, q, q), q)


def test_native_sources_and_recipe_identity_without_importing_vecops():
  base = Path(__file__).parents[1] / "src/sparsefold/backends/vecops/kernels"
  text = (base / "registry.py").read_text()
  tree = ast.parse(text)
  recipes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "jit"]
  assert len(recipes) == 27
  for call in recipes:
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    source = ast.literal_eval(kwargs["source"])
    assert (base / source).is_file()
    assert source.startswith("resources/")
    recipe = kwargs["recipe_id"]
    if isinstance(recipe, ast.Constant):
      assert recipe.value.startswith("sparsefold-v1;")
    else:
      assert isinstance(recipe.func, ast.Name) and recipe.func.id == "_stable_math_recipe"
  assert 'return f"sparsefold-v1;' in text
  assert "alphafold3_native" not in text
  assert text.count('"sparsefold_native::') == 27


def test_optional_native_reports_incompatible_dependency(monkeypatch):
  monkeypatch.setitem(sys.modules, "vecops", SimpleNamespace(is_precompiling=lambda: False))
  config.configure_runtime(backend="vecops", device="cpu", native_ops=["linear"])
  with pytest.raises(RuntimeError, match="missing: jit, ops"):
    backends.init()
  config.configure_runtime(backend="torch")
  backends.init()


@pytest.mark.parametrize("selection", [["*"], ["all"], ["msa_attention"], ["linear", "grid_self_attn"]])
def test_native_fusion_gate_precedes_dependency_import(selection, monkeypatch):
  monkeypatch.delenv("SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION", raising=False)
  def unexpected_check():
    raise AssertionError("Unsafe fused selection must fail before checking dependency")
  monkeypatch.setattr(backends, "_check_vecops_api", unexpected_check)
  config.configure_runtime(backend="vecops", native_ops=selection)
  with pytest.raises(RuntimeError, match="Derived parameters"):
    backends.init()
  config.configure_runtime(backend="torch")
  backends.init()


def test_native_linear_policy_preserves_checkpoint_parameters():
  from sparsefold.backends.vecops.policy import validate_native_ops
  validate_native_ops(["linear"])
  validate_native_ops([])



def test_generic_native_linear_prepares_only_nonpersistent_buffers(monkeypatch):
  from sparsefold.backends.vecops import modules as native_modules
  from sparsefold.weights import load_checkpoint
  monkeypatch.setattr(native_modules.platform, "machine", lambda: "x86_64")
  monkeypatch.setattr(native_modules, "_kernels", lambda: SimpleNamespace(
      pack_linear_b=lambda value, **kwargs: value.to(torch.bfloat16).contiguous()))
  module = native_modules.LinearVecops(3, 2, bias=True, device="cpu", dtype=torch.float32,
                                      name="legacy_linear")
  report = load_checkpoint(module, {"legacy_linear": {
      "weights": torch.arange(6).reshape(3, 2).float(), "bias": torch.zeros(2)}})
  assert report.complete
  parameters = dict(module.named_parameters())
  assert module.prepare_native()
  assert all(dict(module.named_parameters())[name] is value for name, value in parameters.items())
  assert set(module.state_dict()) == {"weight", "bias"}
  assert {"_packed_weight", "_native_bias"} <= module._non_persistent_buffers_set
  assert module._packed_weight.dtype == torch.bfloat16



def test_runtime_snapshot_preserves_existing_settings_and_is_independent():
  config.configure_runtime(backend="custom_backend", device="cpu", logs={"time"},
                           diffusion_precision="bf16", sdpa_implementation="flash",
                           native_ops=["linear"], enable_cache=False,
                           sparse_attention_parameters={"nested": {"samples": [2, 4]}})
  snapshot = config.get_runtime_options()
  snapshot["logs"].add("debug")
  snapshot["native_ops"].append("unused")
  snapshot["sparse_attention_parameters"]["nested"]["samples"].append(8)
  assert config.logs == {"time"}
  assert config.native_ops == ["linear"]
  assert config.sparse_attention_parameters == {"nested": {"samples": [2, 4]}}
  settings = config.get_runtime_options()
  settings["device"] = "cuda"
  config.configure_runtime(**settings)
  assert config.device == util.DEFAULT_DEVICE == "cuda"
  assert config.acclib_implementation == "custom_backend"
  assert config.diffusion_precision == "bf16" and config.sdpa_implementation == "flash"
  assert config.logs == {"time"} and not config.enable_cache
  assert config.native_ops == ["linear"]
  assert config.sparse_attention_parameters == {"nested": {"samples": [2, 4]}}
