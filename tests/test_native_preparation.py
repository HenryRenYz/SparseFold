# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Synthetic native preparation checks. All packing is mocked; no vecops/JIT required."""
import math
import sys
from types import ModuleType

import pytest
import torch

from sparsefold.model import config as model_config
from sparsefold.model.layers import primitives as nn
from sparsefold.model.trunk import blocks
from sparsefold.model.diffusion import transformer as diffusion
from sparsefold.runtime import config
from sparsefold.backends.vecops import modules as native
from sparsefold.backends.vecops import backend
from sparsefold.backends.vecops.policy import validate_native_ops, validate_native_device
from sparsefold.weights import load_checkpoint
from sparsefold.weights.legacy_scopes import ParamVisitor


class SyntheticVisitor(ParamVisitor):
  """Build zeros with the shapes requested by original reference load methods."""
  def __init__(self):
    super().__init__({})
    self.owner = None

  def set(self, module, **kwargs):
    old = self.owner
    self.owner = module
    try:
      return super().set(module, **kwargs)
    finally:
      self.owner = old

  def __contains__(self, name):
    if isinstance(self.owner, native.LinearVecops):
      return name == "weights" or (name == "bias" and self.owner.use_bias)
    if isinstance(self.owner, nn.LayerNorm):
      return name == "scale" or (name == "offset" and self.owner.use_bias)
    return False

  def __getitem__(self, name):
    if isinstance(self.owner, native.LinearVecops):
      shape = self.owner.weight_shape if name == "weights" else self.owner.output_shape
    elif isinstance(self.owner, nn.LayerNorm):
      shape = self.owner.normalized_shape
    elif isinstance(self.owner, native.OuterProductMeanVecops):
      outer = self.owner.left_projection.output_shape
      shape = (*outer, *outer, self.owner.num_output_channel) if name == "output_w" else (self.owner.num_output_channel,)
    else:
      raise AssertionError(f"Unhandled synthetic parameter: {type(self.owner).__name__} {self.prefix}/{name}")
    tensors = self.params.setdefault(self.prefix, {})
    leading = tuple(index + 1 for index in self._slice)
    requested = leading + tuple(shape)
    existing = tensors.get(name)
    if existing is not None:
      requested = tuple(max(a, b) for a, b in zip(requested, existing.shape))
    tensors[name] = torch.zeros(requested)
    return tensors[name][tuple(self._slice)]


@pytest.fixture
def prepared_backend(monkeypatch):
  config.configure_runtime(device="cpu", backend="torch", native_ops=["*"], enable_compile=False)
  monkeypatch.setenv("SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION", "1")
  fake = ModuleType("sparsefold.backends.vecops.kernels")
  def pack(value, **kwargs):
    n, k = value.shape[-2:]
    extent = math.ceil(n / 16) * 16 * math.ceil(k / 32) * 32
    return torch.zeros(value.shape[:-2] + (extent,), device=value.device, dtype=torch.bfloat16)
  fake.pack_matmul_b = pack
  fake.pack_linear_b = pack
  monkeypatch.setitem(sys.modules, fake.__name__, fake)
  import sparsefold.backends.vecops as package
  monkeypatch.setattr(package, "kernels", fake, raising=False)
  monkeypatch.setattr(native, "_kernels", lambda: fake)
  backend.apply()
  yield
  backend.unapply()
  config.reset()


def make_module(kind):
  global_config = model_config.GlobalConfig()
  attention = diffusion.SelfAttention.Config(num_head=2)
  cross = diffusion.CrossAttention.Config(num_head=2, key_dim=16, value_dim=16)
  transition = blocks.TransitionBlock.Config(num_intermediate_factor=2)
  factories = {
      "transition": lambda: native.TransitionBlockVecops(transition, global_config, num_channels=16, name="transition"),
      "msa": lambda: native.MSAAttentionVecops(blocks.MSAAttention.Config(num_head=2), global_config, num_channels=16, num_output_channel=8, name="msa"),
      "outer": lambda: native.OuterProductMeanVecops(blocks.OuterProductMean.Config(num_outer_channel=4), global_config, num_channels=16, num_output_channel=8, name="outer"),
      "grid": lambda: native.GridSelfAttentionVecops(blocks.GridSelfAttention.Config(num_head=2), global_config, False, num_channels=16, name="grid"),
      "triangle": lambda: native.TriangleMultiplicationVecops(blocks.TriangleMultiplication.Config(equation="...ikc,...jkc->...ijc"), global_config, num_channels=16, name="triangle"),
      "single": lambda: native.SingleActUpdateVecops(global_config, 8, 16, attention, transition),
      "diffusion_transition": lambda: native.DiffusionTransitionBlockVecops(global_config, num_channels=16, num_intermediate_factor=2, single_cond_size=16, name="diffusion_transition"),
      "diffusion_attention": lambda: native.DiffusionSelfAttentionVecops(attention, global_config, num_channels=16, single_cond_size=16, name="diffusion_attention"),
      "diffusion_cross": lambda: native.DiffusionCrossAttentionVecops(cross, global_config, num_channels=16, name="diffusion_cross"),
      "transformer": lambda: native.TransformerVecops(diffusion.Transformer.Config(attention=attention, num_blocks=4), global_config, num_channels=16, single_cond_size=16, pair_cond_size=8),
      "cross_transformer": lambda: native.CrossAttTransformerVecops(diffusion.CrossAttTransformer.Config(attention=cross, num_blocks=2, num_intermediate_factor=2), global_config, pair_cond_size=8, num_channels=16, super_name="atom_"),
  }
  return factories[kind]()


@pytest.mark.parametrize("kind", ["transition", "msa", "outer", "grid", "triangle", "single", "diffusion_transition", "diffusion_attention", "diffusion_cross", "transformer", "cross_transformer"])
def test_fused_preparation_preserves_strict_reference_parameters(kind, prepared_backend):
  module = make_module(kind)
  source = SyntheticVisitor()
  source.set(module)
  report = load_checkpoint(module, source.params)
  assert report.complete
  parameters = dict(module.named_parameters())
  values = {name: parameter.detach().clone() for name, parameter in parameters.items()}
  state_keys = set(module.state_dict())
  with pytest.raises(RuntimeError, match="buffers are not prepared"):
    module.forward()
  counts = backend.prepare(module)
  assert counts["opaque"] >= 1
  assert getattr(module, "_native_prepared")
  assert set(module.state_dict()) == state_keys
  assert set(dict(module.named_parameters())) == set(parameters)
  for name, parameter in module.named_parameters():
    assert parameter is parameters[name]
    torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)
  assert list(module.named_buffers())
  for child in module.modules():
    assert set(child._buffers) <= child._non_persistent_buffers_set
  # Re-preparation refreshes buffers without changing the source state contract.
  backend.prepare(module)
  assert set(module.state_dict()) == state_keys


def test_fused_gate_retains_explicit_experimental_path(monkeypatch):
  monkeypatch.delenv("SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION", raising=False)
  with pytest.raises(RuntimeError, match="experimental opt-in"):
    validate_native_ops(["all"])
  monkeypatch.setenv("SPARSEFOLD_EXPERIMENTAL_NATIVE_FUSION", "1")
  validate_native_ops(["all"])
  validate_native_ops(["msa_attention", "diffusion_cross_attention"])
  with pytest.raises(ValueError, match="Unknown"):
    validate_native_ops(["unknown"])
  with pytest.raises(RuntimeError, match="CPU only"):
    validate_native_device("cuda:0")
