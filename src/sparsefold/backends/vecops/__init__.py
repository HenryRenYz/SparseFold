# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Optional native backend exports, resolved only when a hook is requested."""
import importlib

_HOOKS = {"apply", "prepare", "materialize", "precompile", "manual_seed", "is_precompiling", "execution_context"}
_MODULES = {"CrossAttTransformerVecops", "DiffusionCrossAttentionVecops",
            "DiffusionSelfAttentionVecops", "DiffusionTransitionBlockVecops",
            "GridSelfAttentionVecops", "LinearVecops", "MSAAttentionVecops",
            "OuterProductMeanVecops", "SingleActUpdateVecops", "TransformerVecops",
            "TransitionBlockVecops", "TriangleMultiplicationVecops"}
__all__ = sorted(_HOOKS | _MODULES | {"execution_context"})


def __getattr__(name):
  if name in _HOOKS:
    module = importlib.import_module("sparsefold.backends.vecops.backend")
  elif name in _MODULES:
    module = importlib.import_module("sparsefold.backends.vecops.modules")
  else:
    raise AttributeError(name)
  return getattr(module, name)
