# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Backend lifecycle dispatcher. Native vecops is imported only on explicit selection."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import importlib

from sparsefold.runtime import config


@dataclass(frozen=True)
class _Library:
  name: str
  apply: Callable[[], None]
  fn_redirects: dict[str, Callable]


_registered_impl: dict[str, _Library] = {}
_current_library: _Library | None = None
_applied_libraries: set[str] = set()
_HOOKS = ("manual_seed", "prepare", "precompile", "materialize", "is_precompiling", "execution_context")


def register_library(apply_fn: Callable[[], None], name: str,
                     fn_redirects: dict[str, Callable] | None = None):
  if name in _registered_impl:
    raise KeyError(f"Library {name!r} already registered")
  _registered_impl[name] = _Library(name, apply_fn, fn_redirects or {})


def available_libraries() -> list[str]:
  """Declared backends; optional library installation is checked only by init()."""
  return list(dict.fromkeys(["torch", "vecops", *_registered_impl]))


def _check_vecops_api() -> None:
  """Fail before registering bridges when an incompatible dependency is installed."""
  try:
    native = importlib.import_module("vecops")
  except ImportError as error:
    raise RuntimeError("Backend 'vecops' requires an installed compatible vecops package") from error
  required = ("jit", "ops", "is_precompiling", "precompile", "compile_batch", "collect_compile_requests")
  missing = [name for name in required if not hasattr(native, name)]
  if not missing:
    try:
      importlib.import_module("vecops.typing")
    except ImportError:
      missing.append("typing")
  if missing:
    raise RuntimeError(
        "Installed vecops API is incompatible with these migrated C++ kernels; "
        f"missing: {', '.join(missing)}. Use a verified compatible vecops build "
        "or migrate the native kernel API explicitly. The Torch backend remains available."
    )


def _load_library(name: str) -> _Library:
  if name not in _registered_impl and name in {"torch", "vecops"}:
    if name == "vecops":
      _check_vecops_api()
    try:
      module = importlib.import_module(f"sparsefold.backends.{name}")
    except ImportError as error:
      raise RuntimeError(f"Cannot load backend {name!r}: {error}. Install the optional vecops dependency when selecting it.") from error
    register_library(module.apply, name, {hook: getattr(module, hook) for hook in _HOOKS})
  if name not in _registered_impl:
    raise KeyError(f"Backend {name!r} is not registered")
  return _registered_impl[name]


def init(apply: bool = True):
  """Select the configured backend; applying the same backend is idempotent.

  Model construction must follow backend selection. Switching to Torch restores
  class factories; models already constructed with native classes stay native.
  """
  global _current_library
  name = config.normalize_backend(config.acclib_implementation)
  if name == "vecops":
    from sparsefold.backends.vecops.policy import validate_native_device, validate_native_ops
    validate_native_device(config.device)
    validate_native_ops(config.native_ops)
  lib = _load_library(name)
  if _current_library is not None and _current_library.name != name:
    if _current_library.name == "vecops" and "vecops" in _applied_libraries:
      importlib.import_module("sparsefold.backends.vecops.backend").unapply()
      _applied_libraries.discard("vecops")
    elif _current_library.name not in {"torch", "vecops"} and _current_library.name in _applied_libraries:
      raise RuntimeError("Cannot safely switch from an applied custom backend")
  if apply and name not in _applied_libraries:
    try:
      lib.apply()
    except Exception as error:
      if name == "vecops":
        importlib.import_module("sparsefold.backends.vecops.backend").unapply()
      raise RuntimeError(f"Failed to apply backend {name!r}: {error}") from error
    _applied_libraries.add(name)
  _current_library = lib


def _dispatch_redirect_call(name: str, *args, **kwargs):
  # Reference hooks work even before explicit initialization.
  lib = _current_library or _load_library("torch")
  fn = lib.fn_redirects.get(name)
  return fn(*args, **kwargs) if fn is not None else None


def manual_seed(seed: int):
  return _dispatch_redirect_call("manual_seed", seed)


def prepare(module):
  return _dispatch_redirect_call("prepare", module)


def precompile(module, *args, use_real_tensors: bool = False,
               parallelism: int | None = None, capture_graph: bool = False,
               graph_name: str | None = None, **kwargs):
  return _dispatch_redirect_call("precompile", module, *args,
      use_real_tensors=use_real_tensors, parallelism=parallelism,
      capture_graph=capture_graph, graph_name=graph_name, **kwargs)


def materialize(*, parallelism: int | None = None):
  return _dispatch_redirect_call("materialize", parallelism=parallelism)


def is_precompiling() -> bool:
  return bool(_dispatch_redirect_call("is_precompiling"))


def execution_context():
  return _dispatch_redirect_call("execution_context") or nullcontext()
