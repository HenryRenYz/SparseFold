# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Runtime settings, with explicit reference defaults and legacy options support."""
from __future__ import annotations

import ast
import copy
import os
import sys
import warnings
from typing import Any

# Old AF3_* variables remain readable, but are never required for defaults.
_LEGACY: dict[str, tuple[str, Any]] = {
    "device": ("AF3_DEVICE", "cpu"),
    "trace_output_path": ("AF3_TRACE_OUTPUT_PATH", "output/_trace"),
    "logs": ("AF3_LOGS", ""),
    "enable_cache": ("AF3_ENABLE_CACHE", "1"),
    "enable_compile": ("AF3_ENABLE_COMPILE", "0"),
    "enable_sharding": ("AF3_ENABLE_SHARDING", "0"),
    "acclib_implementation": ("AF3_ACCLIB_IMPLEMENTATION", "torch"),
    "native_ops": ("AF3_NATIVE_OPS", "*"),
    "native_matmul_dispatch": ("AF3_NATIVE_MATMUL_DISPATCH", "0"),
    "sdpa_implementation": ("AF3_SDPA_IMPLEMENTATION", "math"),
    "diffusion_precision": ("AF3_DIFFUSION_PRECISION", "fp32"),
    "sparse_attention_parameters": ("AF3_SPARSE_ATTENTION_PARAMETERS", "{}"),
    "enable_gsa_index_reuse": ("AF3_ENABLE_GSA_INDEX_REUSE", "1"),
}
_state: dict[str, Any] | None = None
_BOOL_FIELDS = {"enable_cache", "enable_compile", "enable_sharding",
                "native_matmul_dispatch", "enable_gsa_index_reuse"}


def _split(value: str, sep: str) -> list[str]:
  return [part.strip() for part in value.split(sep) if part.strip()]


def _convert_path(value: str) -> str:
  return os.path.abspath(os.path.expanduser(value))


def normalize_backend(value: str) -> str:
  value = value.lower()
  if value == "none":
    return "torch"
  if value == "cpp":
    warnings.warn("backend 'cpp' is deprecated; use 'vecops'", DeprecationWarning, stacklevel=2)
    return "vecops"
  return value


def _environment_value(name: str) -> Any:
  legacy, default = _LEGACY[name]
  current = legacy.replace("AF3_", "SPARSEFOLD_", 1)
  if current in os.environ:
    raw = os.environ[current]
  elif legacy in os.environ:
    raw = os.environ[legacy]
    warnings.warn(f"{legacy} is deprecated; use {current} or configure_runtime()",
                  DeprecationWarning, stacklevel=3)
  else:
    raw = default
  if name == "logs":
    return set(_split(raw, ","))
  if name == "native_ops":
    return _split(raw, ",")
  if name == "sparse_attention_parameters":
    return ast.literal_eval(raw)
  if name in _BOOL_FIELDS:
    return str(raw).lower() in {"1", "true", "yes", "on"}
  if name == "trace_output_path":
    return _convert_path(raw)
  if name == "acclib_implementation":
    return normalize_backend(raw)
  return raw


def _refresh_consumers() -> None:
  # These legacy modules cache settings at import; keep configure-after-import safe.
  util = sys.modules.get("sparsefold.runtime.util")
  if util is not None:
    util.DEFAULT_DEVICE = __getattr__("device")
  trace = sys.modules.get("sparsefold.runtime.trace")
  if trace is not None:
    logs = __getattr__("logs")
    trace.LOG_IN_COMPILE_REGION = "+compiled" in logs
    trace.LOG_TIME = "time" in logs
    trace.LOG_INFO = "info" in logs or "debug" in logs
    trace.LOG_DEBUG = "debug" in logs


def configure_runtime(**overrides: Any) -> None:
  """Resolve settings once. Passing backend= is an alias for acclib_implementation."""
  global _state
  if "backend" in overrides:
    if "acclib_implementation" in overrides:
      raise ValueError("Pass either backend or acclib_implementation")
    overrides["acclib_implementation"] = overrides.pop("backend")
  unknown = overrides.keys() - _LEGACY.keys()
  if unknown:
    raise TypeError(f"Unknown runtime settings: {sorted(unknown)}")
  state = {name: _environment_value(name) for name in _LEGACY}
  state.update(overrides)
  state["acclib_implementation"] = normalize_backend(state["acclib_implementation"])
  state["trace_output_path"] = _convert_path(state["trace_output_path"])
  state["logs"] = set(state["logs"])
  state["native_ops"] = list(state["native_ops"])
  state["sparse_attention_parameters"] = dict(state["sparse_attention_parameters"])
  _state = state
  _refresh_consumers()


def get_runtime_options() -> dict[str, Any]:
  """Return an independent snapshot suitable for configure_runtime(**settings).

  This preserves the resolved backend, precision and other settings when an API
  caller changes only device. Mutable collections, including nested sparse
  attention settings, never expose internal configuration references.
  """
  return copy.deepcopy({name: __getattr__(name) for name in _LEGACY})


def configure(options) -> None:
  """Accept the resolved options object used by the original worker."""
  configure_runtime(
      device="cpu" if options.model.get("use_cpu", False) else "cuda",
      trace_output_path=options.log.trace_output or os.path.join(options.paths.output, "_trace"),
      logs=set(options.log.channels),
      enable_cache=options.runtime.cache,
      enable_compile=options.runtime.compile,
      enable_sharding=options.runtime.sharding,
      acclib_implementation=options.native.impl,
      native_ops=["*"] if options.native.ops == ["all"] else list(options.native.ops),
      native_matmul_dispatch=options.native.matmul_dispatch,
      sdpa_implementation=options.precision.sdpa,
      diffusion_precision=options.precision.diffusion,
      sparse_attention_parameters=dict(options.native.sparse_attention),
      enable_gsa_index_reuse=options.native.gsa_index_reuse,
  )


def uses_vecops() -> bool:
  return __getattr__("acclib_implementation") == "vecops"


def reset() -> None:
  """Clear the resolved settings; future reads use reference defaults/environment."""
  global _state
  _state = None
  _refresh_consumers()


def __getattr__(name: str) -> Any:
  if name in _LEGACY:
    return _state[name] if _state is not None else _environment_value(name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
