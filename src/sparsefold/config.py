# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Structured SparseFold CLI configuration loaded from TOML files.

A configuration file is a small typed document; layered files combine through
`extends`, command-line `--set` overrides win last. This module deliberately
depends on the standard library only so both the launcher and the
per-rank worker can import it before torch.

Legacy `AF3_*` environment variables are handled by `config.py` as a
deprecation shim on top of the values produced here.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
import typing
from pathlib import Path
from typing import Any

NATIVE_OP_NAMES = frozenset({
    "linear",
    "evoformer_transition_block",
    "msa_attention",
    "outer_product_mean",
    "grid_self_attn",
    "triangle_multiplication",
    "evoformer_single_act_update",
    "diffusion_transition_block",
    "diffusion_self_attention",
    "diffusion_cross_attention",
})

MALLOC_IMPLS = frozenset({"default", "tcmalloc", "jemalloc", "mimalloc"})
OMP_IMPLS = frozenset({"default", "intel"})
LAUNCHERS = frozenset({"torchrun", "openmpi", "intelmpi", "mpich"})
BIND_MODES = frozenset({"none", "numa", "explicit"})
THREAD_PINS = frozenset({"none", "spread", "scatter", "close"})
MEM_POLICIES = frozenset({"local", "hbm-preferred"})


class OptionsError(ValueError):
  """Raised for malformed or inconsistent configuration."""


@dataclasses.dataclass
class PathsConfig:
  input: str
  model: str
  output: str
  database: str | list[str] | None = None
  baseline: str | None = None


@dataclasses.dataclass
class NativeConfig:
  impl: str = "none"
  ops: list[str] = dataclasses.field(default_factory=lambda: ["all"])
  matmul_dispatch: bool = False
  gsa_index_reuse: bool = True
  # Mirrors `IBSAParameters`; consumed as one dict by the sparse attention.
  sparse_attention: dict[str, Any] = dataclasses.field(
      default_factory=lambda: {
          "num_samples": 16,
          "num_rand_blocks": 5,
          "t_prob": 0.90,
          "tr_prob": 0.95,
          "u_prob": 0.02,
      }
  )


@dataclasses.dataclass
class PrecisionConfig:
  sdpa: str = "flash"  # math | flash
  diffusion: str = "bf16"  # fp32 | bf16


@dataclasses.dataclass
class RuntimeConfig:
  malloc: str = "default"
  omp: str = "default"
  # 0 derives the thread count from the applied CPU binding.
  threads: int = 0
  compile: bool = False
  cache: bool = True
  sharding: bool = False
  torchinductor_cache: str = "./cache/sparsefold/inductor"
  ld_library_paths: list[str] = dataclasses.field(default_factory=list)
  # Escape hatch for variables that must exist before the worker starts
  # (e.g. VECOPS_TORCH_BRIDGE_CXX when torch needs a newer GNU compiler).
  env: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class LogConfig:
  channels: list[str] = dataclasses.field(default_factory=lambda: ["info"])
  # Defaults to `<output>/_trace` when unset.
  trace_output: str | None = None


@dataclasses.dataclass
class AffinityConfig:
  mode: str = "none"
  comm_workers: int = 0
  thread_pin: str = "none"
  # "local" membinds a bound rank to its compute NUMA node; "hbm-preferred"
  # prefers the nearest memory-only tier (HBM) and spills to the default
  # policy once that tier is full.
  mem_policy: str = "local"
  # Optional CPU-bearing NUMA node ids, one per local rank.  An empty list
  # uses every visible CPU-bearing node in OS node-id order.
  numa_nodes: list[int] = dataclasses.field(default_factory=list)
  compute_cpu_lists: list[str] = dataclasses.field(default_factory=list)
  comm_cpu_lists: list[str] = dataclasses.field(default_factory=list)
  # Exact memory node per local rank for mode="explicit". Empty means CPU-only
  # binding; use this when the desired memory node cannot be inferred safely.
  memory_node_ids: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LaunchConfig:
  launcher: str = "torchrun"
  backend: str = "gloo"
  num_nodes: int = 1
  node_rank: int = 0
  nproc: int | str = 1
  master_addr: str = "127.0.0.1"
  master_port: int = 29500


@dataclasses.dataclass
class SparseFoldOptions:
  input_name: str | None = None
  paths: PathsConfig = None  # type: ignore[assignment]
  # Free mapping of absl flag names (e.g. `use_cpu`) to values; validated
  # against the registered flags when the worker builds the run.
  model: dict[str, Any] = dataclasses.field(default_factory=dict)
  precision: PrecisionConfig = dataclasses.field(
      default_factory=PrecisionConfig
  )
  native: NativeConfig = dataclasses.field(default_factory=NativeConfig)
  runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
  log: LogConfig = dataclasses.field(default_factory=LogConfig)
  launch: LaunchConfig = dataclasses.field(default_factory=LaunchConfig)
  affinity: AffinityConfig = dataclasses.field(default_factory=AffinityConfig)

  def resolved_nproc(self, cpu_node_count: int) -> int:
    value = self.launch.nproc
    if isinstance(value, str):
      if value != "numa":
        raise OptionsError(
            f"launch.nproc must be an integer or 'numa', got {value!r}"
        )
      if cpu_node_count < 1:
        raise OptionsError("no CPU-bearing NUMA nodes detected")
      return len(self.affinity.numa_nodes) or cpu_node_count
    return int(value)


_SECTION_TYPES: dict[str, type] = {
    "precision": PrecisionConfig,
    "native": NativeConfig,
    "runtime": RuntimeConfig,
    "log": LogConfig,
    "launch": LaunchConfig,
    "affinity": AffinityConfig,
}

_TOP_LEVEL_KEYS = {"input_name", "paths", "model", *_SECTION_TYPES}


def _normalize_legacy_schema(data: dict) -> dict:
  """Translate the retired [dist]/[dist.bind] schema before validation."""
  if "dist" not in data:
    return data
  if "launch" in data or "affinity" in data:
    raise OptionsError(
        "legacy [dist] cannot be combined with [launch] or [affinity]"
    )
  normalized = dict(data)
  legacy = dict(normalized.pop("dist"))
  bind = legacy.pop("bind", None)
  if "procs_per_node" in legacy:
    legacy["nproc"] = legacy.pop("procs_per_node")
  normalized["launch"] = legacy
  if bind is not None:
    normalized["affinity"] = bind
  return normalized


def _deep_merge(base: dict, override: dict) -> dict:
  merged = dict(base)
  for key, value in override.items():
    if (
        key in merged
        and isinstance(merged[key], dict)
        and isinstance(value, dict)
    ):
      merged[key] = _deep_merge(merged[key], value)
    else:
      merged[key] = value
  return merged


def _load_toml(path: Path) -> dict:
  try:
    with path.open("rb") as f:
      return tomllib.load(f)
  except tomllib.TOMLDecodeError as e:
    raise OptionsError(f"{path}: invalid TOML: {e}") from e


def _load_chain(path: Path, seen: tuple[str, ...] = ()) -> dict:
  path = path.resolve()
  if str(path) in seen:
    chain = " -> ".join([*seen, str(path)])
    raise OptionsError(f"circular `extends` chain: {chain}")
  data = _load_toml(path)
  extends = data.pop("extends", None)
  data = _normalize_legacy_schema(data)
  if extends is None:
    return data
  parent = Path(extends)
  if not parent.is_absolute():
    parent = path.parent / parent
  if not parent.exists():
    raise OptionsError(f"{path}: extends target does not exist: {parent}")
  return _deep_merge(_load_chain(parent, (*seen, str(path))), data)


def _build_section(section: str, cls: type, data: dict) -> Any:
  hints = typing.get_type_hints(cls)
  unknown = set(data) - set(hints)
  if unknown:
    raise OptionsError(f"[{section}] unknown keys: {sorted(unknown)}")
  kwargs = {}
  for name, annotation in hints.items():
    if name not in data:
      continue
    value = data[name]
    if (
        dataclasses.is_dataclass(annotation)
        and isinstance(value, dict)
    ):
      value = _build_section(f"{section}.{name}", annotation, value)
    kwargs[name] = value
  try:
    return cls(**kwargs)
  except TypeError as e:
    raise OptionsError(f"[{section}]: {e}") from e


def _validate(options: SparseFoldOptions) -> None:
  if options.runtime.malloc not in MALLOC_IMPLS:
    raise OptionsError(
        f"runtime.malloc must be one of {sorted(MALLOC_IMPLS)}, "
        f"got {options.runtime.malloc!r}"
    )
  if options.runtime.omp not in OMP_IMPLS:
    raise OptionsError(
        f"runtime.omp must be one of {sorted(OMP_IMPLS)}, "
        f"got {options.runtime.omp!r}"
    )
  if options.runtime.threads < 0:
    raise OptionsError("runtime.threads must be >= 0")

  if options.precision.sdpa not in {"math", "flash"}:
    raise OptionsError(
        f"precision.sdpa must be 'math' or 'flash', "
        f"got {options.precision.sdpa!r}"
    )
  if options.precision.diffusion not in {"fp32", "bf16"}:
    raise OptionsError(
        f"precision.diffusion must be 'fp32' or 'bf16', "
        f"got {options.precision.diffusion!r}"
    )

  if options.native.impl not in {"vecops", "none", "cpp"}:
    raise OptionsError(
        "native.impl must be 'vecops' or 'none' ('cpp' is a deprecated alias)"
    )
  ops = options.native.ops
  if not isinstance(ops, list) or not all(isinstance(x, str) for x in ops):
    raise OptionsError("native.ops must be a list of operator names")
  if "all" in ops:
    if len(ops) != 1:
      raise OptionsError("native.ops: 'all' cannot be combined with names")
  else:
    unknown = set(ops) - NATIVE_OP_NAMES
    if unknown:
      raise OptionsError(
          f"native.ops unknown entries: {sorted(unknown)}; "
          f"valid names: {sorted(NATIVE_OP_NAMES)}"
      )

  if options.launch.launcher not in LAUNCHERS:
    raise OptionsError(
        f"launch.launcher must be one of {sorted(LAUNCHERS)}, "
        f"got {options.launch.launcher!r}"
    )
  if options.launch.num_nodes < 1:
    raise OptionsError("launch.num_nodes must be >= 1")
  if not 0 <= options.launch.node_rank < options.launch.num_nodes:
    raise OptionsError("launch.node_rank must be within [0, launch.num_nodes)")
  if options.launch.master_port < 0 or options.launch.master_port > 65535:
    raise OptionsError("launch.master_port must be a valid port")
  if isinstance(options.launch.nproc, int) and options.launch.nproc < 1:
    raise OptionsError("launch.nproc must be >= 1")
  if isinstance(options.launch.nproc, str) and options.launch.nproc != "numa":
    raise OptionsError("launch.nproc must be an integer or 'numa'")

  bind = options.affinity
  if bind.mode not in BIND_MODES:
    raise OptionsError(
        f"affinity.mode must be one of {sorted(BIND_MODES)}, "
        f"got {bind.mode!r}"
    )
  if bind.thread_pin not in THREAD_PINS:
    raise OptionsError(
        f"affinity.thread_pin must be one of {sorted(THREAD_PINS)}, "
        f"got {bind.thread_pin!r}"
    )
  if bind.comm_workers < 0:
    raise OptionsError("affinity.comm_workers must be >= 0")
  if bind.mem_policy not in MEM_POLICIES:
    raise OptionsError(
        f"affinity.mem_policy must be one of {sorted(MEM_POLICIES)}, "
        f"got {bind.mem_policy!r}"
    )
  if not all(isinstance(node, int) and node >= 0 for node in bind.numa_nodes):
    raise OptionsError("affinity.numa_nodes must contain non-negative integers")
  if len(set(bind.numa_nodes)) != len(bind.numa_nodes):
    raise OptionsError("affinity.numa_nodes must not contain duplicates")
  if bind.numa_nodes and bind.mode != "numa":
    raise OptionsError(
        "affinity.numa_nodes is only valid with mode='numa'"
    )
  if not all(
      isinstance(node, int) and node >= 0 for node in bind.memory_node_ids
  ):
    raise OptionsError(
        "affinity.memory_node_ids must contain non-negative integers"
    )
  if bind.memory_node_ids and bind.mode != "explicit":
    raise OptionsError(
        "affinity.memory_node_ids is only valid with mode='explicit'"
    )
  if bind.memory_node_ids and bind.mem_policy != "local":
    raise OptionsError(
        "affinity.memory_node_ids requires mem_policy='local'"
    )
  if bind.mode == "explicit":
    if not bind.compute_cpu_lists:
      raise OptionsError(
          "affinity.mode='explicit' requires affinity.compute_cpu_lists"
      )
  if bind.mode == "numa" and bind.compute_cpu_lists:
    raise OptionsError(
        "affinity.compute_cpu_lists is only valid with mode='explicit'"
    )


def _from_dict(data: dict) -> SparseFoldOptions:
  data = _normalize_legacy_schema(data)
  unknown = set(data) - _TOP_LEVEL_KEYS
  if unknown:
    raise OptionsError(f"unknown top-level keys: {sorted(unknown)}")

  paths_data = data.get("paths", {})
  unknown_paths = set(paths_data) - {
      f.name for f in dataclasses.fields(PathsConfig)
  }
  if unknown_paths:
    raise OptionsError(f"[paths] unknown keys: {sorted(unknown_paths)}")
  required_paths = {"input", "model", "output"} - set(paths_data)
  if required_paths:
    raise OptionsError(f"[paths] missing required keys: {sorted(required_paths)}")

  options = SparseFoldOptions(
      input_name=data.get("input_name"),
      paths=PathsConfig(**paths_data),
      model=data.get("model", {}),
  )
  for section, cls in _SECTION_TYPES.items():
    setattr(
        options, section, _build_section(section, cls, data.get(section, {}))
    )
  _validate(options)
  return options


def load(path: str | Path) -> SparseFoldOptions:
  """Load and validate one configuration file (following `extends`)."""
  return _from_dict(_load_chain(Path(path)))


def parse_value(raw: str) -> Any:
  """Interpret one `--set` value using TOML scalar syntax.

  Values that are not valid TOML scalars (e.g. relative paths like
  `./output`) fall back to plain strings so quoting is optional.
  """
  try:
    return tomllib.loads(f"v = {raw}")["v"]
  except tomllib.TOMLDecodeError:
    return raw


def _coerce(current: Any, value: Any) -> Any:
  """Accept TOML bool/int coercion for string-valued fields and vice versa."""
  if isinstance(current, bool) and isinstance(value, str):
    if value.lower() in {"true", "false"}:
      return value.lower() == "true"
  if isinstance(current, int) and not isinstance(current, bool) \
      and isinstance(value, str):
    try:
      return int(value)
    except ValueError:
      pass
  return value


def apply_override(options: SparseFoldOptions, dotted: str, value: Any) -> None:
  """Apply one `--set key.subkey=value` override in place."""
  if dotted == "dist.procs_per_node":
    dotted = "launch.nproc"
  elif dotted.startswith("dist.bind."):
    dotted = "affinity." + dotted.removeprefix("dist.bind.")
  elif dotted.startswith("dist."):
    dotted = "launch." + dotted.removeprefix("dist.")
  parts = dotted.split(".")
  if len(parts) < 2:
    raise OptionsError(
        f"--set {dotted}: expected a dotted path like 'affinity.mode'"
    )
  target: Any = options
  for part in parts[:-1]:
    if dataclasses.is_dataclass(target) and not isinstance(target, type):
      if not hasattr(target, part):
        raise OptionsError(f"--set {dotted}: unknown section '{part}'")
      target = getattr(target, part)
    elif isinstance(target, dict):
      target = target.setdefault(part, {})
    else:
      raise OptionsError(f"--set {dotted}: cannot descend into '{part}'")
    if not isinstance(target, (dict,)) and not dataclasses.is_dataclass(target):
      raise OptionsError(f"--set {dotted}: '{part}' is not a section")
  key = parts[-1]
  if isinstance(target, dict):
    current = target.get(key)
    target[key] = value if current is None else _coerce(current, value)
  else:
    field_names = {f.name for f in dataclasses.fields(target)}
    if key not in field_names:
      raise OptionsError(f"--set {dotted}: unknown key '{key}'")
    setattr(target, key, _coerce(getattr(target, key), value))


def _format_scalar(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, str):
    return json.dumps(value)  # JSON basic strings are valid TOML strings
  if isinstance(value, list):
    return "[" + ", ".join(_format_scalar(x) for x in value) + "]"
  return repr(value)


def _dump_toml(value: dict[str, Any], prefix: str = "") -> list[str]:
  lines: list[str] = []
  for key, item in value.items():
    if item is None:
      continue
    header = f"{prefix}.{key}" if prefix else key
    if isinstance(item, dict):
      nested = {k: v for k, v in item.items() if v is not None}
      if not nested:
        continue
      lines.append(f"[{header}]")
      lines.extend(_dump_toml(nested, header))
      lines.append("")
    else:
      lines.append(f"{key} = {_format_scalar(item)}")
  return lines


def to_toml(options: SparseFoldOptions) -> str:
  """Serialize resolved options back to TOML (for --print-config)."""
  data = {
      "input_name": options.input_name,
      "paths": dataclasses.asdict(options.paths),
      "model": options.model,
      "precision": dataclasses.asdict(options.precision),
      "native": dataclasses.asdict(options.native),
      "runtime": dataclasses.asdict(options.runtime),
      "log": dataclasses.asdict(options.log),
      "launch": dataclasses.asdict(options.launch),
      "affinity": dataclasses.asdict(options.affinity),
  }
  # Drop None entries so the dump stays valid TOML.
  data["paths"] = {k: v for k, v in data["paths"].items() if v is not None}
  if data["input_name"] is None:
    data.pop("input_name")
  body = _dump_toml(data)
  return "\n".join(body).rstrip() + "\n"

# Compatibility type alias for existing typed configuration callers.
Af3Options = SparseFoldOptions
