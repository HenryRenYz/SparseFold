# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Per-process worker entry for the unified AlphaFold CLI.

This module is the CLI-to-model adapter. It runs inside each inference process
before torch is imported: resolves configuration, applies this rank's
CPU/memory binding, derives the thread environment, then loads the installed
`sparsefold.cli.predict` module and materializes its absl flags from the
`[model]` section.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _detect_rank() -> tuple[int, int, int]:
  """Read (rank, world_size, local_rank) from the launcher environment."""

  def first(*names: str) -> str | None:
    for name in names:
      value = os.environ.get(name)
      if value not in (None, ""):
        return value
    return None

  rank = first("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK")
  world = first("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
  local = first(
      "LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
      "PMI_RANK",
  )
  if rank is None or world is None or local is None:
    print(
        "[sparsefold-worker] unable to determine distributed rank from the "
        "launcher environment",
        file=sys.stderr,
    )
    raise SystemExit(2)
  return int(rank), int(world), int(local)


def _load_prediction_module():
  """Load the installed prediction module after process affinity is fixed."""
  if os.environ.get("AF3_RUN_SCRIPT"):
    raise SystemExit("AF3_RUN_SCRIPT is no longer supported; use the installed sparsefold.cli.predict module")
  from sparsefold.cli import predict
  return predict


def main(argv: list[str] | None = None) -> None:
  raw = list(sys.argv[1:] if argv is None else argv)
  extra: list[str] = []
  if "--" in raw:
    split = raw.index("--")
    extra = raw[split + 1:]
    raw = raw[:split]

  parser = argparse.ArgumentParser(prog="sparsefold-worker")
  parser.add_argument("input", nargs="?", help="input name override")
  parser.add_argument("--input-name", help="input name override")
  parser.add_argument("-c", "--config", required=True)
  parser.add_argument("--set", action="append", default=[])
  parser.add_argument("--numa-nodes", metavar="LIST")
  parser.add_argument(
      "--distributed",
      action="store_true",
      help="this worker belongs to a world with more than one rank",
  )
  parser.add_argument("--nproc-per-node", type=int, default=1)
  parser.add_argument(
      "--run-output-dir",
      action="append",
      default=[],
      help="launcher-resolved fold output directory (repeat for each job)",
  )
  args = parser.parse_args(raw)

  dist_mode = args.distributed
  if dist_mode:
    rank, world_size, local_rank = _detect_rank()
  else:
    rank, world_size, local_rank = 0, 1, 0

  run_output_dirs = [Path(path).absolute() for path in args.run_output_dir]
  console_name = f"console-r-{rank}.log" if dist_mode else "console.log"
  from sparsefold.cli.output import ConsoleTee

  console_tee = ConsoleTee(
      [output_dir / console_name for output_dir in run_output_dirs]
  ).start()
  try:
    _main_after_console(
        args,
        extra,
        dist_mode=dist_mode,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        run_output_dirs=run_output_dirs,
    )
  finally:
    console_tee.close()


def _main_after_console(
    args: argparse.Namespace,
    extra: list[str],
    *,
    dist_mode: bool,
    rank: int,
    world_size: int,
    local_rank: int,
    run_output_dirs: list[Path],
) -> None:

  # Heavy imports stay below the affinity section on purpose: the process
  # CPU mask and thread environment must be fixed before torch loads.
  from sparsefold.cli import affinity
  from sparsefold.cli import options as options_mod

  opts = options_mod.load(args.config)
  for entry in args.set:
    key, _, raw_value = entry.partition("=")
    if not raw_value and "=" not in entry:
      raise SystemExit(f"[sparsefold-worker] --set expects KEY=VALUE, got {entry!r}")
    options_mod.apply_override(
        opts, key, options_mod.parse_value(raw_value)
    )
  options_mod._validate(opts)
  if args.input and args.input_name:
    raise SystemExit(
        "[sparsefold-worker] specify the input either positionally or with "
        "--input-name, not both"
    )
  selected_input = args.input_name or args.input
  if selected_input:
    opts.input_name = selected_input
  if args.numa_nodes:
    opts.affinity.mode = "numa"
    opts.affinity.numa_nodes = affinity.parse_numa_nodes(args.numa_nodes)
    opts.affinity.compute_cpu_lists = []
    opts.affinity.comm_cpu_lists = []
    opts.affinity.memory_node_ids = []
  options_mod._validate(opts)
  if not opts.input_name:
    raise SystemExit(
        "[sparsefold-worker] no input name: set `input_name` or pass it as argument"
    )

  topology = affinity.read_topology()
  bind = opts.affinity
  plan = affinity.plan_affinity(
      topology,
      bind.mode,
      local_rank,
      comm_workers=bind.comm_workers,
      numa_nodes=bind.numa_nodes,
      compute_cpu_lists=bind.compute_cpu_lists,
      comm_cpu_lists=bind.comm_cpu_lists,
      memory_node_ids=bind.memory_node_ids,
      mem_policy=bind.mem_policy,
  )
  affinity.apply(plan)
  threads = affinity.derive_thread_count(
      plan,
      topology,
      configured=opts.runtime.threads,
      procs=args.nproc_per_node,
      comm_workers=bind.comm_workers,
  )
  os.environ.update(
      affinity.derive_env(plan, threads, bind.thread_pin)
  )
  flush_subnormals = os.environ.get("VECOPS_FLUSH_SUBNORMALS")
  if flush_subnormals is not None:
    if flush_subnormals not in {"0", "1"}:
      raise ValueError("VECOPS_FLUSH_SUBNORMALS must be 0 or 1")
    # Set FPCR/MXCSR before importing Torch so its CPU workers inherit the
    # mode. vecops also reapplies it when pre-existing workers enter a task.
    import vecops

    vecops.set_flush_subnormals(flush_subnormals == "1")
  prefix = f"[rank {rank}/{world_size}]" if dist_mode else "[sparsefold]"
  print(
      f"{prefix} {plan.describe()} OMP_NUM_THREADS={threads} "
      f"(native.impl={opts.native.impl}, "
      f"matmul_dispatch={opts.native.matmul_dispatch})",
      flush=True,
  )

  from sparsefold.runtime import config as runtime_config

  runtime_config.configure(opts)

  json_path = (
      Path(opts.paths.input).expanduser()
      / f"{opts.input_name}.json"
  )
  if not json_path.exists():
    raise SystemExit(f"[sparsefold-worker] input file not found: {json_path}")

  from absl import app, flags

  prediction = _load_prediction_module()
  if run_output_dirs:
    prediction.set_output_dir_overrides(run_output_dirs)

  launcher_managed = {
      "json_path", "input_dir", "output_dir", "model_dir", "db_dir",
      "dist", "dist_backend",
  }
  reserved = sorted(set(opts.model) & launcher_managed)
  if reserved:
    raise SystemExit(
        "[sparsefold-worker] launcher-managed [model] entries must be configured "
        f"through [paths]/[launch]: {reserved}"
    )

  model_flags: dict[str, object] = dict(opts.model)
  model_flags.update({
      "json_path": str(json_path),
      "output_dir": str(Path(opts.paths.output).expanduser().absolute()),
      "model_dir": str(Path(opts.paths.model).expanduser()),
  })
  if opts.paths.database:
    database_dirs = (
        opts.paths.database
        if isinstance(opts.paths.database, list)
        else [opts.paths.database]
    )
    model_flags["db_dir"] = [
        str(Path(path).expanduser()) for path in database_dirs
    ]
  if dist_mode:
    model_flags["dist"] = True
    model_flags["dist_backend"] = opts.launch.backend

  known = set(flags.FLAGS)
  unknown: list[str] = []
  resolved_flags: dict[str, object] = {}
  for name, value in model_flags.items():
    if name in known:
      resolved_flags[name] = value
    elif name.startswith("no") and name[2:] in known:
      # absl-style negation shorthand, e.g. `norun_data_pipeline`.
      resolved_flags[name[2:]] = False
    else:
      unknown.append(name)
  if unknown:
    raise SystemExit(
        f"[sparsefold-worker] [model] entries are not SparseFold prediction flags: "
        f"{sorted(unknown)}"
    )
  for name, value in resolved_flags.items():
    flags.FLAGS[name].value = value

  Path(opts.paths.output).expanduser().absolute().mkdir(
      parents=True, exist_ok=True
  )
  app.run(prediction.main, argv=[sys.argv[0], *extra])


if __name__ == "__main__":
  main()
