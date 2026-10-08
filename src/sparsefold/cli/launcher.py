# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Repository-local command-line launcher for SparseFold inference.

The launcher resolves the TOML configuration, derives the environment that
must exist before the worker starts (allocator / OpenMP preloads, inductor
caches) and chooses direct execution or a distributed launcher from the
resolved world size. It deliberately imports nothing but the standard library.

Usage:
  sparsefold [input] --config F [--nproc N] [--set k=v]...
  sparsefold print-config --config F [--set k=v]...
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import affinity
from . import options as options_mod
from . import output as output_mod
from . import run_manifest

_MALLOC_LIBS = {
    "tcmalloc": ("gperftools", "libtcmalloc"),
    "jemalloc": ("jemalloc", "libjemalloc"),
    "mimalloc": ("mimalloc", "libmimalloc"),
}


def _library_search_dirs() -> list[Path]:
  """Library roots, starting with the running interpreter's environment.

  `$CONDA_PREFIX` follows the *activated* environment, which may differ
  from the interpreter actually running this launcher (e.g. an env's python
  invoked directly), so `sys.prefix` takes precedence.
  """
  dirs = [
      Path(sys.prefix) / "lib",
      Path(sys.prefix) / "lib64",
  ]
  prefix = os.environ.get("CONDA_PREFIX")
  if prefix and Path(prefix) != Path(sys.prefix):
    dirs += [Path(prefix) / "lib", Path(prefix) / "lib64"]
  for var_dir in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if var_dir:
      dirs.append(Path(var_dir))
  dirs.extend([
      Path("/usr/local/lib"),
      Path("/usr/lib64"),
      Path("/usr/lib/x86_64-linux-gnu"),
      Path("/usr/lib/aarch64-linux-gnu"),
      Path("/usr/lib"),
      Path("/lib64"),
      Path("/lib"),
  ])
  return dirs


def _find_library(spack_package: str, name: str) -> str | None:
  """Locate a shared library like the legacy `_find_so_library` helper."""
  if shutil.which("spack"):
    result = subprocess.run(
        ["spack", "location", "-i", spack_package],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
      for path in sorted(
          glob.glob(f"{result.stdout.strip()}/**/lib/{name}.so", recursive=True)
      ):
        return path
  candidates = [f"{name}.so", f"{name}.so.*"]
  for directory in _library_search_dirs():
    for candidate in candidates:
      for path in sorted(glob.glob(str(directory / candidate))):
        return path
  return _find_library_ldconfig(name)


def _find_library_ldconfig(name: str) -> str | None:
  try:
    result = subprocess.run(
        ["ldconfig", "-p"], capture_output=True, text=True, check=False
    )
  except OSError:
    return None
  for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[0] == f"{name}.so":
      return parts[-1]
  return None


def _find_intel_omp() -> str | None:
  """Locate libiomp5 (Intel OpenMP) like the legacy setup script."""
  for path in sorted(glob.glob(f"{sys.prefix}/lib/libiomp5.so*")):
    return path
  prefix = os.environ.get("CONDA_PREFIX")
  if prefix and prefix != sys.prefix:
    for path in sorted(glob.glob(f"{prefix}/lib/libiomp5.so*")):
      return path
  if shutil.which("spack"):
    result = subprocess.run(
        ["spack", "location", "-i", "intel-oneapi-compilers"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
      base = result.stdout.strip()
      # The oneAPI spack prefix is enormous (hundreds of thousands of
      # files); a recursive glob blocks for minutes. Probe the two known
      # layout paths instead.
      for candidate in (
          f"{base}/compiler/latest/lib/libiomp5.so",
          f"{base}/lib/libiomp5.so",
      ):
        if os.path.isfile(candidate):
          return candidate
  return _find_library_ldconfig("libiomp5")


def _malloc_tunables(choice: str) -> dict[str, str]:
  if choice == "tcmalloc":
    # 0.01: madvise syscalls showed ~3% of inference time on 608-core ARM;
    # a lower release rate trades slightly higher RSS for fewer page-unmap
    # storms during steady-state loops.
    return {"TCMALLOC_RELEASE_RATE": "0.01"}
  if choice == "jemalloc":
    return {
        "MALLOC_CONF": (
            "oversize_threshold:1,background_thread:true,metadata_thp:auto,"
            "dirty_decay_ms:-1,muzzy_decay_ms:-1"
        )
    }
  if choice == "mimalloc":
    return {
        "MIMALLOC_ARENA_EAGER_COMMIT": "1",
        "MIMALLOC_PURGE_DELAY": "1000",
    }
  return {}


def derive_preload_env(opts: options_mod.Af3Options) -> dict[str, str]:
  """Environment entries derived from runtime.malloc / runtime.omp."""
  env: dict[str, str] = {}
  preloads: list[str] = []
  choice = opts.runtime.malloc
  if choice != "default":
    package, lib = _MALLOC_LIBS[choice]
    path = _find_library(package, lib)
    if path is None:
      raise SystemExit(
          f"[sparsefold] runtime.malloc={choice!r}: library {lib}.so not found"
      )
    preloads.append(path)
    env.update(_malloc_tunables(choice))
    print(f"[sparsefold] Using {choice} (path={path})")
  if opts.runtime.omp == "intel":
    path = _find_intel_omp()
    if path is None:
      raise SystemExit("[sparsefold] runtime.omp=intel: libiomp5.so not found")
    preloads.append(path)
    print(f"[sparsefold] Using Intel OpenMP (path={path})")
  if preloads:
    existing = os.environ.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = ":".join(
        [*preloads, existing] if existing else preloads
    )
  return env


def build_launch_env(opts: options_mod.Af3Options) -> dict[str, str]:
  env = {
      "PYTHONUNBUFFERED": "1",
      "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
      "TORCHDYNAMO_CAPTURE_DYNAMIC_OUTPUT_SHAPE_OPS": "1",
      "TORCHINDUCTOR_CACHE_DIR": str(
          Path(opts.runtime.torchinductor_cache).absolute()
      ),
  }
  # Keep the worker resolving tools (cmake, compilers) from the same
  # environment as this interpreter even when it was started by path.
  env["PATH"] = f"{sys.prefix}/bin{os.pathsep}{os.environ.get('PATH', '')}"
  Path(opts.runtime.torchinductor_cache).expanduser().mkdir(
      parents=True, exist_ok=True
  )
  env.update(derive_preload_env(opts))
  if opts.runtime.ld_library_paths:
    expanded = [str(Path(p).expanduser()) for p in opts.runtime.ld_library_paths]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [*expanded, existing] if existing else expanded
    )
  for name, value in opts.runtime.env.items():
    env[name] = str(Path(value).expanduser()) if value.startswith("~") else value
  return env


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
      "input", nargs="?", help="input name (overrides `input_name`)"
  )
  parser.add_argument(
      "--input-name",
      help="input name override (preferred over the legacy positional form)",
  )
  parser.add_argument(
      "-c", "--config", required=True, help="path to the TOML config file"
  )
  parser.add_argument(
      "--set",
      action="append",
      default=[],
      metavar="KEY=VALUE",
      help="override one config entry, e.g. --set launch.nproc=4",
  )
  parser.add_argument(
      "--nproc",
      metavar="N|numa",
      help=(
          "processes per node; 1 is a true non-distributed run, and 'numa' "
          "uses one process per selected CPU-bearing NUMA node"
      ),
  )
  parser.add_argument(
      "--numa-nodes",
      metavar="LIST",
      help=(
          "bind local ranks to these NUMA nodes, e.g. 6 or 6-9; "
          "overrides affinity.numa_nodes and implies affinity.mode=numa"
      ),
  )
  parser.add_argument(
      "--gdb",
      action="store_true",
      help="run the worker under gdb (world size must be one)",
  )


def _parse_nproc(raw: str) -> int | str:
  if raw == "numa":
    return raw
  try:
    value = int(raw)
  except ValueError as error:
    raise SystemExit("[sparsefold] --nproc expects a positive integer or 'numa'") from error
  if value < 1:
    raise SystemExit("[sparsefold] --nproc must be >= 1")
  return value


def _load_options(
    args: argparse.Namespace, extra: list[str]
) -> options_mod.Af3Options:
  opts = options_mod.load(args.config)
  for entry in args.set:
    if "=" not in entry:
      raise SystemExit(f"[sparsefold] --set expects KEY=VALUE, got {entry!r}")
    key, _, raw = entry.partition("=")
    options_mod.apply_override(opts, key, options_mod.parse_value(raw))
  options_mod._validate(opts)
  if args.input and getattr(args, "input_name", None):
    raise SystemExit(
        "[sparsefold] specify the input either positionally or with --input-name, "
        "not both"
    )
  selected_input = getattr(args, "input_name", None) or args.input
  if selected_input:
    opts.input_name = selected_input
  if getattr(args, "numa_nodes", None):
    opts.affinity.mode = "numa"
    opts.affinity.numa_nodes = affinity.parse_numa_nodes(args.numa_nodes)
    opts.affinity.compute_cpu_lists = []
    opts.affinity.comm_cpu_lists = []
    opts.affinity.memory_node_ids = []
  if getattr(args, "nproc", None) is not None:
    opts.launch.nproc = _parse_nproc(args.nproc)
  if opts.launch.nproc == "numa" and opts.affinity.mode == "none":
    opts.affinity.mode = "numa"
  options_mod._validate(opts)
  return opts


def _worker_arguments(
    args: argparse.Namespace,
    distributed: bool,
    nproc: int,
    output_dirs: list[Path],
) -> list[str]:
  argv = ["--config", args.config]
  for entry in args.set:
    argv += ["--set", entry]
  if getattr(args, "input_name", None):
    argv += ["--input-name", args.input_name]
  elif args.input:
    argv.append(args.input)
  if getattr(args, "numa_nodes", None):
    argv += ["--numa-nodes", args.numa_nodes]
  argv += ["--nproc-per-node", str(nproc)]
  for output_dir in output_dirs:
    argv += ["--run-output-dir", str(output_dir)]
  if distributed:
    argv.append("--distributed")
  argv.append("--")
  argv.extend(getattr(args, "extra", []))
  return argv


def _run_subprocess(
    cmd: list[str], env: dict[str, str], log_path: Path, tee: bool = True
) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"[sparsefold] {' '.join(cmd)}", flush=True)
  with log_path.open("w") as log:
    process = subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        stdout=subprocess.PIPE if tee else None,
        stderr=subprocess.STDOUT if tee else None,
        text=True,
        bufsize=1,
    )
    try:
      if tee:
        assert process.stdout is not None
        for line in process.stdout:
          sys.stdout.write(line)
          sys.stdout.flush()
          log.write(line)
        return process.wait()
      return process.wait()
    except KeyboardInterrupt:
      process.terminate()
      return process.wait()


def _worker_script() -> str:
  """Return the installed worker module; arbitrary file overrides were removed."""
  if os.environ.get("AF3_WORKER_SCRIPT"):
    raise SystemExit("AF3_WORKER_SCRIPT is no longer supported; use sparsefold.cli.worker")
  return "sparsefold.cli.worker"


def _apply_launch_memory_policy(opts: options_mod.Af3Options) -> None:
  """Set the memory policy before spawning so preloaded allocators inherit.

  mempolicy survives fork+exec, so applying it in the launcher equals
  `numactl` semantics: the worker's tcmalloc arenas (LD_PRELOAD) are built
  under the HBM-preferred policy from the very first allocation.
  """
  bind = opts.affinity
  if bind.mode == "none":
    return
  topology = affinity.read_topology()
  plan = affinity.plan_affinity(
      topology,
      bind.mode,
      0,
      comm_workers=bind.comm_workers,
      numa_nodes=bind.numa_nodes,
      compute_cpu_lists=bind.compute_cpu_lists,
      comm_cpu_lists=bind.comm_cpu_lists,
      memory_node_ids=bind.memory_node_ids,
      mem_policy=bind.mem_policy,
  )
  affinity.apply_memory(plan)


def _prime_launch_thread_env(
    opts: options_mod.Af3Options,
    topology: affinity.Topology,
    nproc: int,
    env: dict[str, str],
) -> None:
  """Publish thread settings before torchrun imports Torch.

  Torchrun otherwise injects ``OMP_NUM_THREADS=1`` before it execs workers.
  Updating the value later in ``worker.py`` is too late for any OpenMP runtime
  initialized during launcher imports.  Rank zero is representative for the
  automatic uniform-NUMA layout; every worker still recomputes and overwrites
  its complete rank-specific environment before importing the model.
  """
  bind = opts.affinity
  plan = affinity.plan_affinity(
      topology,
      bind.mode,
      0,
      comm_workers=bind.comm_workers,
      numa_nodes=bind.numa_nodes,
      compute_cpu_lists=bind.compute_cpu_lists,
      comm_cpu_lists=bind.comm_cpu_lists,
      memory_node_ids=bind.memory_node_ids,
      mem_policy=bind.mem_policy,
  )
  threads = affinity.derive_thread_count(
      plan,
      topology,
      configured=opts.runtime.threads,
      procs=nproc,
      comm_workers=bind.comm_workers,
  )
  env["OMP_NUM_THREADS"] = str(threads)
  env["MKL_NUM_THREADS"] = str(threads)
  # Do not publish OMP_PLACES/OMP_PROC_BIND in the launcher. Torchrun imports
  # Torch itself; binding that parent would make every fork/exec worker inherit
  # its first OpenMP place before worker.py can apply the rank-local NUMA mask.
  # The worker publishes placement only after sched_setaffinity succeeds.


def cmd_launch(args: argparse.Namespace, *, force_single: bool = False) -> int:
  opts = _load_options(args, args.extra)
  if force_single:
    opts.launch.nproc = 1
    opts.launch.num_nodes = 1
    opts.launch.node_rank = 0
  env = build_launch_env(opts)

  topology = affinity.read_topology()
  nproc = opts.resolved_nproc(len(topology.nodes))
  if (
      opts.affinity.mode == "numa"
      and opts.affinity.numa_nodes
      and nproc > len(opts.affinity.numa_nodes)
  ):
    raise SystemExit(
        f"[sparsefold] launch.nproc={nproc} exceeds the "
        f"{len(opts.affinity.numa_nodes)} selected NUMA nodes"
    )
  if (
      opts.affinity.mode == "explicit"
      and nproc > len(opts.affinity.compute_cpu_lists)
  ):
    raise SystemExit(
        f"[sparsefold] launch.nproc={nproc} exceeds the "
        f"{len(opts.affinity.compute_cpu_lists)} explicit CPU lists"
    )
  if (
      opts.affinity.memory_node_ids
      and nproc > len(opts.affinity.memory_node_ids)
  ):
    raise SystemExit(
        f"[sparsefold] launch.nproc={nproc} exceeds the "
      f"{len(opts.affinity.memory_node_ids)} explicit memory nodes"
    )
  _prime_launch_thread_env(opts, topology, nproc, env)
  world = opts.launch.num_nodes * nproc
  output = Path(opts.paths.output).expanduser().absolute()
  input_path = (
      Path(opts.paths.input).expanduser() / f"{opts.input_name}.json"
  )
  job_names = output_mod.input_job_names(input_path, opts.input_name)
  output_dirs = output_mod.reserve_output_dirs(
      output,
      job_names,
      force=bool(opts.model.get("force_output_dir", False)),
  )
  print(
      "[sparsefold] Run output " + ", ".join(str(path) for path in output_dirs),
      flush=True,
  )
  if args.gdb and world != 1:
    raise SystemExit("[sparsefold] --gdb requires a single-process launch")
  if world == 1:
    _apply_launch_memory_policy(opts)
  worker = str(_worker_script())
  baseline_root = (
      Path(opts.paths.baseline).expanduser().absolute()
      if opts.paths.baseline else None
  )
  manifest_path, manifest = run_manifest.start_run(
      output, input_path, baseline_root, Path(args.config), job_names,
      output_dirs, settings={
          "nproc_per_node": nproc,
          "num_nodes": opts.launch.num_nodes,
          "native_impl": opts.native.impl,
          "run_data_pipeline": opts.model.get("run_data_pipeline", True),
          "run_inference": opts.model.get("run_inference", True),
          "num_recycles": opts.model.get("num_recycles"),
          "num_diffusion_samples": opts.model.get("num_diffusion_samples"),
          "forwarded_model_args": list(args.extra),
      },
  )
  print(f"[sparsefold] Run record {manifest_path}", flush=True)

  def run_recorded(cmd: list[str], log_path: Path, *, tee: bool = True) -> int:
    try:
      status = _run_subprocess(cmd, env, log_path, tee=tee)
    except BaseException:
      run_manifest.finish_run(manifest_path, manifest, 1, interrupted=True)
      raise
    run_manifest.finish_run(manifest_path, manifest, status)
    return status

  if world == 1:
    cmd = [
        sys.executable, "-m",
        worker,
        *_worker_arguments(
            args, distributed=False, nproc=1, output_dirs=output_dirs
        ),
    ]
    if args.gdb:
      return run_recorded(["gdb", "--args", *cmd], output / "latest.log", tee=False)
    status = run_recorded(cmd, output / "latest.log")
    if status != 0:
      print(f"[sparsefold] launch failed with status {status}", file=sys.stderr)
    return status

  env["MASTER_ADDR"] = opts.launch.master_addr
  env["MASTER_PORT"] = str(opts.launch.master_port)

  d = opts.launch
  if d.launcher == "torchrun":
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nnodes={d.num_nodes}",
        f"--node-rank={d.node_rank}",
        f"--nproc-per-node={nproc}",
        f"--master-addr={d.master_addr}",
        f"--master-port={d.master_port}",
        "--module", worker,
        *_worker_arguments(
            args, distributed=True, nproc=nproc, output_dirs=output_dirs
        ),
    ]
  else:
    worker_cmd = [
        sys.executable, "-m", worker,
        *_worker_arguments(
            args, distributed=True, nproc=nproc, output_dirs=output_dirs
        ),
    ]
    if d.launcher == "openmpi":
      launcher = [
          "mpirun", "-np", str(world), "-npernode", str(nproc),
          "--bind-to", "none",
      ]
      for var in ("LD_PRELOAD", "LD_LIBRARY_PATH", "MASTER_ADDR",
                  "MASTER_PORT", "TORCHINDUCTOR_CACHE_DIR"):
        if env.get(var):
          launcher += ["-x", var]
    elif d.launcher == "intelmpi":
      launcher = [
          "mpirun", "-n", str(world), "-ppn", str(nproc),
          "-genv", "I_MPI_PIN", "0",
      ]
    else:  # mpich
      launcher = ["mpirun", "-np", str(world), "-ppn", str(nproc)]
    cmd = [*launcher, *worker_cmd]

  status = run_recorded(cmd, output / "latest-dist.log")
  if status != 0:
    print(f"[sparsefold] distributed launch failed with status {status}",
          file=sys.stderr)
  return status


def cmd_print_config(args: argparse.Namespace) -> int:
  print(options_mod.to_toml(_load_options(args, [])))
  return 0


def main(argv: list[str] | None = None) -> None:
  raw = list(sys.argv[1:] if argv is None else argv)
  # Split forwarded prediction flags off before argparse sees them so
  # that `-- --flag value` cannot be misread as launcher options.
  extra: list[str] = []
  if "--" in raw:
    split = raw.index("--")
    extra = raw[split + 1:]
    raw = raw[:split]

  force_single = False
  if raw and raw[0] in {"run", "dist"}:
    legacy = raw.pop(0)
    force_single = legacy == "run"
    print(
        f"[sparsefold] deprecated subcommand {legacy!r}; omit it and use --nproc",
        file=sys.stderr,
    )

  print_only = bool(raw and raw[0] == "print-config")
  if print_only:
    raw.pop(0)
  parser = argparse.ArgumentParser(
      prog="sparsefold",
      description="SparseFold inference launcher",
  )
  _add_common_arguments(parser)
  args = parser.parse_args(raw)
  args.extra = extra
  if print_only:
    raise SystemExit(cmd_print_config(args))
  raise SystemExit(cmd_launch(args, force_single=force_single))


if __name__ == "__main__":
  main()
