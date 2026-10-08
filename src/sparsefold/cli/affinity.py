# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""CPU affinity planning and application for AlphaFold CLI workers.

Replaces the former `dist-rank-wrapper.sh`: rank-to-CPU mapping is a pure,
unit-testable function of the machine topology, and each worker applies its
plan itself via `os.sched_setaffinity` before torch is imported. Memory-node
binding uses libnuma through ctypes (the same policy `numactl --membind`
applies) and warns instead of failing when libnuma is unavailable.

Standard library only: this module runs before torch is imported.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import os
from pathlib import Path

_SYSFS_CPU = Path("/sys/devices/system/cpu")
_SYSFS_NODE = Path("/sys/devices/system/node")


class AffinityError(RuntimeError):
  """Raised when a requested binding plan cannot be realized."""


def _parse_cpulist(text: str) -> set[int]:
  cpus: set[int] = set()
  for part in text.strip().split(","):
    if not part:
      continue
    if "-" in part:
      begin, end = part.split("-", 1)
      cpus.update(range(int(begin), int(end) + 1))
    else:
      cpus.add(int(part))
  return cpus


def parse_numa_nodes(text: str) -> list[int]:
  """Parse a Linux-style node list while preserving the requested order."""
  nodes: list[int] = []
  try:
    for part in text.strip().split(","):
      if not part:
        continue
      if "-" in part:
        begin_text, end_text = part.split("-", 1)
        begin, end = int(begin_text), int(end_text)
        if begin < 0 or end < begin:
          raise ValueError
        nodes.extend(range(begin, end + 1))
      else:
        node = int(part)
        if node < 0:
          raise ValueError
        nodes.append(node)
  except ValueError as error:
    raise AffinityError(
        f"invalid NUMA node list {text!r}; expected e.g. '0,2-4'"
    ) from error
  if not nodes:
    raise AffinityError("NUMA node list must not be empty")
  if len(set(nodes)) != len(nodes):
    raise AffinityError(f"NUMA node list contains duplicates: {text!r}")
  return nodes


def _format_cpulist(cpus: list[int]) -> str:
  return ",".join(str(c) for c in cpus)


@dataclasses.dataclass(frozen=True)
class Topology:
  """Logical-CPU view of the machine restricted to the allowed set."""

  allowed: frozenset[int]
  # CPU-bearing NUMA node id -> its logical CPUs (sorted, intersected).
  nodes: dict[int, tuple[int, ...]]
  # Distinct (package, core) pairs covered by the allowed set.
  physical_cores: int
  # One CPU per physical core (the lowest-numbered sibling). Identical to
  # `allowed` on non-SMT machines; on SMT hosts it drops the high
  # siblings so automatic plans give every thread a physical core.
  smt_free: frozenset[int]
  # Nodes with memory but no CPUs (e.g. dedicated HBM tiers), id -> MiB.
  memory_only_nodes: dict[int, int] = dataclasses.field(
      default_factory=dict
  )
  # node id -> distance vector indexed by node id.
  distances: dict[int, tuple[int, ...]] = dataclasses.field(
      default_factory=dict
  )

  def nearest_memory_only_node(self, node: int) -> int | None:
    """Return the memory-only node closest to `node` (its HBM tier)."""
    vector = self.distances.get(node, ())
    best: int | None = None
    best_distance: int | None = None
    for candidate in sorted(self.memory_only_nodes):
      if candidate >= len(vector):
        continue
      distance = vector[candidate]
      if best_distance is None or distance < best_distance:
        best, best_distance = candidate, distance
    return best


def read_topology() -> Topology:
  allowed = frozenset(os.sched_getaffinity(0))
  nodes: dict[int, tuple[int, ...]] = {}
  memory_only: dict[int, int] = {}
  distances: dict[int, tuple[int, ...]] = {}
  if _SYSFS_NODE.is_dir():
    for node_dir in sorted(_SYSFS_NODE.glob("node[0-9]*")):
      node_id = int(node_dir.name[4:])
      distance_file = node_dir / "distance"
      if distance_file.exists():
        try:
          distances[node_id] = tuple(
              int(x) for x in distance_file.read_text().split()
          )
        except (OSError, ValueError):
          pass
      cpulist = node_dir / "cpulist"
      cpus: set[int] = set()
      if cpulist.exists():
        cpus = _parse_cpulist(cpulist.read_text())
      if cpus & allowed:
        nodes[node_id] = tuple(sorted(cpus & allowed))
        continue
      if not cpus:
        try:
          mib = int((node_dir / "meminfo").read_text().split("MemTotal:")[1]
                    .split()[0]) // 1024
        except (OSError, IndexError, ValueError):
          mib = 0
        if mib > 0:
          memory_only[node_id] = mib

  smt_free: set[int] = set()
  seen_cores: set[tuple[int, int]] = set()
  for cpu in sorted(allowed):
    topo = _SYSFS_CPU / f"cpu{cpu}" / "topology"
    try:
      package = int((topo / "physical_package_id").read_text())
      core = int((topo / "core_id").read_text())
      core_key = (package, core)
    except (OSError, ValueError):
      core_key = (-1, cpu)
    if core_key in seen_cores:
      continue
    seen_cores.add(core_key)
    smt_free.add(cpu)
  return Topology(
      allowed=allowed,
      nodes=nodes,
      physical_cores=len(seen_cores),
      smt_free=frozenset(smt_free),
      memory_only_nodes=memory_only,
      distances=distances,
  )


@dataclasses.dataclass(frozen=True)
class BindingPlan:
  local_rank: int
  compute: tuple[int, ...]
  comm: tuple[int, ...]
  memory_node: int | None
  unbound: bool
  # Memory-only tier (HBM) preferred over the default policy, if any.
  preferred_node: int | None = None

  def describe(self) -> str:
    if self.unbound:
      return f"local_rank={self.local_rank} unbound"
    memory = (
        f"prefer={self.preferred_node}"
        if self.preferred_node is not None
        else f"membind={self.memory_node if self.memory_node is not None else '-'}"
    )
    return (
        f"local_rank={self.local_rank} "
        f"compute=[{_format_cpulist(list(self.compute))}] "
        f"comm=[{_format_cpulist(list(self.comm))}] {memory}"
    )


def plan_affinity(
    topology: Topology,
    mode: str,
    local_rank: int,
    *,
    comm_workers: int = 0,
    numa_nodes: list[int] | None = None,
    compute_cpu_lists: list[str] | None = None,
    comm_cpu_lists: list[str] | None = None,
    memory_node_ids: list[int] | None = None,
    mem_policy: str = "local",
) -> BindingPlan:
  """Map one local rank to compute/comm CPU sets following the bind mode."""
  if mode == "none":
    return BindingPlan(local_rank, (), (), None, unbound=True)

  if mode == "explicit":
    if not compute_cpu_lists:
      raise AffinityError("mode='explicit' requires compute_cpu_lists")
    if local_rank >= len(compute_cpu_lists):
      raise AffinityError(
          f"no compute_cpu_lists entry for local rank {local_rank} "
          f"({len(compute_cpu_lists)} entries given)"
      )
    compute = _parse_cpulist(compute_cpu_lists[local_rank]) & topology.allowed
    comm: set[int] = set()
    if comm_cpu_lists:
      if local_rank >= len(comm_cpu_lists):
        raise AffinityError(
            f"no comm_cpu_lists entry for local rank {local_rank} "
            f"({len(comm_cpu_lists)} entries given)"
        )
      comm = _parse_cpulist(comm_cpu_lists[local_rank]) & topology.allowed
    if not compute:
      raise AffinityError(
          f"compute CPU list for local rank {local_rank} is empty after "
          "intersecting with the allowed CPU set"
      )
    memory_node: int | None = None
    if memory_node_ids:
      if local_rank >= len(memory_node_ids):
        raise AffinityError(
            f"no memory_node_ids entry for local rank {local_rank} "
            f"({len(memory_node_ids)} entries given)"
        )
      memory_node = memory_node_ids[local_rank]
      available_memory_nodes = set(topology.nodes) | set(topology.memory_only_nodes)
      if memory_node not in available_memory_nodes:
        raise AffinityError(
            f"memory node {memory_node} is not visible; available nodes: "
            f"{sorted(available_memory_nodes)}"
        )
    return BindingPlan(
        local_rank,
        tuple(sorted(compute)),
        tuple(sorted(comm)),
        memory_node,
        False,
    )

  if mode == "numa":
    node_ids = list(numa_nodes) if numa_nodes else sorted(topology.nodes)
    missing = [node for node in node_ids if node not in topology.nodes]
    if missing:
      raise AffinityError(
          f"requested NUMA nodes are not visible CPU-bearing nodes: {missing}; "
          f"available nodes: {sorted(topology.nodes)}"
      )
    if local_rank >= len(node_ids):
      raise AffinityError(
          f"local rank {local_rank} exceeds the {len(node_ids)} CPU-bearing "
          "NUMA nodes"
      )
    node = node_ids[local_rank]
    # Automatic plans are SMT-free: every thread gets a physical core.
    node_cpus = tuple(
        cpu for cpu in topology.nodes[node] if cpu in topology.smt_free
    )
    if not node_cpus:
      node_cpus = topology.nodes[node]
    if comm_workers < 0 or comm_workers >= len(node_cpus):
      raise AffinityError(
          f"comm_workers={comm_workers} invalid for NUMA node {node} "
          f"with {len(node_cpus)} CPUs"
      )
    # Keep Kutacc communication workers on the lowest-numbered physical CPUs
    # in each NUMA node.  Besides matching the platform placement convention,
    # carving from the front makes the resulting mapping obvious in logs:
    # ``comm=[node_base..]`` and ``compute=[node_base + comm_workers..]``.
    split = comm_workers
    preferred = (
        topology.nearest_memory_only_node(node)
        if mem_policy == "hbm-preferred" else None
    )
    if mem_policy == "hbm-preferred" and preferred is None:
      print(
          "[affinity] no memory-only NUMA tier found; keeping local "
          "memory binding",
          flush=True,
      )
    return BindingPlan(
        local_rank,
        tuple(node_cpus[split:]),
        tuple(node_cpus[:split]),
        node,
        False,
        preferred,
    )

  raise AffinityError(f"unknown bind mode {mode!r}")


def _libnuma() -> ctypes.CDLL | None:
  name = ctypes.util.find_library("numa")
  if not name:
    return None
  try:
    return ctypes.CDLL(name)
  except OSError:
    return None


def apply(plan: BindingPlan) -> None:
  """Restrict the current process to the plan's compute CPUs (and memory)."""
  if plan.unbound:
    return
  apply_cpu(plan)
  apply_memory(plan)


def apply_cpu(plan: BindingPlan) -> None:
  """Restrict the current process to the plan's compute CPU set."""
  if plan.unbound:
    return
  os.sched_setaffinity(0, set(plan.compute))


def apply_memory(plan: BindingPlan) -> None:
  """Apply the plan's NUMA memory policy.

  Called from the launcher before spawning a worker so that preloaded
  allocators (e.g. tcmalloc) build their arenas under the policy, exactly
  like `numactl`; a later in-process re-apply is a harmless no-op.
  """
  if plan.unbound:
    return
  if plan.preferred_node is not None:
    lib = _libnuma()
    if lib is not None and hasattr(lib, "numa_set_preferred"):
      lib.numa_set_preferred.argtypes = [ctypes.c_int]
      lib.numa_set_preferred.restype = None
      lib.numa_set_preferred(plan.preferred_node)
      return
    print(
        "[affinity] libnuma unavailable; skipping HBM-preferred policy",
        flush=True,
    )
    return
  if plan.memory_node is None:
    return

  lib = _libnuma()
  if lib is None:
    print(
        "[affinity] libnuma not found; skipping memory-node binding "
        f"(compute binding to node {plan.memory_node} CPUs still applied)",
        flush=True,
    )
    return
  if not hasattr(lib, "numa_set_membind") or not hasattr(
      lib, "numa_allocate_nodemask"
  ):
    print(
        "[affinity] libnuma lacks membind symbols; skipping memory binding",
        flush=True,
    )
    return
  lib.numa_available.restype = ctypes.c_int
  if lib.numa_available() < 0:
    print("[affinity] NUMA policy unavailable; skipping memory binding",
          flush=True)
    return

  lib.numa_allocate_nodemask.restype = ctypes.c_void_p
  lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
  lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
  lib.numa_set_membind.argtypes = [ctypes.c_void_p]
  lib.numa_set_membind.restype = ctypes.c_int

  mask = lib.numa_allocate_nodemask()
  if not mask:
    raise AffinityError("numa_allocate_nodemask failed")
  try:
    lib.numa_bitmask_setbit(mask, plan.memory_node)
    if lib.numa_set_membind(mask) != 0:
      print(
          f"[affinity] numa_set_membind(node {plan.memory_node}) failed; "
          "memory stays unbound",
          flush=True,
      )
  finally:
    lib.numa_bitmask_free(mask)


def derive_thread_count(
    plan: BindingPlan,
    topology: Topology,
    *,
    configured: int,
    procs: int,
    comm_workers: int = 0,
) -> int:
  """Resolve the OpenMP team size honouring an explicit override."""
  if configured > 0:
    return configured
  if not plan.unbound:
    return len(plan.compute)
  if procs > 1:
    share = topology.physical_cores // procs - comm_workers
  else:
    share = min(len(topology.allowed), topology.physical_cores)
  return max(share, 1)


def derive_env(
    plan: BindingPlan, threads: int, thread_pin: str
) -> dict[str, str]:
  """Environment for OpenMP/Kutacc sized from the applied plan.

  `thread_pin='none'` keeps the historical behaviour: OpenMP-level affinity
  is disabled and threads may float within the process CPU set. The other
  modes enable standard per-thread binding via OMP_PLACES/OMP_PROC_BIND;
  note that JIT subprocesses may then inherit a narrower mask.
  """
  env: dict[str, str] = {
      "OMP_NUM_THREADS": str(threads),
      "MKL_NUM_THREADS": str(threads),
  }
  if not plan.unbound:
    # OMP_PROC_BIND may pin the Python/master thread to one place before a JIT
    # compiler is spawned. Vecops uses this full rank-local list to restore
    # CMake/Ninja subprocess affinity without relaxing model-process binding.
    env["VECOPS_BUILD_CPU_LIST"] = _format_cpulist(list(plan.compute))
    env["VECOPS_THREAD_POOL_CPU_LIST"] = _format_cpulist(list(plan.compute))
    env["VECOPS_NUM_THREADS"] = str(threads)
    if thread_pin == "none":
      env["KMP_AFFINITY"] = "disabled"
      for var in ("GOMP_CPU_AFFINITY", "OMP_PLACES", "OMP_PROC_BIND"):
        os.environ.pop(var, None)
    else:
      env["OMP_PLACES"] = "cores"
      env["OMP_PROC_BIND"] = thread_pin
      # Spin before sleeping: the model's steady state issues hundreds of
      # short parallel_for regions; futex sleep/wake syscalls accounted
      # for ~25% of inference time on many-core ARM. Overriding lets the
      # runtime tune the spin count per machine (unset = library default).
      spin = os.environ.get("AF3_OMP_SPINCOUNT")
      if spin is not None:
        env["GOMP_SPINCOUNT"] = spin
      if os.environ.get("AF3_OMP_WAIT_ACTIVE") == "1":
        env["OMP_WAIT_POLICY"] = "ACTIVE"
      os.environ.pop("KMP_AFFINITY", None)
      os.environ.pop("GOMP_CPU_AFFINITY", None)
    if plan.comm:
      comm_list = _format_cpulist(list(plan.comm))
      # Kutacc's serial context/progress thread runs on the first communication
      # CPU and participates in payload copies. The remaining CPUs host copy
      # helpers; counting the context CPU as another helper oversubscribes the
      # first core and delays collective sequence publication.
      copy_helpers = max(len(plan.comm) - 1, 0)
      env["KUTACC_SHM_COMM_CPU_LIST"] = comm_list
      env["KUTACC_SHM_COPY_THREADS"] = str(copy_helpers)
      env["KUTACC_SHM_COPY_POOL_THREADS"] = str(copy_helpers)
      env["KUTACC_TORCH_MAX_COPY_THREADS"] = str(copy_helpers)
  return env
