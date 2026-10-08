# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Tests for the TOML options loader and the affinity planner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sparsefold.cli import affinity, options


@pytest.fixture
def base_toml(tmp_path: Path) -> Path:
  path = tmp_path / "base.toml"
  path.write_text(
      """
input_name = "436aa_1MUU"

[paths]
input = "/tmp/inputs"
model = "/tmp/model"
output = "/tmp/out"

[model]
use_cpu = true

[native]
ops = ["all"]

[launch]
nproc = 2
"""
  )
  return path


def test_load_and_defaults(base_toml: Path):
  opts = options.load(base_toml)
  assert opts.input_name == "436aa_1MUU"
  assert opts.runtime.malloc == "default"
  assert opts.native.matmul_dispatch is False
  assert opts.affinity.mode == "none"
  assert opts.precision.sdpa == "flash"


def test_extends_deep_merge(tmp_path: Path, base_toml: Path):
  child = tmp_path / "child.toml"
  child.write_text(
      f"""
extends = "{base_toml.name}"

[runtime]
malloc = "tcmalloc"

[launch]
master_port = 29700

[affinity]
mode = "numa"
comm_workers = 2
"""
  )
  opts = options.load(child)
  assert opts.input_name == "436aa_1MUU"  # inherited
  assert opts.runtime.malloc == "tcmalloc"
  assert opts.launch.nproc == 2  # inherited, not replaced
  assert opts.launch.master_port == 29700
  assert opts.affinity.mode == "numa"
  assert opts.affinity.comm_workers == 2


def test_unknown_top_level_key(base_toml: Path):
  path = base_toml.parent / "bad.toml"
  path.write_text(base_toml.read_text() + "\n[typo]\nx = 1\n")
  with pytest.raises(options.OptionsError, match="unknown top-level"):
    options.load(path)


def test_unknown_native_op(base_toml: Path):
  content = base_toml.read_text().replace(
      'ops = ["all"]', 'ops = ["grid_self_attn", "evoformer_single_act"]'
  )
  path = base_toml.parent / "badops.toml"
  path.write_text(content)
  with pytest.raises(options.OptionsError, match="unknown entries"):
    options.load(path)


def test_set_override(base_toml: Path):
  opts = options.load(base_toml)
  options.apply_override(opts, "launch.nproc", 4)
  options.apply_override(opts, "runtime.compile", "true")
  options.apply_override(
      opts, "native.ops", options.parse_value('["linear", "msa_attention"]')
  )
  assert opts.launch.nproc == 4
  assert opts.runtime.compile is True
  assert opts.native.ops == ["linear", "msa_attention"]


def test_numa_nodes_config_controls_auto_process_count(tmp_path: Path):
  path = tmp_path / "nodes.toml"
  path.write_text(
      """
[paths]
input = "/tmp/inputs"
model = "/tmp/model"
output = "/tmp/out"

[launch]
nproc = "numa"

[affinity]
mode = "numa"
numa_nodes = [6, 7, 9]
"""
  )
  opts = options.load(path)
  assert opts.resolved_nproc(cpu_node_count=16) == 3


def test_legacy_dist_schema_and_override_are_translated(tmp_path: Path):
  path = tmp_path / "legacy.toml"
  path.write_text(
      """
[paths]
input = "/tmp/inputs"
model = "/tmp/model"
output = "/tmp/out"

[dist]
procs_per_node = 2
backend = "gloo"

[dist.bind]
mode = "numa"
"""
  )
  opts = options.load(path)
  assert opts.launch.nproc == 2
  assert opts.affinity.mode == "numa"
  options.apply_override(opts, "dist.procs_per_node", 4)
  assert opts.launch.nproc == 4


def test_toml_roundtrip(base_toml: Path, tmp_path: Path):
  opts = options.load(base_toml)
  rendered = options.to_toml(opts)
  round_path = tmp_path / "roundtrip.toml"
  round_path.write_text(rendered)
  reloaded = options.load(round_path)
  assert reloaded == opts


# ---------------------------------------------------------------------------
# Affinity planning


def _topology(smt_free=None, nodes=None):
  # 8 logical CPUs; by default no SMT. With `smt_free`, CPUs 0-3 hold one
  # thread of every core and 4-7 their siblings (x86-style high siblings).
  if nodes is None:
    nodes = {i: tuple(range(i * 4, i * 4 + 4)) for i in range(2)}
  return affinity.Topology(
      allowed=frozenset(range(8)),
      nodes=nodes,
      physical_cores=len(smt_free) if smt_free is not None else 8,
      smt_free=frozenset(range(8)) if smt_free is None else frozenset(smt_free),
  )


def test_plan_numa_splits_comm_workers():
  plan = affinity.plan_affinity(
      _topology(), "numa", local_rank=1, comm_workers=1
  )
  assert plan.compute == (5, 6, 7)
  assert plan.comm == (4,)
  assert plan.memory_node == 1


def test_parse_and_select_numa_nodes():
  assert affinity.parse_numa_nodes("6,8-9") == [6, 8, 9]
  topo = _topology(nodes={6: (0, 1, 2, 3), 9: (4, 5, 6, 7)})
  plan = affinity.plan_affinity(
      topo, "numa", local_rank=1, numa_nodes=[9, 6]
  )
  assert plan.memory_node == 6
  assert plan.compute == (0, 1, 2, 3)


def test_numa_node_selection_rejects_invisible_node():
  with pytest.raises(affinity.AffinityError, match="not visible"):
    affinity.plan_affinity(
        _topology(), "numa", local_rank=0, numa_nodes=[7]
    )


def test_plan_numa_excludes_high_smt_siblings():
  topo = _topology(
      smt_free=[0, 1, 2, 3],
      nodes={0: (0, 1, 4, 5), 1: (2, 3, 6, 7)},
  )
  plan = affinity.plan_affinity(topo, "numa", local_rank=0, comm_workers=1)
  # Only the low siblings (0, 1) remain; the lowest CPU is the comm core.
  assert plan.compute == (1,)
  assert plan.comm == (0,)
  assert plan.memory_node == 0


def test_plan_explicit_keeps_manual_smt_cpus():
  topo = _topology(smt_free=[0, 1, 2, 3])
  plan = affinity.plan_affinity(
      topo, "explicit", local_rank=0, compute_cpu_lists=["0,4"]
  )
  assert plan.compute == (0, 4)  # explicit lists are honoured as-is


def test_plan_explicit_lists():
  plan = affinity.plan_affinity(
      _topology(),
      "explicit",
      local_rank=0,
      compute_cpu_lists=["0-1", "2-3"],
      comm_cpu_lists=["7", "6"],
  )
  assert plan.compute == (0, 1)
  assert plan.comm == (7,)


def test_plan_explicit_memory_node():
  topo = _topology()
  plan = affinity.plan_affinity(
      topo,
      "explicit",
      local_rank=1,
      compute_cpu_lists=["0-1", "4-5"],
      memory_node_ids=[0, 1],
  )
  assert plan.compute == (4, 5)
  assert plan.memory_node == 1


def test_plan_explicit_rank_out_of_range():
  with pytest.raises(affinity.AffinityError, match="no compute_cpu_lists"):
    affinity.plan_affinity(
        _topology(), "explicit", local_rank=5, compute_cpu_lists=["0-1"]
    )


def test_plan_numa_rank_exceeds_nodes():
  with pytest.raises(affinity.AffinityError, match="NUMA nodes"):
    affinity.plan_affinity(_topology(), "numa", local_rank=2)


def test_plan_none_is_unbound():
  plan = affinity.plan_affinity(_topology(), "none", local_rank=0)
  assert plan.unbound


def test_derive_env_thread_counts(monkeypatch):
  for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_AFFINITY"):
    monkeypatch.delenv(var, raising=False)
  plan = affinity.BindingPlan(0, (0, 1, 2), (3,), None, unbound=False)
  env = affinity.derive_env(plan, threads=3, thread_pin="none")
  assert env["OMP_NUM_THREADS"] == "3"
  assert env["KMP_AFFINITY"] == "disabled"
  assert env["VECOPS_BUILD_CPU_LIST"] == "0,1,2"
  assert env["VECOPS_THREAD_POOL_CPU_LIST"] == "0,1,2"
  assert env["VECOPS_NUM_THREADS"] == "3"
  assert env["KUTACC_SHM_COMM_CPU_LIST"] == "3"
  assert env["KUTACC_SHM_COPY_THREADS"] == "0"
  assert env["KUTACC_SHM_COPY_POOL_THREADS"] == "0"
  assert env["KUTACC_TORCH_MAX_COPY_THREADS"] == "0"

  two_comm = affinity.BindingPlan(
      0, (0, 1, 2), (3, 4), None, unbound=False
  )
  two_comm_env = affinity.derive_env(
      two_comm, threads=3, thread_pin="none"
  )
  assert two_comm_env["KUTACC_SHM_COPY_THREADS"] == "1"
  assert two_comm_env["KUTACC_SHM_COPY_POOL_THREADS"] == "1"
  assert two_comm_env["KUTACC_TORCH_MAX_COPY_THREADS"] == "1"

  pinned = affinity.derive_env(plan, threads=3, thread_pin="scatter")
  assert pinned["OMP_PLACES"] == "cores"
  assert pinned["OMP_PROC_BIND"] == "scatter"


def test_derive_thread_count_auto(monkeypatch):
  monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
  topo = _topology()  # 8 physical cores, no SMT
  bound = affinity.BindingPlan(0, (0, 1), (), 0, unbound=False)
  assert affinity.derive_thread_count(bound, topo, configured=0, procs=1) == 2
  unbound = affinity.BindingPlan(0, (), (), None, unbound=True)
  assert affinity.derive_thread_count(
      unbound, topo, configured=0, procs=2
  ) == 4
  assert affinity.derive_thread_count(
      unbound, topo, configured=7, procs=2
  ) == 7


def test_read_topology_on_linux():
  if not Path("/sys/devices/system/node").is_dir():
    pytest.skip("requires sysfs NUMA topology")
  topo = affinity.read_topology()
  assert topo.allowed == frozenset(os.sched_getaffinity(0))
  assert topo.physical_cores >= 1


def test_nearest_memory_only_node():
  topo = affinity.Topology(
      allowed=frozenset(range(4)),
      nodes={0: (0, 1, 2, 3)},
      physical_cores=4,
      smt_free=frozenset(range(4)),
      memory_only_nodes={16: 4096, 20: 4096},
      distances={0: tuple([10, 20] + [0] * 14 + [30, 40, 50, 60])},
  )
  assert topo.nearest_memory_only_node(0) == 16  # distance 30 < 40


def test_plan_numa_hbm_preferred():
  topo = affinity.Topology(
      allowed=frozenset(range(8)),
      nodes={0: (0, 1, 2, 3)},
      physical_cores=8,
      smt_free=frozenset(range(8)),
      memory_only_nodes={16: 4096},
      distances={0: tuple([10] + [0] * 15 + [30])},
  )
  plan = affinity.plan_affinity(
      topo, "numa", local_rank=0, mem_policy="hbm-preferred"
  )
  assert plan.preferred_node == 16
