# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Tests for the unified direct/distributed launch decision."""

from __future__ import annotations

import datetime
from pathlib import Path
import subprocess
import sys

import pytest

from sparsefold.cli import affinity
from sparsefold.cli import launcher
from sparsefold.cli import output


def _config(tmp_path: Path, nproc: int) -> Path:
  path = tmp_path / "run.toml"
  path.write_text(
      f"""
input_name = "case"

[paths]
input = "{tmp_path}/input"
model = "{tmp_path}/model"
output = "{tmp_path}/output"

[launch]
nproc = {nproc}
launcher = "torchrun"
backend = "gloo"

[affinity]
mode = "none"
"""
  )
  return path


def _capture_launch(monkeypatch, tmp_path: Path, nproc: int) -> list[str]:
  config = _config(tmp_path, nproc)
  worker = tmp_path / "worker.py"
  worker.write_text("")
  captured: list[list[str]] = []

  monkeypatch.setattr(launcher, "build_launch_env", lambda opts: {})
  monkeypatch.setattr(launcher, "_worker_script", lambda: worker)
  monkeypatch.setattr(
      launcher.affinity,
      "read_topology",
      lambda: affinity.Topology(
          allowed=frozenset(range(8)),
          nodes={0: tuple(range(8))},
          physical_cores=8,
          smt_free=frozenset(range(8)),
      ),
  )

  def fake_run(cmd, env, log_path, tee=True):
    del env, log_path, tee
    captured.append(cmd)
    return 0

  monkeypatch.setattr(launcher, "_run_subprocess", fake_run)
  with pytest.raises(SystemExit) as result:
    launcher.main(["--config", str(config)])
  assert result.value.code == 0
  return captured[0]


def test_nproc_one_directly_executes_worker(monkeypatch, tmp_path: Path):
  command = _capture_launch(monkeypatch, tmp_path, nproc=1)
  assert "torch.distributed.run" not in command
  assert "--distributed" not in command
  assert command[command.index("--nproc-per-node") + 1] == "1"
  assert command[command.index("--run-output-dir") + 1].endswith(
      "/output/case"
  )
  assert command[-1] == "--"


def test_nproc_two_uses_torchrun(monkeypatch, tmp_path: Path):
  command = _capture_launch(monkeypatch, tmp_path, nproc=2)
  assert "torch.distributed.run" in command
  assert "--nproc-per-node=2" in command
  assert "--distributed" in command


def test_launcher_primes_openmp_before_torchrun(tmp_path: Path):
  opts = launcher.options_mod.load(_config(tmp_path, nproc=2))
  opts.affinity.mode = "numa"
  opts.affinity.comm_workers = 1
  opts.affinity.thread_pin = "spread"
  topology = affinity.Topology(
      allowed=frozenset(range(8)),
      nodes={0: (0, 1, 2, 3), 1: (4, 5, 6, 7)},
      physical_cores=8,
      smt_free=frozenset(range(8)),
  )
  env = {}
  launcher._prime_launch_thread_env(opts, topology, nproc=2, env=env)
  assert env["OMP_NUM_THREADS"] == "3"
  assert env["MKL_NUM_THREADS"] == "3"
  assert "OMP_PLACES" not in env
  assert "OMP_PROC_BIND" not in env


def test_cli_nproc_one_overrides_distributed_config(monkeypatch, tmp_path: Path):
  config = _config(tmp_path, nproc=4)
  worker = tmp_path / "worker.py"
  worker.write_text("")
  captured = []
  monkeypatch.setattr(launcher, "build_launch_env", lambda opts: {})
  monkeypatch.setattr(launcher, "_worker_script", lambda: worker)
  monkeypatch.setattr(
      launcher.affinity,
      "read_topology",
      lambda: affinity.Topology(
          allowed=frozenset(range(8)),
          nodes={0: tuple(range(8))},
          physical_cores=8,
          smt_free=frozenset(range(8)),
      ),
  )
  monkeypatch.setattr(
      launcher,
      "_run_subprocess",
      lambda cmd, env, log_path, tee=True: captured.append(cmd) or 0,
  )
  with pytest.raises(SystemExit) as result:
    launcher.main(["--config", str(config), "--nproc", "1"])
  assert result.value.code == 0
  assert "torch.distributed.run" not in captured[0]
  assert "--distributed" not in captured[0]


def test_output_directory_uses_json_name_and_timestamp(tmp_path: Path):
  input_path = tmp_path / "case.json"
  input_path.write_text('{"name": "436aa 1MUU"}')
  assert output.input_job_names(input_path, "case") == ["436aa_1MUU"]

  root = tmp_path / "output"
  existing = root / "436aa_1MUU"
  existing.mkdir(parents=True)
  (existing / "result.cif").write_text("old")
  chosen = output.reserve_output_dirs(
      root,
      ["436aa_1MUU"],
      now=datetime.datetime(2026, 9, 10, 15, 50, 9),
  )
  assert chosen == [root / "436aa_1MUU_20260910_155009"]
  assert chosen[0].is_dir()


def test_second_launch_does_not_reuse_an_empty_reserved_directory(tmp_path: Path):
  root = tmp_path / "output"
  first = output.reserve_output_dirs(root, ["case"])
  second = output.reserve_output_dirs(root, ["case"])
  assert first == [root / "case"]
  assert second[0] != first[0]


def test_console_tee_captures_python_and_fd_output(tmp_path: Path):
  path = tmp_path / "console.log"
  program = f"""
from pathlib import Path
import os
import subprocess
import sys
from sparsefold.cli.output import ConsoleTee
with ConsoleTee([Path({str(path)!r})]):
  print('python stdout', flush=True)
  os.write(2, b'native stderr\\n')
  subprocess.run(
      [sys.executable, '-c', "import os; os.write(1, b'compiler child\\\\n')"],
      check=True,
  )
"""
  completed = subprocess.run(
      [sys.executable, "-c", program], capture_output=True, text=True, check=True
  )
  assert "python stdout" in completed.stdout
  assert "native stderr" in completed.stderr
  assert "compiler child" in completed.stdout
  assert set(path.read_text().splitlines()) == {
      "python stdout", "native stderr", "compiler child"
  }
