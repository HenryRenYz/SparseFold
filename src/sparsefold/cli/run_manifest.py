# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Durable, launcher-owned record of the paths used by an AF3 invocation."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time
import uuid
from pathlib import Path


SCHEMA_VERSION = 1


def _write(path: Path, document: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(
      mode="w", encoding="utf-8", dir=path.parent, prefix=".run-",
      suffix=".tmp", delete=False,
  ) as stream:
    temporary = Path(stream.name)
    try:
      json.dump(document, stream, indent=2)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    except BaseException:
      temporary.unlink(missing_ok=True)
      raise
  os.replace(temporary, path)


def _signature(path: Path) -> list[int] | None:
  try:
    stat = path.stat()
  except FileNotFoundError:
    return None
  return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def start_run(
    output_root: Path, input_json: Path, baseline_root: Path | None,
    config_path: Path, job_names: list[str], output_dirs: list[Path],
    settings: dict | None = None,
) -> tuple[Path, dict]:
  started_ns = time.time_ns()
  run_id = f"{started_ns}-{uuid.uuid4().hex[:12]}"
  jobs = []
  for name, directory in zip(job_names, output_dirs, strict=True):
    model = directory / f"{name}_model.cif"
    jobs.append({
        "job_name": name,
        "output_dir": str(directory),
        "model_cif": str(model),
        "model_cif_before": _signature(model),
        "scorable": False,
    })
  document = {
      "schema_version": SCHEMA_VERSION,
      "run_id": run_id,
      "status": "running",
      "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
      "started_ns": started_ns,
      "completed_at": None,
      "completed_ns": None,
      "exit_code": None,
      "config": str(config_path.resolve()),
      "input_json": str(input_json.resolve()),
      "output_root": str(output_root.resolve()),
      "baseline_root": str(baseline_root.resolve()) if baseline_root else None,
      "settings": settings or {},
      "jobs": jobs,
  }
  path = output_root / ".runs" / f"{run_id}.json"
  _write(path, document)
  return path, document


def finish_run(path: Path, document: dict, exit_code: int, *, interrupted: bool = False) -> None:
  document["status"] = "interrupted" if interrupted else ("completed" if exit_code == 0 else "failed")
  document["exit_code"] = exit_code
  document["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
  document["completed_ns"] = time.time_ns()
  for job in document["jobs"]:
    model = Path(job["model_cif"])
    current = _signature(model)
    previous = job.pop("model_cif_before")
    job["scorable"] = bool(
        exit_code == 0 and not interrupted and current is not None
        and current != previous
    )
  _write(path, document)


def load_run(path: Path) -> dict:
  with path.open(encoding="utf-8") as stream:
    document = json.load(stream)
  if document.get("schema_version") != SCHEMA_VERSION:
    raise ValueError(f"unsupported run record schema: {path}")
  return document


def select_run(output_root: Path, *, latest_attempt: bool = False) -> tuple[dict, bool]:
  paths = sorted((output_root / ".runs").glob("*.json"))
  if not paths:
    raise ValueError(f"no run records found under {output_root}")
  runs = [load_run(path) for path in paths]
  attempts = sorted(runs, key=lambda run: run["started_ns"])
  if latest_attempt:
    return attempts[-1], False
  ready = [
      run for run in runs
      if run["status"] == "completed"
      and any(job["scorable"] for job in run["jobs"])
  ]
  if not ready:
    raise ValueError(f"no completed run with a new model CIF under {output_root}")
  selected = max(ready, key=lambda run: run["completed_ns"])
  newer_attempt = attempts[-1]["started_ns"] > selected["started_ns"]
  return selected, newer_attempt
