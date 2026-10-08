# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Output-directory reservation and process-wide console capture."""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import string
import sys
import threading
from pathlib import Path
from typing import BinaryIO


def _sanitise_name(name: str) -> str:
  """Match ``folding_input.Input.sanitised_name`` without importing AF3."""
  allowed = set(string.ascii_letters + string.digits + "_-.")
  return "".join(char for char in name.replace(" ", "_") if char in allowed)


def input_job_names(json_path: Path, fallback: str) -> list[str]:
  """Read fold-job names using only the standard library.

  AlphaFold JSON is either one mapping or an AlphaFold Server list of
  mappings. Malformed inputs are left to the model's full parser; the filename
  stem is sufficient to give their diagnostics a durable console log.
  """
  fallback_name = _sanitise_name(fallback) or "alphafold_run"
  try:
    document = json.loads(json_path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError):
    return [fallback_name]
  jobs = document if isinstance(document, list) else [document]
  names = [
      _sanitise_name(job["name"])
      for job in jobs
      if isinstance(job, dict) and isinstance(job.get("name"), str)
      and _sanitise_name(job["name"])
  ]
  return names or [fallback_name]


def reserve_output_dirs(
    output_root: Path,
    job_names: list[str],
    *,
    force: bool = False,
    now: datetime.datetime | None = None,
) -> list[Path]:
  """Create the exact directories used by this launch.

  Existing non-empty directories receive the same timestamp suffix used by
  ``sparsefold.cli.predict``. A numeric suffix makes simultaneous/same-second
  launches collision-safe.
  """
  output_root.mkdir(parents=True, exist_ok=True)
  stamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
  reserved: list[Path] = []
  # A newly created directory is otherwise empty until a worker starts.
  # Keep the check, mkdir, and marker creation under one cross-process lock.
  with (output_root / ".af3-reservation.lock").open("a+") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for job_name in job_names:
      base = output_root / job_name
      candidate = base
      if not force and candidate.exists() and any(candidate.iterdir()):
        candidate = output_root / f"{job_name}_{stamp}"
        collision = 1
        while candidate.exists():
          candidate = output_root / f"{job_name}_{stamp}_{collision}"
          collision += 1
      candidate.mkdir(parents=True, exist_ok=True)
      (candidate / ".af3-reserved").touch()
      reserved.append(candidate)
  return reserved


def _write_all(fd: int, data: bytes) -> None:
  view = memoryview(data)
  while view:
    try:
      written = os.write(fd, view)
    except (BrokenPipeError, OSError):
      return
    view = view[written:]


class ConsoleTee:
  """Tee fd-level stdout and stderr to one or more console log files.

  File-descriptor interception, rather than replacing ``sys.stdout``, also
  captures native code and compiler subprocesses launched by vecops.
  """

  def __init__(self, paths: list[Path]):
    self._paths = list(dict.fromkeys(paths))
    self._logs: list[BinaryIO] = []
    self._original: dict[int, int] = {}
    self._readers: list[tuple[int, threading.Thread]] = []
    self._lock = threading.Lock()
    self._closed = False

  def start(self) -> "ConsoleTee":
    if not self._paths:
      return self
    for path in self._paths:
      path.parent.mkdir(parents=True, exist_ok=True)
      self._logs.append(path.open("wb", buffering=0))
    sys.stdout.flush()
    sys.stderr.flush()
    try:
      # Always intercept the process-standard descriptors. Test runners and
      # logging proxies may expose wrapper objects whose ``fileno()`` does not
      # identify the underlying stdout/stderr slot.
      for fd in (1, 2):
        original = os.dup(fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)
        thread = threading.Thread(
            target=self._pump,
            args=(read_fd, original),
            name=f"af3-console-tee-{fd}",
            daemon=True,
        )
        self._original[fd] = original
        self._readers.append((read_fd, thread))
        thread.start()
    except Exception:
      self.close()
      raise
    return self

  def _pump(self, read_fd: int, original_fd: int) -> None:
    try:
      while True:
        chunk = os.read(read_fd, 64 * 1024)
        if not chunk:
          break
        _write_all(original_fd, chunk)
        with self._lock:
          for log_file in self._logs:
            _write_all(log_file.fileno(), chunk)
    finally:
      os.close(read_fd)

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      sys.stdout.flush()
      sys.stderr.flush()
    except Exception:
      pass
    for fd, original in self._original.items():
      try:
        os.dup2(original, fd)
      except OSError:
        pass
    for _, thread in self._readers:
      thread.join(timeout=5)
    for original in self._original.values():
      os.close(original)
    for log_file in self._logs:
      log_file.close()

  def __enter__(self) -> "ConsoleTee":
    return self.start()

  def __exit__(self, *_exc_info) -> None:
    self.close()
