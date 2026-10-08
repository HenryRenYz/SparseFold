# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Keep the documented TOML template synchronized with run_alphafold flags."""

from __future__ import annotations

import ast
from pathlib import Path
import re

from sparsefold.cli import options


_ROOT = Path(__file__).resolve().parents[1]
_RESERVED_MODEL_FLAGS = {
    "json_path",
    "input_dir",
    "output_dir",
    "model_dir",
    "db_dir",
    "dist",
    "dist_backend",
}


def _run_alphafold_flags() -> set[str]:
  tree = ast.parse((_ROOT / "src" / "sparsefold" / "cli" / "flags.py").read_text())
  result: set[str] = set()
  for node in tree.body:
    value = getattr(node, "value", None)
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "flags"
        and value.func.attr.startswith("DEFINE_")
    ):
      continue
    result.add(ast.literal_eval(value.args[0]))
  return result


def test_template_loads_and_documents_every_model_flag():
  path = _ROOT / "configs" / "template.toml"
  loaded = options.load(path)
  assert loaded.launch.nproc == 1

  model_text = path.read_text().split("[model]", 1)[1]
  mentioned = set(
      re.findall(r"^#?\s*([a-z][a-z0-9_]*)\s*=", model_text, re.MULTILINE)
  )
  all_flags = _run_alphafold_flags()
  assert all_flags <= mentioned | _RESERVED_MODEL_FLAGS
  assert set(loaded.model) <= all_flags - _RESERVED_MODEL_FLAGS
