# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import Atom, Chain, MMCIFIO, Model, Residue, Structure

from sparsefold.cli import run_manifest
from sparsefold.cli.structure_compare import main


def _cif(path: Path) -> None:
  structure = Structure.Structure(path.stem)
  model = Model.Model(0)
  chain = Chain.Chain("A")
  for index in range(4):
    residue = Residue.Residue((" ", index + 1, " "), "ALA", " ")
    residue.add(Atom.Atom(
        "CA", np.array([index * 3.8, 0, 0]), 1.0, 1.0,
        " ", " CA ", index + 1, "C",
    ))
    chain.add(residue)
  model.add(chain)
  structure.add(model)
  writer = MMCIFIO()
  writer.set_structure(structure)
  writer.save(str(path))


def test_explicit_compare_same_json_on_stdout_and_disk(tmp_path: Path, capsys):
  model, reference = tmp_path / "model.cif", tmp_path / "reference.cif"
  _cif(model)
  _cif(reference)
  output = tmp_path / "comparison.json"
  assert main(["compare", str(model), str(reference), "--format", "json",
               "--output", str(output)]) == 0
  stdout = capsys.readouterr().out
  assert stdout == output.read_text()
  result = json.loads(stdout)
  assert len(result) == 1
  assert result[0]["rmsd"] == 0.0
  assert result[0]["tm_score"] == 1.0
  assert result[0]["lddt"] == 1.0


def test_recorded_run_uses_actual_paths_and_skips_failed_attempt(tmp_path: Path, capsys):
  root = tmp_path / "overridden-output"
  baseline = tmp_path / "baseline"
  baseline_model = baseline / "case" / "case_model.cif"
  baseline_model.parent.mkdir(parents=True)
  _cif(baseline_model)
  output_dir = root / "case_timestamp"
  output_dir.mkdir(parents=True)
  config = tmp_path / "run.toml"
  config.write_text(f'''input_name = "case"
[paths]
input = "{tmp_path / 'input'}"
model = "{tmp_path / 'weights'}"
output = "{tmp_path / 'wrong-output'}"
baseline = "{baseline}"
''')
  path, run = run_manifest.start_run(
      root, tmp_path / "input" / "case.json", baseline, config,
      ["case"], [output_dir],
  )
  _cif(output_dir / "case_model.cif")
  run_manifest.finish_run(path, run, 0)
  failed_dir = root / "case_newer"
  failed_dir.mkdir()
  failed_path, failed = run_manifest.start_run(
      root, tmp_path / "input" / "case.json", baseline, config,
      ["case"], [failed_dir],
  )
  run_manifest.finish_run(failed_path, failed, 1)
  assert main(["compare", "--config", str(config),
               "--set", f"paths.output={root}", "--format", "json"]) == 0
  captured = capsys.readouterr()
  assert "newer attempt" in captured.err
  result = json.loads(captured.out)[0]
  assert result["run_id"] == run["run_id"]
  assert result["model"] == str(output_dir / "case_model.cif")
  assert main(["compare", "--run", str(failed_path),
               "--baseline-root", str(baseline)]) == 2
  assert "no completed scorable CIF" in capsys.readouterr().err


def test_reused_old_cif_is_not_scorable(tmp_path: Path):
  output_dir = tmp_path / "case"
  output_dir.mkdir()
  _cif(output_dir / "case_model.cif")
  path, run = run_manifest.start_run(
      tmp_path, tmp_path / "case.json", None, tmp_path / "run.toml",
      ["case"], [output_dir],
  )
  run_manifest.finish_run(path, run, 0)
  assert not run["jobs"][0]["scorable"]


def test_run_selection_supports_multiple_jobs(tmp_path: Path, capsys):
  root = tmp_path / "results"
  baseline = tmp_path / "baseline"
  directories = [root / "first_20260101", root / "second_20260101"]
  for name, directory in zip(("first", "second"), directories, strict=True):
    directory.mkdir(parents=True)
    reference = baseline / name / f"{name}_model.cif"
    reference.parent.mkdir(parents=True)
    _cif(reference)
  path, run = run_manifest.start_run(
      root, tmp_path / "input.json", baseline, tmp_path / "run.toml",
      ["first", "second"], directories,
  )
  for name, directory in zip(("first", "second"), directories, strict=True):
    _cif(directory / f"{name}_model.cif")
  run_manifest.finish_run(path, run, 0)
  assert main(["compare", "--run", str(path), "--format", "json"]) == 0
  assert [record["job_name"] for record in json.loads(capsys.readouterr().out)] == [
      "first", "second",
  ]
  assert main(["compare", "--run", str(path), "--job", "second",
               "--format", "json"]) == 0
  assert len(json.loads(capsys.readouterr().out)) == 1


def test_batch_csv_and_casefold_reference(tmp_path: Path, capsys):
  input_dir = tmp_path / "input" / "CASE_20260101"
  input_dir.mkdir(parents=True)
  _cif(input_dir / "CASE_model.cif")
  baseline = tmp_path / "baseline" / "case"
  baseline.mkdir(parents=True)
  _cif(baseline / "case_model.cif")
  output = tmp_path / "scores.csv"
  assert main(["batch", "--input-root", str(tmp_path / "input"),
               "--reference-root", str(tmp_path / "baseline"),
               "--format", "csv", "--output", str(output)]) == 0
  stdout = capsys.readouterr().out
  assert stdout == output.read_text()
  row = list(csv.DictReader(stdout.splitlines()))[0]
  assert row["job_name"] == "CASE"
  assert row["status"] == "ok"


def test_batch_reports_missing_model_and_selected_metric(tmp_path: Path, capsys):
  input_root = tmp_path / "input"
  good = input_root / "good"
  missing = input_root / "missing"
  good.mkdir(parents=True)
  missing.mkdir(parents=True)
  _cif(good / "good_model.cif")
  (missing / "missing_ranking_scores.csv").write_text("seed,sample,ranking_score\n")
  refs = tmp_path / "references"
  refs.mkdir()
  _cif(refs / "good.cif")
  _cif(refs / "missing.cif")
  assert main(["batch", "--input-root", str(input_root),
               "--reference-root", str(refs), "--metrics", "tm_score",
               "--format", "json"]) == 1
  records = json.loads(capsys.readouterr().out)
  assert [record["job_name"] for record in records] == ["good", "missing"]
  assert records[0]["rmsd"] is None and records[0]["lddt"] is None
  assert records[0]["tm_score"] == 1.0
  assert records[1]["status"] == "error"
  assert "does not exist" in records[1]["error"]
