# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Compare predicted mmCIF structures with explicit or recorded references."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
from pathlib import Path

from . import options, run_manifest
from .structure_metrics import (
    ComparisonError, calculate_lddt, calculate_rmsd, calculate_tmscore,
    load_structure, match_for_lddt,
)

FIELDS = (
    "run_id", "job_name", "model", "reference", "ca_atoms",
    "matched_residues", "rmsd", "tm_score", "lddt", "status", "error",
)
METRICS = ("rmsd", "tm_score", "lddt")


def _metrics(raw: str) -> tuple[str, ...]:
  selected = tuple(dict.fromkeys(part.strip() for part in raw.split(",")))
  if not selected or any(metric not in METRICS for metric in selected):
    raise argparse.ArgumentTypeError(
        "--metrics must be a comma-separated subset of rmsd,tm_score,lddt"
    )
  return selected


def _common(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--metrics", type=_metrics, default=METRICS)
  parser.add_argument("--cycles", type=int, default=5)
  parser.add_argument("--cutoff", type=float, default=2.0)
  parser.add_argument("--no-align", action="store_true")
  parser.add_argument("--format", choices=("human", "csv", "json"), default="human")
  parser.add_argument("--output", type=Path, help="write the same format as stdout")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(prog="compare_structures")
  commands = parser.add_subparsers(dest="command", required=True)
  compare = commands.add_parser("compare", help="compare a pair or a recorded run")
  compare.add_argument("model", nargs="?", type=Path)
  compare.add_argument("reference", nargs="?", type=Path)
  compare.add_argument("-c", "--config", type=Path)
  compare.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
  compare.add_argument("--run", type=Path, help="exact .runs/<run_id>.json path")
  compare.add_argument("--output-root", type=Path, help="override the config output root")
  compare.add_argument("--baseline-root", type=Path, help="override the reference root")
  compare.add_argument("--job", help="select one job from a multi-job run")
  compare.add_argument("--latest-attempt", action="store_true")
  _common(compare)
  batch = commands.add_parser("batch", help="compare every model under a result tree")
  batch.add_argument("--input-root", type=Path, required=True)
  batch.add_argument("--reference-root", type=Path, required=True)
  batch.add_argument("--artifact-root", type=Path, help="optional copied artifact directory")
  _common(batch)
  return parser


def _reference(root: Path, job_name: str) -> Path:
  if root.is_file():
    return root
  if not root.is_dir():
    raise ValueError(f"reference root does not exist: {root}")
  exact = root / job_name / f"{job_name}_model.cif"
  if exact.is_file():
    return exact
  flat = root / f"{job_name}.cif"
  if flat.is_file():
    return flat
  candidates = [
      path for path in root.glob("*/*_model.cif")
      if path.parent.name.casefold() == job_name.casefold()
      and path.stem.removesuffix("_model").casefold() == job_name.casefold()
  ]
  candidates += [
      path for path in root.glob("*.cif")
      if path.stem.casefold() == job_name.casefold()
  ]
  if len(candidates) == 1:
    return candidates[0]
  if len(candidates) > 1:
    raise ValueError(f"ambiguous reference for {job_name}: {candidates}")
  raise ValueError(f"reference for {job_name} not found under {root}")


def _record(model_path: Path, reference_path: Path, *, job_name: str,
            run_id: str | None, args: argparse.Namespace) -> dict:
  result = dict.fromkeys(FIELDS)
  result.update(run_id=run_id, job_name=job_name, model=str(model_path),
                reference=str(reference_path), status="ok")
  try:
    model = load_structure(model_path)
    reference = load_structure(reference_path)
    result["ca_atoms"] = len(model.coords)
    if "rmsd" in args.metrics:
      result["rmsd"] = calculate_rmsd(
          model.coords, reference.coords, cycles=args.cycles, cutoff=args.cutoff
      )
    if "tm_score" in args.metrics:
      result["tm_score"] = calculate_tmscore(model.coords, reference.coords)
    if "lddt" in args.metrics:
      matched_model, matched_reference = match_for_lddt(
          model, reference, align=not args.no_align
      )
      result["matched_residues"] = len(matched_model)
      result["lddt"] = calculate_lddt(matched_model, matched_reference)
    for metric in args.metrics:
      if not math.isfinite(result[metric]):
        raise ValueError(f"{metric} is non-finite")
  except (ComparisonError, ValueError, OSError) as exc:
    result["status"] = "error"
    result["error"] = str(exc)
  return result


def _render(records: list[dict], fmt: str) -> str:
  if fmt == "json":
    return json.dumps(records, indent=2, allow_nan=False) + "\n"
  if fmt == "csv":
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue()
  lines = []
  for record in records:
    lines.append(f"Job: {record['job_name']}" + (
        f"  Run: {record['run_id']}" if record["run_id"] else ""
    ))
    lines += [f"Model: {record['model']}", f"Reference: {record['reference']}"]
    for metric, label in (("rmsd", "RMSD (Å)"), ("tm_score", "TM-score"),
                          ("lddt", "lDDT")):
      if record[metric] is not None:
        lines.append(f"{label}: {record[metric]:.4f}")
    lines.append(f"Status: {record['status']}")
    if record["error"]:
      lines.append(f"Error: {record['error']}")
    lines.append("")
  return "\n".join(lines).rstrip() + "\n"


def _compare(args: argparse.Namespace) -> list[dict]:
  if (args.model is None) != (args.reference is None):
    raise ValueError("provide both MODEL and REFERENCE, or neither")
  if args.model is not None:
    if any((args.config, args.run, args.output_root, args.baseline_root,
            args.job, args.latest_attempt, args.set)):
      raise ValueError("explicit MODEL REFERENCE cannot be combined with run selection")
    return [_record(args.model, args.reference,
                    job_name=args.model.stem.removesuffix("_model"),
                    run_id=None, args=args)]
  if not (args.config or args.run):
    raise ValueError("provide MODEL REFERENCE, --config, or --run")
  if args.run and (args.latest_attempt or args.output_root):
    raise ValueError("--run cannot be combined with --latest-attempt or --output-root")
  opts = options.load(args.config) if args.config else None
  if args.set and opts is None:
    raise ValueError("--set requires --config")
  if opts:
    for setting in args.set:
      if "=" not in setting:
        raise ValueError(f"--set expects KEY=VALUE: {setting}")
      key, _, raw = setting.partition("=")
      options.apply_override(opts, key, options.parse_value(raw))
  if args.run:
    run = run_manifest.load_run(args.run)
    newer = False
  else:
    root = args.output_root or Path(opts.paths.output).expanduser()
    run, newer = run_manifest.select_run(root, latest_attempt=args.latest_attempt)
  if newer:
    print("compare: a newer attempt exists without a completed scorable result",
          file=sys.stderr)
  if run["status"] != "completed" or not any(job["scorable"] for job in run["jobs"]):
    raise ValueError(f"run {run['run_id']} has no completed scorable CIF")
  baseline = args.baseline_root or (
      Path(opts.paths.baseline).expanduser() if opts and opts.paths.baseline else None
  ) or (Path(run["baseline_root"]) if run.get("baseline_root") else None)
  if baseline is None:
    raise ValueError("no baseline root; set paths.baseline or pass --baseline-root")
  jobs = [job for job in run["jobs"] if not args.job or job["job_name"] == args.job]
  if not jobs:
    raise ValueError(f"job {args.job!r} is absent from run {run['run_id']}")
  if baseline.is_file() and len(jobs) != 1:
    raise ValueError("a single reference CIF requires --job for a multi-job run")
  records = []
  for job in jobs:
    if not job["scorable"]:
      raise ValueError(f"job {job['job_name']} has no new CIF in run {run['run_id']}")
    records.append(_record(
        Path(job["model_cif"]), _reference(baseline, job["job_name"]),
        job_name=job["job_name"], run_id=run["run_id"], args=args,
    ))
  return records


def _batch(args: argparse.Namespace) -> list[dict]:
  if not args.input_root.is_dir():
    raise ValueError(f"input root does not exist: {args.input_root}")
  if not args.reference_root.is_dir():
    raise ValueError(f"reference root does not exist: {args.reference_root}")
  candidates: dict[tuple[Path, str], Path] = {}
  for suffix in ("_model.cif", "_ranking_scores.csv", "_confidences.json"):
    for path in args.input_root.rglob(f"*{suffix}"):
      if args.artifact_root and args.artifact_root.resolve() in path.resolve().parents:
        continue
      name = path.name.removesuffix(suffix)
      candidates[path.parent, name] = path.parent / f"{name}_model.cif"
  if not candidates:
    raise ValueError(f"no model or ranking artifacts found under {args.input_root}")
  records = []
  for (directory, name), model_path in sorted(candidates.items()):
    try:
      reference = _reference(args.reference_root, name)
      record = _record(model_path, reference, job_name=name, run_id=None, args=args)
    except ValueError as exc:
      record = dict.fromkeys(FIELDS)
      record.update(job_name=name, model=str(model_path), reference="",
                    status="error", error=str(exc))
    records.append(record)
    if args.artifact_root and record["status"] == "ok":
      destination = args.artifact_root / directory.relative_to(args.input_root)
      destination.mkdir(parents=True, exist_ok=True)
      shutil.copy2(model_path, destination / "model.cif")
      for source_suffix, target_name in (
          ("_ranking_scores.csv", "ranking_scores.csv"),
          ("_confidences.json", "confidences.json"),
      ):
        source = directory / f"{name}{source_suffix}"
        if source.is_file():
          shutil.copy2(source, destination / target_name)
  return records


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    records = _compare(args) if args.command == "compare" else _batch(args)
    rendered = _render(records, args.format)
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 1 if any(record["status"] != "ok" for record in records) else 0
  except (ValueError, OSError, ComparisonError) as exc:
    print(f"compare: error: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
