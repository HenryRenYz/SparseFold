# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Reusable structure comparison metrics.

The command line layer intentionally lives elsewhere.  These functions return
data or raise a comparison exception, which makes them usable from tests and from
the batch runner without parsing or printing text.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from Bio.PDB import MMCIFParser
    from Bio.SVDSuperimposer import SVDSuperimposer
except ImportError as exc:  # pragma: no cover - depends on installation extras
    MMCIFParser = None
    SVDSuperimposer = None
    _BIO_IMPORT_ERROR = exc
else:
    _BIO_IMPORT_ERROR = None

try:
    import tmtools
except ImportError:  # pragma: no cover - exercised in dependency checks
    tmtools = None


DEFAULT_R0 = 15.0
DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "ASX": "B", "GLX": "Z", "SEC": "U", "PYL": "O",
    "UNK": "X",
}


class ComparisonError(Exception):
    """Base class for expected comparison failures."""


class ComparisonInputError(ComparisonError):
    """The input structure is missing, unreadable, or unusable."""


class ComparisonComputationError(ComparisonError):
    """A metric could not be computed for an otherwise valid input."""


@dataclass(frozen=True)
class Residue:
    number: int
    insertion_code: str
    name: str
    coord: np.ndarray


@dataclass(frozen=True)
class StructureData:
    path: Path
    coords: np.ndarray
    chains: dict[str, tuple[Residue, ...]]


def _parse_structure(path: Path):
    if MMCIFParser is None:
        raise ComparisonInputError(
            "CIF comparison requires the optional 'compare' dependencies; "
            "install with `pip install -e '.[compare]'`"
        ) from _BIO_IMPORT_ERROR
    if not path.is_file():
        raise ComparisonInputError(f"structure file does not exist: {path}")
    parser = MMCIFParser(QUIET=True)
    try:
        return parser.get_structure("struct", str(path))
    except Exception as exc:  # Biopython exposes several parser exceptions.
        raise ComparisonInputError(f"cannot parse CIF {path}: {exc}") from exc


def load_structure(path: str | Path) -> StructureData:
    """Load CA coordinates and chain residues from a mmCIF file."""
    path = Path(path)
    structure = _parse_structure(path)
    coords: list[np.ndarray] = []
    chains: defaultdict[str, list[Residue]] = defaultdict(list)

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" not in residue:
                    continue
                coord = np.asarray(residue["CA"].get_coord(), dtype=float)
                coords.append(coord)
                residue_id = residue.id
                hetero = residue_id[0].strip()
                if hetero and hetero.startswith("H_"):
                    continue
                chains[chain.id].append(
                    Residue(
                        number=int(residue_id[1]),
                        insertion_code=residue_id[2].strip(),
                        name=residue.resname.strip(),
                        coord=coord,
                    )
                )

    if not coords:
        raise ComparisonInputError(f"no C-alpha atoms found in {path}")

    for chain_residues in chains.values():
        chain_residues.sort(key=lambda item: (item.number, item.insertion_code))

    return StructureData(
        path=path,
        coords=np.asarray(coords, dtype=float),
        chains={key: tuple(value) for key, value in chains.items()},
    )


def load_ca_coordinates(path: str | Path) -> np.ndarray:
    """Compatibility-friendly helper returning only C-alpha coordinates."""
    return load_structure(path).coords


def calculate_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    *,
    cycles: int = 5,
    cutoff: float = 2.0,
) -> float:
    """Compute superposed RMSD with optional iterative outlier rejection."""
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if SVDSuperimposer is None:
        raise ComparisonComputationError(
            "RMSD requires the optional 'biopython' package"
        ) from _BIO_IMPORT_ERROR
    if coords1.shape[0] != coords2.shape[0] or coords1.shape[0] < 3:
        raise ComparisonComputationError(
            "RMSD requires equal C-alpha counts and at least three atoms"
        )

    sub_coords1 = coords1
    sub_coords2 = coords2
    sup = SVDSuperimposer()
    try:
        sup.set(sub_coords1, sub_coords2)
        sup.run()
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ComparisonComputationError(f"RMSD superposition failed: {exc}") from exc
    rmsd = float(sup.get_rms())

    for _ in range(cycles):
        if rmsd == 0:
            break
        transformed = sup.get_transformed()
        distances = np.linalg.norm(sub_coords1 - transformed, axis=1)
        mask = distances <= cutoff * rmsd
        if np.all(mask):
            break
        sub_coords1 = sub_coords1[mask]
        sub_coords2 = sub_coords2[mask]
        if len(sub_coords1) < 3:
            break
        sup.set(sub_coords1, sub_coords2)
        sup.run()
        rmsd = float(sup.get_rms())
    return rmsd


def calculate_tmscore(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute the larger of tmtools' two normalized TM-scores."""
    if tmtools is None:
        raise ComparisonComputationError("TM-score requires the optional 'tmtools' package")
    seq1 = "A" * len(coords1)
    seq2 = "A" * len(coords2)
    try:
        result = tmtools.tm_align(coords1, coords2, seq1, seq2)
        return float(max(result.tm_norm_chain1, result.tm_norm_chain2))
    except Exception as exc:
        raise ComparisonComputationError(f"TM-score alignment failed: {exc}") from exc


def _aa(residue: Residue) -> str:
    return THREE_TO_ONE.get(residue.name, "X")


def _find_residue_number_offset(
    model_residues: tuple[Residue, ...],
    reference_residues: tuple[Residue, ...],
) -> int:
    reference_by_number = {_res.number: _aa(_res) for _res in reference_residues}
    model_aas = [(res.number, _aa(res)) for res in model_residues]
    best_offset = 0
    best_identity = 0.0
    best_total = 0
    min_model = min(res.number for res in model_residues)
    max_model = max(res.number for res in model_residues)
    min_reference = min(res.number for res in reference_residues)
    max_reference = max(res.number for res in reference_residues)

    for offset in range(min_reference - max_model - 1, max_reference - min_model + 2):
        matches = 0
        total = 0
        for number, amino_acid in model_aas:
            reference_number = number + offset
            if reference_number in reference_by_number:
                total += 1
                matches += amino_acid == reference_by_number[reference_number]
        # Small synthetic/peptide structures still need a usable offset.  For
        # larger structures retain the historical minimum-overlap guard.
        minimum_overlap = min(5, len(model_residues), len(reference_residues))
        if total >= max(1, minimum_overlap):
            identity = matches / total
            if total > best_total or (total == best_total and identity > best_identity):
                best_total = total
                best_identity = identity
                best_offset = offset
    return best_offset


def match_residues_by_sequence(
    model_chains: dict[str, tuple[Residue, ...]],
    reference_chains: dict[str, tuple[Residue, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    model_coords: list[np.ndarray] = []
    reference_coords: list[np.ndarray] = []
    for chain_id in sorted(set(model_chains) & set(reference_chains)):
        model_residues = model_chains[chain_id]
        reference_residues = reference_chains[chain_id]
        if len(model_residues) < 2 or len(reference_residues) < 2:
            continue
        offset = _find_residue_number_offset(model_residues, reference_residues)
        reference_by_number = {res.number: res for res in reference_residues}
        for model_residue in model_residues:
            reference_residue = reference_by_number.get(model_residue.number + offset)
            if reference_residue and _aa(model_residue) == _aa(reference_residue):
                model_coords.append(model_residue.coord)
                reference_coords.append(reference_residue.coord)
    return _coordinate_pair(model_coords, reference_coords)


def match_residues_by_number(
    model_chains: dict[str, tuple[Residue, ...]],
    reference_chains: dict[str, tuple[Residue, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    model_coords: list[np.ndarray] = []
    reference_coords: list[np.ndarray] = []
    for chain_id in sorted(set(model_chains) & set(reference_chains)):
        reference_by_key = {
            (res.number, res.insertion_code): res for res in reference_chains[chain_id]
        }
        for model_residue in model_chains[chain_id]:
            reference_residue = reference_by_key.get(
                (model_residue.number, model_residue.insertion_code)
            )
            if reference_residue:
                model_coords.append(model_residue.coord)
                reference_coords.append(reference_residue.coord)
    return _coordinate_pair(model_coords, reference_coords)


def _coordinate_pair(
    model_coords: Iterable[np.ndarray], reference_coords: Iterable[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    model = list(model_coords)
    reference = list(reference_coords)
    if not model:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.asarray(model, dtype=float), np.asarray(reference, dtype=float)


def calculate_lddt(
    coords_model: np.ndarray,
    coords_reference: np.ndarray,
    *,
    r0: float = DEFAULT_R0,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> float:
    """Compute global mean lDDT for already matched residue coordinates."""
    n = coords_reference.shape[0]
    if n < 2 or coords_model.shape[0] != n:
        raise ComparisonComputationError("lDDT requires at least two matched residues")
    thresholds = tuple(thresholds)
    reference_diff = coords_reference[:, None, :] - coords_reference[None, :, :]
    reference_dist = np.sqrt(np.sum(reference_diff**2, axis=-1))
    model_diff = coords_model[:, None, :] - coords_model[None, :, :]
    model_dist = np.sqrt(np.sum(model_diff**2, axis=-1))

    per_residue = np.full(n, np.nan)
    for index in range(n):
        neighbors = np.where(reference_dist[index] < r0)[0]
        neighbors = neighbors[neighbors != index]
        if len(neighbors) == 0:
            continue
        differences = np.abs(reference_dist[index, neighbors] - model_dist[index, neighbors])
        per_residue[index] = np.mean(
            [np.mean(differences < threshold) for threshold in thresholds]
        )
    valid = ~np.isnan(per_residue)
    if not np.any(valid):
        raise ComparisonComputationError("lDDT has no residue pairs within the distance cutoff")
    return float(np.mean(per_residue[valid]))


def match_for_lddt(
    model: StructureData,
    reference: StructureData,
    *,
    align: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if align:
        return match_residues_by_sequence(model.chains, reference.chains)
    return match_residues_by_number(model.chains, reference.chains)
