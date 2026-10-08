# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Explicit numeric wire schema checks, performed before dtype conversion."""
from collections.abc import Mapping
import numpy as np
import torch

INTEGER_KEYS = frozenset("msa deletion_matrix num_alignments template_aatype residue_index token_index aatype entity_id asym_id sym_id seq_length residue_center_index ref_element ref_atom_name_chars ref_space_uid".split())
BOOL_KEYS = frozenset("msa_mask template_atom_mask seq_mask is_protein is_rna is_dna is_ligand is_nonstandard_polymer_chain is_water pred_dense_atom_mask ref_mask frames_mask".split())
FLOAT_KEYS = frozenset("profile deletion_mean template_atom_positions ref_pos ref_charge".split())
GATHER_PREFIXES = ("tokens_to_polymer_ligand_bonds", "token_atoms_to_polymer_ligand_bonds", "tokens_to_ligand_ligand_bonds", "token_atoms_to_pseudo_beta", "token_atoms_to_queries", "tokens_to_queries", "tokens_to_keys", "queries_to_keys", "queries_to_token_atoms")
REQUIRED_KEYS = INTEGER_KEYS | BOOL_KEYS | FLOAT_KEYS | frozenset(f"{p}:{s}" for p in GATHER_PREFIXES for s in ("gather_idxs", "gather_mask", "input_shape"))
HOST_KEYS = frozenset("cleaned_struc token_atoms_layout flat_output_layout empty_output_struc polymer_ligand_bonds ligand_ligand_bonds".split())


def _kind(value):
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool: return 'b'
        if value.is_floating_point(): return 'f'
        if value.is_complex(): return 'c'
        return 'i'
    return np.asarray(value).dtype.kind


def validate_mapping(batch: Mapping, *, strict=False):
    missing = REQUIRED_KEYS - batch.keys()
    if missing: raise ValueError(f"Missing model features: {sorted(missing)}")
    if strict and (extra := batch.keys() - REQUIRED_KEYS - HOST_KEYS):
        raise ValueError(f"Unrecognized feature schema fields: {sorted(extra)}")
    for key in REQUIRED_KEYS:
        kind = _kind(batch[key])
        expected = 'b' if key in BOOL_KEYS or key.endswith(':gather_mask') else 'f' if key in FLOAT_KEYS else 'i'
        if kind not in ({'i', 'u'} if expected == 'i' else {expected}):
            raise TypeError(f"Feature {key} requires {expected} dtype, got {kind}")
    def shape(key): return tuple(batch[key].shape)
    nshape = shape('aatype')
    if len(nshape) != 1: raise ValueError('aatype must have shape [tokens]')
    n = nshape[0]
    for key in (INTEGER_KEYS | BOOL_KEYS) - {'msa','msa_mask','deletion_matrix','num_alignments','template_aatype','template_atom_mask','seq_length','pred_dense_atom_mask','ref_mask','ref_element','ref_atom_name_chars','ref_space_uid'}:
        if shape(key) != (n,): raise ValueError(f'{key} must have shape [{n}]')
    if shape('seq_length') != () or shape('num_alignments') != ():
        raise ValueError('seq_length and num_alignments must be scalar')
    if len(shape('msa')) != 2 or shape('msa')[1] != n: raise ValueError('msa must have shape [rows,tokens]')
    for key in ('msa_mask','deletion_matrix'):
        if shape(key) != shape('msa'): raise ValueError(f'{key} must match msa shape')
    if shape('profile') != (n,31): raise ValueError('profile must have shape [tokens,31]')
    if shape('deletion_mean') != (n,): raise ValueError('deletion_mean must have shape [tokens]')
    dense = shape('pred_dense_atom_mask')
    if len(dense)!=2 or dense[0]!=n: raise ValueError('dense atom mask must have shape [tokens,atoms]')
    for key in ('ref_mask','ref_element','ref_charge','ref_space_uid'):
        if shape(key)!=dense: raise ValueError(f'{key} must match dense atom shape')
    if shape('ref_pos')!=dense+(3,): raise ValueError('ref_pos must end in xyz')
    if shape('ref_atom_name_chars')!=dense+(4,): raise ValueError('ref_atom_name_chars must end in 4 characters')
    ts=shape('template_aatype')
    if len(ts)!=2 or ts[1]!=n: raise ValueError('template_aatype must have shape [templates,tokens]')
    if shape('template_atom_mask')!=ts+(24,) or shape('template_atom_positions')!=ts+(24,3):
        raise ValueError('template atoms must use AF3 24-atom layout')
    for p in GATHER_PREFIXES:
        idx = torch.as_tensor(batch[p+':gather_idxs'])
        mask = torch.as_tensor(batch[p+':gather_mask'])
        src = torch.as_tensor(batch[p+':input_shape'])
        if idx.shape != mask.shape: raise ValueError(f'{p}: gather indices/mask shape mismatch')
        if src.ndim != 1 or not src.numel() or bool((src < 0).any()): raise ValueError(f'{p}: invalid input_shape')
        size = int(src.to(torch.int64).prod().item())
        if bool(((idx[mask] < 0) | (idx[mask] >= size)).any()): raise ValueError(f'{p}: valid gather index out of bounds')
        if bool((idx < 0).any()) or bool((idx >= max(size,1)).any()): raise ValueError(f'{p}: unsafe masked gather index')
    for key in FLOAT_KEYS:
        values=torch.as_tensor(batch[key])
        if not bool(torch.isfinite(values).all()): raise ValueError(f'{key}: nonfinite values')
        if bool((values.abs()>torch.finfo(torch.float32).max).any()): raise ValueError(f'{key}: exceeds float32 consumer range')
    for key in ('msa','deletion_matrix'):
        value=torch.as_tensor(batch[key]).to(torch.int64)
        if bool(((value<0)|(value>255)).any()): raise ValueError(f'{key}: values cannot be cast to uint8 losslessly')
    for key in ('msa','aatype','template_aatype'):
        values=torch.as_tensor(batch[key]).to(torch.int64)
        if bool(((values<0)|(values>=31)).any()): raise ValueError(f'{key}: invalid AF3 residue alphabet index')
    for key in REQUIRED_KEYS-INTEGER_KEYS-BOOL_KEYS-FLOAT_KEYS:
        if key.endswith(':gather_mask'): continue
        values=torch.as_tensor(batch[key]).to(torch.int64)
        if bool(((values<0)|(values>torch.iinfo(torch.int32).max)).any()): raise ValueError(f'{key}: exceeds int32 gather range')
    for key in INTEGER_KEYS-{'msa','deletion_matrix','aatype'}:
        values=torch.as_tensor(batch[key]).to(torch.int64)
        limits=torch.iinfo(torch.int32)
        if bool(((values<limits.min)|(values>limits.max)).any()): raise ValueError(f'{key}: exceeds int32 consumer range')
    return batch


def model_dtype(key):
    """Canonical PyTorch consumer dtype for every versioned numerical key."""
    if key in BOOL_KEYS or key.endswith(':gather_mask'): return torch.bool
    if key in FLOAT_KEYS: return torch.float32
    if key in ('msa','deletion_matrix'): return torch.uint8
    if key == 'aatype': return torch.int64
    return torch.int32
