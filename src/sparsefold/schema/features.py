# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: Apache-2.0

"""Numerical model features; upstream feature generation stays external."""
from typing import Self, TypeAlias, Any
import torch
from sparsefold.runtime import tree
from sparsefold.schema import atom_layout

BatchDict: TypeAlias = dict[str, Any]
TorchBatchDict: TypeAlias = dict[str, torch.Tensor]

@tree.dataclass(mappable_dataclass=False, frozen=True)
class MSA:
    rows: torch.ByteTensor
    mask: torch.BoolTensor
    deletion_matrix: torch.ByteTensor
    profile: torch.FloatTensor
    deletion_mean: torch.FloatTensor
    num_alignments: torch.IntTensor

    def index_msa_rows(self, indices: torch.Tensor) -> Self:
        assert indices.ndim == 1
        return MSA(rows=self.rows[indices, :], mask=self.mask[indices, :], deletion_matrix=self.deletion_matrix[indices, :], profile=self.profile, deletion_mean=self.deletion_mean, num_alignments=self.num_alignments)

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        output = cls(rows=torch.as_tensor(batch['msa'], dtype=torch.uint8), mask=torch.as_tensor(batch['msa_mask'], dtype=torch.bool), deletion_matrix=torch.as_tensor(batch['deletion_matrix'], dtype=torch.uint8), profile=torch.as_tensor(batch['profile'], dtype=torch.float32), deletion_mean=torch.as_tensor(batch['deletion_mean'], dtype=torch.float32), num_alignments=torch.as_tensor(batch['num_alignments'], dtype=torch.int32))
        return output

    def as_data_dict(self) -> BatchDict:
        return {'msa': self.rows, 'msa_mask': self.mask, 'deletion_matrix': self.deletion_matrix, 'profile': self.profile, 'deletion_mean': self.deletion_mean, 'num_alignments': self.num_alignments}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class Templates:
    aatype: torch.IntTensor
    atom_positions: torch.FloatTensor
    atom_mask: torch.BoolTensor

    def __getitem__(self, index: int | slice | torch.Tensor) -> Self:
        return Templates(self.aatype[index], self.atom_positions[index], self.atom_mask[index])

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        """Make Template from batch dictionary."""
        return cls(aatype=torch.as_tensor(batch['template_aatype'], dtype=torch.int32), atom_positions=torch.as_tensor(batch['template_atom_positions'], dtype=torch.float32), atom_mask=torch.as_tensor(batch['template_atom_mask'], dtype=torch.bool))

    def as_data_dict(self) -> BatchDict:
        return {'template_aatype': self.aatype, 'template_atom_positions': self.atom_positions, 'template_atom_mask': self.atom_mask}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class TokenFeatures:
    residue_index: torch.IntTensor
    token_index: torch.IntTensor
    aatype: torch.LongTensor
    mask: torch.BoolTensor
    seq_length: torch.IntTensor
    asym_id: torch.IntTensor
    entity_id: torch.IntTensor
    sym_id: torch.IntTensor
    is_protein: torch.BoolTensor
    is_rna: torch.BoolTensor
    is_dna: torch.BoolTensor
    is_ligand: torch.BoolTensor
    is_nonstandard_polymer_chain: torch.BoolTensor
    is_water: torch.BoolTensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(residue_index=torch.as_tensor(batch['residue_index'], dtype=torch.int32), token_index=torch.as_tensor(batch['token_index'], dtype=torch.int32), aatype=torch.as_tensor(batch['aatype'], dtype=torch.int64), mask=torch.as_tensor(batch['seq_mask'], dtype=torch.bool), entity_id=torch.as_tensor(batch['entity_id'], dtype=torch.int32), asym_id=torch.as_tensor(batch['asym_id'], dtype=torch.int32), sym_id=torch.as_tensor(batch['sym_id'], dtype=torch.int32), seq_length=torch.as_tensor(batch['seq_length'], dtype=torch.int32), is_protein=torch.as_tensor(batch['is_protein'], dtype=torch.bool), is_rna=torch.as_tensor(batch['is_rna'], dtype=torch.bool), is_dna=torch.as_tensor(batch['is_dna'], dtype=torch.bool), is_ligand=torch.as_tensor(batch['is_ligand'], dtype=torch.bool), is_nonstandard_polymer_chain=torch.as_tensor(batch['is_nonstandard_polymer_chain'], dtype=torch.bool), is_water=torch.as_tensor(batch['is_water'], dtype=torch.bool))

    def as_data_dict(self) -> BatchDict:
        return {'residue_index': self.residue_index, 'token_index': self.token_index, 'aatype': self.aatype, 'seq_mask': self.mask, 'entity_id': self.entity_id, 'asym_id': self.asym_id, 'sym_id': self.sym_id, 'seq_length': self.seq_length, 'is_protein': self.is_protein, 'is_rna': self.is_rna, 'is_dna': self.is_dna, 'is_ligand': self.is_ligand, 'is_nonstandard_polymer_chain': self.is_nonstandard_polymer_chain, 'is_water': self.is_water}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class PredictedStructureInfo:
    atom_mask: torch.BoolTensor
    residue_center_index: torch.IntTensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(atom_mask=torch.as_tensor(batch['pred_dense_atom_mask'], dtype=torch.bool), residue_center_index=torch.as_tensor(batch['residue_center_index'], dtype=torch.int32))

    def as_data_dict(self) -> BatchDict:
        return {'pred_dense_atom_mask': self.atom_mask, 'residue_center_index': self.residue_center_index}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class PolymerLigandBondInfo:
    tokens_to_polymer_ligand_bonds: atom_layout.GatherInfo
    token_atoms_to_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: TorchBatchDict) -> Self:
        return cls(tokens_to_polymer_ligand_bonds=atom_layout.GatherInfo.from_dict(batch, key_prefix='tokens_to_polymer_ligand_bonds'), token_atoms_to_bonds=atom_layout.GatherInfo.from_dict(batch, key_prefix='token_atoms_to_polymer_ligand_bonds'))

    def as_data_dict(self) -> TorchBatchDict:
        return {**self.tokens_to_polymer_ligand_bonds.as_dict(key_prefix='tokens_to_polymer_ligand_bonds'), **self.token_atoms_to_bonds.as_dict(key_prefix='token_atoms_to_polymer_ligand_bonds')}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class LigandLigandBondInfo:
    tokens_to_ligand_ligand_bonds: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(tokens_to_ligand_ligand_bonds=atom_layout.GatherInfo.from_dict(batch, key_prefix='tokens_to_ligand_ligand_bonds'))

    def as_data_dict(self) -> BatchDict:
        return {**self.tokens_to_ligand_ligand_bonds.as_dict(key_prefix='tokens_to_ligand_ligand_bonds')}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class PseudoBetaInfo:
    token_atoms_to_pseudo_beta: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(token_atoms_to_pseudo_beta=atom_layout.GatherInfo.from_dict(batch, key_prefix='token_atoms_to_pseudo_beta'))

    def as_data_dict(self) -> BatchDict:
        return {**self.token_atoms_to_pseudo_beta.as_dict(key_prefix='token_atoms_to_pseudo_beta')}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class RefStructure:
    positions: torch.FloatTensor
    mask: torch.BoolTensor
    element: torch.IntTensor
    charge: torch.FloatTensor
    atom_name_chars: torch.IntTensor
    ref_space_uid: torch.IntTensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(positions=torch.as_tensor(batch['ref_pos'], dtype=torch.float32), mask=torch.as_tensor(batch['ref_mask'], dtype=torch.bool), element=torch.as_tensor(batch['ref_element'], dtype=torch.int32), charge=torch.as_tensor(batch['ref_charge'], dtype=torch.float32), atom_name_chars=torch.as_tensor(batch['ref_atom_name_chars'], dtype=torch.int32), ref_space_uid=torch.as_tensor(batch['ref_space_uid'], dtype=torch.int32))

    def as_data_dict(self) -> BatchDict:
        return {'ref_pos': self.positions, 'ref_mask': self.mask, 'ref_element': self.element, 'ref_charge': self.charge, 'ref_atom_name_chars': self.atom_name_chars, 'ref_space_uid': self.ref_space_uid}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class AtomCrossAtt:
    token_atoms_to_queries: atom_layout.GatherInfo
    tokens_to_queries: atom_layout.GatherInfo
    tokens_to_keys: atom_layout.GatherInfo
    queries_to_keys: atom_layout.GatherInfo
    queries_to_token_atoms: atom_layout.GatherInfo

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(token_atoms_to_queries=atom_layout.GatherInfo.from_dict(batch, key_prefix='token_atoms_to_queries'), tokens_to_queries=atom_layout.GatherInfo.from_dict(batch, key_prefix='tokens_to_queries'), tokens_to_keys=atom_layout.GatherInfo.from_dict(batch, key_prefix='tokens_to_keys'), queries_to_keys=atom_layout.GatherInfo.from_dict(batch, key_prefix='queries_to_keys'), queries_to_token_atoms=atom_layout.GatherInfo.from_dict(batch, key_prefix='queries_to_token_atoms'))

    def as_data_dict(self) -> BatchDict:
        return {**self.token_atoms_to_queries.as_dict(key_prefix='token_atoms_to_queries'), **self.tokens_to_queries.as_dict(key_prefix='tokens_to_queries'), **self.tokens_to_keys.as_dict(key_prefix='tokens_to_keys'), **self.queries_to_keys.as_dict(key_prefix='queries_to_keys'), **self.queries_to_token_atoms.as_dict(key_prefix='queries_to_token_atoms')}

@tree.dataclass(mappable_dataclass=False, frozen=True)
class Frames:
    mask: torch.BoolTensor

    @classmethod
    def from_data_dict(cls, batch: BatchDict) -> Self:
        return cls(mask=torch.as_tensor(batch['frames_mask'], dtype=torch.bool))

    def as_data_dict(self) -> BatchDict:
        return {'frames_mask': self.mask}
