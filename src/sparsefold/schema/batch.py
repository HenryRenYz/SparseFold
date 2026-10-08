# SPDX-FileCopyrightText: 2024 DeepMind Technologies Limited
# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: Apache-2.0

"""Typed numerical input, independent of AF3 host-side structure objects."""
from typing import Self
from sparsefold.schema import features
from sparsefold.runtime import tree

@tree.dataclass(mappable_dataclass=False, frozen=True)
class ModelBatch:
    """Dataclass containing batch."""
    msa: features.MSA
    templates: features.Templates
    token_features: features.TokenFeatures
    ref_structure: features.RefStructure
    predicted_structure_info: features.PredictedStructureInfo
    polymer_ligand_bond_info: features.PolymerLigandBondInfo
    ligand_ligand_bond_info: features.LigandLigandBondInfo
    pseudo_beta_info: features.PseudoBetaInfo
    atom_cross_att: features.AtomCrossAtt
    frames: features.Frames

    @property
    def num_res(self) -> int:
        return self.token_features.aatype.shape[-1]

    @classmethod
    def from_data_dict(cls, batch: features.BatchDict) -> Self:
        """Construct batch object from dictionary."""
        from sparsefold.schema.validation import validate_mapping
        if isinstance(batch, cls):
            import numpy as np
            import torch
            numeric = tree.map(lambda value: torch.as_tensor(value) if isinstance(value, (np.ndarray, np.generic)) else value, batch)
            data = numeric.as_data_dict()
            validate_mapping(data, strict=True)
            from sparsefold.schema.validation import model_dtype
            if all(isinstance(value, torch.Tensor) and value.dtype == model_dtype(key) for key, value in data.items()) and all(isinstance(value, torch.Tensor) for value in tree.leaves(batch)):
                return batch
            batch = data
        validate_mapping(batch, strict=True)
        return cls(msa=features.MSA.from_data_dict(batch), templates=features.Templates.from_data_dict(batch), token_features=features.TokenFeatures.from_data_dict(batch), ref_structure=features.RefStructure.from_data_dict(batch), predicted_structure_info=features.PredictedStructureInfo.from_data_dict(batch), polymer_ligand_bond_info=features.PolymerLigandBondInfo.from_data_dict(batch), ligand_ligand_bond_info=features.LigandLigandBondInfo.from_data_dict(batch), pseudo_beta_info=features.PseudoBetaInfo.from_data_dict(batch), atom_cross_att=features.AtomCrossAtt.from_data_dict(batch), frames=features.Frames.from_data_dict(batch))

    def as_data_dict(self) -> features.BatchDict:
        """Converts batch object to dictionary."""
        output = {**self.msa.as_data_dict(), **self.templates.as_data_dict(), **self.token_features.as_data_dict(), **self.ref_structure.as_data_dict(), **self.predicted_structure_info.as_data_dict(), **self.polymer_ligand_bond_info.as_data_dict(), **self.ligand_ligand_bond_info.as_data_dict(), **self.pseudo_beta_info.as_data_dict(), **self.atom_cross_att.as_data_dict(), **self.frames.as_data_dict()}
        return output

    def to_torch(self, device=None) -> Self:
        """Move numerical leaves only; host metadata is held separately."""
        import torch
        canonical = type(self).from_data_dict(self)
        return tree.map(lambda value: value.to(device=device) if isinstance(value, torch.Tensor) else value, canonical)

    def to(self, device=None) -> Self:
        return self.to_torch(device)

Batch = ModelBatch
