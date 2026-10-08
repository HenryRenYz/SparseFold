# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Host-only reconstruction context; never part of model input or device transfer."""
from dataclasses import dataclass, fields
from typing import Any
import numpy as np

@dataclass(frozen=True)
class OutputContext:
    cleaned_struc: Any = None
    token_atoms_layout: Any = None
    flat_output_layout: Any = None
    empty_output_struc: Any = None
    polymer_ligand_bonds: Any = None
    ligand_ligand_bonds: Any = None

    @classmethod
    def from_data_dict(cls, data):
        def unwrap(value):
            if isinstance(value,np.ndarray) and value.dtype==object and value.shape==(): return value.item()
            return value
        return cls(**{f.name:unwrap(data.get(f.name)) for f in fields(cls)})

    def as_data_dict(self):
        return {f.name:getattr(self,f.name) for f in fields(self)}
