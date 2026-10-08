# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
"""Optional CPU native-output test against the fixed, separately installed AF3."""
import os
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sparsefold.integrations.af3.provider import AF3Provider
from sparsefold.schema.context import OutputContext
from sparsefold.output import decode_predictions, write_outputs, ResultsForSeed


@pytest.mark.skipif(not os.environ.get('SPARSEFOLD_AF3_SOURCE'), reason='Separate verified official AF3 environment required')
def test_official_atom_reordering_confidences_mmcif_roundtrip(tmp_path):
  provider = AF3Provider(source_root=os.environ['SPARSEFOLD_AF3_SOURCE'])
  structure = provider.module('alphafold3.structure')
  chemical = provider.module('alphafold3.structure.chemical_components')
  layout = provider.module('alphafold3.model.atom_layout.atom_layout')
  def obj(x):
    return np.array(x,dtype=object)
  chemical_data = chemical.ChemicalComponentsData(chem_comp={
      'LIG': chemical.ChemCompEntry(type='non-polymer',name='SYNTHETIC TEST LIGAND')})
  empty = structure.from_atom_arrays(name='synthetic', res_id=np.ones(4,dtype=np.int32),
      chain_id=obj(['A','A','B','B']), chain_type=obj(['non-polymer']*4),
      res_name=obj(['LIG']*4), atom_name=obj(['C1','N1','C1','N1']),
      atom_element=obj(['C','N','C','N']), chemical_components_data=chemical_data)
  token_layout = layout.AtomLayout(atom_name=obj([['N1','C1'],['N1','C1']]),
      res_id=np.ones((2,2),dtype=np.int32),chain_id=obj([['A','A'],['B','B']]),
      res_name=obj([['LIG']*2]*2),atom_element=obj([['N','C']]*2),chain_type=obj([['non-polymer']*2]*2))
  flat_layout = layout.AtomLayout(atom_name=obj(['C1','N1','C1','N1']),
      res_id=np.ones(4,dtype=np.int32),chain_id=obj(['A','A','B','B']),
      res_name=obj(['LIG']*4),atom_element=obj(['C','N','C','N']),chain_type=obj(['non-polymer']*4))
  context = OutputContext(token_atoms_layout=token_layout,flat_output_layout=flat_layout,empty_output_struc=empty)
  coords = torch.arange(12,dtype=torch.float32).reshape(1,2,2,3)
  coords = torch.cat([coords, coords + 100.], dim=0)
  batch = SimpleNamespace(frames=SimpleNamespace(mask=torch.ones(2,dtype=torch.bool)),
      token_features=SimpleNamespace(seq_length=torch.tensor(2),asym_id=torch.tensor([1,2]),residue_index=torch.tensor([1,1])))
  matrix = torch.stack([torch.full((2,2),.8),torch.full((2,2),.6)])
  result = {'diffusion_samples':{'atom_positions':coords},'predicted_lddt':torch.tensor([[[80.,90.],[70.,95.]],[[50.,60.],[40.,65.]]]),
      'full_pae':matrix*4,'full_pde':matrix*2,'tmscore_adjusted_pae_global':matrix,
      'tmscore_adjusted_pae_interface':matrix,'distogram':{'contact_probs':torch.ones(2,2)},
      'average_pde':torch.tensor([1.6,1.2]),'__identifier__':b'synthetic'}
  outputs = decode_predictions(batch,result,context,provider=provider)
  assert len(outputs) == 2
  expected_metadata = {'predicted_distance_error','ranking_score','fraction_disordered','has_clash',
      'predicted_tm_score','interface_predicted_tm_score','chain_pair_pde_mean','chain_pair_pde_min',
      'chain_pair_pae_min','ptm','iptm','ptm_iptm_average','intra_chain_single_pde','cross_chain_single_pde',
      'pae_ichain','pae_xchain','ranking_confidence','ranking_confidence_pae','chain_pair_iptm',
      'iptm_ichain','iptm_xchain','token_chain_ids','token_res_ids'}
  for i, output in enumerate(outputs):
    assert set(output.metadata) == expected_metadata
    assert set(output.numerical_data) == {'full_pde','full_pae','contact_probs'}
    assert output.model_id == b'synthetic'
    np.testing.assert_array_equal(output.predicted_structure.coords, coords[i].numpy().reshape(4,3)[[1,0,3,2]])
  np.testing.assert_array_equal(outputs[0].predicted_structure.atom_b_factor, [90.,80.,95.,70.])
  assert outputs[0].metadata['fraction_disordered'] == 0
  assert outputs[0].metadata['has_clash'] == False
  write_outputs([ResultsForSeed(seed=7,inference_results=outputs)], tmp_path,'synthetic','Test terms')
  assert (tmp_path/'seed-7_sample-1/synthetic_seed-7_sample-1_model.cif').exists()
  restored = structure.from_mmcif((tmp_path/'synthetic_model.cif').read_text(),include_other=True,include_water=True,include_bonds=True)
  np.testing.assert_allclose(restored.coords, outputs[0].predicted_structure.coords, rtol=0,atol=0)
  np.testing.assert_array_equal(restored.atom_name, outputs[0].predicted_structure.atom_name)
