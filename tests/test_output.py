# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
"""CPU output boundary tests; structures below are synthetic protocol fixtures."""
import csv
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sparsefold.output import confidence
from sparsefold.output.decode import to_numpy, _compute_ptm, _compute_chain_pair_iptm, decode_predictions
from sparsefold.output.types import InferenceResult, ResultsForSeed
from sparsefold.output import writers


def test_tensor_host_conversion_detaches_bfloat16_and_preserves_identifier():
  values = {'samples': [torch.tensor([1.5], dtype=torch.bfloat16, requires_grad=True)], 'id': b'weights'}
  out = to_numpy(values)
  assert out['samples'][0].dtype == np.float32
  np.testing.assert_array_equal(out['samples'][0], [1.5])
  assert out['id'] == b'weights'


def test_ptm_interface_and_padded_chain_pair_crop():
  tm = np.array([[[.9,.3,0], [.7,.8,0], [0,0,0]]])
  result = {'tmscore_adjusted_pae_global': tm}
  mask = np.ones((3,3), dtype=bool)
  ids = np.array([1,2])
  np.testing.assert_allclose(_compute_ptm(result, 2, ids, mask, False), [.75])
  np.testing.assert_allclose(_compute_ptm(result, 2, ids, mask, True), [.7])
  pair = _compute_chain_pair_iptm(2, np.array([1,2,0]), mask, tm)
  np.testing.assert_allclose(pair, [[[.9,.7],[.7,.8]]])


def test_single_chain_interface_undefined_and_ranking_formula():
  score = confidence.predicted_tm_score(np.ones((2,2)), np.ones((2,2), dtype=bool), np.ones(2), True)
  assert np.isnan(score)
  assert confidence.get_ranking_score(.8, score, .2, False) == pytest.approx(.9)
  assert confidence.get_ranking_score(.8, .6, .2, True) == pytest.approx(-99.26)


class SyntheticCif(dict):
  def copy_and_update(self, values):
    return SyntheticCif(self | values)
  def to_string(self):
    return 'data_synthetic\n# metadata\n' + json.dumps(self)


class SyntheticStructure:
  chain_id = np.array(['A','B'])
  atom_b_factor = np.array([82.345, 95.123])
  def iter_atoms(self):
    return iter([{'chain_id': 'A'}, {'chain_id': 'B'}])
  def to_mmcif_dict(self):
    return SyntheticCif({'_struct_asym.id': ['A','B'], '_struct_asym.entity_id': ['1','2'],
      '_atom_site.B_iso_or_equiv': ['82.345','95.123'], '_atom_site.type_symbol': ['C','N']})
  def unstack(self):
    return [self]


def synthetic_result(score):
  return InferenceResult(predicted_structure=SyntheticStructure(),
    numerical_data={'full_pae': np.array([[0.,3.],[4.,0.]]), 'contact_probs': np.ones((2,2))},
    metadata={'ptm': .8,'iptm': np.nan,'ranking_score':score,'fraction_disordered':.1,'has_clash':False,
      'chain_pair_pae_min':np.array([[0.,3.],[4.,0.]]),'chain_pair_iptm':np.array([[.8,.6],[.6,.9]]),
      'iptm_ichain':np.array([.8,.9]),'iptm_xchain':np.array([.6,.6]),
      'token_chain_ids':['A','B'],'token_res_ids':np.array([1,2])}, model_id=b'synthetic')


def test_writer_all_samples_best_ranking_terms_embeddings_distogram(tmp_path):
  results = [ResultsForSeed(seed=3, inference_results=[synthetic_result(.1),synthetic_result(.9)],
    embeddings={'single': torch.tensor([[1.,2.]], requires_grad=True)}, distogram=torch.ones(2,2,3))]
  writers.write_outputs(results, tmp_path, 'job', 'Synthetic test terms')
  summary = json.loads((tmp_path/'job_summary_confidences.json').read_text())
  assert summary['ranking_score'] == .9
  assert summary['iptm'] is None
  assert (tmp_path/'TERMS_OF_USE.md').read_text() == 'Synthetic test terms'
  assert (tmp_path/'seed-3_sample-0/job_seed-3_sample-0_model.cif').is_file()
  cif = (tmp_path/'job_model.cif').read_text()
  assert 'Non-commercial' in cif and 'SparseFold' in cif and 'DeepMind' in cif
  full = json.loads((tmp_path/'job_confidences.json').read_text())
  assert full['atom_chain_ids'] == ['A','B'] and full['token_res_ids'] == [1,2]
  assert full['atom_plddts'] == [82.34,95.12]
  with (tmp_path/'job_ranking_scores.csv').open() as f:
    assert list(csv.reader(f)) == [['seed','sample','ranking_score'],['3','0','0.1'],['3','1','0.9']]
  with np.load(tmp_path/'seed-3_embeddings/job_seed-3_embeddings.npz', allow_pickle=False) as f:
    np.testing.assert_array_equal(f['single'], [[1.,2.]])
  with np.load(tmp_path/'seed-3_distogram/job_seed-3_distogram.npz', allow_pickle=False) as f:
    assert f['distogram'].shape == (2,2,3) and f['distogram'].dtype == np.float16


def test_writer_empty_results_and_unsafe_name(tmp_path):
  writers.write_outputs([], tmp_path, 'job')
  assert not list(tmp_path.iterdir())
  with pytest.raises(ValueError, match='filename'):
    writers.write_outputs([], tmp_path, '../job')


def test_decode_complete_numerical_schema_at_structure_boundary(monkeypatch):
  import sparsefold.output.decode as decode
  from sparsefold.output.types import InferenceResult
  class Provider:
    def verify(self):
      pass
    def module(self, name):
      assert name == 'alphafold3.structure.mmcif'
      return SimpleNamespace(int_id_to_str_id=lambda i: 'AB'[int(i)-1])
  class SyntheticEnsemble:
    def unstack(self):
      return [SyntheticStructure(), SyntheticStructure()]
  monkeypatch.setattr(decode, 'get_predicted_structure', lambda **kwargs: SyntheticEnsemble())
  monkeypatch.setattr(confidence, 'has_clash', lambda s: False)
  monkeypatch.setattr(confidence, 'fraction_disordered', lambda s, **kwargs: .2)
  batch = SimpleNamespace(frames=SimpleNamespace(mask=torch.ones(3,dtype=torch.bool)),
      token_features=SimpleNamespace(seq_length=torch.tensor(2),asym_id=torch.tensor([1,2,0]),residue_index=torch.tensor([1,2,0])))
  matrix = torch.stack([torch.full((3,3), .8), torch.full((3,3), .6)])
  result = {'full_pae': matrix*4, 'full_pde': matrix*2,
    'tmscore_adjusted_pae_global':matrix,'tmscore_adjusted_pae_interface':matrix,
    'distogram':{'contact_probs':torch.ones(3,3)}, 'average_pde': torch.tensor([1.6,1.2]), '__identifier__':b'synthetic'}
  outputs = decode_predictions(batch,result,None,provider=Provider())
  assert len(outputs) == 2 and all(isinstance(o, InferenceResult) for o in outputs)
  assert outputs[0].metadata['ranking_score'] == pytest.approx(.9)
  assert outputs[0].metadata['token_chain_ids'] == ['A','B']
  assert outputs[0].metadata['chain_pair_iptm'].shape == (2,2)
  assert outputs[0].numerical_data['full_pae'].shape == (2,2)
  expected_metadata = {'predicted_distance_error','ranking_score','fraction_disordered','has_clash',
      'predicted_tm_score','interface_predicted_tm_score','chain_pair_pde_mean','chain_pair_pde_min',
      'chain_pair_pae_min','ptm','iptm','ptm_iptm_average','intra_chain_single_pde','cross_chain_single_pde',
      'pae_ichain','pae_xchain','ranking_confidence','ranking_confidence_pae','chain_pair_iptm',
      'iptm_ichain','iptm_xchain','token_chain_ids','token_res_ids'}
  assert outputs[1].metadata['ranking_score'] == pytest.approx(.7)
  for output in outputs:
    assert set(output.metadata) == expected_metadata
    assert set(output.numerical_data) == {'full_pde','full_pae','contact_probs'}
    assert output.model_id == b'synthetic'
    json.loads(writers.post_process_inference_result(output).structure_confidence_summary_json)


def test_output_import_does_not_load_official_structure_or_model():
  import subprocess, os
  proc = subprocess.run([sys.executable, '-c',
      "import sparsefold.output, sys; assert not any(n.startswith('alphafold3') for n in sys.modules); assert 'sparsefold.model.network' not in sys.modules"],
      env=os.environ.copy(), capture_output=True, text=True)
  assert proc.returncode == 0, proc.stderr
