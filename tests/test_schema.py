# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

import dataclasses
import numpy as np
import pytest
import torch
from sparsefold.schema.batch import ModelBatch
from sparsefold.schema.atom_layout import GatherInfo, convert
from sparsefold.schema.context import OutputContext
from sparsefold.schema.validation import INTEGER_KEYS,BOOL_KEYS,FLOAT_KEYS,GATHER_PREFIXES,REQUIRED_KEYS,validate_mapping
from sparsefold.integrations.af3.artifact import save_prepared,load_numeric
from sparsefold.integrations.af3.provider import PreparedExample


def numeric_fixture(n=3,a=2):
    data={key:np.zeros(n,dtype=np.int32) for key in INTEGER_KEYS}
    data.update({key:np.ones(n,dtype=bool) for key in BOOL_KEYS})
    data.update({key:np.zeros(n,dtype=np.float32) for key in FLOAT_KEYS})
    data['msa']=np.zeros((4,n),dtype=np.uint8)
    data['msa_mask']=np.ones((4,n),dtype=bool)
    data['deletion_matrix']=np.zeros((4,n),dtype=np.uint8)
    data['num_alignments']=np.array(4,dtype=np.int32)
    data['seq_length']=np.array(n,dtype=np.int32)
    data['profile']=np.zeros((n,31),dtype=np.float32)
    data['template_aatype']=np.zeros((2,n),dtype=np.int32)
    data['template_atom_positions']=np.zeros((2,n,24,3),dtype=np.float32)
    data['template_atom_mask']=np.ones((2,n,24),dtype=bool)
    for key in ['pred_dense_atom_mask','ref_mask']: data[key]=np.ones((n,a),dtype=bool)
    for key in ['ref_element','ref_space_uid']: data[key]=np.zeros((n,a),dtype=np.int32)
    data['ref_charge']=np.zeros((n,a),dtype=np.float32)
    data['ref_pos']=np.zeros((n,a,3),dtype=np.float32)
    data['ref_atom_name_chars']=np.zeros((n,a,4),dtype=np.int32)
    for p in GATHER_PREFIXES:
        data[p+':gather_idxs']=np.arange(n*a,dtype=np.int32).reshape(n,a)
        data[p+':gather_mask']=np.ones((n,a),dtype=bool)
        data[p+':input_shape']=np.array([n,a],dtype=np.int32)
    return data


def test_numeric_schema_roundtrip_and_host_separation():
    data=numeric_fixture(); marker=object();data['empty_output_struc']=np.array(marker,dtype=object)
    context=OutputContext.from_data_dict(data)
    batch=ModelBatch.from_data_dict(data)
    assert context.empty_output_struc is marker
    assert ModelBatch.from_data_dict(batch) is batch
    assert set(batch.as_data_dict())==set(REQUIRED_KEYS)
    assert batch.num_res==3
    assert not hasattr(batch,'convert_model_output')
    moved=batch.to_torch('cpu')
    assert all(isinstance(x,torch.Tensor) for x in moved.as_data_dict().values())
    for key,value in batch.as_data_dict().items(): torch.testing.assert_close(value,moved.as_data_dict()[key])
    indices=torch.tensor([3,1])
    reduced=batch.msa.index_msa_rows(indices)
    assert reduced.rows.shape==(2,3)
    assert reduced.profile is batch.msa.profile
    assert reduced.deletion_mean is batch.msa.deletion_mean
    assert batch.templates[0].aatype.shape==(3,)


@pytest.mark.parametrize('change,error',[
    (lambda d:d.pop('ref_pos'),ValueError),
    (lambda d:d.update(msa=np.full((4,3),300,dtype=np.int32)),ValueError),
    (lambda d:d.update(seq_mask=np.ones(3,dtype=np.int32)),TypeError),
    (lambda d:d.update(ref_pos=np.zeros((3,2,4),np.float32)),ValueError),
    (lambda d:d.update(profile=np.zeros((3,32),np.float32)),ValueError),
    (lambda d:d.update(ref_charge=np.full((3,2),np.nan,np.float32)),ValueError),
    (lambda d:d.update(msa=np.full((4,3),'bad',dtype=object)),TypeError),
    (lambda d:d.update({'tokens_to_keys:gather_idxs':np.full((3,2),6,np.int32)}),ValueError),
])
def test_schema_rejects_invalid_feature_semantics(change,error):
    data=numeric_fixture();change(data)
    with pytest.raises(error):ModelBatch.from_data_dict(data)


def test_gather_matches_layout_mask_and_preserves_autograd():
    gather=GatherInfo(torch.tensor([[2,0],[1,0]],dtype=torch.int32),torch.tensor([[1,1],[1,0]],dtype=torch.bool),torch.tensor([3],dtype=torch.int32))
    arr=torch.arange(9,dtype=torch.float32).reshape(3,3).requires_grad_()
    actual=convert(gather,arr)
    torch.testing.assert_close(actual,torch.tensor([[[6.,7.,8.],[0.,1.,2.]],[[3.,4.,5.],[0.,0.,0.]]]))
    actual.sum().backward();torch.testing.assert_close(arr.grad,torch.ones_like(arr))
    with pytest.raises(ValueError):convert(gather,torch.zeros(2,3))
    with pytest.raises(ValueError):convert(gather,arr,layout_axes=())
    with pytest.raises(ValueError):convert(gather,arr,layout_axes=(1,0))


def test_npz_numeric_artifact_roundtrip_and_corruption(tmp_path):
    batch=ModelBatch.from_data_dict(numeric_fixture())
    prepared=PreparedExample(11,batch,OutputContext(),{'revision':'test','source_tree_sha256':'test','schema_id':'sparsefold.model-batch.v1'})
    stem=tmp_path/'seed_11';npz,meta=save_prepared(prepared,stem)
    with np.load(npz,allow_pickle=False) as archive:
        assert set(archive.files)==set(REQUIRED_KEYS)
        assert all(not archive[name].dtype.hasobject for name in archive.files)
    restored,payload=load_numeric(stem)
    assert payload['seed']==11
    for key,value in batch.as_data_dict().items():torch.testing.assert_close(value,restored.as_data_dict()[key])
    with pytest.raises(FileExistsError):save_prepared(prepared,stem)
    npz.write_bytes(npz.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):load_numeric(stem)


def test_direct_numpy_typed_batch_converts_and_validates():
    from sparsefold.runtime import tree
    batch=ModelBatch.from_data_dict(numeric_fixture())
    typed_numpy=tree.map(lambda tensor:tensor.numpy(),batch)
    moved=typed_numpy.to_torch('cpu')
    assert all(isinstance(x,torch.Tensor) for x in tree.leaves(moved))
    assert moved.msa.rows.dtype==torch.uint8
    assert moved.token_features.aatype.dtype==torch.int64
    for key,value in batch.as_data_dict().items():torch.testing.assert_close(value,moved.as_data_dict()[key])
    bad_msa=dataclasses.replace(batch.msa,mask=torch.ones((4,3),dtype=torch.int32))
    with pytest.raises(TypeError,match='msa_mask'):ModelBatch.from_data_dict(dataclasses.replace(batch,msa=bad_msa))
    gather=batch.atom_cross_att.tokens_to_queries
    bad_gather=dataclasses.replace(gather,gather_mask=torch.ones(gather.gather_mask.shape,dtype=torch.float32))
    bad_cross=dataclasses.replace(batch.atom_cross_att,tokens_to_queries=bad_gather)
    with pytest.raises(TypeError,match='gather_mask'):dataclasses.replace(batch,atom_cross_att=bad_cross).to_torch()


def test_typed_tensor_noncanonical_dtype_is_repaired_without_bypassing_validation():
    batch=ModelBatch.from_data_dict(numeric_fixture())
    token=dataclasses.replace(batch.token_features,aatype=batch.token_features.aatype.int())
    ref=dataclasses.replace(batch.ref_structure,positions=batch.ref_structure.positions.double())
    typed=dataclasses.replace(batch,token_features=token,ref_structure=ref)
    converted=typed.to_torch()
    assert converted.token_features.aatype.dtype==torch.int64
    assert converted.ref_structure.positions.dtype==torch.float32


def test_atom_identity_context_roundtrip_retains_empty_bond_layout_shape():
    from types import SimpleNamespace
    from sparsefold.integrations.af3.artifact import encode_context,decode_context
    class Layout:
        def __init__(self,**fields):self.__dict__.update(fields)
        @property
        def shape(self):return self.atom_name.shape
    def make(shape):
        return Layout(atom_name=np.full(shape,'CA',object),chain_id=np.full(shape,'A',object),res_id=np.ones(shape,np.int32),atom_element=np.full(shape,'C',object),res_name=np.full(shape,'ALA',object),chain_type=np.full(shape,'polypeptide(L)',object))
    context=OutputContext(token_atoms_layout=make((3,2)),flat_output_layout=make((6,)),polymer_ligand_bonds=make((0,2)))
    provider=SimpleNamespace(module=lambda name:SimpleNamespace(AtomLayout=Layout))
    restored=decode_context(encode_context(context),provider)
    assert restored.polymer_ligand_bonds.shape==(0,2)
    for name in ['token_atoms_layout','flat_output_layout','polymer_ligand_bonds']:
        original=getattr(context,name);actual=getattr(restored,name)
        for field in ['atom_name','chain_id','res_id','atom_element','res_name','chain_type']:
            np.testing.assert_array_equal(getattr(original,field),getattr(actual,field))


def test_context_checks_dense_mask_and_structural_atom_order():
    from types import SimpleNamespace
    from sparsefold.integrations.af3.artifact import validate_context
    batch=ModelBatch.from_data_dict(numeric_fixture())
    token=SimpleNamespace(atom_name=np.array([['N','CA']]*3,object),chain_id=np.full((3,2),'A',object),res_id=np.repeat(np.arange(1,4)[:,None],2,axis=1))
    flat=SimpleNamespace(atom_name=token.atom_name.ravel(),chain_id=token.chain_id.ravel(),res_id=token.res_id.ravel())
    structure=SimpleNamespace(atom_name=flat.atom_name.copy(),chain_id=flat.chain_id.copy(),res_id=flat.res_id.copy())
    context=OutputContext(token_atoms_layout=token,flat_output_layout=flat,empty_output_struc=structure)
    assert validate_context(context,batch) is context
    structure.atom_name=structure.atom_name[::-1]
    with pytest.raises(ValueError,match='ordered atom_name'):validate_context(context,batch)
    structure.atom_name=flat.atom_name.copy()
    token.atom_name=token.atom_name.copy();token.atom_name[0,1]=''
    with pytest.raises(ValueError,match='prediction mask'):validate_context(context,batch)


@pytest.mark.parametrize('change,match',[
    (lambda data:data.update({'unused_loss':np.array(1,np.float32)}),'Unrecognized'),
    (lambda data:data.update(msa=np.full((4,3),31,np.uint8)),'alphabet'),
    (lambda data:data.update(ref_space_uid=np.full((3,2),2**31,np.int64)),'int32'),
    (lambda data:data.update(ref_charge=np.full((3,2),1e300,np.float64)),'float32'),
])
def test_schema_rejects_unknown_fields_and_lossy_numeric_casts(change,match):
    data=numeric_fixture();change(data)
    with pytest.raises(ValueError,match=match):ModelBatch.from_data_dict(data)


def test_official_signed_int8_msa_and_deletions_do_not_overflow_range_checks():
    data=numeric_fixture()
    data['msa']=np.full((4,3),21,np.int8)
    data['deletion_matrix']=np.full((4,3),127,np.int8)
    batch=ModelBatch.from_data_dict(data)
    assert batch.msa.rows.dtype==torch.uint8
    assert bool((batch.msa.rows==21).all())
    assert bool((batch.msa.deletion_matrix==127).all())
    data['deletion_matrix']=np.full((4,3),-1,np.int8)
    with pytest.raises(ValueError,match='losslessly'):ModelBatch.from_data_dict(data)


def test_gather_accepts_equal_volume_views_but_rejects_stride_changing_padding():
    gather=GatherInfo(torch.tensor([[0,4],[5,1]],dtype=torch.int32),torch.ones((2,2),dtype=torch.bool),torch.tensor([3,2],dtype=torch.int32))
    original=torch.arange(6,dtype=torch.float32).reshape(3,2)
    expected=convert(gather,original,layout_axes=(0,1))
    torch.testing.assert_close(convert(gather,original.reshape(2,3),layout_axes=(0,1)),expected)
    torch.testing.assert_close(convert(gather,original.flatten(),layout_axes=(0,)),expected)
    padded=torch.arange(8,dtype=torch.float32).reshape(4,2)
    torch.testing.assert_close(convert(gather,padded,layout_axes=(0,1)),expected)
    for invalid in [torch.arange(9).reshape(3,3),torch.arange(12).reshape(4,3),torch.arange(4).reshape(2,2)]:
        with pytest.raises(ValueError,match='incompatible'):convert(gather,invalid,layout_axes=(0,1))
