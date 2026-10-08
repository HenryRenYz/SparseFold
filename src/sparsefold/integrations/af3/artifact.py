# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT

"""Versioned numerical NPZ + JSON context; never deserialize Python objects."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sparsefold.schema.batch import ModelBatch
from sparsefold.schema.context import OutputContext
from sparsefold.schema.validation import validate_mapping, REQUIRED_KEYS
from sparsefold.integrations.af3.provider import PreparedExample, ProviderError, SCHEMA_ID

ARTIFACT_VERSION = 1
LAYOUT_FIELDS = ('atom_name','res_id','chain_id','atom_element','res_name','chain_type')
STRUCTURE_FIELDS = frozenset({'cleaned_struc','empty_output_struc'})


def _encode_array(value):
    array=np.asarray(value)
    if array.dtype.kind=='O':
        if not all(v is None or isinstance(v,(str,int,float,bool)) for v in array.flat):
            raise TypeError('Context object arrays must contain only JSON scalar atom identities')
        dtype='object'
    else:
        if array.dtype.kind not in 'iufbUS': raise TypeError(f'Unsupported context array dtype {array.dtype}')
        dtype=array.dtype.str
    return {'dtype':dtype,'shape':list(array.shape),'values':array.tolist()}


def _decode_array(data):
    dtype=np.dtype(data['dtype'])
    if dtype.kind not in 'OiufbUS': raise ValueError('Unsafe context dtype')
    array=np.asarray(data['values'],dtype=dtype)
    expected_shape=tuple(data['shape'])
    if any(not isinstance(x,int) or x<0 for x in expected_shape): raise ValueError('Invalid context array dimensions')
    if array.size==0 and 0 in expected_shape:
        array=array.reshape(expected_shape)
    if array.shape!=expected_shape: raise ValueError('Context array shape does not match JSON schema')
    if dtype.kind=='O' and not all(v is None or isinstance(v,(str,int,float,bool)) for v in array.flat):
        raise ValueError('Context arrays may contain JSON scalars only')
    return array


def _structure_identity(value):
    return {name:_encode_array(getattr(value,name)) for name in ('chain_id','res_id','res_name','atom_name','atom_element')}


def encode_context(context):
    encoded={}
    for field in fields(context):
        value=getattr(context,field.name)
        if value is None: encoded[field.name]=None
        elif field.name in STRUCTURE_FIELDS:
            encoded[field.name]={'kind':'mmcif','mmcif':value.to_mmcif(),'identity':_structure_identity(value),'name':value.name}
        else:
            encoded[field.name]={'kind':'atom-layout','fields':{name:None if getattr(value,name) is None else _encode_array(getattr(value,name)) for name in LAYOUT_FIELDS}}
    return encoded


def decode_context(encoded,provider):
    expected={f.name for f in fields(OutputContext)}
    if set(encoded)!=expected: raise ValueError('OutputContext JSON field set is incompatible')
    restored={}
    for name,data in encoded.items():
        if data is None: restored[name]=None;continue
        if name in STRUCTURE_FIELDS:
            if data['kind']!='mmcif': raise ValueError('Structure context requires mmCIF encoding')
            structure=provider.module('alphafold3.structure')
            value=structure.from_mmcif(data['mmcif'],name=data['name'],include_water=True,include_other=True,include_bonds=True)
            for field,identity in data['identity'].items():
                if not np.array_equal(getattr(value,field),_decode_array(identity)):
                    raise ValueError(f'{name}: mmCIF restoration changed ordered atom identity field {field}')
            restored[name]=value
        else:
            if data['kind']!='atom-layout' or set(data['fields'])!=set(LAYOUT_FIELDS): raise ValueError('Invalid atom layout schema')
            layout=provider.module('alphafold3.model.atom_layout.atom_layout')
            restored[name]=layout.AtomLayout(**{f:None if a is None else _decode_array(a) for f,a in data['fields'].items()})
    return OutputContext(**restored)


def validate_context(context,model_batch):
    """Check the host atom identities against dense masks and output ordering."""
    token=context.token_atoms_layout
    flat=context.flat_output_layout
    structure=context.empty_output_struc
    if token is None or flat is None or structure is None:
        raise ValueError('Required output atom identities/structure are missing')
    expected_mask=model_batch.predicted_structure_info.atom_mask
    if isinstance(expected_mask,torch.Tensor):expected_mask=expected_mask.detach().cpu().numpy()
    atom_mask=np.asarray(token.atom_name).astype(bool)
    if not np.array_equal(atom_mask,expected_mask):
        raise ValueError('Token atom identities do not match dense prediction mask')
    if np.asarray(flat.atom_name).ndim!=1:
        raise ValueError('Flat output atom identity layout must be one-dimensional')
    for field in ('chain_id','res_id','atom_name'):
        if not np.array_equal(getattr(flat,field),getattr(structure,field)):
            raise ValueError(f'Flat atom layout and output structure differ in ordered {field}')
    def unique(layout,mask):
        ids=list(zip(np.asarray(layout.chain_id)[mask].tolist(),np.asarray(layout.res_id)[mask].tolist(),np.asarray(layout.atom_name)[mask].tolist()))
        if len(ids)!=len(set(ids)):
            raise ValueError('Output context contains duplicate active atom identities')
        return set(ids)
    token_ids=unique(token,atom_mask)
    flat_ids=unique(flat,np.asarray(flat.atom_name).astype(bool))
    if not flat_ids.issubset(token_ids):
        raise ValueError('Output atom identities cannot be reconstructed from model token atoms')
    return context


def save_prepared(example,path):
    """Write stem.npz and stem.json, overwriting neither existing artifact."""
    stem=Path(path)
    npz=stem.with_suffix('.npz');meta=stem.with_suffix('.json')
    if npz.exists() or meta.exists(): raise FileExistsError(f'Feature artifact already exists: {stem}')
    data=example.model_batch.as_data_dict()
    validate_mapping(data,strict=True)
    arrays={name:value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value) for name,value in data.items()}
    if any(a.dtype.hasobject for a in arrays.values()): raise TypeError('Object arrays forbidden in numerical NPZ')
    if example.output_context.token_atoms_layout is not None:
        validate_context(example.output_context,example.model_batch)
    context=encode_context(example.output_context)
    # Serialize before writing so malformed context cannot leave a partial NPZ.
    payload={'artifact_version':ARTIFACT_VERSION,'schema_id':SCHEMA_ID,'seed':int(example.seed),'provider':dict(example.provider_fingerprint),'context':context}
    json.dumps(payload,allow_nan=False)
    with npz.open('xb') as stream: np.savez_compressed(stream,**arrays)
    payload['npz_sha256']=hashlib.sha256(npz.read_bytes()).hexdigest()
    with meta.open('x',encoding='utf-8') as stream: json.dump(payload,stream,ensure_ascii=False,allow_nan=False,indent=2)
    return npz,meta


def load_numeric(path):
    """Read numeric input without importing official AF3, JAX, RDKit or structure."""
    stem=Path(path);npz=stem.with_suffix('.npz');meta=stem.with_suffix('.json')
    payload=json.loads(meta.read_text(encoding='utf-8'))
    if payload.get('artifact_version')!=ARTIFACT_VERSION or payload.get('schema_id')!=SCHEMA_ID:
        raise ValueError('Unsupported feature artifact version/schema')
    if hashlib.sha256(npz.read_bytes()).hexdigest()!=payload.get('npz_sha256'):
        raise ValueError('Feature artifact NPZ checksum mismatch')
    with np.load(npz,allow_pickle=False) as archive:
        if set(archive.files)!=set(REQUIRED_KEYS): raise ValueError('Numerical archive does not match explicit feature key schema')
        data={name:archive[name] for name in archive.files}
    validate_mapping(data,strict=True)
    return ModelBatch.from_data_dict(data),payload


def load_prepared(path,provider):
    model_batch,payload=load_numeric(path)
    fingerprint=provider.verify()
    saved=payload['provider']
    for key in ('revision','source_tree_sha256','schema_id'):
        if fingerprint.get(key)!=saved.get(key): raise ProviderError(f'Feature artifact upstream {key} differs from installed official provider')
    context=decode_context(payload['context'],provider)
    if tuple(context.token_atoms_layout.shape)!=tuple(model_batch.predicted_structure_info.atom_mask.shape):
        raise ValueError('Restored atom identity layout incompatible with model input')
    validate_context(context,model_batch)
    return PreparedExample(payload['seed'],model_batch,context,fingerprint)
