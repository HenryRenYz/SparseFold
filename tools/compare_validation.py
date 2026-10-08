# SPDX-FileCopyrightText: 2026 HenryRenYz
# SPDX-License-Identifier: MIT
"""All paired samples, fixed atom identities and untrimmed C-alpha rigid fit."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser
import tmtools


def atoms(path):
    model = next(MMCIFParser(QUIET=True).get_structure('prediction',str(path)).get_models())
    result = {}
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.element.upper() in {'H','D'}:
                    continue
                key = (chain.id,residue.id[1],residue.id[2].strip(),residue.resname,atom.name)
                if key in result:
                    raise ValueError(f'Duplicate selected atom identity: {key}')
                result[key] = np.asarray(atom.coord, dtype=np.float64)
    return result


def rigid_fit(moving, reference):
    cx,cy=moving.mean(axis=0),reference.mean(axis=0)
    u,_,vt=np.linalg.svd((moving-cx).T@(reference-cy))
    correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    translation=cy-cx@rotation
    rmsd=float(np.sqrt(np.mean(np.sum((moving@rotation+translation-reference)**2,axis=-1))))
    return rotation,translation,rmsd


def compare(prediction,baseline):
    p,b=atoms(prediction),atoms(baseline)
    shared=sorted(p.keys()&b.keys()); ca=[k for k in shared if k[-1]=='CA']
    if len(ca)<3:raise ValueError('Fewer than three matching C-alpha atoms')
    x=np.stack([p[k] for k in ca]);y=np.stack([b[k] for k in ca])
    rot,shift,rmsd=rigid_fit(x,y)
    full=np.stack([p[k] for k in shared])@rot+shift-np.stack([b[k] for k in shared])
    heavy_rmsd=float(np.sqrt(np.mean(np.sum(full**2,axis=-1))))
    from sparsefold.cli.structure_metrics import THREE_TO_ONE
    seq=''.join(THREE_TO_ONE.get(k[-2],'X') for k in ca)
    tm=tmtools.tm_align(np.ascontiguousarray(x),np.ascontiguousarray(y),seq,seq)
    return dict(matched_ca=len(ca),prediction_ca=sum(k[-1]=='CA' for k in p),baseline_ca=sum(k[-1]=='CA' for k in b),
                matched_heavy_atoms=len(shared),prediction_only_heavy_atoms=len(p.keys()-b.keys()),baseline_only_heavy_atoms=len(b.keys()-p.keys()),
                ca_rmsd_A=rmsd,ca_fit_heavy_rmsd_A=heavy_rmsd,tm_norm_prediction=float(tm.tm_norm_chain1),tm_norm_baseline=float(tm.tm_norm_chain2))


def model_files(directory):
    return sorted({*directory.glob("*_model.cif"), *directory.glob("model.cif")})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); rows=[]
    for job in ('107aa_1tce','436aa_1muu','740aa_4a5s'):
        candidates=[p for p in args.predictions.iterdir() if p.is_dir() and (p.name.lower()==job or p.name.lower().startswith(job+'_')) and all(len(model_files(p/f'seed-1_sample-{i}'))==1 for i in range(5))]
        if len(candidates)!=1:raise ValueError(f'Need exactly one successful target directory for {job}: {candidates}')
        target=candidates[0]
        for sample in range(5):
            slot=f'seed-1_sample-{sample}'
            predicted=model_files(target/slot)
            baseline=model_files(args.baseline/job/slot)
            if len(predicted)!=1 or len(baseline)!=1:raise ValueError(f'Missing or ambiguous paired sample: {job}/{slot}')
            row=dict(target=job,seed=1,sample=sample,prediction=str(predicted[0]),baseline=str(baseline[0]))
            row.update(compare(predicted[0],baseline[0]))
            row.update(prediction_sha256=hashlib.sha256(predicted[0].read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(baseline[0].read_bytes()).hexdigest())
            rows.append(row)
    report={'protocol':{'pairing':'same target, seed=1, sample=0..4; all pairs included','atoms':'first model, same chain/residue number/insertion/residue name/atom name; Bio.PDB-selected altloc; heavy atoms exclude H/D','rmsd':'Kabsch fit on ALL shared CA, no outlier removal; heavy RMSD uses the same CA fit','tm':'tmtools TM-align over all identity-matched CA, actual amino acid sequence; both normalization scores reported','baseline_limits':'historical baseline lacks exact upstream SHA/recycle provenance; Torch/JAX random streams differ','acceptance_threshold':'not declared; measurements are not an automatic numerical-equivalence claim'},'rows':rows,'summary':{}}
    for job in ('107aa_1tce','436aa_1muu','740aa_4a5s'):
        subset=[r for r in rows if r['target']==job]
        report['summary'][job]={key:{'mean':float(np.mean([r[key] for r in subset])),'min':min(r[key] for r in subset),'max':max(r[key] for r in subset)} for key in ('ca_rmsd_A','ca_fit_heavy_rmsd_A','tm_norm_prediction','tm_norm_baseline')}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'paired_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'paired_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
