"""Check supplied experimental records and recompute selected numerical targets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from core import CASES
from certify import betting_upper
from experiment import quadrature
from enclosures import load_weights
ROOT=Path(__file__).resolve().parent

def verify(check_hashes=True):
 if check_hashes:
  manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
  for name,want in manifest.items():
   got=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
   assert got==want,('Hash mismatch',name)
 folders=[]
 for p in (ROOT/'results/trajectories').glob('*/config.json'):
  c=json.loads(p.read_text())
  if 101<=c['seed']<=105:folders.append(p.parent)
 assert len(folders)==80,len(folders)
 count=0;bounds=0
 for folder in folders:
  for p in sorted((ROOT/'results/audits'/folder.name).glob('*.json')):
   a=json.loads(p.read_text());m=128 if CASES[a['case']]['dim']==1 else 64
   if a['m']!=m:continue
   count+=1;z=np.load(p.with_suffix('.npz'))
   assert np.all(z['lower']<=z['upper']) and np.all(z['lower']>=0)
   np.testing.assert_allclose(z['upper'].mean(),a['deterministic_upper'],rtol=1e-13)
   for sm,method in [('importance','betting'),('uniform','betting'),('importance','eb'),('importance','hoeffding')]:
    assert len(z[sm])==8192 and np.all((z[sm]>=0)&(z[sm]<=1))
    for r in a['methods'][sm]['rows']:
     assert r['bounds'][method]+1e-10>=a['reference']['residual_mse'];bounds+=1
   if a['step']==3200 and folder.name.endswith('w24d2_uniform_s101'):
    best=a['deterministic_upper'];B=a['methods']['importance']['B']
    for row in a['methods']['importance']['rows']:
     best=min(best,a['A']+B*betting_upper(z['importance'][:row['n']],a['alpha']))
     np.testing.assert_allclose(best,row['bounds']['betting'],rtol=1e-10,atol=1e-12)
    q=quadrature(load_weights(folder/'checkpoint_03200.npz'),a['case'],192)
    for key in q:np.testing.assert_allclose(q[key],a['reference'][key],rtol=1e-9,atol=1e-12)
 assert count==400 and bounds==14400,(count,bounds)
 stress=json.loads((ROOT/'results/stress/summary.json').read_text())
 assert len(stress['results'])==20 and all(x['repetitions']==20000 for x in stress['results'])
 print('Verified: 80 trajectories, 400 audits, 14,400 bounds, 80,000 monitoring repetitions, and selected independently recomputed numerical targets.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--numerical-only',action='store_true');a=p.parse_args();verify(not a.numerical_only)
