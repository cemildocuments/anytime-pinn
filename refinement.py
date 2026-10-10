"""Deterministic refinement baseline and partition-resolution ablations."""
from pathlib import Path
import json,time,concurrent.futures
import numpy as np
from core import CASES
from enclosures import partition,load_weights,centered_bounds,squared_residual_bounds
ROOT=Path(__file__).resolve().parent

def chunk_bounds(lo,hi,ws,c,chunk=512):
 lowers=[];uppers=[]
 for start in range(0,len(lo),chunk):
  l=lo[start:start+chunk];h=hi[start:start+chunk];a,b=centered_bounds(l,h,ws,c);aa,bb=squared_residual_bounds(l,h,ws,c)
  lowers.append(np.maximum(a,aa));uppers.append(np.minimum(b,bb))
 return np.concatenate(lowers),np.concatenate(uppers)

def one(name):
 folder=ROOT/'results/trajectories'/name;cfg=json.loads((folder/'config.json').read_text());c=cfg['case'];dim=CASES[c]['dim'];ws=load_weights(folder/'checkpoint_03200.npz')
 lam=np.pi**2*CASES[c]['space']+CASES[c]['c'];rows=[];total=0
 for m in ([32,64,128,256,512,1024] if dim==1 else [8,16,32,64,128,256]):
  lo,hi=partition(dim,m);t=time.perf_counter();a,b=chunk_bounds(lo,hi,ws,c);dt=time.perf_counter()-t;total+=dt
  bound=float(np.sqrt(b.mean())/lam)
  rows.append({'m':m,'cells':len(a),'upper_mse':float(b.mean()),'width':float((b-a).mean()),'error_bound':bound,'seconds':dt,'cumulative_seconds':total})
  if bound<=.01:break
 out={'name':name,'case':c,'target':.01,'rows':rows,'certified':rows[-1]['error_bound']<=.01}
 dest=ROOT/'results/refinement';dest.mkdir(parents=True,exist_ok=True);(dest/(name+'.json')).write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 jobs=[f'{c}_w{w}d{d}_uniform_s101' for c in CASES for w,d in [(32,1),(24,2)]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  for r in ex.map(one,jobs):print(json.dumps(r),flush=True)
