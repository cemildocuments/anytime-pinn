"""Reproduce the predefined certificate benchmark on trained checkpoints."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,argparse,concurrent.futures
import numpy as np
from scipy.special import logsumexp
from core import CASES,evaluate_numpy
from enclosures import load_weights,partition,centered_bounds,squared_residual_bounds
from certify import betting_upper
ROOT=Path(__file__).resolve().parent
STEPS=[500,1000,2000,3000,3200]

def quadrature(ws,case,n):
 z,w=np.polynomial.legendre.leggauss(n);z=(z+1)/2;w=w/2;d=CASES[case]['dim']
 if d==1:x=z[:,None];ww=w
 else:a,b=np.meshgrid(z,z,indexing='ij');x=np.stack([a.ravel(),b.ravel()],axis=1);ww=np.outer(w,w).ravel()
 u,r,exact=evaluate_numpy(x,ws,case)
 return {'residual_mse':float(np.dot(ww,r*r)),'l2_error':float(np.sqrt(np.dot(ww,(u-exact)**2)))}

def one(job):
 name,step,m_override,nmax=job
 folder=ROOT/'results/trajectories'/name;cfg=json.loads((folder/'config.json').read_text());c=cfg['case'];k=STEPS.index(step)+1;alpha=.05/(k*(k+1));m=m_override or (128 if CASES[c]['dim']==1 else 64)
 dest=ROOT/'results/audits'/name;dest.mkdir(parents=True,exist_ok=True);f=dest/f'audit_{step:05d}_m{m}.json'
 if f.exists():return {'cached':str(f)}
 ws=load_weights(folder/f'checkpoint_{step:05d}.npz');lo,hi=partition(CASES[c]['dim'],m)
 t=time.perf_counter();an,bn=squared_residual_bounds(lo,hi,ws,c);natural_sec=time.perf_counter()-t
 t=time.perf_counter();ac,bc=centered_bounds(lo,hi,ws,c);center_sec=time.perf_counter()-t
 aa=np.maximum(an,ac);bb=np.minimum(bn,bc)
 if np.any(aa>bb):raise ValueError('Inconsistent interval enclosures')
 A=float(aa.mean());D=float((bb-aa).mean());det=float(bb.mean());bounds_sec=natural_sec+center_sec
 ref=quadrature(ws,c,128);ref2=quadrature(ws,c,192)
 rng=np.random.default_rng(100000*cfg['seed']+100*step+cfg['width']+cfg['depth']);results={};samples={}
 lam=np.pi**2*CASES[c]['space']+CASES[c]['c'];nnseq=[32*2**j for j in range(20) if 32*2**j<=nmax]
 for sampling in ['importance','uniform']:
  widths=bb-aa;M=len(aa);vol=1/M
  if D==0:p=np.full(M,vol);B=0.
  elif sampling=='importance':p=widths/widths.sum();B=D
  else:p=np.full(M,vol);B=float(widths.max())
  zz=[];rows=[];best={'betting':det,'eb':det,'hoeffding':det};sample_seconds=0;bound_seconds={'betting':0.,'eb':0.,'hoeffding':0.}
  for j,nn in enumerate(nnseq,1):
   add=nn-len(zz);ids=rng.choice(M,size=add,p=p);x=lo[ids]+rng.random((add,CASES[c]['dim']))*(hi[ids]-lo[ids])
   t=time.perf_counter();_,y=squared_residual_bounds(x,x,ws,c);sample_seconds+=time.perf_counter()-t
   y=np.minimum(y,bb[ids]);new=(y-aa[ids])/(M*p[ids]*B) if B>0 else np.zeros(add)
   if np.any(new< -1e-7) or np.any(new>1+1e-7):raise ValueError('Invalid normalized audit observations')
   zz.extend(np.clip(new,0,1).tolist());z=np.array(zz)
   t=time.perf_counter();bet=A+B*betting_upper(z,alpha);bound_seconds['betting']+=time.perf_counter()-t
   delta=alpha/(j*(j+1));t=time.perf_counter();ell=np.log(2/delta)
   eb=A+B*min(1,z.mean()+np.sqrt(2*z.var(ddof=1)*ell/nn)+7*ell/(3*(nn-1)));bound_seconds['eb']+=time.perf_counter()-t
   t=time.perf_counter();hf=A+B*min(1,z.mean()+np.sqrt(np.log(1/delta)/(2*nn)));bound_seconds['hoeffding']+=time.perf_counter()-t
   for method,value in [('betting',bet),('eb',eb),('hoeffding',hf)]:best[method]=min(best[method],value)
   rows.append({'n':nn,'mean_mse':A+B*float(z.mean()),'bounds':dict(best),'error_bounds':{me:float(np.sqrt(v)/lam) for me,v in best.items()},'seconds':{me:bounds_sec+sample_seconds+bound_seconds[me] for me in best}})
  results[sampling]={'B':B,'rows':rows};samples[sampling]=np.array(zz)
 out={'name':name,'case':c,'step':step,'checkpoint_index':k,'alpha':alpha,'m':m,'cells':len(aa),'A':A,'D':D,'deterministic_upper':det,'natural_upper':float(bn.mean()),'natural_seconds':natural_sec,'centered_seconds':center_sec,'reference':ref2,'quadrature_difference':{q:abs(ref[q]-ref2[q]) for q in ref},'methods':results}
 f.write_text(json.dumps(out,indent=2));np.savez_compressed(dest/f'audit_{step:05d}_m{m}.npz',lower=aa,upper=bb,natural_lower=an,natural_upper=bn,**samples)
 return {'name':name,'step':step,'m':m,'bound':results['importance']['rows'][-1]['error_bounds']['betting'],'actual':ref2['l2_error'],'D':D}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--nmax',type=int,default=8192);p.add_argument('--m',type=int,default=0);p.add_argument('--final-only',action='store_true');p.add_argument('--pilot',action='store_true');a=p.parse_args()
 folders=[]
 for f in sorted((ROOT/'results/trajectories').glob('*/config.json')):
  cfg=json.loads(f.read_text())
  if 101<=cfg['seed']<=105:folders.append(f.parent.name)
 if a.pilot:folders=[f'{c}_w24d2_uniform_s101' for c in CASES]
 jobs=[(f,s,a.m,a.nmax) for f in folders for s in ([3200] if a.final_only or a.pilot else STEPS)]
 print(json.dumps({'jobs':len(jobs),'workers':a.workers,'nmax':a.nmax}),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for r in ex.map(one,jobs):print(json.dumps(r),flush=True)
