"""Train reproducible PDE trajectories; each checkpoint remains immutable."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,platform,concurrent.futures
from pathlib import Path
import numpy as np
import torch
from core import CASES,PINN,residual,weights,evaluate_numpy
ROOT=Path(__file__).resolve().parent

def run(cfg):
 case,seed,width,depth,sampling,device,steps=cfg
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 torch.manual_seed(seed);np.random.seed(seed)
 dtype=torch.float32 if device=='mps' else torch.float64
 model=PINN(CASES[case]['dim'],width,depth).to(device=device,dtype=dtype)
 opt=torch.optim.Adam(model.parameters(),lr=.003)
 name=f'{case}_w{width}d{depth}_{sampling}_s{seed}'
 out=ROOT/'results'/'trajectories'/name;out.mkdir(parents=True,exist_ok=True)
 checkpoints=sorted(set([0,100,250,500,1000,2000,steps]));hist=[];start=time.perf_counter()
 gridn=300 if CASES[case]['dim']==1 else 64
 gg=np.meshgrid(*[np.linspace(0,1,gridn) for _ in range(CASES[case]['dim'])],indexing='ij');gx=np.stack([a.ravel() for a in gg],axis=1)
 def save(step,loss):
  ws=weights(model); arr={}
  for i,(w,b) in enumerate(ws):arr[f'w{i}']=w;arr[f'b{i}']=b
  np.savez_compressed(out/f'checkpoint_{step:05d}.npz',**arr)
  u,r,exact=evaluate_numpy(gx,ws,case)
  row={'step':step,'loss':float(loss),'seconds':time.perf_counter()-start,'l2_error_grid':float(np.sqrt(np.mean((u-exact)**2))),'residual_rms_grid':float(np.sqrt(np.mean(r*r)))};hist.append(row)
  (out/'history.json').write_text(json.dumps(hist,indent=2))
 save(0,0.)
 for it in range(1,steps+1):
  x=torch.rand((256,CASES[case]['dim']),device=device,dtype=dtype)
  if sampling=='adaptive' and it%10==0:
   pool=torch.rand((1024,CASES[case]['dim']),device=device,dtype=dtype)
   rr=residual(model,pool,case).detach().abs().flatten();pr=.2/1024+.8*rr/(rr.sum()+1e-30)
   x=pool[torch.multinomial(pr,256,replacement=True)]
  opt.zero_grad();r=residual(model,x,case);loss=(r*r).mean();loss.backward();opt.step()
  if it==int(.6*steps):
   for pg in opt.param_groups:pg['lr']=.001
  if it in checkpoints:save(it,loss.item())
 # Fixed collocation refinement; all cases receive the same budget.
 x=torch.rand((1024,CASES[case]['dim']),device=device,dtype=dtype)
 opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
 for j in range(1,5):
  def closure():
   opt.zero_grad();rr=residual(model,x,case);loss=(rr*rr).mean();loss.backward();return loss
  loss=opt.step(closure);save(steps+50*j,float(loss))
 meta={'case':case,'seed':seed,'width':width,'depth':depth,'sampling':sampling,'device':device,'dtype':str(dtype),'steps':steps,'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'platform':platform.platform(),'seconds':time.perf_counter()-start}
 (out/'config.json').write_text(json.dumps(meta,indent=2));return {'run':name,'seconds':meta['seconds'],'final':hist[-1]}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=3000);p.add_argument('--seed-start',type=int,default=101);p.add_argument('--seeds',type=int,default=5);p.add_argument('--cases',nargs='+',default=list(CASES));a=p.parse_args()
 if a.pilot:jobs=[('poisson1',100,24,2,'uniform',a.device,500)]
 else:jobs=[(c,s,w,d,sm,a.device,a.steps) for c in a.cases for s in range(a.seed_start,a.seed_start+a.seeds) for w,d in [(32,1),(24,2)] for sm in ['uniform','adaptive']]
 jobs=[j for j in jobs if not (ROOT/'results'/'trajectories'/f'{j[0]}_w{j[2]}d{j[3]}_{j[4]}_s{j[1]}'/'config.json').exists()]
 print(json.dumps({'jobs':len(jobs),'workers':a.workers,'device':a.device}),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for result in ex.map(run,jobs):print(json.dumps(result),flush=True)
