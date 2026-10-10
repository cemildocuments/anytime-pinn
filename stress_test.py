"""Exact-mean repeated-monitoring stress test for bounded audit observations."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta
from certify import BET_FRACTIONS
ROOT=Path(__file__).resolve().parent

def run(reps=20000,seed=78213):
 rng=np.random.default_rng(seed);ns=32*2**np.arange(8);ps=[.001,.01,.1,.5];out=[];allcounts={}
 for p in ps:
  events={me:np.zeros(reps,dtype=bool) for me in ['mixture','spent_eb','spent_hoeffding','repeated_fixed_cp','repeated_normal']};counts=[]
  for k in range(1,6):
   s=np.zeros(reps,dtype=int);last=0;one=[];alpha=.05/(k*(k+1))
   for j,n in enumerate(ns,1):
    s+=rng.binomial(n-last,p,size=reps);last=n;one.append(s.copy());mean=s/n;var=mean*(1-mean)*n/(n-1)
    cc=BET_FRACTIONS[:,None];logs=s[None,:]*np.log1p(-cc)+(n-s[None,:])*np.log1p(cc*p/(1-p))
    events['mixture']|=logsumexp(logs,axis=0)-np.log(len(BET_FRACTIONS))>=np.log(1/alpha)
    delta=alpha/(j*(j+1));ell=np.log(2/delta)
    events['spent_eb']|=mean+np.sqrt(2*var*ell/n)+7*ell/(3*(n-1))<p
    events['spent_hoeffding']|=mean+np.sqrt(np.log(1/delta)/(2*n))<p
    up=beta.ppf(.95,s+1,n-s);up[s==n]=1.
    events['repeated_fixed_cp']|=up<p
    events['repeated_normal']|=mean+1.6448536269514722*np.sqrt(var/n)<p
   counts.append(np.array(one))
  allcounts[f'p{p}']=np.array(counts)
  for me,e in events.items():
   q=int(e.sum());ci=[0 if q==0 else beta.ppf(.025,q,reps-q+1),1 if q==reps else beta.ppf(.975,q+1,reps-q)]
   out.append({'p':p,'method':me,'repetitions':reps,'failures':q,'rate':q/reps,'ci95':[float(v) for v in ci]})
 dest=ROOT/'results/stress';dest.mkdir(parents=True,exist_ok=True)
 (dest/'summary.json').write_text(json.dumps({'seed':seed,'checkpoints':5,'sample_sizes':ns.tolist(),'results':out},indent=2));np.savez_compressed(dest/'counts.npz',**allcounts)
 for x in out:print(json.dumps(x),flush=True)
if __name__=='__main__':run()
