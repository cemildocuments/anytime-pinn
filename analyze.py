"""Aggregate experimental results into reusable CSV and JSON tables."""
from pathlib import Path
import json,csv
import numpy as np
from core import CASES
ROOT=Path(__file__).resolve().parent

def collect():
 audits=[]
 for p in sorted((ROOT/'results/audits').glob('*/*.json')):
  a=json.loads(p.read_text());m=128 if CASES[a['case']]['dim']==1 else 64
  if a['m']==m:audits.append(a)
 return audits

def summarize():
 audits=collect();names=sorted(set(a['name'] for a in audits));out={'audits':len(audits),'trajectories':len(names),'cases':{},'stopping':[],'refinement':[]};flat=[]
 for a in audits:
  lam=np.pi**2*CASES[a['case']]['space']+CASES[a['case']]['c']
  for sm,meth in [('importance','betting'),('importance','eb'),('importance','hoeffding'),('uniform','betting')]:
   for r in a['methods'][sm]['rows']:
    flat.append({'name':a['name'],'case':a['case'],'step':a['step'],'sampling':sm,'method':meth,'n':r['n'],'upper_mse':r['bounds'][meth],'error_bound':r['error_bounds'][meth],'seconds':r['seconds'][meth],'l2_error':a['reference']['l2_error'],'residual_mse':a['reference']['residual_mse']})
 for c in CASES:
  arr=[a for a in audits if a['case']==c and a['step']==3200]
  if not arr:continue
  def stats(vals):return {'median':float(np.median(vals)),'q25':float(np.quantile(vals,.25)),'q75':float(np.quantile(vals,.75)),'min':float(np.min(vals)),'max':float(np.max(vals))}
  lam=np.pi**2*CASES[c]['space']+CASES[c]['c']
  q={'runs':len(arr),'actual_error':stats([a['reference']['l2_error'] for a in arr]),'residual_bound':stats([np.sqrt(a['reference']['residual_mse'])/lam for a in arr]),'deterministic':stats([np.sqrt(a['deterministic_upper'])/lam for a in arr]),'natural_to_centered':stats([a['natural_upper']/a['deterministic_upper'] for a in arr])}
  for sm,meth in [('importance','betting'),('importance','eb'),('importance','hoeffding'),('uniform','betting')]:
   key=sm+'_'+meth;q[key]=stats([a['methods'][sm]['rows'][-1]['error_bounds'][meth] for a in arr]);q[key]['certified_001']=sum(a['methods'][sm]['rows'][-1]['error_bounds'][meth]<=.01 for a in arr)
   ns=[]
   for a in arr:
    if np.sqrt(a['deterministic_upper'])/lam<=.01:ns.append(0)
    else:
     hits=[r['n'] for r in a['methods'][sm]['rows'] if r['error_bounds'][meth]<=.01]
     if hits:ns.append(hits[0])
   q[key]['median_n_certified']=float(np.median(ns)) if ns else None;q[key]['n_certified']=len(ns)
  out['cases'][c]=q
 for eps in [.01,.02,.05]:
  for sm,meth in [('importance','betting'),('importance','eb'),('importance','hoeffding'),('uniform','betting')]:
   for name in names:
    seq=sorted([a for a in audits if a['name']==name],key=lambda a:a['step']);hist=json.loads((ROOT/'results/trajectories'/name/'history.json').read_text());ht={r['step']:r for r in hist};spent=0;points=0;hit=None
    for a in seq:
     lam=np.pi**2*CASES[a['case']]['space']+CASES[a['case']]['c']
     if np.sqrt(a['deterministic_upper'])/lam<=eps:
      spent+=a['natural_seconds']+a['centered_seconds'];hit={'step':a['step'],'audit_points':0,'bound':float(np.sqrt(a['deterministic_upper'])/lam),'actual':a['reference']['l2_error']};break
     rows=a['methods'][sm]['rows'];rr=next((r for r in rows if r['error_bounds'][meth]<=eps),rows[-1]);spent+=rr['seconds'][meth];points+=rr['n']
     if rr['error_bounds'][meth]<=eps:hit={'step':a['step'],'audit_points':rr['n'],'bound':rr['error_bounds'][meth],'actual':a['reference']['l2_error']};break
    out['stopping'].append({'name':name,'case':seq[0]['case'],'epsilon':eps,'sampling':sm,'method':meth,'certified':hit is not None,'stop':hit,'audit_seconds':spent,'audit_points_total':points,'training_seconds_at_stop':ht[hit['step']]['seconds'] if hit else ht[3200]['seconds'],'full_training_seconds':ht[3200]['seconds']})
 out['stopping_full_budget']=out['stopping']
 out['stopping']=[]
 for old in out['stopping_full_budget']:
  eps=old['epsilon'];sm=old['sampling'];meth=old['method'];name=old['name'];spent=0.;points=0;hit=None
  seq=sorted([a for a in audits if a['name']==name],key=lambda a:a['step']);hist=json.loads((ROOT/'results/trajectories'/name/'history.json').read_text());ht={r['step']:r for r in hist}
  for a in seq:
   lam=np.pi**2*CASES[a['case']]['space']+CASES[a['case']]['c'];tau2=(lam*eps)**2;enc=a['natural_seconds']+a['centered_seconds']
   if a['deterministic_upper']<=tau2:
    spent+=enc;hit={'step':a['step'],'audit_points':0,'bound':float(np.sqrt(a['deterministic_upper'])/lam),'actual':a['reference']['l2_error']};break
   if a['A']>tau2:
    spent+=enc;continue
   rows=a['methods'][sm]['rows'];rr=rows[-1]
   for rr in rows:
    if rr['error_bounds'][meth]<=eps or rr['mean_mse']>tau2:break
   spent+=rr['seconds'][meth];points+=rr['n']
   if rr['error_bounds'][meth]<=eps:
    hit={'step':a['step'],'audit_points':rr['n'],'bound':rr['error_bounds'][meth],'actual':a['reference']['l2_error']};break
  out['stopping'].append({'name':name,'case':old['case'],'epsilon':eps,'sampling':sm,'method':meth,'certified':hit is not None,'stop':hit,'audit_seconds':spent,'audit_points_total':points,'training_seconds_at_stop':ht[hit['step']]['seconds'] if hit else ht[3200]['seconds'],'full_training_seconds':ht[3200]['seconds']})
 out['violations']={'solution':sum(r['error_bound']+1e-12<r['l2_error'] for r in flat),'residual':sum(r['upper_mse']+1e-10<r['residual_mse'] for r in flat),'evaluated_bounds':len(flat),'max_quadrature_difference':max(a['quadrature_difference']['residual_mse'] for a in audits)}
 out['refinement']=[json.loads(p.read_text()) for p in sorted((ROOT/'results/refinement').glob('*.json'))]
 dest=ROOT/'results/summary';dest.mkdir(parents=True,exist_ok=True)
 (dest/'summary.json').write_text(json.dumps(out,indent=2))
 with (dest/'certificate_curves.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 with (dest/'stopping.csv').open('w') as f:
  fields=['name','case','epsilon','sampling','method','certified','step','audit_points_total','audit_seconds','training_seconds_at_stop','full_training_seconds']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in out['stopping']:
   z={k:r[k] for k in fields if k!='step'};z['step']=r['stop']['step'] if r['stop'] else '';w.writerow(z)
 print(json.dumps({k:v for k,v in out.items() if k not in ['stopping','stopping_full_budget','refinement']},indent=2))
 return out
if __name__=='__main__':summarize()
