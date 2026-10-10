"""Manufactured PDEs and fully trainable, hard-constrained tanh networks."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import numpy as np
import torch
from torch import nn

CASES={
 'poisson1':{'dim':1,'space':1,'c':0.,'modes':[(1.,[1],0),(.15,[3],0)]},
 'reaction1':{'dim':1,'space':1,'c':10.,'modes':[(1.,[1],0),(.10,[5],0)]},
 'poisson2':{'dim':2,'space':2,'c':0.,'modes':[(1.,[1,1],0),(.10,[3,2],0)]},
 'heat1':{'dim':2,'space':1,'c':0.,'modes':[(1.,[1],1),(.15,[3],2)]},
}

def truth(x,case):
 cfg=CASES[case]; out=torch.zeros_like(x[:,:1]); force=out.clone()
 for amp,ks,p in cfg['modes']:
  s=torch.ones_like(out)
  for j,k in enumerate(ks): s=s*torch.sin(np.pi*k*x[:,j:j+1])
  time=x[:,-1:]**p if p else 1.
  out=out+amp*s*time
  force=force+amp*s*time*(np.pi**2*sum(k*k for k in ks)+cfg['c'])
  if p: force=force+amp*s*p*x[:,-1:]**(p-1)
 return out,force

class PINN(nn.Module):
 def __init__(self,dim,width=32,depth=2):
  super().__init__(); ls=[]; old=dim
  for _ in range(depth):
   a=nn.Linear(old,width);nn.init.xavier_normal_(a.weight);nn.init.zeros_(a.bias);ls.extend([a,nn.Tanh()]);old=width
  a=nn.Linear(old,1);nn.init.xavier_normal_(a.weight);nn.init.zeros_(a.bias);ls.append(a)
  self.net=nn.Sequential(*ls)
 def forward(self,x,case):
  cfg=CASES[case]; g=torch.ones_like(x[:,:1])
  for j in range(cfg['space']): g=g*4*x[:,j:j+1]*(1-x[:,j:j+1])
  if case=='heat1':g=g*x[:,-1:]
  return g*self.net(2*x-1)

def residual(model,x,case):
 x=x.detach().requires_grad_(True);u=model(x,case)
 du=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]
 lap=torch.zeros_like(u)
 for j in range(CASES[case]['space']):
  lap=lap+torch.autograd.grad(du[:,j:j+1],x,torch.ones_like(u),create_graph=True)[0][:,j:j+1]
 f=truth(x,case)[1];r=-lap+CASES[case]['c']*u-f
 if case=='heat1':r=r+du[:,-1:]
 return r

def weights(model):
 return [(l.weight.detach().cpu().double().numpy(),l.bias.detach().cpu().double().numpy()) for l in model.net if isinstance(l,nn.Linear)]

def evaluate_numpy(x,ws,case):
 """Analytic forward differentiation (independent of autograd)."""
 n,d=x.shape;h=2*x-1; dh=np.broadcast_to(2*np.eye(d),(n,d,d)).copy();d2=np.zeros_like(dh)
 for i,(w,b) in enumerate(ws):
  h=h@w.T+b; dh=np.einsum('njd,kj->nkd',dh,w);d2=np.einsum('njd,kj->nkd',d2,w)
  if i<len(ws)-1:
   t=np.tanh(h);a=1-t*t;bb=-2*t*a;d2=a[:,:,None]*d2+bb[:,:,None]*dh*dh;dh=a[:,:,None]*dh;h=t
 N=h[:,0];N1=dh[:,0,:];N2=d2[:,0,:];cfg=CASES[case]
 fs=[4*x[:,j]*(1-x[:,j]) for j in range(cfg['space'])]
 if case=='heat1':fs.append(x[:,1])
 g=np.prod(fs,axis=0);g1=np.zeros((n,d));g2=np.zeros((n,d))
 for j in range(d):
  prod=np.prod([fs[k] for k in range(d) if k!=j],axis=0) if d>1 else np.ones(n)
  if j<cfg['space']:g1[:,j]=4*(1-2*x[:,j])*prod;g2[:,j]=-8*prod
  else:g1[:,j]=prod
 u=g*N;uxx=g2*N[:,None]+2*g1*N1+g[:,None]*N2
 r=-np.sum(uxx[:,:cfg['space']],axis=1)+cfg['c']*u
 if case=='heat1':r=r+g1[:,1]*N+g*N1[:,1]
 exact=np.zeros(n);f=np.zeros(n)
 for amp,ks,p in cfg['modes']:
  s=np.prod(np.sin(np.pi*x[:,:cfg['space']]*np.array(ks)),axis=1);tp=x[:,-1]**p if p else 1.
  exact+=amp*s*tp;f+=amp*s*tp*(np.pi**2*sum(k*k for k in ks)+cfg['c'])
  if p:f+=amp*s*p*x[:,-1]**(p-1)
 return u,r-f,exact
