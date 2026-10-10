"""Arb ball-arithmetic enclosures of neural PDE residuals and derivatives."""
import numpy as np
from flint import arb,arb_mat,ctx
from core import CASES
ctx.prec=80

def _mat(a):return arb_mat(a)
def _had(a,b):return _mat([[a[i,j]*b[i,j] for j in range(a.ncols())] for i in range(a.nrows())])
def _scale(a,s):return _mat([[a[i,j]*s for j in range(a.ncols())] for i in range(a.nrows())])
def _addbias(a,b):return _mat([[a[i,j]+b[j] for j in range(a.ncols())] for i in range(a.nrows())])

def squared_residual_bounds(lo,hi,ws,case):
 """Bound r(x)^2 on every input box, including arithmetic roundoff."""
 ctx.prec=80;n,d=lo.shape
 # Each input float is represented exactly; radius is rounded outward by Arb.
 xs=[[arb(float(lo[i,j]))+(arb(float(hi[i,j]))-arb(float(lo[i,j])))*arb(.5,.5) for j in range(d)] for i in range(n)]
 h=_mat([[2*v-1 for v in row] for row in xs])
 dh=[_mat([[arb(2 if j==k else 0) for j in range(d)] for _ in range(n)]) for k in range(d)]
 d2=[_mat([[arb(0) for _ in range(d)] for _ in range(n)]) for k in range(d)]
 for z,(w,b) in enumerate(ws):
  W=_mat([[arb(float(v)) for v in row] for row in w]).transpose();bb=[arb(float(v)) for v in b]
  h=_addbias(h*W,bb);dh=[a*W for a in dh];d2=[a*W for a in d2]
  if z<len(ws)-1:
   h=_mat([[h[i,j].tanh() for j in range(h.ncols())] for i in range(n)])
   a=_mat([[1-h[i,j]**2 for j in range(h.ncols())] for i in range(n)])
   b2=_scale(_had(h,a),-2)
   d2=[_had(a,v)+_had(b2,_had(q,q)) for v,q in zip(d2,dh)]
   dh=[_had(a,q) for q in dh]
 cfg=CASES[case];pi=arb.pi();low=[];high=[]
 for i,x in enumerate(xs):
  fs=[4*x[j]*(1-x[j]) for j in range(cfg['space'])]
  if case=='heat1':fs.append(x[1])
  g=arb(1)
  for v in fs:g*=v
  N=h[i,0];u=g*N;r=cfg['c']*u
  for j in range(cfg['space']):
   prod=arb(1)
   for k,v in enumerate(fs):
    if j!=k:prod*=v
   g1=4*(1-2*x[j])*prod;g2=-8*prod
   r-=g2*N+2*g1*dh[j][i,0]+g*d2[j][i,0]
  if case=='heat1':r+=fs[0]*N+g*dh[1][i,0]
  for amp,ks,p in cfg['modes']:
   s=arb(str(amp))
   for j,k in enumerate(ks):s*=(pi*k*x[j]).sin()
   tp=x[-1]**p if p else arb(1)
   r-=s*tp*(pi*pi*sum(k*k for k in ks)+cfg['c'])
   if p:r-=s*p*x[-1]**(p-1)
  low.append(max(0.,np.nextafter(float(r.abs_lower()**2),-np.inf)))
  high.append(np.nextafter(float(r.abs_upper()**2),np.inf))
 return np.array(low),np.array(high)

def partition(dim,m):
 if dim not in (1,2) or not isinstance(m,int) or m<1 or m&(m-1):
  raise ValueError("Use dimension 1 or 2 and a positive power-of-two partition")
 q=np.arange(m,dtype=float)/m
 mesh=np.meshgrid(*[q]*dim,indexing='ij');lo=np.stack([v.ravel() for v in mesh],axis=1)
 return lo,lo+1/m

def load_weights(path):
 z=np.load(path);return [(z[f'w{i}'],z[f'b{i}']) for i in range(len(z.files)//2)]

# Centered mean-value enclosures use third derivatives to preserve residual
# cancellation more effectively than a direct natural interval extension.
def centered_bounds(lo,hi,ws,case):
 import itertools,math
 ctx.prec=80;n,d=lo.shape
 xs=[[arb(float(lo[i,j]))+(arb(float(hi[i,j]))-arb(float(lo[i,j])))*arb(.5,.5) for j in range(d)] for i in range(n)]
 zero=(0,)*d;keys=[a for a in itertools.product(range(4),repeat=d) if sum(a)<=3];keys.sort(key=sum)
 J={zero:_mat([[2*v-1 for v in row] for row in xs])}
 for a in keys[1:]:
  J[a]=_mat([[arb(2 if sum(a)==1 and a[j]==1 else 0) for j in range(d)] for _ in range(n)])
 def key(ids):return tuple(ids.count(i) for i in range(d))
 for iz,(w,b) in enumerate(ws):
  W=_mat([[arb(float(v)) for v in row] for row in w]).transpose()
  J={a:q*W for a,q in J.items()};J[zero]=_addbias(J[zero],[arb(float(v)) for v in b])
  if iz==len(ws)-1:break
  t=_mat([[J[zero][i,j].tanh() for j in range(J[zero].ncols())] for i in range(n)])
  a1=_mat([[1-t[i,j]**2 for j in range(t.ncols())] for i in range(n)])
  a2=_scale(_had(t,a1),-2)
  a3=_mat([[-2+8*t[i,j]**2-6*t[i,j]**4 for j in range(t.ncols())] for i in range(n)])
  out={zero:t}
  for a in keys[1:]:
   ids=[i for i,v in enumerate(a) for _ in range(v)];v=_had(a1,J[a])
   if len(ids)==2:
    i,j=ids;v=v+_had(a2,_had(J[key([i])],J[key([j])]))
   if len(ids)==3:
    i,j,k=ids
    pair=_had(J[key([i,j])],J[key([k])])+_had(J[key([i,k])],J[key([j])])+_had(J[key([j,k])],J[key([i])])
    v=v+_had(a2,pair)+_had(a3,_had(_had(J[key([i])],J[key([j])]),J[key([k])]))
   out[a]=v
  J=out
 cfg=CASES[case];pi=arb.pi();grad=[]
 for i,x in enumerate(xs):
  def mask_der(a):
   v=arb(1)
   for j,o in enumerate(a):
    if j<cfg['space']:v*=([4*x[j]*(1-x[j]),4-8*x[j],arb(-8),arb(0)][o])
    else:v*=([x[j],arb(1),arb(0),arb(0)][o])
   return v
  def ud(a):
   val=arb(0)
   for beta in itertools.product(*[range(v+1) for v in a]):
    gam=tuple(a[k]-beta[k] for k in range(d));coef=math.prod(math.comb(a[k],beta[k]) for k in range(d))
    val+=coef*mask_der(beta)*J[gam][i,0]
   return val
  rr=[]
  for j in range(d):
   rj=cfg['c']*ud(key([j]))
   for k in range(cfg['space']):rj-=ud(key([k,k,j]))
   if case=='heat1':rj+=ud(key([1,j]))
   for amp,ks,p in cfg['modes']:
    s=arb(str(amp))
    for k,kk in enumerate(ks):s*=pi*kk*(pi*kk*x[k]).cos() if j==k else (pi*kk*x[k]).sin()
    eig=pi*pi*sum(k*k for k in ks)+cfg['c']
    if j<cfg['space']:
     rj-=s*(x[-1]**p if p else 1)*eig
     if p:rj-=s*p*x[-1]**(p-1)
    else:
     if p:rj-=s*p*x[-1]**(p-1)*eig
     if p>=2:rj-=s*p*(p-1)*x[-1]**(p-2)
   rr.append(rj.abs_upper())
  grad.append(rr)
 mid=(lo+hi)/2
 # Evaluate the signed center residual with an arbitrarily narrow interval:
 # its magnitude suffices for a symmetric centered enclosure.
 cl,ch=squared_residual_bounds(mid,mid,ws,case)
 low=[];high=[]
 for i in range(n):
  radius=sum((arb(float(hi[i,j]))-arb(float(lo[i,j])))*grad[i][j]/2 for j in range(d))
  centerlo=arb(float(cl[i])).sqrt().lower();centerhi=arb(float(ch[i])).sqrt().upper()
  ll=centerlo-radius
  low.append(0. if not ll>0 else max(0.,np.nextafter(float(ll.lower()**2),-np.inf)))
  high.append(np.nextafter(float((centerhi+radius).upper()**2),np.inf))
 return np.array(low),np.array(high)
