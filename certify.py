"""Interval-assisted, importance-weighted anytime certificates."""
import numpy as np
from scipy.special import logsumexp

BET_FRACTIONS=0.99*2.**(-np.arange(12))

def betting_upper(x,alpha):
 """Invert a mixture of nonnegative test martingales for a bounded mean."""
 x=np.asarray(x,dtype=float)
 if x.ndim!=1 or not len(x) or np.any(~np.isfinite(x)) or np.any(x<0) or np.any(x>1):
  raise ValueError("Observations must be a nonempty finite vector in [0,1]")
 if not 0<alpha<1:raise ValueError("alpha must lie in (0,1)")
 def loge(m):
  z=1+BET_FRACTIONS[:,None]*(m-x[None,:])/(1-m)
  return float(logsumexp(np.sum(np.log(z),axis=1))-np.log(len(BET_FRACTIONS)))
 l=float(np.mean(x));h=1-1e-14;threshold=np.log(1/alpha)
 if loge(h)<threshold:return 1.
 for _ in range(38):
  m=(l+h)/2
  if loge(m)>=threshold:h=m
  else:l=m
 return h
