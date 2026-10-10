"""Independent numerical checks of differentiation and enclosure containment."""
import unittest
import numpy as np
import torch
from core import CASES,PINN,residual,weights,evaluate_numpy
from enclosures import partition,squared_residual_bounds,centered_bounds
from certify import betting_upper

class NumericalChecks(unittest.TestCase):
 def test_derivatives_and_bounds(self):
  torch.set_num_threads(1);torch.manual_seed(519);rng=np.random.default_rng(812)
  for c,cfg in CASES.items():
   model=PINN(cfg['dim'],8,2).double();ws=weights(model);x=rng.random((24,cfg['dim']))
   rt=residual(model,torch.tensor(x,dtype=torch.float64),c).detach().numpy().ravel()
   un,rn,_=evaluate_numpy(x,ws,c);np.testing.assert_allclose(rt,rn,atol=2e-12,rtol=2e-12)
   np.testing.assert_allclose(un,model(torch.tensor(x),c).detach().numpy().ravel(),atol=1e-13)
   lo,hi=partition(cfg['dim'],8);an,bn=squared_residual_bounds(lo,hi,ws,c);ac,bc=centered_bounds(lo,hi,ws,c)
   for _ in range(20):
    q=lo+rng.random(lo.shape)*(hi-lo);v=evaluate_numpy(q,ws,c)[1]**2
    self.assertTrue(np.all(v>=np.maximum(an,ac)-1e-10));self.assertTrue(np.all(v<=np.minimum(bn,bc)+1e-10))
 def test_betting(self):
  rng=np.random.default_rng(251)
  for n in [32,128,1024]:
   x=rng.beta(.5,3,size=n);a=betting_upper(x,.05);b=betting_upper(x,.01)
   self.assertGreaterEqual(a,x.mean());self.assertGreaterEqual(b,a)
  self.assertLess(betting_upper(np.zeros(1024),.05),.006)
if __name__=='__main__':unittest.main()
