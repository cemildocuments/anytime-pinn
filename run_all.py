"""Run the complete research workflow using resumable primary records."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 if a.workers<1:raise SystemExit('--workers must be positive')
 for cmd in [['train.py','--workers',str(a.workers),'--device','cpu','--steps','3000','--seed-start','101','--seeds','5'],['experiment.py','--workers',str(a.workers),'--nmax','8192'],['stress_test.py'],['refinement.py'],['analyze.py'],['verify_results.py','--numerical-only']]:
  print('Running',*cmd,flush=True);subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)
