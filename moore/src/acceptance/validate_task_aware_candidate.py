"""Complete a registered pilot, then run fresh pairs only if both tasks succeed."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
from acceptance.common import load, dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--pilot-cases',type=Path,required=True)
    p.add_argument('--validation-cases',type=Path,required=True)
    a=p.parse_args()
    with a.pilot_cases.open() as stream:pilot=list(csv.DictReader(stream))
    assert len(pilot)==2 and all(x['profile']=='precision' for x in pilot)
    assert len({x['policy_path'] for x in pilot})==1
    subprocess.run([sys.executable,'-m','acceptance.run_matrix','--campaign',str(a.campaign),
                    '--cases',str(a.pilot_cases),'--resume'],check=True)
    results=[load(a.campaign/'runs'/x['run_id']/'run.json') for x in pilot]
    passed=all(x['status']=='PASS' and x['task_success'] is True for x in results)
    dump(a.campaign/'reports'/('pilot_gate_'+a.pilot_cases.stem+'.json'),
         {'status':'PASS' if passed else 'REJECTED','runs':[x['run_id'] for x in results],
          'success':[x['task_success'] for x in results],
          'criterion':'Both pilot tasks succeed; action equality not required',
          'reference':'Historical frozen BF16 successes, not fresh paired timing'})
    if not passed:raise SystemExit(2)
    print('PILOT_PASS; starting pre-registered fresh pairs',flush=True)
    subprocess.run([sys.executable,'-m','acceptance.run_matrix','--campaign',str(a.campaign),
                    '--cases',str(a.validation_cases)],check=True)


if __name__=='__main__':main()
