"""Run one instrumented shadow replay after the timing campaign has ended."""
import argparse
from pathlib import Path
import subprocess
import time
from acceptance.common import dump,load
from acceptance.run_matrix import make_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    a=p.parse_args();root=Path('/root/KERV');out=a.campaign/'diagnostics/shadow_same_input'
    out.mkdir(parents=True,exist_ok=False)
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    (out/'device_before.log').write_text(device)
    if 'No running processes found' not in device:raise RuntimeError('GPU0 must be idle')
    case={'profile':'calibrated','policy_path':str(a.policy),'kind':'replay','timeout_seconds':'1800',
          'max_steps':'220','planning_steps':'220','task_id':'0','init_state_id':'0','seed':'17'}
    command=make_command(root,case,out/'context.json','kerv-musa-runtime')
    command=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
    command=command[:2]+['-e','KERV_REPLAY_PROFILE=shadow','-e','KERV_REPLAY_CAPTURE=/workspace/'+str(a.capture.relative_to(root)),
                        '-e','KERV_REPLAY_OUTPUT=/workspace/'+str(out.relative_to(root))]+command[2:]
    command+=['--override','inference.runtime.entrypoint=/workspace/src/acceptance/calibrated_shadow_replay_entry.py']
    dump(out/'command.json',{'argv':command})
    with (out/'launcher.log').open('w') as log:
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            for path in [out/'ready.json',out/'done_0.json']:
                while not path.exists():
                    if process.poll() is not None:raise RuntimeError('Worker exited before '+path.name)
                    if time.monotonic()-start>1200:raise TimeoutError(str(path))
                    time.sleep(.2)
                if path.name=='ready.json':dump(out/'request_0.json',{'round':0})
        finally:
            (out/'stop').touch()
            code=process.wait(timeout=60)
            record=load(out/'command.json');record['exit_code']=code;dump(out/'command.json',record)
        if code:raise RuntimeError('Shadow worker failed')
    proof=load(out/'shadow_operator_checks.json')
    print({k:v for k,v in proof.items() if k not in ('calls','topk_checks')},
          'calls',len(proof['calls']),'topk_calls',len(proof['topk_checks']))


if __name__=='__main__':main()
