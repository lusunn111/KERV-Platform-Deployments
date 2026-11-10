"""One explicitly instrumented replay worker, excluded from performance claims."""
import argparse
from pathlib import Path
import subprocess
import time
from acceptance.common import dump,load,sha256
from acceptance.run_matrix import make_command

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--entrypoint',required=True)
    p.add_argument('--rounds',type=int,choices=(1,2,3),default=1)
    p.add_argument('--tree-buffers',action='store_true',help='Use batched host buffers and scalar Kalman without pruning')
    p.add_argument('--worker-env',action='append',default=[],help='Explicit non-secret diagnostic environment NAME=value')
    a=p.parse_args();r=Path('/root/KERV');out=a.output;out.mkdir(parents=True,exist_ok=False)
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True);(out/'device_before.log').write_text(device)
    if 'No running processes found' not in device:raise RuntimeError('GPU0 must be idle')
    entry=r/Path(a.entrypoint).relative_to('/workspace')
    (out/'entrypoint_source.py').write_bytes(entry.read_bytes())
    dump(out/'source_manifest.json',{str(f.relative_to(r)):sha256(f) for f in sorted((r/'src').rglob('*.py'))
                                   if '__pycache__' not in f.parts})
    case={'profile':'calibrated','policy_path':str(a.policy),'kind':'replay','timeout_seconds':'1800',
          'max_steps':'220','planning_steps':'220','task_id':'0','init_state_id':'0','seed':'17'}
    cmd=make_command(r,case,out/'context.json','kerv-musa-runtime')
    cmd=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in cmd]
    cmd=cmd[:2]+['-e','KERV_REPLAY_PROFILE=diagnostic','-e','KERV_REPLAY_CAPTURE=/workspace/'+str(a.capture.relative_to(r)),
                  '-e','KERV_REPLAY_OUTPUT=/workspace/'+str(out.relative_to(r))]+cmd[2:]
    for setting in a.worker_env:
        if not setting.startswith('KERV_STABILITY_') or '=' not in setting:
            raise ValueError('Only explicit KERV_STABILITY_ diagnostic settings are accepted')
        cmd=cmd[:2]+['-e',setting]+cmd[2:]
    cmd+=['--override','inference.runtime.entrypoint='+a.entrypoint];dump(out/'command.json',{'argv':cmd})
    if a.tree_buffers:
        cmd+=['--override','inference.runtime.tree_builder.enabled=true',
              '--override','inference.runtime.tree_builder.static_runtime=false',
              '--override','inference.runtime.tree_builder.kalman_batch=false']
        dump(out/'command.json',{'argv':cmd})
    with (out/'launcher.log').open('w') as f:
        proc=subprocess.Popen(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            for marker in ['ready.json']+['done_%d.json'%i for i in range(a.rounds)]:
                while not (out/marker).exists():
                    if proc.poll() is not None:raise RuntimeError('Exited before '+marker)
                    if list(out.glob('*.mudmp')):raise RuntimeError('Device fault dump detected')
                    if time.monotonic()-start>1200:raise TimeoutError(marker)
                    time.sleep(.2)
                if marker=='ready.json':dump(out/'request_0.json',{'round':0})
                elif int(marker.split('_')[1].split('.')[0])+1<a.rounds:
                    next_round=int(marker.split('_')[1].split('.')[0])+1
                    dump(out/('request_%d.json'%next_round),{'round':next_round})
        finally:
            (out/'stop').touch()
            if list(out.glob('*.mudmp')) and (out/'worker_start.json').exists() and proc.poll() is None:
                pid=int(load(out/'worker_start.json')['pid'])
                dump(out/'fault_termination.json',{'reason':'Device fault dump; terminate only this owned worker','container_pid':pid})
                subprocess.run(['docker','exec','kerv-musa-runtime','kill','-TERM',str(pid)],check=False)
            code=proc.wait(timeout=60)
            d=load(out/'command.json');d['exit_code']=code;dump(out/'command.json',d)
        if code:raise RuntimeError('Diagnostic failed')

if __name__=='__main__':main()
