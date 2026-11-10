"""Serial controller for resident BF16/INT8/exact-storage accuracy screening."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import time
from acceptance.common import dump,load,rows,distribution,sha256
from acceptance.run_matrix import make_command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=Path('/root/KERV');c=a.campaign
    modes=['bf16','old_int8','conservative_up8','conservative_up4','exact_bf16']
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    assert 'No running processes found' in device
    (c/'device_before.log').write_text(device)
    capture=root/'outputs/calibrated_quantization/20261026T045030Z_v2/inputs/diagnostic_capture'
    protocol={'modes':modes,'rounds':1,'warmup_full_passes_per_mode':1,'frames':len(rows(capture/'steps.jsonl')),
              'capture':str(capture),'capture_sha256':sha256(capture/'observations.npz'),
              'runtime':'native BF16 GEMM, guarded Softmax, host tree, shared scalar Kalman history',
              'scope':'previously seen development data, fixed history; co-resident selection only, no memory peak or SR acceptance'}
    dump(c/'screening_protocol.json',protocol)
    case={'profile':'native','timeout_seconds':'7200','max_steps':'220','planning_steps':'220','task_id':'0','init_state_id':'0','seed':'17'}
    command=make_command(root,case,c/'bootstrap/context.json','kerv-musa-runtime')
    command=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
    env=['-e','KERV_PRECISION_ROOT=/workspace/'+str(c.relative_to(root)),
         '-e','KERV_REPLAY_CAPTURE=/workspace/'+str(capture.relative_to(root)),
         '-e','KERV_REPLAY_OUTPUT=/workspace/'+str((c/'bootstrap').relative_to(root))]
    command=command[:2]+env+command[2:]
    command+=['--override','inference.runtime.entrypoint=/workspace/src/acceptance/precision_search_entry.py',
              '--override','inference.runtime.linear_fusion.enabled=false',
              '--override','inference.runtime.tree_builder.enabled=true',
              '--override','inference.runtime.tree_builder.static_runtime=false',
              '--override','inference.runtime.tree_builder.kalman_batch=true']
    dump(c/'screening_command.json',{'argv':command,'start':datetime.now(timezone.utc).isoformat()})
    with (c/'screening_launcher.log').open('w') as log:
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        def wait(path):
            start=time.monotonic()
            while not path.exists():
                if process.poll() is not None:raise RuntimeError('Worker exited before '+str(path))
                if time.monotonic()-start>1800:raise TimeoutError(str(path))
                time.sleep(.2)
        try:
            for label in modes:
                out=c/'screening'/label;print('WARM',label,flush=True);wait(out/'ready.json')
                dump(out/'request_0.json',{'round':0});print('MEASURE',label,flush=True);wait(out/'done_0.json')
                (out/'stop').touch();wait(out/'execution_proof.json')
                baseline=rows(c/'screening/bf16/measure_0.jsonl');actual=rows(out/'measure_0.jsonl')
                assert len(baseline)==len(actual)==protocol['frames']
                assert all(x['observation_hash']==y['observation_hash'] for x,y in zip(baseline,actual))
                errors=[abs(x-y) for a,b in zip(baseline,actual) for x,y in zip(a['raw_action'],b['raw_action'])]
                result={'mode':label,'frames':len(actual),'exact_actions':sum(x['action_tokens']==y['action_tokens'] for x,y in zip(baseline,actual)),
                        'exact_raw_actions':sum(x['raw_action']==y['raw_action'] for x,y in zip(baseline,actual)),
                        'gripper_matches':sum(x['executed_action'][-1]==y['executed_action'][-1] for x,y in zip(baseline,actual)),
                        'raw_error':distribution(errors),'max_abs':max(errors),'manifest':load(out/'precision_manifest.json'),
                        'model_seconds':distribution(x['model_seconds'] for x in actual),
                        'pipeline_seconds':distribution(x['pipeline_seconds'] for x in actual),
                        'different_frames':[i for i,(x,y) in enumerate(zip(baseline,actual)) if x['action_tokens']!=y['action_tokens']]}
                dump(out/'comparison.json',result)
                print('RESULT',label,result['exact_actions'],result['raw_error']['mean'],result['max_abs'],flush=True)
            assert process.wait(timeout=60)==0
        finally:
            for label in modes:
                out=c/'screening'/label
                if out.exists():(out/'stop').touch()
            if process.poll() is None:
                pid=int(load(c/'screening/worker_start.json')['pid'])
                subprocess.run(['docker','exec','kerv-musa-runtime','kill','-TERM',str(pid)],check=True)
                process.wait(timeout=60)
    dump(c/'screening_exit.json',{'exit_code':process.returncode,'end':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
