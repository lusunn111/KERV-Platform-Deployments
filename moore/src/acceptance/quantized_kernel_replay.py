"""Paired replay: SAME INT8 policy, old decoder versus new row decoder.

Two resident workers compute sequentially; round order alternates. No native
BF16 speedup, closed-loop success or cold-memory claim can be inferred here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from acceptance.common import dump,load,rows,sha256,distribution
from acceptance.run_matrix import make_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--rounds',type=int,choices=(1,2,3),default=3)
    p.add_argument('--reference-backend',choices=('reference','rows','safe_rows'),default='reference')
    p.add_argument('--candidate-backend',choices=('rows','prefetch','tree_rows','tree_prefetch','shared_history'),default='rows')
    p.add_argument('--experiment-name',default='same_policy_rows_v1')
    a=p.parse_args();root=Path('/root/KERV')
    if Path(a.experiment_name).name!=a.experiment_name:raise ValueError('Single experiment directory name required')
    experiment=a.campaign/'replays'/a.experiment_name
    experiment.mkdir(parents=True,exist_ok=False)
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    (experiment/'device_before.log').write_text(device)
    if 'No running processes found' not in device:raise RuntimeError('GPU0 must be idle')
    frames=rows(a.capture/'steps.jsonl');labels=['quant_reference','quant_kernel']
    protocol={'capture':str(a.capture),'capture_sha256':sha256(a.capture/'observations.npz'),
        'policy':str(a.policy),'policy_sha256':sha256(a.policy),'frames':len(frames),
        'warmup_full_passes':1,'measured_rounds':a.rounds,'seed':17,
        'order':[labels if i%2==0 else list(reversed(labels)) for i in range(a.rounds)],
        'history':'identical captured observations, raw actions and Kalman-token prefixes before every frame',
        'scope':'operator-only comparison on one fixed lossy INT8 policy; memory co-resident, no SR',
        'reference_backend':a.reference_backend,'candidate_backend':a.candidate_backend,
        'backend':'same policy, native BF16 GEMM; explicit decode/host-tree backends',
        'timing':'same existing replay worker; logging/profiling off during measured passes; warmup frame77 debug instrumentation disclosed'}
    dump(experiment/'protocol.json',protocol)
    sources={}
    snapshot=experiment/'source';snapshot.mkdir()
    for path in sorted((root/'src').rglob('*.py')):
        if any(x in path.parts for x in ('.git','__pycache__','.venv')):continue
        rel=str(path.relative_to(root));sources[rel]=sha256(path)
    dump(experiment/'source_manifest.json',sources)
    for path in [root/'src/runtime_opt/KERVRuntimeOptimization/calibrated_w8a16_ops.py',
                 root/'src/runtime_opt/KERVRuntimeOptimization/calibrated_native_decode.py',
                 root/'src/runtime_opt/KERVRuntimeOptimization/csrc/calibrated_decode.mu',
                 root/'src/runtime_opt/KERVRuntimeOptimization/musa_safe_softmax.py',
                 root/'src/openvla/prismatic/extern/hf/modeling_speculation.py',
                 root/'src/openvla/experiments/robot/libero/run_calibrated_shared_libero.py',
                 root/'src/openvla/experiments/robot/libero/run_calibrated_safe_libero.py',
                 root/'src/openvla/experiments/robot/libero/run_calibrated_kernel_libero.py',
                 root/'src/runtime_opt/KERVRuntimeOptimization/calibrated_prefetch.py',
                 root/'src/openvla/experiments/robot/libero/run_calibrated_prefetch_libero.py',Path(__file__)]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    processes={};handles={}
    def wait(path,label):
        start=time.monotonic()
        while not path.exists():
            if processes[label].poll() is not None:raise RuntimeError(label+' exited before '+path.name)
            if list(path.parent.glob('*.mudmp')):raise RuntimeError(label+' produced a MUSA fault dump')
            if time.monotonic()-start>900:raise TimeoutError(str(path))
            time.sleep(.2)
    try:
        for label in labels:
            output=experiment/label;output.mkdir()
            case={'profile':'calibrated','policy_path':str(a.policy),'kind':'replay','timeout_seconds':'3600',
                  'max_steps':'220','planning_steps':'220','task_id':'0','init_state_id':'0','seed':'17'}
            backend=a.candidate_backend if label=='quant_kernel' else a.reference_backend
            case['kernel_backend']=backend
            command=make_command(root,case,output/'context.json','kerv-musa-runtime')
            command=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
            env=['-e','KERV_REPLAY_PROFILE='+label,'-e','KERV_REPLAY_CAPTURE=/workspace/'+str(a.capture.relative_to(root)),
                 '-e','KERV_REPLAY_OUTPUT=/workspace/'+str(output.relative_to(root))]
            command=command[:2]+env+command[2:]
            entry=('calibrated_shared_replay_entry.py' if backend == 'shared_history' else
                   'calibrated_safe_replay_entry.py' if backend == 'safe_rows' else
                   'calibrated_prefetch_replay_entry.py' if backend in ('prefetch','tree_prefetch') else
                   'calibrated_tree_replay_entry.py' if backend == 'tree_rows' else
                   'calibrated_kernel_replay_entry.py' if backend == 'rows' else 'calibrated_replay_entry.py')
            command+=['--override','inference.runtime.entrypoint=/workspace/src/acceptance/'+entry]
            dump(output/'command.json',{'argv':command,'start_timestamp':datetime.now(timezone.utc).isoformat()})
            handles[label]=(output/'launcher.log').open('w')
            processes[label]=subprocess.Popen(command,cwd=root,stdout=handles[label],stderr=subprocess.STDOUT)
            print('WARM',label,flush=True);wait(output/'ready.json',label);print('READY',label,flush=True)
        pairs=[]
        for i,order in enumerate(protocol['order']):
            for label in order:
                output=experiment/label;dump(output/('request_%d.json'%i),{'round':i})
                print('MEASURE',i,label,flush=True);wait(output/('done_%d.json'%i),label)
            left,right=[rows(experiment/label/('measure_%d.jsonl'%i)) for label in labels]
            assert len(left)==len(right)==len(frames)
            for index,(x,y,captured) in enumerate(zip(left,right,frames)):
                assert x['frame']==y['frame']==index
                assert x['observation_hash']==y['observation_hash']==captured['observation_hash']
            errors=[abs(x-y) for u,v in zip(left,right) for x,y in zip(u['raw_action'],v['raw_action'])]
            times={label:{k:distribution(x[k] for x in data) for k in ['model_seconds','pipeline_seconds']}
                   for label,data in zip(labels,[left,right])}
            pair={'round':i,'frames':len(frames),'times':times,
                'exact_action_tokens':sum(x['action_tokens']==y['action_tokens'] for x,y in zip(left,right)),
                'exact_raw_actions':sum(x['raw_action']==y['raw_action'] for x,y in zip(left,right)),
                'gripper_matches':sum(x['executed_action'][-1]==y['executed_action'][-1] for x,y in zip(left,right)),
                'raw_action_error':distribution(errors),'max_abs':max(errors),
                'model_speedup':times[labels[0]]['model_seconds']['mean']/times[labels[1]]['model_seconds']['mean'],
                'pipeline_speedup':times[labels[0]]['pipeline_seconds']['mean']/times[labels[1]]['pipeline_seconds']['mean']}
            pairs.append(pair);dump(experiment/'results.json',{'protocol':protocol,'rounds':pairs})
            print('ROUND',i,'exact',pair['exact_action_tokens'],'speedup',pair['model_speedup'],flush=True)
        # The same native parameters and exact quantized-buffer hashes must load.
        assert load(experiment/labels[0]/'parameter_hashes.json')==load(experiment/labels[1]/'parameter_hashes.json')
        ready=[load(experiment/l/'ready.json') for l in labels]
        assert ready[0]['quantization_manifest']['policy_sha256']==ready[1]['quantization_manifest']['policy_sha256']==protocol['policy_sha256']
        dump(experiment/'identity_checks.json',{'same_native_parameters':True,'same_policy_sha256':True,
             'observations_checked':len(frames)*a.rounds,'both_workers_int8_buffers':[r['int8_weight_buffer_count'] for r in ready]})
    finally:
        for label in processes:(experiment/label/'stop').touch()
        for label,process in processes.items():
            if process.poll() is None and list((experiment/label).glob('*.mudmp')):
                pid=int(load(experiment/label/'worker_start.json')['pid'])
                subprocess.run(['docker','exec','kerv-musa-runtime','kill','-TERM',str(pid)],check=True)
            try:code=process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pid=int(load(experiment/label/'worker_start.json')['pid'])
                subprocess.run(['docker','exec','kerv-musa-runtime','kill','-TERM',str(pid)],check=True)
                code=process.wait(timeout=30)
            handles[label].close();record=load(experiment/label/'command.json')
            record.update(exit_code=code,end_timestamp=datetime.now(timezone.utc).isoformat())
            dump(experiment/label/'command.json',record)


if __name__=='__main__':main()
