"""Three fresh same-input replay rounds in AB/BA/AB order, on one idle chip."""
import argparse, datetime, json, os, subprocess
from pathlib import Path
from acceptance.preflight import ROOT, SHARED, write, sha
from acceptance.run_matrix import devices
from acceptance.aggregate import stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--device',type=int,required=True);a=p.parse_args()
    root=ROOT/'outputs/acceptance'/a.campaign
    plan={'orders':['AB','BA','AB'],'samples':157,'measured_sequences_per_round':1,
        'full_warmup_sequences_per_profile_per_round':1,'physical_device':a.device,
        'algorithm':'KERV seven, KF enabled, horizon220, threshold15->3, seed7, BF16, original TF',
        'scope':'same-observation runtime self-test, not closed-loop success or Spec-VLA speedup'}
    if (root/'replay_plan.json').exists():raise FileExistsError('Replay plan already exists; preserve prior campaign')
    write(root/'replay_plan.json',plan)
    (root/'replay_plan.sha256').write_text(sha(root/'replay_plan.json')+'\n')
    completed=[]
    for repeat,order in enumerate(plan['orders'],1):
        dest=root/'replays'/f'round{repeat}_{order}';dest.mkdir(parents=True,exist_ok=False)
        smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True).stdout
        (dest/'npu_before.txt').write_text(smi);state=devices(smi).get(a.device)
        assert state and state['used_mb']<6000 and state['total_mb']-state['used_mb']>36000, 'NPU occupied'
        container_out=f'/data/data/KERV/outputs/acceptance/{a.campaign}/replays/{dest.name}'
        cmd=[str(ROOT/'scripts/benchmark_kerv_moore_ascend.sh'),'--samples','157','--repeats','1',
            '--profile-order',order,'--input-dir','/data/data/KERV/outputs/npu_ar_tensorflow/observations','--output',container_out]
        env=os.environ.copy();env['KERV_NPU_ID']=str(a.device)
        meta={'run_id':dest.name,'command':cmd,'cwd':str(ROOT),'environment':{'KERV_NPU_ID':str(a.device)},
            'start_timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RUNNING'}
        (dest/'source_manifest.sha256').write_text((root/'inventory/source_after_manifest.sha256').read_text())
        write(dest/'run.json',meta)
        with (dest/'stdout.log').open('w') as f:
            proc=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        shared=SHARED/'outputs/acceptance'/a.campaign/'replays'/dest.name
        if shared.exists():subprocess.run(['rsync','-a',str(shared)+'/',str(dest)+'/'],check=True)
        meta.update(exit_code=proc.returncode,end_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status='PASS' if proc.returncode==0 else 'FAIL')
        write(dest/'run.json',meta)
        if proc.returncode:raise SystemExit('Replay failed; retained '+str(dest))
        report=json.loads((dest/'benchmark.json').read_text());assert report['actions_exact']
        completed.append(report)
        print(dest.name,'PASS',report['model_speedup'],flush=True)
    first=completed[0]
    for report in completed:
        for key in ['observation_sha256','action_protocol','dtype','seed','planning_horizon','physical_device','processor_token_cache','rope','linear_fusion','graphs']:
            assert report[key]==first[key],key
        for profile in ['native','runtime']:
            assert report['profiles'][profile]['actions']==first['profiles'][profile]['actions'],'cross-round action drift'
    combined={}
    for profile in ['native','runtime']:
        combined[profile]={metric:stats([x for report in completed for x in report['profiles'][profile][key]])
            for metric,key in [('model_ms','raw_model_ms'),('pipeline_ms','raw_pipeline_ms')]}
        combined[profile]['per_round_model_mean_ms']=[r['profiles'][profile]['model_mean_ms'] for r in completed]
    summary={'scope':plan['scope'],'rounds':3,'profile_order':['AB','BA','AB'],
        'measured_actions_per_profile':471,'all_actions_exact':True,'max_action_difference':0,
        'physical_device':a.device,'profiles':combined,
        'model_speedup':combined['native']['model_ms']['mean']/combined['runtime']['model_ms']['mean'],
        'pipeline_speedup':combined['native']['pipeline_ms']['mean']/combined['runtime']['pipeline_ms']['mean'],
        'concurrent_other_work':'closed-loop/transfer diagnostic on chip4 and heavy CANN profiling on chip8 during parts of rounds1/2; shared host CPU not affinity-isolated; no same-device concurrent model',
        'formal_acceptance_status':'BLOCKED: B0/A100/M2/P2/P3 and decisions missing'}
    write(root/'reports/replay_results.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
