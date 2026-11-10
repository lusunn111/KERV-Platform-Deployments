"""Execute registered local Ascend cases, serially on one physical chip."""
import argparse, datetime, json, os, re, signal, subprocess, threading, time
from pathlib import Path
from acceptance.preflight import ROOT, SHARED, sha, write


def devices(text):
    result={}
    for line in text.splitlines():
        parts=line.strip().strip('|').split('|')
        if len(parts)>=3 and '0000:' in parts[1]:
            ids=parts[0].split(); pairs=re.findall(r'(\d+)\s*/\s*(\d+)',parts[-1])
            if len(ids)==2 and pairs:
                used,total=map(int,pairs[-1]);result[int(ids[1])]={'used_mb':used,'total_mb':total}
    return result


def monitor(out,stop):
    with (out/'vendor_memory.jsonl').open('w') as f:
        while not stop.is_set():
            start=time.monotonic();p=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
            f.write(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'sample_interval_seconds_requested':10,'query_seconds':time.monotonic()-start,
                'devices':devices(p.stdout),'raw':p.stdout,'exit_code':p.returncode})+'\n');f.flush()
            stop.wait(10)


def execute(campaign,case):
    dest=ROOT/'outputs/acceptance'/campaign/'runs'/case['run_id']
    dest.mkdir(parents=True,exist_ok=False)
    current=subprocess.run(['npu-smi','info'],capture_output=True,text=True).stdout
    (dest/'npu_before.txt').write_text(current)
    state=devices(current).get(int(case['physical_device_id']))
    if state is None or state['used_mb']>6000 or state['total_mb']-state['used_mb']<36000:
        write(dest/'run.json',{'case':case,'status':'BLOCKED','error':'physical device not idle or insufficient free HBM','device':state})
        return False
    metrics=f"/data/data/KERV/outputs/acceptance/{campaign}/runs/{case['run_id']}/metrics"
    command=[str(ROOT/'scripts/run_kerv_moore_ascend.sh'),'--output-dir',f"outputs/acceptance/{campaign}/runs/{case['run_id']}/model",
        '--max-tasks',str(case['max_tasks']),'--trials',str(case['trials']),
        '--max-episode-steps',str(case['max_steps']),
        '--override','inference.runtime.max_planning_steps=220',
        '--override',f"inference.runtime.seed={case['seed']}"]
    for value in case.get('extra_overrides',[]):command.extend(['--override',value])
    env=os.environ.copy(); extra={'KERV_NPU_ID':str(case['physical_device_id']),
        'KERV_MOORE_PROFILE':case['runtime_profile'],'KERV_ACTION_PROTOCOL':'seven',
        'KERV_ACCEPTANCE_DIR':metrics,'KERV_ACCEPTANCE_RUN_ID':case['run_id'],
        'KERV_ACCEPTANCE_TRANSFERS':str(int(case.get('transfer_diagnostic',False)))}
    extra['KERV_NPU_QUANT']=case.get('quantization','none')
    extra['KERV_NPU_QUANT_GROUP_SIZE']=str(case.get('quant_group_size',128))
    extra['KERV_NPU_QUANT_LAYERS']=case.get('quant_layers','all')
    extra['KERV_NPU_QUANT_POLICY']=case.get('quant_policy','')
    extra['KERV_LIBERO_INITIAL_STATE_OFFSET']=str(case.get('initial_state_offset',0))
    extra['KERV_QUANT_CALIBRATION_DIR']=str(Path(metrics).parent/'calibration') if case.get('calibration') else ''
    env.update(extra)
    frozen=(ROOT/'outputs/acceptance'/campaign/'inventory/source_after_manifest.sha256').read_text()
    (dest/'source_manifest.sha256').write_text(frozen)
    meta={'campaign_id':campaign,**case,'status':'RUNNING','command':command,'cwd':str(ROOT),'environment':extra,
        'source_manifest':f'runs/{case["run_id"]}/source_manifest.sha256','weight_manifest':'inventory/weight_manifest.json',
        'action_protocol':'seven','start_timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'container':'kerv-ascend','model_id':'M1','platform_id':'P1','chip_id':'H2',
        'mode':'diagnostic' if case.get('transfer_diagnostic') or case.get('calibration') or any('profile_case.py' in x for x in case.get('extra_overrides',[])) else 'low_interference_self_test'}
    write(dest/'run.json',meta)
    stop=threading.Event();thread=threading.Thread(target=monitor,args=(dest,stop),daemon=True);thread.start()
    try:
        with (dest/'stdout.log').open('w') as f:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,
                timeout=case.get('timeout_seconds',1200))
        meta['exit_code']=p.returncode
        shared=SHARED/'outputs/acceptance'/campaign/'runs'/case['run_id']
        if shared.exists():subprocess.run(['rsync','-a',str(shared)+'/',str(dest)+'/'],check=True)
        meta['status']='PASS' if p.returncode==0 and (dest/'metrics/episodes.jsonl').exists() else 'FAIL'
    except Exception as e:
        meta.update(status='FAIL',exit_code=None,error=repr(e))
        if isinstance(e,subprocess.TimeoutExpired):
            # Only terminate processes carrying this exact run ID; never reset
            # the card or match other users by process name.
            cleanup="import os,signal; key=b'KERV_ACCEPTANCE_DIR='+os.environ['OWNED_METRICS_DIR'].encode(); me=os.getpid(); targets=[]\nfor x in os.listdir('/proc'):\n if x.isdigit() and int(x)!=me:\n  try:\n   if key in open('/proc/'+x+'/environ','rb').read().split(b'\\0'): targets.append(int(x))\n  except (OSError,ProcessLookupError):pass\nfor pid in targets:\n try:os.kill(pid,signal.SIGTERM)\n except ProcessLookupError:pass\nprint(targets)"
            stopped=subprocess.run(['docker','exec','-e','OWNED_METRICS_DIR='+metrics,'kerv-ascend',
                '/data/data/KERV/.venv/bin/python','-c',cleanup],capture_output=True,text=True)
            (dest/'timeout_cleanup.log').write_text(stopped.stdout+stopped.stderr)
    finally:
        stop.set();thread.join(timeout=15)
        meta['end_timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta['interpretation']='PASS means execution completed; task success and data completeness are evaluated separately'
        write(dest/'run.json',meta)
    print(json.dumps({'run_id':case['run_id'],'status':meta['status'],'exit_code':meta.get('exit_code')}),flush=True)
    return meta['status']=='PASS'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--only',nargs='*',help='Registered run IDs');a=p.parse_args()
    cases=json.loads(a.cases.read_text()); ids=[x['run_id'] for x in cases]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate run IDs')
    chosen=[x for x in cases if a.only is None or x['run_id'] in a.only]
    if a.only and set(a.only)!=set(x['run_id'] for x in chosen):raise ValueError('unknown run ID')
    lock=ROOT/'outputs/acceptance'/a.campaign/(a.cases.stem+'.sha256')
    expected=sha(a.cases)
    if lock.exists():assert lock.read_text().strip()==expected,'Cases changed after registration'
    else:lock.write_text(expected+'\n')
    for case in chosen:
        dest=ROOT/'outputs/acceptance'/a.campaign/'runs'/case['run_id']
        if dest.exists():
            print('SKIP existing evidence:',case['run_id'],flush=True);continue
        if not execute(a.campaign,case):raise SystemExit(1)

if __name__=='__main__':main()
