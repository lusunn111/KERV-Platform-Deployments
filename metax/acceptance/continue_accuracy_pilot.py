"""Advance registered independent stages as their device/data dependencies finish."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from acceptance.run_matrix import devices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];base=root/'outputs/acceptance'/a.campaign
    shared=Path('/public-flash/zhiyuan/liyuzhuo/data/KERV');runtime='/data/data/KERV'
    remote=f'{runtime}/outputs/acceptance/{a.campaign}'
    launched={};records={};logs=[]

    def status(name):
        f=base/'runs'/name/'run.json'
        try:return json.loads(f.read_text()).get('status') if f.exists() else None
        except json.JSONDecodeError:return 'WRITING'

    def idle(device):
        states=devices(subprocess.check_output(['npu-smi','info'],text=True));s=states.get(device)
        assert s and s['used_mb']<6000 and s['total_mb']-s['used_mb']>36000,('device occupied',device,s)

    def start(name,command,device=None):
        if device is not None:idle(device)
        f=(base/f'{name}_driver.log').open('w');logs.append(f)
        proc=subprocess.Popen(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        launched[name]=proc;records[name]={'command':command,'pid':proc.pid,'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (base/'orchestration.json').write_text(json.dumps(records,indent=2)+'\n')
        print('START',name,proc.pid,flush=True)

    def validation(name):
        start(name,['python','-m','acceptance.run_matrix','--campaign',a.campaign,'--cases',str(base/'validation.json'),'--only',name])

    def docker(script,args,device,cpu=False):
        env={'ASCEND_RT_VISIBLE_DEVICES':str(device),'KERV_DEVICE_BACKEND':'cpu' if cpu else 'npu',
            'KERV_ACTION_PROTOCOL':'seven','KERV_NPU_DRAFT_RMS_REFERENCE':'0','KERV_NPU_QUANT':'none',
            'OMP_NUM_THREADS':'16','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1',
            'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','MAX_JOBS':'2',
            'PYTHONPATH':':'.join(runtime+'/'+p for p in ['acceptance','ports/moore_ascend/runtime_opt','ports/moore_ascend','ports/moore_ascend/openvla'])}
        cmd=['docker','exec','-w',runtime+'/ports/moore_ascend']
        for k,v in env.items():cmd+=['-e',k+'='+v]
        # Structured argv keeps paths and arguments literal; PATH is only set
        # for the C++ extension build helper inherited from the existing launcher.
        cmd+=['kerv-ascend','bash','-c','export PATH="/data/data/KERV/.venv/bin:$PATH"; exec python "$@"',
              'accuracy-pilot',runtime+'/acceptance/'+script,*args]
        return cmd

    while True:
        for name,proc in list(launched.items()):
            code=proc.poll()
            if code is not None and 'exit_code' not in records[name]:
                records[name].update(exit_code=code,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                (base/'orchestration.json').write_text(json.dumps(records,indent=2)+'\n')
                print('END',name,code,flush=True)
                if code:raise RuntimeError(('stage failed; other independent jobs retained',name,code))
        development=['bf16_calibration_init0','int8_g128_init0','int8_g32_init0']
        for name in development:
            if status(name) in ['FAIL','BLOCKED']:raise RuntimeError(('development failed',name,status(name)))
        if status(development[0])=='PASS':
            if 'bf16_init1' not in launched:validation('bf16_init1')
            if 'calibration' not in launched:
                start('calibration',docker('calibrate_quant_policy.py',[
                    '--inputs',remote+'/runs/bf16_calibration_init0/calibration/inputs.pt',
                    '--output',remote+'/calibrated','--max-upgrades','24'],6,cpu=True),6)
        if status(development[1])=='PASS' and 'int8_g128_init1' not in launched:validation('int8_g128_init1')
        if status(development[2])=='PASS':
            if 'int8_g32_init1' not in launched:validation('int8_g32_init1')
            if 'timing_g32' not in launched:
                start('timing_g32',docker('interleave_quantization.py',[
                    '--output',remote+'/timing_g32','--samples','64','--repeats','2','--quant-group-size','32'],14),14)
        if records.get('calibration',{}).get('exit_code')==0 and records.get('timing_g32',{}).get('exit_code')==0:
            if 'int8_calibrated_init1' not in launched:
                subprocess.run(['rsync','-a',str(shared/'outputs/acceptance'/a.campaign/'calibrated')+'/',str(base/'calibrated')+'/'],check=True)
                validation('int8_calibrated_init1')
        if status('int8_calibrated_init1')=='PASS' and 'timing_calibrated' not in launched:
            start('timing_calibrated',docker('interleave_quantization.py',[
                '--output',remote+'/timing_calibrated','--samples','64','--repeats','2',
                '--quant-policy',remote+'/calibrated/policy.json'],14),14)
        completed=['bf16_init1','int8_g128_init1','int8_g32_init1','int8_calibrated_init1','calibration','timing_g32','timing_calibrated']
        if all(records.get(name,{}).get('exit_code')==0 for name in completed):break
        time.sleep(5)
    subprocess.run(['rsync','-a',str(shared/'outputs/acceptance'/a.campaign)+'/',str(base)+'/'],check=True)
    subprocess.run(['python','-m','acceptance.report_accuracy_pilot',str(base)],cwd=root,check=True)
    print('PILOT COMPLETE',flush=True)


if __name__=='__main__':main()
