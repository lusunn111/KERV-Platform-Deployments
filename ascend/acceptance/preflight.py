"""Freeze the local Ascend campaign without changing the runtime environment."""
import argparse, datetime, hashlib, json, os, platform, shutil, socket, subprocess, tarfile, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SHARED=Path('/public-flash/zhiyuan/liyuzhuo/data/KERV')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def command(args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    return {'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',help='New campaign ID; must not exist')
    p.add_argument('--freeze',action='store_true',help='Create inventory, complete source archive and weight/asset hashes')
    a=p.parse_args()
    name=a.campaign or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'outputs/acceptance'/name
    out.mkdir(parents=True,exist_ok=False);inv=out/'inventory';inv.mkdir()
    checks={'taskbook':ROOT/'docs/acceptance/sources/2025-1215-课题任务书.pdf',
        'verifier_index':SHARED/'checkpoints/openvla-libero-goal/model.safetensors.index.json',
        'drafter':SHARED/'checkpoints/kerv-drafter/pytorch_model.bin',
        'libero':SHARED/'third_party/LIBERO/libero/libero/benchmark/__init__.py',
        'venv':SHARED/'.venv/bin/python'}
    report={'campaign_id':name,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':socket.gethostname(),'platform':platform.platform(),'workspace':str(ROOT),'runtime_copy':str(SHARED),
        'container':'kerv-ascend','paths':{k:{'path':str(v),'exists':v.exists()} for k,v in checks.items()},
        'taskbook_sha256':sha(checks['taskbook']),'scope':'Ascend local self-test only; not third-party acceptance'}
    assert report['taskbook_sha256']=='94c0d5812d9bf43b48cd5689083d408f41753010aa0bfac537b109fdca0f3f4c'
    write(inv/'inventory.json',report)
    (inv/'npu_before.txt').write_text(command(['npu-smi','info'])['stdout'])
    probe="import json,sys,torch,torch_npu,importlib.metadata as m; print(json.dumps({'python':sys.version,'torch':torch.__version__,'torch_npu':m.version('torch_npu'),'packages':{x:m.version(x) for x in ['transformers','tensorflow','numpy','mujoco','robosuite','psutil']}}))"
    env=command(['docker','exec','kerv-ascend','/data/data/KERV/.venv/bin/python','-c',probe])
    write(inv/'environment.json',env)
    inspect=json.loads(command(['docker','inspect','kerv-ascend'])['stdout'])[0]
    write(inv/'container.json',{k:inspect[k] for k in ['Id','Image','Mounts']})
    freeze=[]
    for base in ['openvla','runtime_opt','ports','scripts','tests','configs','acceptance','third_party/FlagScale','third_party/LIBERO','docs']:
        freeze += [f for f in (ROOT/base).rglob('*') if f.is_file() and not any(x in f.parts for x in ['.git','__pycache__','.venv'])]
    freeze += [f for f in ROOT.iterdir() if f.is_file() and f.suffix in ['.py','.md','.txt','.toml','.yaml','.yml','.sh']]
    freeze=sorted(set(freeze))
    hashes={str(f.relative_to(ROOT)):sha(f) for f in freeze}
    (inv/'source_manifest.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    if a.freeze:
        with tarfile.open(inv/'source_before.tar.gz','w:gz') as archive:
            for f in freeze:archive.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    weights={}
    for f in sorted((SHARED/'checkpoints').rglob('*')):
        if f.is_file():weights[str(f.relative_to(SHARED))]={'bytes':f.stat().st_size,'sha256':sha(f)}
    write(inv/'weight_manifest.json',weights)
    index=json.loads(checks['verifier_index'].read_text())
    missing=[x for x in sorted(set(index['weight_map'].values())) if not (checks['verifier_index'].parent/x).is_file()]
    report['missing_verifier_shards']=missing
    report['source_file_count']=len(freeze)
    report['source_manifest_sha256']=sha(inv/'source_manifest.sha256')
    # The venv interpreter symlink points inside the container, not the host.
    report['venv_container_probe_passed']=env['exit_code']==0
    report['status']='PASS' if not missing and all(x['exists'] for k,x in report['paths'].items() if k!='venv') and env['exit_code']==0 else 'BLOCKED'
    write(inv/'inventory.json',report)
    print(json.dumps({'campaign_id':name,'status':report['status'],'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
