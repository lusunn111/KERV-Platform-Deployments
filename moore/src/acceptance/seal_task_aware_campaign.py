"""Archive source/documents and seal a completed local quantization campaign.

The verification log is a sibling of the campaign, keeping every file inside
covered by checksums.sha256 except the checksum manifest itself.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from acceptance.common import load,dump,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign.resolve();root=Path('/root/KERV')
    if c.parent!=root/'outputs/task_aware_quantization':raise ValueError('Unexpected campaign parent')
    if (c/'checksums.sha256').exists():raise RuntimeError('Already sealed; work in a new campaign or copy')
    assert load(c/'inventory/selection.json')['status']=='RETAIN_VERIFIED_EXACT_BF16'
    assert load(c/'reports/tests_final.json')['status']=='PASS'
    assert load(c/'reports/dependencies_checked.json')['status']=='PASS'
    for file in (c/'operations').glob('*/command.json'):
        assert load(file).get('end_timestamp'),str(file)
    runs=[load(f) for f in sorted((c/'runs').glob('*/run.json'))]
    assert len(runs)==14 and all(x['status']=='PASS' and x['end_timestamp'] for x in runs)
    status=subprocess.check_output(['git','status','--short'],cwd=root/'src',text=True)
    (c/'inventory/git_status_final.txt').write_text(status)
    (c/'inventory/git_head_final.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'src',text=True))
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    (c/'inventory/device_after.log').write_text(device)
    registry={}
    for file in sorted((c/'inventory').glob('cases*.csv')):
        with file.open() as stream:
            for row in csv.DictReader(stream):registry.setdefault(row['run_id'],row)
    actual={r['run_id']:r for r in runs};cases=[]
    for name,row in sorted(registry.items()):
        result=actual.get(name)
        checked=list((c/'resource_checks').glob(name+'_*.log'))
        state=result['status'] if result else 'RESOURCE_CHECK_REJECTED' if checked else 'REGISTERED_NOT_EXECUTED'
        cases.append({'run_id':name,'execution':state,'task_success':result.get('task_success') if result else None})
    dump(c/'inventory/execution_status.json',{'registered_cases':len(registry),'actual_full_episodes':len(runs),
         'case_status':cases,'note':'Task success per policy; registered/unstarted cases excluded from task denominators'})
    dependencies={}
    for relative in ('outputs/precision_refinement/20261027T084306Z/checksums.sha256',
                     'outputs/calibrated_quantization/20261026T045030Z_v2/checksums.sha256'):
        file=root/relative
        if file.exists():dependencies[relative]={'bytes':file.stat().st_size,'sha256':sha256(file)}
    dump(c/'inventory/external_dependencies.json',{'prior_seal_manifests':dependencies,
         'checkpoint_dependencies':'inventory/weight_manifest.json;11 files rehashed at campaign start',
         'runtime':'Existing kerv-musa-runtime container and installed MUSA stack; not included in source archive'})
    sources=[f for f in sorted((root/'src').rglob('*')) if (f.is_file() or f.is_symlink())
             and not any(x in ('.git','__pycache__','.pytest_cache') for x in f.relative_to(root).parts)]
    sources += [root/'README.md',root/'TestOutline.md',root/'docs/acceptance/MOORE_TASK_AWARE_QUANTIZATION_20261028.md']
    manifest={}
    with tarfile.open(c/'inventory/source_final.tar.gz','w:gz',dereference=False) as archive:
        for file in sources:
            rel=str(file.relative_to(root))
            manifest[rel]={'symlink':os.readlink(file)} if file.is_symlink() else {'bytes':file.stat().st_size,'sha256':sha256(file)}
            archive.add(file,arcname=rel,recursive=False)
    dump(c/'inventory/source_final_manifest.json',manifest)
    dump(c/'inventory/seal.json',{'checksum_generation_started_at':datetime.now(timezone.utc).isoformat(),
         'argv':sys.argv,'source_files':len(sources),'full_episodes':len(runs),'final_tests':load(c/'reports/tests_final.json'),
         'gpu_idle_on_completion':'No running processes found' in device,
         'checksum_scope':'Every regular file inside campaign except checksums.sha256; verification log is outside campaign',
         'source_archive_scope':'Current src tree plus root README,TestOutline,and this report; excludes git/cache and external checkpoints/container'})
    files=[f for f in sorted(c.rglob('*')) if f.is_file() and f.name!='checksums.sha256']
    hashes={str(f.relative_to(c)):sha256(f) for f in files}
    (c/'checksums.sha256').write_text(''.join(digest+'  '+name+'\n' for name,digest in hashes.items()))
    print('SEALED',len(files),'files; source archive',len(sources),'entries',flush=True)
    failures=[name for name,digest in hashes.items() if sha256(c/name)!=digest]
    log={'completed_at':datetime.now(timezone.utc).isoformat(),'campaign':str(c),'status':'PASS' if not failures else 'FAIL',
         'files_checked':len(files),'failures':failures,'manifest_sha256':sha256(c/'checksums.sha256')}
    dump(c.parent/(c.name+'.integrity_check.json'),log)
    print(log,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
