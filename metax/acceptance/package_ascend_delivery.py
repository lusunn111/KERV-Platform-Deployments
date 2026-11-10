"""Package the Ascend documents and raw statistics; validate on a CPU-only path."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from acceptance.preflight import ROOT

PREFIX='KERV_Ascend_Technical_Evaluation_20261027'
DOCUMENTS=[
 'docs/acceptance/ASCEND_TECHNICAL_OUTLINE_20261027.md',
 'docs/acceptance/ASCEND_TECHNICAL_OUTLINE_20261027.docx',
 'docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.md',
 'docs/acceptance/reports/ASCEND_EVALUATION_REPORT_20261027.docx',
 'docs/acceptance/README.md','docs/acceptance/reports/README.md','README_ASCEND.md',
]
TOOLS=['acceptance/'+s for s in ['__init__.py','preflight.py','aggregate.py','test_statistics.py',
 'evaluate_ascend_delivery.py','render_ascend_evaluation.py','export_technical_docx.py','package_ascend_delivery.py']]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True);a=p.parse_args();out=a.evidence.resolve()
    assert not a.archive.exists(),'Use a new archive path'
    inputs=json.loads((out/'inventory/input_manifest.json').read_text())
    for name,entry in inputs.items():assert sha(ROOT/name)==entry['sha256'],name
    docs=out/'documents';docs.mkdir(exist_ok=True)
    for name in DOCUMENTS[:4]:shutil.copy2(ROOT/name,docs/Path(name).name)
    shutil.copy2(__file__,out/'inventory/package_ascend_delivery.py')
    files={ROOT/name for name in inputs}|{ROOT/name for name in DOCUMENTS+TOOLS}
    files|={f for f in out.rglob('*') if f.is_file()}
    # All file names are rooted in this project. No weights or inference
    # environment are required to recompute the recorded statistics.
    with tempfile.TemporaryDirectory(prefix='kerv_statistics_only_') as directory:
        staged=Path(directory)/PREFIX;staged.mkdir()
        for path in sorted(files):
            dest=staged/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        empty=Path(directory)/'no_device_tools';empty.mkdir()
        command=[sys.executable,'-m','acceptance.evaluate_ascend_delivery','--output','outputs/acceptance/offline_recomputed_new']
        result=subprocess.run(command,cwd=staged,env={'PATH':str(empty),'LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1'},capture_output=True,text=True)
        (out/'inventory/offline_recompute.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
        reproduced=json.loads((staged/'outputs/acceptance/offline_recomputed_new/evaluation.json').read_text())
        expected=json.loads((out/'evaluation.json').read_text())
        reproduced.pop('recomputed_at');expected.pop('recomputed_at')
        assert reproduced==expected,'Offline statistics differ'
        marker=(staged/'outputs/acceptance/offline_recomputed_new/inventory/npu_state.txt').read_text()
        assert 'statistics-only host' in marker
    validation=dict(status='PASS',command=command,npu_tools_hidden=True,model_runs=0,
        all_evaluation_fields_equal_except_timestamp=True,old_source_campaigns_modified=False,
        scope='Independent directory containing only packaged files; standard Python, no Torch or NPU required for statistics')
    dump(out/'offline_recompute_validation.json',validation)
    instructions=f'''# 昇腾技术大纲与测评交付包

本包是项目内部技术文档与离线统计复算材料，包含已引用的原始记录、哈希、Word/Markdown和统计工具。
不包含完整模型权重、NPU运行环境或全部历史源码归档，不作为完整推理部署镜像。

解压后进入本目录，用Python3.9及以上版本复算（仅标准库，不需要Torch/NPU）：

```bash
python -m acceptance.evaluate_ascend_delivery --output outputs/acceptance/recomputed_new
python -m unittest acceptance.test_statistics -v
```

文档位于docs/acceptance/；源证据索引与CSV位于{out.relative_to(ROOT)}。
复算输出目录必须不存在；原始数据不覆盖。P95是nearest-rank。
已在独立目录隐藏全部NPU工具，核对evaluation.json除时间戳外完全相等，见offline_recompute_validation.json。
BUNDLE_MANIFEST.json给出每个归档文件的SHA256；传输包的整包SHA256在外部同名.sha256文件。

本机范围：Ascend、OpenVLA/SpecVLA drafter、LIBERO Goal。正式验收未完成项见报告第9节。
原始120回合由6个配置复用20个task/init/seed组合，不是120个独立质量样本。
'''
    (out/'OFFLINE_RECOMPUTE.md').write_text(instructions)
    delivery=dict(version='1.0',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Ascend local technical outline and evaluation, not formal third-party acceptance',
        documents=DOCUMENTS[:4],evidence_root=str(out.relative_to(ROOT)),archive=str(a.archive),
        new_model_runs=0,offline_statistics_validation='PASS',
        primary_source_files=len(inputs),formal_acceptance='BLOCKED',recommended_quantized_profile='mixed',default_precision='BF16')
    dump(out/'delivery_manifest.json',delivery)
    # This checksum list seals this new delivery; historical seals stay intact.
    paths=sorted(f for f in out.rglob('*') if f.is_file() and f.name!='delivery_checksums.sha256')
    (out/'delivery_checksums.sha256').write_text(''.join(sha(f)+'  '+str(f.relative_to(out))+'\n' for f in paths))
    files|={f for f in out.rglob('*') if f.is_file()}
    entries=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)]
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for entry in entries:archive.write(ROOT/entry['path'],PREFIX+'/'+entry['path'])
        archive.writestr(PREFIX+'/README_DELIVERY.md',instructions)
        archive.writestr(PREFIX+'/BUNDLE_MANIFEST.json',json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(a.archive) as archive:
        assert archive.testzip() is None
        for entry in entries:
            with archive.open(PREFIX+'/'+entry['path']) as f:
                h=hashlib.sha256()
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==entry['sha256'],entry['path']
    a.archive.with_suffix(a.archive.suffix+'.sha256').write_text(sha(a.archive)+'  '+a.archive.name+'\n')
    print(json.dumps(dict(archive=str(a.archive),archive_bytes=a.archive.stat().st_size,files=len(entries),offline_validation='PASS',zip_validation='PASS'),ensure_ascii=False))

if __name__=='__main__':main()
