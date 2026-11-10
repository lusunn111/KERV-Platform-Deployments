"""Check run IDs, completeness, metrics and local source synchronization."""
import argparse, json
from pathlib import Path
from acceptance.aggregate import summarize
from acceptance.preflight import sha, ROOT, SHARED


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign-dir',required=True,type=Path);a=p.parse_args()
    errors=[];cases=[]
    for source in sorted(a.campaign_dir.glob('cases*.json')):
        cases.extend(json.loads(source.read_text()))
        frozen=source.with_suffix('.sha256')
        if frozen.exists() and frozen.read_text().strip()!=sha(source):errors.append('registered cases modified: '+source.name)
    ids=[x['run_id'] for x in cases]
    if len(set(ids))!=len(ids):errors.append('duplicate cases')
    statuses={}
    for case in cases:
        d=a.campaign_dir/'runs'/case['run_id']
        if not (d/'run.json').exists():statuses[case['run_id']]='NOT_STARTED';continue
        meta=json.loads((d/'run.json').read_text());statuses[case['run_id']]=meta['status']
        if meta['status']=='RUNNING':continue
        try:
            summary=summarize(d)
            if meta['status']=='PASS':
                assert meta['exit_code']==0
                for f in ['steps.jsonl','verify.jsonl','memory.jsonl','episodes.jsonl','collector.json']:assert (d/'metrics'/f).is_file(),f
                assert summary['steps']>0 and summary['verify_events']>0
                for f in ['command','source_manifest','weight_manifest','start_timestamp','end_timestamp']:assert f in meta,f
                assert (d/'stdout.log').exists()
                configs=list((d/'model/flagscale').rglob('inference.yaml'))
                assert len(configs)==1,'missing or ambiguous resolved config'
                assert (d/'metrics/collector.json').is_file()
        except Exception as e:errors.append(case['run_id']+': '+str(e))
    records=json.loads((a.campaign_dir/'inventory/sync_records.json').read_text())
    for r in records:
        if sha(ROOT/r['path'])!=r['after_sha256']:errors.append('workspace source drift: '+r['path'])
        if sha(SHARED/r['path'])!=r['after_sha256']:errors.append('runtime source drift: '+r['path'])
    result={'status':'PASS' if not errors else 'FAIL','scope':'completed-run structural evidence only','errors':errors,'case_status':statuses,
        'formal_acceptance_status':'BLOCKED','missing':['Spec-VLA B0 parity','A100','M2','P2/P3','exact H2 SKU','decisions confirmation','DMA direction/bytes correlation','independent third party']}
    (a.campaign_dir/'reports').mkdir(exist_ok=True)
    (a.campaign_dir/'reports/evidence_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
