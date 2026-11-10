"""Compare cold BF16 and exact-MLP runs without treating execution PASS as SR."""
import argparse
from pathlib import Path
from acceptance.common import dump,load,rows,distribution,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--task-id',type=int,required=True);p.add_argument('--state-id',type=int,required=True)
    p.add_argument('--candidate-run-id')
    a=p.parse_args();c=a.campaign
    names=[f'full_{label}_task{a.task_id}_state{a.state_id}_s7' for label in ['bf16','exact_mlp']]
    if a.candidate_run_id:names[1]=a.candidate_run_id
    data={};manifests={}
    for name in names:
        out=c/'runs'/name;run=load(out/'run.json');assert run['status']=='PASS' and run['exit_code']==0
        steps=rows(out/'steps.jsonl');memory=rows(out/'memory.jsonl')
        data[name]={'run':run,'steps':steps,'episodes':rows(out/'episodes.jsonl'),
            'peak_allocated_bytes':max(x.get('peak_allocated_bytes') or 0 for x in memory),
            'peak_reserved_bytes':max(x.get('peak_reserved_bytes') or 0 for x in memory),
            'model_seconds_all':distribution(x['model_seconds'] for x in steps),
            'model_seconds_excluding_first':distribution(x['model_seconds'] for x in steps[1:]),
            'pipeline_seconds_all':distribution(x['pipeline_seconds'] for x in steps)}
    left,right=[data[n]['steps'] for n in names]
    common=min(len(left),len(right));errors=[abs(x-y) for u,v in zip(left,right) for x,y in zip(u['raw_action'],v['raw_action'])]
    proof=rows(c/'runs'/names[1]/'precision_storage.jsonl');loaded=[x for x in proof if x['phase']=='loaded'][0];final=[x for x in proof if x['phase']=='final'][0]
    assert not loaded['lossy_quantization'] and loaded['manifest']['int8_module_count']==0
    assert loaded['manifest']['exact_module_count']==96
    originals={x['name']:x for x in load(c/'screening/weight_proof.json')['original_weights_and_exact_codec']}
    for entry in loaded['modules']:
        assert entry['original_sha256']==entry['decoded_sha256']==entry['gpu_decoded_sha256']==originals[entry['name']]['original_sha256']
    assert len(final['modules'])==96 and all(x['forward_calls']>0 for x in final['modules'])
    assert final['kalman_batch_hits']>0 and len(final['softmax_guard_calls'])==32
    assert all(x>0 for x in final['softmax_guard_calls'].values())
    ref=[x for x in rows(c/'runs'/names[0]/'precision_reference.jsonl') if x['phase']=='final'][0]
    assert ref['kalman_batch_hits']>0 and all(x>0 for x in ref['softmax_guard_calls'].values())
    policy=Path(data[names[1]]['run']['policy_path']);assert sha256(policy)==loaded['policy_sha256']
    summary={'task_id':a.task_id,'state_id':a.state_id,'execution':'PASS','steps':[len(left),len(right)],
             'task_success':[data[n]['run']['task_success'] for n in names],
             'same_length':len(left)==len(right),'exact_actions':sum(x['action_tokens']==y['action_tokens'] for x,y in zip(left,right)),
             'exact_raw_actions':sum(x['raw_action']==y['raw_action'] for x,y in zip(left,right)),
             'exact_observations':sum(x['observation_hash']==y['observation_hash'] for x,y in zip(left,right)),
             'common_steps':common,'raw_error':distribution(errors),'max_abs':max(errors),
             'decoded_original_weight_hashes_checked':len(loaded['modules']),
             'static_weight_saving_bytes':loaded['manifest']['estimated_weight_storage_saving_bytes'],
             'allocated_peak_saving_bytes':data[names[0]]['peak_allocated_bytes']-data[names[1]]['peak_allocated_bytes'],
             'reserved_peak_saving_bytes':data[names[0]]['peak_reserved_bytes']-data[names[1]]['peak_reserved_bytes'],
             'runs':{n:{k:v for k,v in data[n].items() if k not in ('steps','run')} for n in names},
             'scope':'One frozen task/state/seed pair; no suite success-rate claim. Original BF16 weights, same guarded/shared runtime.'}
    summary['precision_parity']='MATCH' if summary['same_length'] and summary['exact_actions']==summary['exact_raw_actions']==summary['exact_observations']==common else 'DIFF_FOUND'
    dump(c/'reports'/f'closedloop_task{a.task_id}_state{a.state_id}.json',summary)
    print({k:v for k,v in summary.items() if k!='runs'})
    if summary['precision_parity']!='MATCH':raise SystemExit(1)


if __name__=='__main__':main()
