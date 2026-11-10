"""Recompute same-policy replay claims from per-frame evidence, on CPU."""
import argparse
import math
from pathlib import Path
from acceptance.common import dump,load,rows,sha256,distribution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();e=a.experiment;result=load(e/'results.json');protocol=result['protocol']
    captured=rows(Path(protocol['capture'])/'steps.jsonl')
    assert sha256(Path(protocol['capture'])/'observations.npz')==protocol['capture_sha256']
    assert sha256(protocol['policy'])==protocol['policy_sha256']
    labels=['quant_reference','quant_kernel'];all_data={l:[] for l in labels};checks=[]
    for pair in result['rounds']:
        rid=pair['round'];left,right=[rows(e/l/('measure_%d.jsonl'%rid)) for l in labels]
        assert len(left)==len(right)==len(captured)==pair['frames']
        for i,(x,y,z) in enumerate(zip(left,right,captured)):
            assert x['frame']==y['frame']==i
            assert x['observation_hash']==y['observation_hash']==z['observation_hash']
            for item in [x,y]:
                assert len(item['action_tokens'])==len(item['raw_action'])==len(item['executed_action'])==7
                assert all(math.isfinite(t) for t in item['raw_action'])
                assert all(math.isfinite(item[k]) and item[k]>0 for k in ['model_seconds','pipeline_seconds'])
        exact=sum(x['action_tokens']==y['action_tokens'] for x,y in zip(left,right))
        errors=[abs(x-y) for u,v in zip(left,right) for x,y in zip(u['raw_action'],v['raw_action'])]
        assert exact==pair['exact_action_tokens']
        assert sum(x['raw_action']==y['raw_action'] for x,y in zip(left,right))==pair['exact_raw_actions']
        assert sum(x['executed_action'][-1]==y['executed_action'][-1] for x,y in zip(left,right))==pair['gripper_matches']
        assert max(errors)==pair['max_abs']
        assert distribution(errors)==pair['raw_action_error']
        for label,data in zip(labels,[left,right]):
            for key in ['model_seconds','pipeline_seconds']:
                assert distribution(x[key] for x in data)==pair['times'][label][key]
            all_data[label]+=data
        for key,out in [('model_seconds','model_speedup'),('pipeline_seconds','pipeline_speedup')]:
            ratio=sum(x[key] for x in left)/sum(x[key] for x in right)
            assert math.isclose(ratio,pair[out],rel_tol=1e-12)
        checks.append({'round':rid,'frames':len(left),'exact_actions':exact,'exact_raw_actions':pair['exact_raw_actions'],'gripper_matches':pair['gripper_matches'],'model_speedup':pair['model_speedup'],
                       'pipeline_speedup':pair['pipeline_speedup'],'max_abs':max(errors)})
    assert len(checks)==protocol['measured_rounds']
    prefetch=protocol.get('candidate_backend') in ('prefetch','tree_prefetch')
    proof_file='calibrated_prefetch.jsonl' if prefetch else 'calibrated_kernel.jsonl'
    loaded=[x for x in rows(e/'quant_kernel'/proof_file) if x['phase']=='loaded']
    final=[x for x in rows(e/'quant_kernel'/proof_file) if x['phase']=='final']
    assert len(loaded)==len(final)==1
    modules=load(protocol['policy'])['modules'];expected={x['name']:x['decoded_weight_sha256'] for x in modules}
    assert {x['name']:x['decoded_sha256'] for x in loaded[0]['modules']}==expected
    assert len(final[0]['modules'])==len(modules)
    if prefetch:
        assert final[0]['fallback_calls']==0
        assert all(x['forward_calls']==x['prefetch_forward_calls']>0 for x in final[0]['modules'])
        assert final[0]['stats']['inference']['linears']==final[0]['stats']['inference']['decodes']
        assert final[0]['stats']['inference']['linears']==final[0]['stats']['inference']['passes']*len(modules)
        assert loaded[0]['pool_slots']==2
    else:
        assert all(x['forward_calls']>0 and x['decode_calls']==x['forward_calls']+1 for x in final[0]['modules'])
        assert all(x['bit_exact'] for x in loaded[0]['modules'])
        assert loaded[0]['persistent_decoded_weight_bytes']==0
    assert load(e/'quant_reference/parameter_hashes.json')==load(e/'quant_kernel/parameter_hashes.json')
    baseline_proof=[x for x in rows(e/'quant_reference/calibrated_quantization.jsonl') if x['phase']=='loaded'][0]
    assert {x['name']:x['decoded_sha256'] for x in baseline_proof['modules']}==expected
    for label in labels:assert load(e/label/'command.json')['exit_code']==0
    tree_hits=None
    shared_history=None
    if protocol.get('candidate_backend')=='tree_rows':
        audit=rows(e/'quant_kernel/calibrated_tree_runtime.jsonl')
        assert len(audit)==1
        audit=audit[0]
        assert audit['tree_builder_enabled'] and audit['tree_builder_hits']>0
        assert not audit['static_tree_runtime'] and not audit['kalman_batch_enabled']
        assert audit['policy_sha256']==protocol['policy_sha256']
        tree_hits=audit['tree_builder_hits']
    if protocol.get('candidate_backend')=='shared_history':
        audit=rows(e/'quant_kernel/calibrated_shared_history.jsonl')
        loaded_shared=[x for x in audit if x['phase']=='loaded']
        finals_shared=[x for x in audit if x['phase']=='final']
        assert len(loaded_shared)==len(finals_shared)==1
        shared_history=finals_shared[0]
        assert shared_history['calls']>0 and shared_history['chains']>shared_history['calls']
        assert shared_history['tree_builder_enabled'] and shared_history['kalman_batch_enabled']
        assert not shared_history['static_tree_runtime']
        assert not shared_history['quantization_policy_changed']
        assert shared_history['persistent_decoded_weight_bytes']==0
        assert shared_history['policy_sha256']==protocol['policy_sha256']
        guards=[]
        for label in labels:
            audit=rows(e/label/'safe_softmax.jsonl')
            loaded_guard=[x for x in audit if x['phase']=='loaded'][0]
            final_guard=[x for x in audit if x['phase']=='final'][0]
            assert len(loaded_guard['modules'])==32
            assert all(x>0 for x in final_guard['calls'].values())
            guards.append(loaded_guard['implementation_file_sha256'])
        assert guards[0]==guards[1]
    drift={}
    for label in labels:
        passes=[rows(e/label/('measure_%d.jsonl'%i)) for i in range(protocol['measured_rounds'])]
        drift[label]=[sum(x['action_tokens']==y['action_tokens'] for x,y in zip(passes[0],later)) for later in passes[1:]]
    pooled={key:sum(x[key] for x in all_data[labels[0]])/sum(x[key] for x in all_data[labels[1]]) for key in ['model_seconds','pipeline_seconds']}
    report={'status':'PASS','scope':'evidence integrity and recomputed numbers, separate from model quality acceptance',
        'rounds':checks,'pooled_speedup':pooled,'same_policy':True,'decoded_weights_bit_exact':len(modules),
        'actual_operator_modules':len(final[0]['modules']),
        'lookup_storage_bytes':load(e/'quant_kernel/ready.json')['quantization_manifest'].get('lookup_storage_bytes',sum(x.get('lookup_bytes',0) for x in loaded[0]['modules'])),
        'workspace_bytes':loaded[0].get('workspace_bytes',0),
        'tree_builder_hits':tree_hits,
        'shared_history_audit':shared_history,
        'total_frames':len(all_data[labels[0]]),'exact_actions':sum(x['exact_actions'] for x in checks),
        'exact_raw_actions':sum(x['exact_raw_actions'] for x in checks),
        'gripper_matches':sum(x['gripper_matches'] for x in checks),
        'within_profile_round0_token_matches':drift,
        'speedup_reference':protocol.get('reference_backend','reference')+' calibrated decoder, NOT native BF16',
        'precision_scope':'operator adds no weight quantization; action parity only on this fixed replay; original policy not quality-approved'}
    dump(e/'evidence_checks.json',report)
    print(report)


if __name__=='__main__':main()
