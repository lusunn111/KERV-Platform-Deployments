"""Independently recompute precision screening and logical BF16 weight identity."""
import argparse
from pathlib import Path
from acceptance.common import dump,load,rows,distribution,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign;protocol=load(c/'screening_protocol.json')
    assert load(c/'screening_exit.json')['exit_code']==0
    capture=Path(protocol['capture']);assert sha256(capture/'observations.npz')==protocol['capture_sha256']
    captured=rows(capture/'steps.jsonl');base=rows(c/'screening/bf16/measure_0.jsonl')
    parameters=load(c/'screening/bf16/parameter_hashes.json')
    proof=load(c/'screening/weight_proof.json');weights={x['name']:x for x in proof['original_weights_and_exact_codec']}
    quant={x['name']:x for x in proof['quantized_modules']}
    for n,e in weights.items():
        assert parameters[n+'.weight']['sha256']==e['original_sha256']==e['gpu_decoded_sha256']
        assert parameters[n+'.weight']['shape']==e['shape']
    previous_q={};previous_exact={};results=[]
    for label in protocol['modes']:
        out=c/'screening'/label;actual=rows(out/'measure_0.jsonl');r=load(out/'comparison.json');manifest=load(out/'precision_manifest.json')
        assert len(base)==len(actual)==len(captured)==protocol['frames']
        assert all(x['frame']==y['frame']==i and x['observation_hash']==y['observation_hash']==z['observation_hash'] for i,(x,y,z) in enumerate(zip(base,actual,captured)))
        errors=[abs(x-y) for u,v in zip(base,actual) for x,y in zip(u['raw_action'],v['raw_action'])]
        assert distribution(errors)==r['raw_error'] and max(errors)==r['max_abs']
        assert sum(x['action_tokens']==y['action_tokens'] for x,y in zip(base,actual))==r['exact_actions']
        assert sum(x['raw_action']==y['raw_action'] for x,y in zip(base,actual))==r['exact_raw_actions']
        for e in manifest['modules']:
            expected=quant[e['name']]['decoded_weight_sha256'] if e['storage']=='int8_bf16_groups' else weights[e['name']]['original_sha256']
            assert e['decoded_sha256']==expected
        for n,e in load(out/'parameter_hashes.json').items():assert e==parameters[n],n
        counters=load(out/'execution_proof.json');deltas={}
        for key,previous in [('quantized_forward_calls',previous_q),('exact_forward_calls',previous_exact)]:
            deltas[key]={n:count-previous.get(n,0) for n,count in counters[key].items()}
            assert all(count>0 for count in deltas[key].values())
            previous.update(counters[key])
        results.append({'mode':label,'frames':len(actual),'exact_actions':r['exact_actions'],
                        'exact_raw_actions':r['exact_raw_actions'],'raw_MAE':r['raw_error']['mean'],'raw_max_abs':r['max_abs'],
                        'static_saving_bytes':manifest['estimated_weight_storage_saving_bytes'],
                        'int8_modules':manifest['int8_modules'],'exact_modules':manifest['exact_codec_modules'],
                        'model_mean_ms':r['model_seconds']['mean']*1000,'pipeline_mean_ms':r['pipeline_seconds']['mean']*1000,
                        'actual_calls_in_this_mode':deltas})
    warm=rows(c/'screening/bf16/warmup.jsonl')
    checks={'status':'PASS','logical_original_weights_checked':len(weights),'results':results,
            'bf16_warmup_measured_action_matches':sum(x['action_tokens']==y['action_tokens'] for x,y in zip(warm,base)),
            'scope':'Same resident model, seen development capture; full closed-loop and cold memory remain separate.',
            'reference':'Actual original BF16 weights with guarded Softmax and shared history, not previous INT8'}
    dump(c/'reports/screening_checks.json',checks)
    print({k:v for k,v in checks.items() if k!='results'})
    for r in results:print({k:v for k,v in r.items() if k!='actual_calls_in_this_mode'})


if __name__=='__main__':main()
