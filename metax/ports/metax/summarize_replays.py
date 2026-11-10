"""Recompute AB/BA warm replay time and per-round deployment memory comparisons."""
import argparse
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('rounds',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
reports=[json.loads((r/'benchmark.json').read_text()) for r in a.rounds]
comparison=reports[0].get('comparison','native_runtime')
assert all(r.get('comparison','native_runtime')==comparison for r in reports)
assert all(r['observation_sha256']==reports[0]['observation_sha256'] for r in reports)
reference,test={'native_runtime':('native','runtime'),'runtime_w8a16':('runtime','quantized'),'runtime_lossless':('runtime','lossless'),'runtime_v2':('runtime','v2'),'native_v2':('native','v2'),'native_runtime_v2':('native','v2'),'v2_compression':('v2','v2_lossless_new'),'numerical_fix':('native','v2')}[comparison]
assert all(r.get('history_mode')==reports[0].get('history_mode') and r.get('frozen_history_sha256')==reports[0].get('frozen_history_sha256') for r in reports)
result=dict(comparison=comparison,real_input_count=reports[0]['samples'],warmup_sequences_per_round=1,
    profiles={},rounds=[],all_actions_exact=all(r['actions_exact'] for r in reports),
    note='Same recorded observations, complete sequence warmup excluded. No simulator or B0. Quantization changes are reported separately.')
for name in reports[0]['profiles']:
    result['profiles'][name]={}
    for metric in ['model','pipeline']:
        seq=sum((r['profiles'][name][f'raw_{metric}_ms'] for r in reports),[])
        result['profiles'][name][metric]=dict(count=len(seq),mean_ms=statistics.mean(seq),
            median_ms=statistics.median(seq),p95_ms=sorted(seq)[math.ceil(.95*len(seq))-1],raw_ms=seq)
for r,path in zip(reports,a.rounds):
    item=dict(source=str(path),order=r['profile_order'],model_speedup=r['model_speedup'],
        pipeline_speedup=r['pipeline_speedup'],actions_exact=r['actions_exact'],max_action_difference=r['max_action_difference'])
    item['per_profile_memory']={name:{metric:profile.get(f'{metric}_peak_bytes') for metric in ['allocated','reserved']} for name,profile in r['profiles'].items()}
    for metric in ['allocated','reserved']:
        key=f'{metric}_peak_bytes';left=r['profiles'][reference].get(key);right=r['profiles'][test].get(key)
        saving=left-right if left is not None and right is not None else None
        item[metric]=dict(reference_peak_bytes=left,test_peak_bytes=right,saving_bytes=saving,
            saving_GB=saving/1e9 if saving is not None else None,saving_GiB=saving/2**30 if saving is not None else None,
            at_least_2GB=saving>=2_000_000_000 if saving is not None else None,
            at_least_2GiB=saving>=2**31 if saving is not None else None)
    result['rounds'].append(item)
for metric in ['model','pipeline']:
    ratio=result['profiles'][reference][metric]['mean_ms']/result['profiles'][test][metric]['mean_ms']
    result[f'{metric}_speedup']=ratio;result[f'{metric}_time_reduction_percent']=100*(1-1/ratio)
result['pairwise_ratios']={}
for left in result['profiles']:
    for right in result['profiles']:
        if left==right:continue
        result['pairwise_ratios'][left+'/'+right]={metric:result['profiles'][left][metric]['mean_ms']/result['profiles'][right][metric]['mean_ms'] for metric in ['model','pipeline']}
result['history_mode']=reports[0].get('history_mode')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='profiles'},indent=2))
