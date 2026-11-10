"""Check cross-run workload consistency and summarize the quantization diagnosis."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args();base=a.campaign
    ab=read(base/'roundAB/replay.json');ba=read(base/'roundBA/replay.json')
    inter=read(base/'interleaved/results.json')
    assert ab['plan']['input_sha256']==ba['plan']['input_sha256']==inter['plan']['input_sha256']
    assert ab['plan']['physical_device']==ba['plan']['physical_device']==inter['plan']['physical_device']=='14'
    for label in ['bf16','int8']:
        reference=[r['action'] for r in ab['profiles'][label]['rows']]
        assert reference==[r['action'] for r in ba['profiles'][label]['rows']]
        assert reference==[r['action'] for r in inter['rows'][label]]
    assert all(r['profiles']['int8']['quantization']['module_count']==54 for r in [ab,ba])
    b=inter['summary']['bf16'];q=inter['summary']['int8']
    result={'scope':'Ascend chip14, fixed64 observations, warm64 + measured128 actions per precision per run; AB, BA, interleaved',
        'cross_run_same_inputs':True,'cross_run_same_actions_within_each_precision':True,
        'interleaved':inter['summary'],
        'interleaved_int8_latency_increase_percent':100*(q['mean_model_ms']/b['mean_model_ms']-1),
        'interleaved_bf16_over_int8_latency_ratio':b['mean_model_ms']/q['mean_model_ms'],
        'blocked_order_runs':{name:{label:r['profiles'][label]['model_mean_ms'] for label in ['bf16','int8']} for name,r in [('AB',ab),('BA',ba)]},
        'limitations':['shared host CPU and power conditions not isolated','each precision has its own generated history/tree',
            '64-observation timing diagnostic, not a new LIBERO success evaluation','dual model memory is not deployment memory',
            'device trace has profiler overhead; no exact millisecond decomposition of low-interference latency']}
    br=inter['rows']['bf16'];qr=inter['rows']['int8']
    matched=[(x,y) for x,y in zip(br,qr) if x['verify']==y['verify'] and x['verifier_shapes']==y['verifier_shapes']]
    result['matched_work_signatures']={'pairs':len(matched),'bf16_mean_ms':statistics.mean(x['model_ms'] for x,y in matched),
        'int8_mean_ms':statistics.mean(y['model_ms'] for x,y in matched),
        'note':'matching signature is not proof of bitwise identical internal tensors'}
    probes=read(base/'group_probe/results.json');result['group_probe']=[]
    for row in probes:
        times=row['median_device_ms']
        result['group_probe'].append({'projection':row['projection'],'shape':row['input_shape'],
            'median_ms':times,'group128_latency_increase_percent':100*(times['group128']/times['bf16']-1),
            'relative_l2_vs_bf16':row['relative_l2_vs_bf16']})
    result['kernel_traces']=[]
    for f in sorted((base/'kernel_trace/traces').rglob('kernel_details.csv')):
        with f.open() as handle:rows=list(csv.DictReader(handle))
        groups=defaultdict(list)
        for r in rows:groups[r['Name']].append(float(r['Duration(us)']))
        rel=f.relative_to(base/'kernel_trace/traces')
        result['kernel_traces'].append({'projection':rel.parts[0],'mode':rel.parts[1],
            'kernels':{name:{'count':len(values),'mean_us':statistics.mean(values)} for name,values in groups.items()}})
    result['model_traces']={}
    for label in ['bf16','int8']:
        folder=base/'interleaved/traces'/label
        with next(folder.rglob('kernel_details.csv')).open() as handle:kernels=list(csv.DictReader(handle))
        with next(folder.rglob('operator_details.csv')).open() as handle:ops=list(csv.DictReader(handle))
        counts=Counter(r['Name'] for r in ops)
        result['model_traces'][label]={'device_kernel_count':len(kernels),
            'quantized_matmul_kernels':sum('WeightQuantBatchMatmulV2' in r['Name'] for r in kernels),
            'host_operator_counts':{name:counts[name] for name in ['aten::t','aten::transpose','aten::reshape','aten::view',
                'aten::as_strided','npu::get_npu_format','aten::_to_copy','npu::npu_weight_quant_batchmatmul']},
            'warning':'Profiler diagnostics; kernel durations and nested host scopes must not be added as total wall time'}
    result['conclusions']=[
        'Current W8A16 linear kernel is slower at the real M279 shapes, confirmed with identical real inputs and original weights.',
        'Raw posterior acceptance and verification-cycle counts do not deteriorate in the middle18 comparison.',
        'Quantized wrapper adds host metadata/view operations; measured traces do not establish their exact share of the 8.8ms difference.',
        'The original14.4% closed-loop difference is not a stable isolated quantization penalty; interleaving measures about3.6% here.',
        'Pretransposing and changing scale granularity do not recover BF16 kernel speed in these probes; NZ layout probe failed environment initialization and remains unmeasured.'
    ]
    (base/'diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['kernel_traces','model_traces']},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
