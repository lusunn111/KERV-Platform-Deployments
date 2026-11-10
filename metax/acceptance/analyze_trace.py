"""Summarize real CANN traces without inferring unsupported transfer directions."""
import argparse, collections, csv, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('profile_directory',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    file=next(a.profile_directory.rglob('trace_view.json'))
    data=json.loads(file.read_text());events=data['traceEvents'] if isinstance(data,dict) else data
    scopes=collections.defaultdict(list)
    for e in events:
        if e.get('ph')=='X' and e.get('name','').startswith('KERV::'):
            scopes[e['name']].append(float(e.get('dur',0)))
    kernels=next(a.profile_directory.rglob('kernel_details.csv'))
    with kernels.open() as f:device_kernels=list(csv.DictReader(f))
    copies=[x for x in events if x.get('name')=='MEMCPY_ASYNC' and x.get('ph')=='X']
    report={'source':str(file),'scope':'separate heavy diagnostic, includes loading and video; not formal timing',
        'device_ids':sorted(set(x['Device_id'] for x in device_kernels)),
        'device_kernel_count':len(device_kernels),
        'top_kernels':collections.Counter(x['Name'] for x in device_kernels).most_common(12),
        'cpu_ranges':{k:{'count':len(v),'total_host_duration_us':sum(v)} for k,v in scopes.items()},
        'sdma_event_count':len(copies),'sdma_total_observed_duration_us':sum(float(x['dur']) for x in copies),
        'sdma_direction_and_bytes':None,'sdma_warning':'MEMCPY_ASYNC includes unknown directions; do not equate all SDMA with H2D/D2H',
        'formal_bandwidth_status':'BLOCKED: bytes/direction/time correlations and D05 remain unconfirmed',
        'actual_cpu_kalman_scope_observed':'KERV::CPU_Kalman_HistoryEstimate' in scopes,
        'actual_tree_verify_scope_observed':'KERV::TreeVerify' in scopes}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['top_kernels','cpu_ranges']},ensure_ascii=False))

if __name__=='__main__':main()
