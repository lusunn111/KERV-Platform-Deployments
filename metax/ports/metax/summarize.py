"""Recompute standalone run metrics from retained raw events (no historical data)."""
import argparse
import json
from pathlib import Path
import statistics
import re


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def summarize(root):
    raw=root/'raw'
    steps=rows(raw/'steps.jsonl'); verifies=rows(raw/'verify.jsonl'); episodes=rows(raw/'episodes.jsonl')
    memory=rows(raw/'memory.jsonl'); memory=[r for r in memory if 'error' not in r]
    complete=[e for e in episodes if e['termination_reason']!='truncated_smoke']
    def peak(key):
        value=max((m[key] for m in memory),default=None)
        return dict(bytes=value,GB=value/1e9 if value is not None else None,GiB=value/2**30 if value is not None else None)
    def timing(key):
        seq=[float(s[key]) for s in steps]
        return dict(count=len(seq),sum_seconds=sum(seq),mean_seconds=statistics.mean(seq) if seq else None,
                    first_seconds=seq[0] if seq else None,
                    p95_seconds=sorted(seq)[max(0,__import__('math').ceil(.95*len(seq))-1)] if seq else None)
    vendor_text=(root/'vendor-memory.log').read_text() if (root/'vendor-memory.log').exists() else ''
    vendor_kb=[int(x) for x in re.findall(r'\n\s+vram used\s+: (\d+) KB',vendor_text)]
    vendor_process_mib=[int(x) for x in re.findall(r'\|\s+\d+\s+\d+\s+\S+\s+(\d+)\s+\|',vendor_text)]
    return dict(vendor_observed_device_peak_KB=max(vendor_kb,default=None),
        vendor_observed_process_peak_MiB=max(vendor_process_mib,default=None),
        vendor_observed_device_peak_bytes=max(vendor_kb)*1024 if vendor_kb else None,
        vendor_observed_process_peak_bytes=max(vendor_process_mib)*2**20 if vendor_process_mib else None,
        vendor_unit_conversion='vendor KB interpreted as 1024B, consistent with reported 67108864KB=64GiB total; MiB=2**20B; derived, not raw byte API',
        vendor_sampling_seconds=1.0 if vendor_text else None,
        run=str(root),exit_code=(root/'exit_code.txt').read_text().strip() if (root/'exit_code.txt').exists() else None,
        successful_episodes=sum(e['success'] for e in complete),complete_episodes=len(complete),smoke_episodes=len(episodes)-len(complete),
        task_success_rate=sum(e['success'] for e in complete)/len(complete) if complete else None,
        policy_steps=len(steps),verify_calls=len(verifies),zero_accept_calls=sum(v.get('committed_prefix_length')==0 for v in verifies),zero_raw_posterior_calls=sum(v['accepted_prefix_length']==0 for v in verifies),
        acceptance_metric='committed consecutive draft prefix after seven boundary clipping; excludes root/compensation/padding',
        mean_raw_posterior_prefix=statistics.mean([v['accepted_prefix_length'] for v in verifies]) if verifies else None,
        mean_accepted_prefix=statistics.mean([v['committed_prefix_length'] for v in verifies]) if verifies and all(v.get('committed_prefix_length') is not None for v in verifies) else None,
        mean_committed_prefix=statistics.mean([v['committed_prefix_length'] for v in verifies]) if verifies and all(v.get('committed_prefix_length') is not None for v in verifies) else None,
        tree_root_tokens=sum(v.get('root_length',0) for v in verifies),
        root_tokens=sum(v.get('root_length',0) for v in verifies if v.get('action_tokens_before',0)==0),
        bonus_tokens=sum(v.get('root_length',0) for v in verifies if v.get('action_tokens_before',0)>0),
        compensation_tokens=sum(v.get('compensation_length') or 0 for v in verifies),padding_tokens=sum(v.get('padding_length',0) for v in verifies),
        model=timing('model_seconds'),policy_pipeline=timing('pipeline_seconds'),simulation=timing('env_seconds'),
        closed_loop_seconds=sum(e['closed_loop_wall_seconds'] for e in episodes),
        allocated_peak=peak('allocated_peak_bytes'),reserved_peak=peak('reserved_peak_bytes'),
        note='Cold first action retained; no warmed performance claim. Smoke excluded from task success denominator. Vendor memory is separate raw sampling.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('runs',nargs='+',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();results=[summarize(r) for r in args.runs]
    rendered=json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(rendered)
    print(rendered)

if __name__=='__main__':main()
