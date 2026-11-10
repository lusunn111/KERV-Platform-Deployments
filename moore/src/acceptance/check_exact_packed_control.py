"""Isolate the packed runtime with all decoded BF16 weights equal to originals."""
import argparse
from pathlib import Path
from acceptance.common import rows, dump
from acceptance.check_task_aware_pair import check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();c=a.campaign
    reference=c/'runs/control_fresh_bf16_task0_state0_s7'
    candidate=c/'runs/control_exact_packed8_task0_state0_s7'
    result=check(reference,candidate)
    loaded=next(x for x in rows(candidate/'precision_storage.jsonl') if x['phase']=='loaded')
    assert all(x['original_sha256']==x['decoded_sha256']==x['gpu_decoded_sha256'] for x in loaded['modules'])
    left,right=[rows(p/'steps.jsonl') for p in (reference,candidate)]
    comparisons={}
    for key in ('observation_hash','action_tokens','raw_action','executed_action'):
        first=next((i for i,(u,v) in enumerate(zip(left,right)) if u[key]!=v[key]),None)
        comparisons[key]={'first_difference_index':first,'equal_frames':sum(u[key]==v[key] for u,v in zip(left,right))}
    exact=len(left)==len(right) and all(x['equal_frames']==len(left) for x in comparisons.values())
    result.update(original_bf16_weight_hashes_checked=len(loaded['modules']),comparisons=comparisons,
                  exact_trajectory=exact,
                  control_scope='All-original BF16 decoded weights in packed runtime; redundant storage; not a deployment candidate')
    dump(c/'reports/exact_packed_control.json',result)
    print({k:v for k,v in result.items() if k not in ('reference','candidate')})
    if not exact or not result['both_succeeded']:raise SystemExit(2)


if __name__=='__main__':main()
