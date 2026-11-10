"""Independently recompute action identity and actual Softmax fix coverage."""
import argparse
import json
from pathlib import Path
from acceptance.common import load,rows,dump,sha256,distribution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=Path('/root/KERV');c=a.campaign
    current=c/'guarded_full_replay_v1'
    historical=root/'outputs/quantized_pipeline/20261027T044701Z_prefetch/replays/same_policy_tree_v1/quant_reference'
    data=[rows(current/('measure_%d.jsonl'%i)) for i in range(3)]
    captured=rows(root/'outputs/quantized_pipeline/20261027T044701Z_prefetch/inputs/replay_capture/steps.jsonl')
    old=rows(historical/'measure_0.jsonl')
    assert len(captured)==141 and all(len(r)==141 for r in data)
    for run in data:
        for i,row in enumerate(run):
            assert row['frame']==i and row['observation_hash']==captured[i]['observation_hash']
    assert load(current/'parameter_hashes.json')==load(historical/'parameter_hashes.json')
    loaded=rows(current/'calibrated_kernel.jsonl')[0]
    assert len(loaded['modules'])==75 and all(m['bit_exact'] for m in loaded['modules'])
    assert {m['name']:m['decoded_sha256'] for m in loaded['modules']}=={
        m['name']:m['decoded_sha256'] for m in rows(historical/'calibrated_kernel.jsonl')[0]['modules']}
    softmax=rows(current/'safe_softmax.jsonl')
    assert softmax[0]['phase']=='loaded' and softmax[-1]['phase']=='final'
    proof=softmax[-1]
    assert len(proof['modules'])==32 and all(v>0 for v in proof['calls'].values())
    implementation=root/'src/runtime_opt/KERVRuntimeOptimization/musa_safe_softmax.py'
    assert proof['implementation_file_sha256']==sha256(implementation)
    assert proof['implementation_file_sha256']==load(current/'source_manifest.json')[str(implementation.relative_to(root))]
    assert not list(current.glob('*.mudmp'))
    report={'scope':'one captured trajectory, repeated observations/history; no closed-loop SR or paired performance claim',
        'measured_actions':423,'same_native_parameters':True,'same_75_decoded_quantized_weights':True,
        'softmax_covered_modules':32,'softmax_calls_including_warmup':sum(proof['calls'].values()),
        'repaired_rows_including_warmup':sum(proof['repaired_rows'].values()),
        'intra_exact_tokens_vs_round0':[sum(x['action_tokens']==y['action_tokens'] for x,y in zip(data[0],r)) for r in data],
        'intra_exact_raw_vs_round0':[sum(x['raw_action']==y['raw_action'] for x,y in zip(data[0],r)) for r in data],
        'historical_round0_exact_tokens':[sum(x['action_tokens']==y['action_tokens'] for x,y in zip(old,r)) for r in data],
        'historical_round0_exact_raw':[sum(x['raw_action']==y['raw_action'] for x,y in zip(old,r)) for r in data],
        'historical_round0_difference_frames':[[x['frame'] for x,y in zip(old,r) if x['action_tokens']!=y['action_tokens']] for r in data],
        'model_seconds':distribution(x['model_seconds'] for r in data for x in r),
        'pipeline_seconds':distribution(x['pipeline_seconds'] for r in data for x in r),
        'commands_exit_codes':load(current/'command.json')['exit_code']}
    dump(c/'guarded_replay_analysis.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
