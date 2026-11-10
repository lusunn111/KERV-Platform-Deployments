"""Audit seven-action conservation and prefix padding from retained raw data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from summarize import rows
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
reports=[]
for root in a.runs:
    steps=rows(root/'raw/steps.jsonl');events=rows(root/'raw/verify.jsonl');grouped=defaultdict(list)
    for event in events:
        grouped[event['episode_id'],event['step_id']].append(event)
        assert not any(x<0 for x in event['selected_candidate_tokens'][1:1+event['accepted_prefix_length']])
    assert len(grouped)==len(steps)
    for step in steps:
        assert len(step['action_tokens'])==7
        selected=grouped[step['episode_id'],step['step_id']]
        assert sum(v['root_length']+v['committed_prefix_length']+v['compensation_length'] for v in selected)==7
        assert [v['verify_id'] for v in selected]==list(range(len(selected)))
        assert sum(v['action_tokens_before']==0 for v in selected)==1
        for event in selected:
            before=event['action_tokens_before'];length=event['committed_prefix_length']
            assert step['action_tokens'][before:before+1+length]==event['selected_candidate_tokens'][:1+length]

    reports.append(dict(run=str(root),steps=len(steps),verify_calls=len(events),
        seven_action_conservation=True,padding_in_accepted_prefix=False,
        raw_posterior_prefix_sum=sum(v['accepted_prefix_length'] for v in events),
        actual_committed_prefix_sum=sum(v['committed_prefix_length'] for v in events),
        tree_root_tokens=sum(v['root_length'] for v in events),
        initial_action_root_tokens=sum(v['action_tokens_before']==0 for v in events),
        committed_verifier_bonus_tokens=sum(v['action_tokens_before']>0 for v in events),
        compensation_tokens=sum(v['compensation_length'] for v in events)))
a.output.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
