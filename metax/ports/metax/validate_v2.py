"""Integration audit of retained round-two results, independent of B0 decisions."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from ports.metax.summarize import summarize

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/metax_v2_20261104'));a=p.parse_args();root=a.root
 assert subprocess.check_output(['git','rev-parse','metax-round1-bf179f6'],text=True).startswith('bf179f6')
 base=root/'pointwise_goal3_init01';comp=root/'compression_goal3_init01'
 A=summarize(base);B=summarize(comp)
 assert A['exit_code']==B['exit_code']=='0' and A['complete_episodes']==B['complete_episodes']==6
 sa=rows(base/'raw/steps.jsonl');sb=rows(comp/'raw/steps.jsonl');assert len(sa)==len(sb)>0
 fields=['episode_id','step_id','task_id','init_state_id','observation_hash','action_tokens','raw_action','executed_action','done','threshold']
 checks={k:all(x[k]==y[k] for x,y in zip(sa,sb)) for k in fields};assert all(checks.values()),checks
 assert A['successful_episodes']==B['successful_episodes']==5
 event_checks={}
 for run in [base,comp]:
  subprocess.run(['python3','ports/metax/audit_events.py',str(run),'--output',str(root/(run.name+'-audit.json'))],check=True,capture_output=True,text=True)
  events=rows(run/'raw/verify.jsonl');assert events and any(x['committed_prefix_length']==0 for x in events)
  assert all(len(x['action_tokens'])==7 for x in rows(run/'raw/steps.jsonl'))
  event_checks[run.name]=dict(calls=len(events),accepted=sum(x['committed_prefix_length'] for x in events),bonus=sum(x['action_tokens_before']>0 for x in events),compensation=sum(x['compensation_length'] for x in events))
 assert event_checks[base.name]==event_checks[comp.name]
 quant=json.loads((comp/'raw/quantization.json').read_text());assert quant['lossless'] and len(quant['modules'])==94 and quant['forward_calls']>0
 for run in [base,comp]:
  launch=json.loads((run/'launch.json').read_text());assert 'metax' in launch['torch'] and launch['logical_device']=='cuda:0'
  assert any('FlagScale/run.py' in str(x) for x in launch['command']) and launch['planning_horizon']==220
  hits=json.loads((run/'raw/v2.json').read_text());assert set(hits['features'])=={'rms','rope','vision_graph'}
  for key in ['rms_hits','verifier_rope_hits','draft_rope_hits','vision_replays']:assert hits['hits'][key]>0,(run,key)
 replay_checks=[]
 for name in ['selected8_AB','selected8_BA','selected220_AB','selected220_BA','compression8_AB','compression8_BA','compression220_AB','compression220_BA']:
  r=json.loads((root/name/'benchmark.json').read_text());assert r['warmup_sequences']==1 and r['planning_horizon']==220
  assert r['history_mode']=='frozen reference raw actions and seven token prefix'
  assert r['torch'].find('metax')>=0 and r['physical_device']=='0'
  profiles=r['profiles']
  if name.startswith('selected'):
   assert np.array_equal(profiles['native']['actions'],profiles['runtime']['actions'])
  else:
   assert r['actions_exact']
   assert all(v['pre_load_allocated_bytes']==0 and v['post_release_allocated_bytes']==0 and v['cleared_process_blas_workspaces'] for v in profiles.values())
   assert np.array_equal(profiles['v2']['actions'],profiles['v2_lossless_old']['actions'])
   assert profiles['v2']['allocated_peak_bytes']-profiles['v2_lossless_new']['allocated_peak_bytes']>=2_000_000_000
  replay_checks.append(dict(run=name,samples=r['samples'],repeats=r['repeats'],actions_exact=r['actions_exact'],source_sha256=hashlib.sha256((root/name/'source-state.json').read_bytes()).hexdigest()))
 saving=A['allocated_peak']['bytes']-B['allocated_peak']['bytes'];assert saving>=2_000_000_000
 report=dict(status='passed',round1_tag='metax-round1-bf179f6',base=A,compression=B,compression_trajectory_exact=checks,events=event_checks,replays=replay_checks,saving_bytes=saving,saving_GB=saving/1e9,saving_GiB=saving/2**30,quality_scope='v2 5/6, trajectories differ from round1; compression must be exact relative to v2; six episodes cannot establish population quality',B0='unchanged independent unresolved upstream boundary; not a blocker')
 (root/'delivery-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in {'base','compression','replays'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
