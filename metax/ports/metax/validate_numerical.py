"""Retained-data integration audit of numerical repair, not just an action smoke test."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from ports.metax.summarize import summarize

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def compare_steps(a,b):
 assert len(a)==len(b)==696
 fields=['episode_id','step_id','task_id','init_state_id','observation_hash','image_hash','action_tokens','raw_action','executed_action','done','success','threshold','termination_reason']
 result={key:all(x[key]==y[key] for x,y in zip(a,b)) for key in fields}
 for key in ['raw_action','executed_action']:
  result[key+'_float64_bits']=bool(np.array_equal(np.asarray([x[key] for x in a],dtype=np.float64).view(np.uint64),np.asarray([x[key] for x in b],dtype=np.float64).view(np.uint64)))
 assert all(result.values()),result
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/metax_numerical_20261105'));a=p.parse_args();r=a.root
 paths={'native':Path('outputs/metax_native_goal3_init01_02'),'runtime':Path('outputs/metax_optimized_goal3_init01_02'),'old_v2':Path('outputs/metax_v2_20261104/pointwise_goal3_init01'),'fallback':r/'native_rms_goal3_init01','fixed':r/'fixed_goal3_init01','fixed_repeat':r/'fixed_goal3_init01_repeat'}
 stats={name:summarize(path) for name,path in paths.items()}
 assert all(x['exit_code']=='0' and x['complete_episodes']==6 for x in stats.values())
 reference=rows(paths['runtime']/'raw/steps.jsonl');step_checks={name:compare_steps(reference,rows(paths[name]/'raw/steps.jsonl')) for name in ['native','fallback','fixed','fixed_repeat']}
 ve=rows(paths['runtime']/'raw/verify.jsonl');verify_checks={}
 for name in ['fallback','fixed','fixed_repeat']:
  path=paths[name];test=rows(path/'raw/verify.jsonl');assert len(test)==len(ve)==1928
  fields=['episode_id','step_id','verify_id','accepted_prefix_length','committed_prefix_length','root_length','compensation_length','padding_length','action_tokens_before','selected_candidate_tokens','threshold','candidate_shape']
  result={key:all(x[key]==y[key] for x,y in zip(ve,test)) for key in fields};assert all(result.values()),(name,result);verify_checks[name]=result
  subprocess.run(['python3','ports/metax/audit_events.py',str(path),'--output',str(r/(name+'-event-audit.json'))],check=True,capture_output=True,text=True)
  episodes=rows(path/'raw/episodes.jsonl');assert len(episodes)==6 and sum(x['success'] for x in episodes)==5
  assert all(x['planning_horizon']==220 and x['step_limit']==220 for x in episodes)
  place=rows(path/'raw/placement.jsonl')
  for module in ['verifier_language','drafter']:
   assert any(x.get('module')==module and x.get('input_devices')==['cuda:0'] and x.get('event')=='actual_forward_pre_hook' for x in place),module
  launch=json.loads((path/'launch.json').read_text());assert 'metax' in launch['torch'] and launch['logical_device']=='cuda:0'
  assert any('FlagScale/run.py' in str(x) for x in launch['command'])
 audit=[];calls=0;shapes=set()
 for folder in ['all_layer_diag02','all_layer_long_diag03']:
  data=json.loads((r/folder/'rms-stages.json').read_text())['samples'];calls+=len(data);shapes.update(tuple(x['shape']) for x in data)
  assert len({x['layer'] for x in data})==65
  assert all(x['different']['native_mean_tail']==x['different']['native_inv_tail']==0 for x in data)
  assert all(x['different']['native_mean_inverse/rsqrt']==0 for x in data)
  assert json.loads((r/folder/'completed.json').read_text())['global_restore_exact']
  state=json.loads((r/folder/'state-checks.json').read_text())
  assert all(sum(x['different'])==0 if isinstance(x['different'],list) else x['different']==0 for x in state)
  audit.append(dict(run=folder,calls=len(data),layers=65,steps=sorted({x['step'] for x in data})))
 frozen=json.loads((r/'fixed220_AB/benchmark.json').read_text())['observation_sha256'];replays=[]
 for name in ['fixed220_AB','fixed220_BA']:
  d=json.loads((r/name/'benchmark.json').read_text());assert d['observation_sha256']==frozen
  assert d['frozen_history_sha256'] and d['planning_horizon']==220 and d['samples']==220 and d['warmup_sequences']==1
  assert d['actions_exact']
  profiles=d['profiles'];assert set(profiles)=={'native','runtime','old_v2','v2'}
  assert np.array_equal(profiles['native']['actions'],profiles['runtime']['actions'])
  assert np.array_equal(profiles['native']['actions'],profiles['v2']['actions'])
  assert not np.array_equal(profiles['native']['actions'],profiles['old_v2']['actions'])
  assert set(profiles['v2']['v2_features'])=={'rms_native_mean','rope','vision_graph'}
  for mode,profile in profiles.items():
   assert profile['pre_load_allocated_bytes']==profile['post_release_allocated_bytes']==0
   assert profile['restore_audit']['rms_restored'] and profile['restore_audit']['globals_restored']
   if mode in {'old_v2','v2'}:assert profile['restore_audit']['rms_modules']==65
   if mode in {'native','runtime','v2'}:
    warm=rows(r/name/(mode+'_runtime.jsonl'));assert len(warm)==220
    reference_tokens=[x['action_tokens'] for x in rows(Path(d['input_dir']).parent/'raw/steps.jsonl')]
    assert [x['action_token_ids'] for x in warm]==reference_tokens
  assert profiles['v2']['v2_hits']['rms_native_hits']>0 and profiles['v2']['v2_hits'].get('rms_native_fallbacks',0)==0
  replays.append(dict(run=name,actions_exact=True,token_reference_exact=True,model_speedup=d['model_speedup']))
 assert stats['fixed']['successful_episodes']==stats['fixed_repeat']['successful_episodes']==stats['fallback']['successful_episodes']==5
 result=dict(status='passed',step_checks=step_checks,verify_checks=verify_checks,diagnostics=audit,real_rms_calls=calls,distinct_shapes=sorted(shapes),replays=replays,runs=stats,scope='All sampled real RMS outputs bit-equal and six complete trajectories equal; not a universal bitwise claim for all inputs. No compression or B0 expansion.')
 (r/'numerical-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in {'runs','step_checks','verify_checks'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
