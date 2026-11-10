"""Recompute per-episode warm statistics from frozen raw steps, never drop terminal actions."""
import argparse,json,math,statistics
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=[]
def stats(x):
 return dict(n=len(x),sum_s=sum(x),mean_ms=1000*statistics.mean(x),median_ms=1000*statistics.median(x),p95_ms=1000*sorted(x)[math.ceil(.95*len(x))-1])
for run in a.runs:
 groups=defaultdict(list)
 for line in (run/'raw/steps.jsonl').read_text().splitlines():
  row=json.loads(line);groups[row['episode_id']].append(row)
 episodes=[];warm=[];allrows=[]
 for key,seq in groups.items():
  seq.sort(key=lambda r:r['step_id']);assert seq[0]['step_id']==0
  warm.extend(seq[1:]);allrows.extend(seq)
  episodes.append(dict(episode=key,first_model_ms=1000*seq[0]['model_seconds'],
   including_first={k:stats([r[k] for r in seq]) for k in ['model_seconds','pipeline_seconds']},
   excluding_first={k:stats([r[k] for r in seq[1:]]) for k in ['model_seconds','pipeline_seconds']}))
 result.append(dict(run=str(run),episodes=episodes,including_first={k:stats([r[k] for r in allrows]) for k in ['model_seconds','pipeline_seconds']},
  excluding_first_weighted={k:stats([r[k] for r in warm]) for k in ['model_seconds','pipeline_seconds']},
  closed_loop_s=sum(json.loads(x)['closed_loop_wall_seconds'] for x in (run/'raw/episodes.jsonl').read_text().splitlines())))
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='episodes'} for r in result],indent=2))
