"""Attribute device intervals by CPU launch scope; never add CPU and GPU time."""
import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('traces',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def union(intervals):
 total=0;end=-float('inf')
 for start,stop in sorted(intervals):
  total+=max(0,stop-max(start,end));end=max(stop,end)
 return total
reports=[]
labels={'KERV::PromptInitialDraft','KERV::TreeVerify','KERV::PosteriorAccept','KERV::KVCommitNextDraft','KERV::KalmanCompletion'}
for path in a.traces:
 events=json.loads(path.read_text())['traceEvents']
 scopes=[e for e in events if e.get('name') in labels and e.get('cat')=='user_annotation' and e.get('dur') is not None]
 api={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime'}
 assigned=collections.defaultdict(list);names=collections.defaultdict(collections.Counter)
 for e in events:
  if e.get('cat') not in {'kernel','gpu_memcpy','gpu_memset'}:continue
  launch=api.get(e.get('args',{}).get('correlation'))
  if launch is None:continue
  matching=[s for s in scopes if s['tid']==launch['tid'] and s['ts']<=launch['ts']<=s['ts']+s['dur']]
  label=min(matching,key=lambda s:s['dur'])['name'] if matching else 'outside_KERV_scopes'
  assigned[label].append((e['ts'],e['ts']+e['dur']));names[label][e['cat']]+=1
 rows=[]
 for label,intervals in assigned.items():
  scope_intervals=[(s['ts'],s['ts']+s['dur']) for s in scopes if s['name']==label]
  rows.append(dict(label=label,cpu_scope_wall_ms=union(scope_intervals)/1000,device_active_union_ms=union(intervals)/1000,device_span_ms=(max(y for x,y in intervals)-min(x for x,y in intervals))/1000,events=dict(names[label])))
 reports.append(dict(trace=str(path),diagnostic_only=True,note='Device active union excludes gaps; spans can overlap other scopes. CPU and GPU times are not additive; repeated phases interleave.',phases=rows))
a.output.write_text(json.dumps(reports,indent=2)+'\n')
