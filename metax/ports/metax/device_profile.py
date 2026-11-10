"""Heavy real-input placement/CPU/kernel diagnostic, excluded from speed results."""
import json
from pathlib import Path
import sys
import torch
from ports.metax import benchmark

output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=False)
original=benchmark.get_vla_action
reports=[]

def observed(model,*args,**kwargs):
    model._metax_diag_calls=getattr(model,'_metax_diag_calls',0)+1
    name='optimized' if model._flagos_system_optimization_enabled else 'native'
    if model._metax_diag_calls not in {5,8}:return original(model,*args,**kwargs)
    step=model._metax_diag_calls-5
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True,profile_memory=True,with_stack=False) as profiler:
        with torch.profiler.record_function(f'MetaX::{name}::RealObservationPolicy'):
            result=original(model,*args,**kwargs)
        torch.cuda.synchronize()
    trace=output/f'{name}_step{step}_trace.json';profiler.export_chrome_trace(str(trace))
    averages=[]
    for item in profiler.key_averages():
        averages.append(dict(name=item.key,count=item.count,cpu_self_us=item.self_cpu_time_total,
            device_self_us=getattr(item,'self_device_time_total',None),device_total_us=getattr(item,'device_time_total',None)))
    events=json.loads(trace.read_text())['traceEvents']
    kernels=[event for event in events if event.get('cat')=='kernel']
    reports.append(dict(profile=name,step=step,periodic_kalman=step==3,real_observation=True,diagnostic_only=True,
        kernel_event_count=len(kernels),kernel_names=sorted({event['name'] for event in kernels}),
        parameter_devices=sorted({str(x.device) for x in model.parameters()}),
        top_cpu=sorted(averages,key=lambda r:r['cpu_self_us'],reverse=True)[:40],
        top_device=sorted(averages,key=lambda r:r['device_self_us'] or 0,reverse=True)[:40]))
    (output/'placement_and_costs.json').write_text(json.dumps(reports,indent=2)+'\n')
    return result

benchmark.get_vla_action=observed
sys.argv=['benchmark.py','--samples','4','--repeats','1','--profile-order','AB',
          '--input-dir','/workspace/outputs/metax_optimized_smoke_01/observations','--output',str(output/'diagnostic_replay')]
benchmark.main()
assert len(reports)==4
assert all(r['kernel_event_count']>0 for r in reports),'Device kernel capture unavailable; do not claim trace placement'
