"""Isolate native Softmax tail errors; no KERV, quantizer, or Triton imports."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
import torch_musa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',type=Path)
    p.add_argument('--repeats',type=int,default=2000)
    p.add_argument('--variants',default='native,out_nan,out_quarter,mask_1e4,mask_inf,pad1024')
    p.add_argument('--rows',type=int,default=8128)
    p.add_argument('--columns',type=int,default=537)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'worker_start.json').write_text(json.dumps({'pid':os.getpid()}))
    torch.manual_seed(117)
    if a.input:
        source=torch.load(a.input,weights_only=False)
        original=source['old'][0][1].float()
    else:
        original=torch.randn(a.rows,a.columns)*.5
        original[:,min(256,a.columns//2):]=-3.3895313892515355e38
        original[:,0]=1.
    results={}
    for variant in a.variants.split(','):
        cpu=original.clone()
        if variant=='mask_1e4':cpu[cpu < -1e30]=-10000.
        if variant=='mask_inf':cpu[cpu < -1e30]=-float('inf')
        n=cpu.shape[-1]
        if variant=='pad1024':cpu=torch.nn.functional.pad(cpu,(0,1024-n),value=-float('inf'))
        expected=cpu.softmax(-1)
        x=cpu.to('musa');masked=x < -1e30
        out=torch.empty_like(x)
        failures=[]
        start=time.monotonic()
        for i in range(a.repeats):
            if variant.startswith('out_'):
                out.fill_(float('nan') if variant=='out_nan' else .25)
                y=torch.ops.aten._softmax.out(x,-1,False,out=out)
            else:y=torch.softmax(x,-1)
            invalid=((y.sum(-1)-1).abs()>1e-5).any() | (~y.isfinite()).any()
            if variant!='mask_1e4':invalid=invalid | (y.masked_select(masked)!=0).any()
            if bool(invalid.cpu()):
                host=y.cpu()
                error=(host-expected).abs()
                bad_rows=((error>1e-5) | ~host.isfinite()).any(-1).nonzero().tolist()
                details=[]
                for row in bad_rows[:5]:
                    index=tuple(row);bad=(error[index]>1e-5) | ~host[index].isfinite()
                    cols=bad.nonzero().flatten()
                    details.append({'row':row,'columns':cols.tolist(),'input':cpu[index][cols].tolist(),
                        'output':host[index][cols].tolist(),'reference':expected[index][cols].tolist(),
                        'reference_max_probability':float(expected[index].max()),
                        'output_sum':float(host[index].sum())})
                failures.append({'iteration':i,'rows':details})
                if len(failures)<=2:torch.save({'input':cpu,'output':host,'reference':expected},
                      a.output/('%s_failure_%d.pt'%(variant,i)))
            if (i+1)%500==0:print(variant,i+1,'failures',len(failures),flush=True)
        assert torch.equal(x.cpu(),cpu),'Input tensor was mutated'
        result={'shape':list(cpu.shape),'iterations':a.repeats,'failures':failures,
                'seconds_diagnostic_only':time.monotonic()-start,'input_unchanged':True}
        results[variant]=result
        (a.output/'results.json').write_text(json.dumps(results,indent=2,allow_nan=True)+'\n')
        print('RESULT',variant,'failures',len(failures),flush=True)
    if a.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1]) as prof:
            for _ in range(3):torch.softmax(original.to('musa'),-1)
            torch.musa.synchronize()
        prof.export_chrome_trace(str(a.output/'trace.json'))


if __name__=='__main__':main()
