"""Screen rounded Ascend SwiGLU tiles without changing the rounding graph."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from experiments.robot.ascend_dynamic_quantization import NPUPrepackedWeightOnlyLinear
from experiments.robot.ascend_mlp_ops import rounded_swiglu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    payload=torch.load(a.inputs,weights_only=False,map_location='cpu')
    w=torch.cat([payload['weights']['gate_proj'],payload['weights']['up_proj']])
    linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);linear.weight.data=w
    gate=NPUPrepackedWeightOnlyLinear(linear,{'forward_calls':0},'gate_proj',32).npu()
    proxy=next(x for (name,shape),x in payload['inputs'].items() if name=='gate_proj' and shape[-2]==279)
    results=[]
    with torch.inference_mode():
        for rows in [8,232,279,320]:
            combined=gate(proxy.repeat(1,2,1)[:,:rows].contiguous().npu())
            width=combined.shape[-1]//2
            reference=torch.nn.functional.silu(combined[...,:width])*combined[...,width:]
            variants={block:dict(samples_ms=[]) for block in [512,1024,2048,4096,8192,16384]}
            for block,v in variants.items():
                y=rounded_swiglu(combined,block=block)
                v['mismatches']=int((y!=reference).sum())
            for repeat in range(6):
                for block in list(variants) if repeat%2==0 else list(reversed(variants)):
                    for _ in range(15):rounded_swiglu(combined,block=block)
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(100):rounded_swiglu(combined,block=block)
                    torch.npu.synchronize();variants[block]['samples_ms'].append(1000*(time.perf_counter()-start)/100)
            for v in variants.values():v['median_ms']=statistics.median(v['samples_ms'])
            results.append(dict(rows=rows,variants=variants));print(results[-1],flush=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
