"""Isolate scaling granularity on real weights/inputs; no quality claim."""
import argparse
import gc
import time

import numpy as np
import torch
import torch_npu

from diagnose_quantization import bench, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True,type=__import__('pathlib').Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    data=torch.load(a.inputs,weights_only=False,map_location='cpu');results=[]
    with torch.inference_mode():
        for name in ['gate_proj','down_proj']:
            weight=data['weights'][name];n,k=weight.shape
            shape,cpu_x=next((s,x) for (p,s),x in data['inputs'].items() if p==name and s[-2]==279)
            x=cpu_x.npu();w=weight.npu();reference=torch.nn.functional.linear(x,w)
            variants={'bf16':lambda:torch.nn.functional.linear(x,w)};buffers=[];errors={}
            for group in [0,32,64,128]:
                block=k if group==0 else group
                wf=weight.float().reshape(n,-1,block)
                scale=(wf.abs().amax(-1)/127).clamp_min(1e-8).bfloat16()
                q=(wf/scale.float().unsqueeze(-1)).round().clamp(-127,127).to(torch.int8).reshape(n,k).npu()
                s=scale.reshape(n).npu() if group==0 else scale.npu().t()
                def call(q=q,s=s,group=group):
                    return torch_npu.npu_weight_quant_batchmatmul(x.reshape(-1,k),q.t(),s,
                        antiquant_group_size=group).reshape(*shape[:-1],n)
                buffers.extend([q,s]);variants[f'group{group}']=call
                actual=call();errors[f'group{group}']=float((actual.float()-reference.float()).norm()/reference.float().norm())
            samples={mode:[] for mode in variants}
            for repeat in range(4):
                for mode in list(variants) if repeat%2==0 else list(reversed(variants)):
                    samples[mode].append(bench(variants[mode]))
            result={'projection':name,'input_shape':list(shape),'weight_shape':list(weight.shape),
                'relative_l2_vs_bf16':errors,'samples':samples,
                'median_device_ms':{mode:float(np.median([r['device_ms'] for r in rows])) for mode,rows in samples.items()},
                'scope':'scaling-granularity kernel ablation only; per-channel quality not evaluated in LIBERO'}
            results.append(result);write(a.output/'results.json',results);print(result,flush=True)
            del variants,buffers,q,s,w,x,actual,reference,wf,scale;gc.collect();torch.npu.empty_cache()


if __name__=='__main__':main()
