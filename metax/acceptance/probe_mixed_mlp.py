"""Real-input screening of mixed W8A8/W8A16 MLP and rounded SwiGLU."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from experiments.robot.ascend_dynamic_quantization import NPUDynamicInt8Linear, NPUPrepackedWeightOnlyLinear
from experiments.robot.ascend_mlp_ops import rounded_swiglu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.inputs,weights_only=False,map_location='cpu')
    policy=torch.load(a.policy,weights_only=True,map_location='cpu')
    settings={x['name'].rsplit('.',1)[-1]:x for x in policy['modules'] if '.layers.7.' in x['name']}
    x=next(x for (name,shape),x in data['inputs'].items() if name=='gate_proj' and shape[-2]==279).npu()
    with torch.inference_mode():
        weights={n:w.npu() for n,w in data['weights'].items()}
        w=torch.cat([data['weights']['gate_proj'],data['weights']['up_proj']])
        linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);linear.weight.data=w
        gate=NPUDynamicInt8Linear(linear,settings['gate_proj']['smooth'],{'forward_calls':0},'gate_proj').npu()
        gate16=NPUPrepackedWeightOnlyLinear(linear,{'forward_calls':0},'gate_proj',32).npu()
        w=data['weights']['down_proj'];linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);linear.weight.data=w
        down=NPUPrepackedWeightOnlyLinear(linear,{'forward_calls':0},'down_proj',32).npu()
        def plain(value):
            width=value.shape[-1]//2
            return torch.nn.functional.silu(value[...,:width])*value[...,width:]
        def baseline():
            g=torch.nn.functional.linear(x,weights['gate_proj']);u=torch.nn.functional.linear(x,weights['up_proj'])
            return torch.nn.functional.linear(torch.nn.functional.silu(g)*u,weights['down_proj'])
        functions={'bf16':baseline,'mixed_native':lambda:down(plain(gate(x))),
                   'mixed_rounded_swiglu':lambda:down(rounded_swiglu(gate(x))),
                   'weight_only_native':lambda:down(plain(gate16(x))),
                   'weight_only_rounded':lambda:down(rounded_swiglu(gate16(x)))}
        combined=gate(x);reference=plain(combined);results={};failures={}
        try:
            actual=rounded_swiglu(combined)
            correctness=dict(equal_fraction=float((actual==reference).float().mean()),
                             mismatch_count=int((actual!=reference).sum()),exact=bool(torch.equal(actual,reference)),
                             relative_l2=float((actual.float()-reference.float()).norm()/reference.float().norm()))
        except Exception as e:
            correctness=dict(error=str(e));functions.pop('mixed_rounded_swiglu')
        ref=baseline()
        for name,fn in functions.items():
            try:
                y=fn();torch.npu.synchronize()
                results[name]=dict(relative_l2_vs_bf16=float((y.float()-ref.float()).norm()/ref.float().norm()),samples_ms=[])
            except Exception as e:failures[name]=str(e)
        for repeat in range(6):
            for name in list(results) if repeat%2==0 else list(reversed(results)):
                fn=functions[name]
                for _ in range(15):fn()
                torch.npu.synchronize();start=time.perf_counter()
                for _ in range(100):fn()
                torch.npu.synchronize();results[name]['samples_ms'].append(1000*(time.perf_counter()-start)/100)
        for value in results.values():value['median_ms']=statistics.median(value['samples_ms'])
        out=dict(results=results,rounded_swiglu=correctness,failures=failures)
        a.output.write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)


if __name__=='__main__':main()
