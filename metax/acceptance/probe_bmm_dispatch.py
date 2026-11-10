"""Measure full three-projection module dispatch on actual attention inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from safetensors import safe_open
from KERVRuntimeOptimization.adaptive_linear_fusion import _replace_group
from experiments.robot.ascend_qkv_ops import transfer_packed_qkv


class Attention(torch.nn.Module):
    def forward(self,x):return self.q_proj(x),self.k_proj(x),self.v_proj(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=torch.load(a.inputs,map_location='cpu',weights_only=False)
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    models={}
    for backend in ['separate','bmm','bmm_scripted']:
        model=Attention()
        for name in ['q_proj','k_proj','v_proj']:
            key=f'language_model.model.layers.7.self_attn.{name}.weight'
            with safe_open(a.checkpoint/index[key],framework='pt',device='cpu') as f:w=f.get_tensor(key)
            layer=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=w.dtype)
            layer.weight.data=w.clone();setattr(model,name,layer)
        if backend=='separate':model.npu()
        else:
            group=_replace_group(model,('q_proj','k_proj','v_proj'),'qkv_linear',list(range(257,321)))
            group.projection_backend=backend
            transfer_packed_qkv([group],torch.device('npu:0'))
        models[backend]=model
    rows=[]
    with torch.inference_mode():
        for m in [202,257,279,295,303,320]:
            host=data[7,279];x=host.repeat(1,2,1)[:,:m].contiguous().npu()
            ref=models['separate'](x)
            row=dict(rows=m,mismatches={},samples_ms={key:[] for key in models})
            for name in ['bmm','bmm_scripted']:
                actual=models[name](x);row['mismatches'][name]=sum(int((u!=v).sum()) for u,v in zip(ref,actual))
            for repeat in range(6):
                for name in (list(models) if repeat%2==0 else list(reversed(models))):
                    fn=models[name]
                    for _ in range(15):fn(x)
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(100):fn(x)
                    torch.npu.synchronize();row['samples_ms'][name].append(1000*(time.perf_counter()-start)/100)
            row['median_ms']={k:statistics.median(v) for k,v in row['samples_ms'].items()}
            rows.append(row);print(json.dumps(row),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2)+'\n')
    assert all(all(v==0 for v in r['mismatches'].values()) for r in rows)


if __name__=='__main__':main()
