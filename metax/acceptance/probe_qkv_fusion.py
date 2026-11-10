"""Screen FlagScale's BF16 QKV packing using checkpoint weights and proxy inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from safetensors import safe_open
from KERVRuntimeOptimization.adaptive_linear_fusion import fuse_transformer_linears
from experiments.robot.ascend_qkv_ops import native_qkv


class Attention(torch.nn.Module):
    def forward(self,x):
        return self.q_proj(x),self.k_proj(x),self.v_proj(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    original=Attention();packed=Attention()
    for name in ['q_proj','k_proj','v_proj']:
        key='language_model.model.layers.7.self_attn.'+name+'.weight'
        with safe_open(a.checkpoint/index[key],framework='pt',device='cpu') as f:w=f.get_tensor(key)
        for model in [original,packed]:
            layer=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=w.dtype)
            layer.weight.data=w.clone();setattr(model,name,layer)
    original.npu();packed.npu()
    grouped_weights=[getattr(original,name).weight.t() for name in ['q_proj','k_proj','v_proj']]
    manifest=fuse_transformer_linears(packed,qkv_rows=list(range(1,513)),qkv_input_sizes=[4096])
    data=torch.load(a.inputs,weights_only=False,map_location='cpu')
    proxy=next(x for (name,shape),x in data['inputs'].items() if name=='gate_proj' and shape[-2]==279)
    rows=[]
    with torch.inference_mode():
        for m in [1,8,232,279,320]:
            x=proxy.repeat(1,2,1)[:,:m].contiguous().npu()
            ref=original(x);actual=packed(x)
            result=dict(rows=m,mismatches=sum(int((a!=b).sum()) for a,b in zip(ref,actual)),
                        relative_l2=max(float((a.float()-b.float()).norm()/a.float().norm()) for a,b in zip(ref,actual)),
                        samples_ms={'separate':[],'packed':[]})
            functions={'separate':original,'packed':packed}
            functions['scripted']=lambda value:native_qkv(value,*[getattr(original,n).weight for n in ['q_proj','k_proj','v_proj']])
            y=functions['scripted'](x)
            result['scripted_mismatches']=sum(int((a!=b).sum()) for a,b in zip(ref,y))
            result['samples_ms']['scripted']=[]
            try:
                def grouped(value):
                    flat=value.reshape(-1,value.shape[-1])
                    values=torch_npu.npu_grouped_matmul([flat]*3,grouped_weights,split_item=0,group_type=-1)
                    return tuple(y.reshape(*value.shape[:-1],y.shape[-1]) for y in values)
                y=grouped(x)
                result['grouped_mismatches']=sum(int((a!=b).sum()) for a,b in zip(ref,y))
                result['grouped_relative_l2']=max(float((a.float()-b.float()).norm()/a.float().norm()) for a,b in zip(ref,y))
                functions['grouped']=grouped;result['samples_ms']['grouped']=[]
            except Exception as error:result['grouped_error']=str(error)
            for repeat in range(6):
                for name in (list(functions) if repeat%2==0 else list(reversed(functions))):
                    fn=functions[name]
                    for _ in range(15):fn(x)
                    torch.npu.synchronize();start=time.perf_counter()
                    for _ in range(100):fn(x)
                    torch.npu.synchronize();result['samples_ms'][name].append(1000*(time.perf_counter()-start)/100)
            result['median_ms']={k:statistics.median(v) for k,v in result['samples_ms'].items()}
            rows.append(result);print(result,flush=True)
    a.output.write_text(json.dumps(dict(manifest=manifest,rows=rows,input_scope='MLP activation proxy,not actual attention activation capture'),indent=2)+'\n')


if __name__=='__main__':main()
