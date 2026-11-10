"""AB/BA hot decode/Linear benchmark on original BF16 layer15 MLP weights."""
import argparse
from pathlib import Path
import time
import torch
import torch_musa
from torch import nn
from safetensors import safe_open
from acceptance.common import load,dump,distribution
from KERVRuntimeOptimization.lossless_bf16 import LosslessBF16Linear
from KERVRuntimeOptimization.hash_exact_bf16 import HashExactBF16Linear
import copy


def measure_pair(functions,repeats,batch):
    for fn in functions.values():
        for _ in range(4):fn()
    torch.musa.synchronize();samples={k:[] for k in functions}
    for iteration in range(repeats):
        order=list(functions) if iteration%2==0 else list(reversed(functions))
        for name in order:
            started=time.perf_counter()
            for _ in range(batch):value=functions[name]()
            torch.musa.synchronize()
            samples[name].append((time.perf_counter()-started)*1000/batch)
    return {k:{'samples_ms':v,**distribution(v)} for k,v in samples.items()}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=8);p.add_argument('--batch',type=int,default=20)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);torch.set_num_threads(8);torch.manual_seed(456)
    checkpoint=Path('/workspace/checkpoints/openvla-libero-goal');index=load(checkpoint/'model.safetensors.index.json')['weight_map']
    result={'scope':'Hot isolated real-weight operators; not model or task speedup',
            'timing':'Synchronized wall-clock batch boundaries; four warmups excluded; alternating AB/BA',
            'repeats':a.repeats,'batch':a.batch,'cases':[]}
    for projection in ('gate_proj','up_proj','down_proj'):
        key=f'language_model.model.layers.15.mlp.{projection}.weight'
        with safe_open(str(checkpoint/index[key]),framework='pt',device='cpu') as f:weight=f.get_tensor(key)
        linear=nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
        linear.weight=nn.Parameter(weight,requires_grad=False)
        old=LosslessBF16Linear(linear);new=HashExactBF16Linear(linear)
        expected=weight.to('musa');old.to('musa');new.to('musa')
        assert torch.equal(old.dequantize().view(torch.int16),expected.view(torch.int16))
        assert torch.equal(new.dequantize().view(torch.int16),expected.view(torch.int16))
        info={'projection':projection,'shape':list(weight.shape),'original_sha256':new.original_sha256,
              'old_exceptions':old.escape_values.numel(),'new_exceptions':new.exception_count,
              'hash_slots':[new.exception_keys1.numel(),new.exception_keys2.numel()],
              'old_storage_bytes':old.storage_bytes,'new_storage_bytes':new.storage_bytes,'results':[]}
        variants={}
        for block,warps in ((256,4),(512,4),(1024,4),(2048,4),(4096,4),(512,8),(1024,8),(2048,8)):
            q=copy.copy(new);q.decode_block_size=block;q.decode_num_warps=warps
            assert torch.equal(q.dequantize().view(torch.int16),expected.view(torch.int16))
            variants[f'hash_{block}_{warps}']=q
        modes={'old':old.dequantize,**{k:q.dequantize for k,q in variants.items()}}
        data=measure_pair(modes,a.repeats,a.batch)
        info['results'].append({'kind':'decode','rows':None,'times':data,'speedup_median':{k:data['old']['median']/v['median'] for k,v in data.items() if k!='old'}})
        for m in (1,8,279):
            x=torch.randn(m,linear.in_features,dtype=torch.bfloat16).to('musa')
            oracle=torch.nn.functional.linear(x,expected)
            assert torch.equal(old(x).view(torch.int16),oracle.view(torch.int16))
            assert torch.equal(new(x).view(torch.int16),oracle.view(torch.int16))
            data=measure_pair({'old':lambda:old(x),**{k:(lambda q=q:q(x)) for k,q in variants.items()}},a.repeats,a.batch)
            info['results'].append({'kind':'decode_native_gemm','rows':m,'times':data,'speedup_median':{k:data['old']['median']/v['median'] for k,v in data.items() if k!='old'}})
        result['cases'].append(info);dump(a.output/'results.json',result)
        print(projection,[(x['kind'],x['rows'],{k:round(v,4) for k,v in x['speedup_median'].items()}) for x in info['results']],flush=True)
        del old,new,linear,weight,expected,x,oracle,variants,modes,q
        torch.musa.empty_cache()


if __name__=='__main__':main()
