"""Check whether batch-preserving QKV dispatch matches separate native GEMMs.

Uses actual attention inputs, checkpoint BF16 weights, and a single chip. No
runtime replacement is installed. Reject numerically different or slow paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shape-sweep',action='store_true');a=p.parse_args()
    data=torch.load(a.inputs,map_location='cpu',weights_only=False)
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    rows=[]
    with torch.inference_mode():
        for layer in sorted(set(k[0] for k in data)):
            host_weights=[]
            for name in ['q_proj','k_proj','v_proj']:
                key=f'language_model.model.layers.{layer}.self_attn.{name}.weight'
                with safe_open(a.checkpoint/index[key],framework='pt',device='cpu') as f:host_weights.append(f.get_tensor(key))
            weights=torch.stack(host_weights).npu()
            if a.shape_sweep:
                originals=[v for (i,m),v in sorted(data.items()) if i==layer and m in [279,295]]
                assert len(originals)==2
                samples=[(m,sample,host.repeat(1,2,1)[:,:m].contiguous())
                         for m in range(257,321) for sample,host in enumerate(originals)]
            else:samples=[(m,0,host) for (i,m),host in sorted(data.items()) if i==layer]
            for m,sample,host_x in samples:
                x=host_x.npu();flat=x.reshape(-1,x.shape[-1]);shared=flat.unsqueeze(0).expand(3,-1,-1)
                def separate():return tuple(torch.nn.functional.linear(x,w) for w in weights.unbind(0))
                def bmm():return torch.bmm(shared,weights.transpose(1,2)).unbind(0)
                ref=separate();actual=bmm()
                row=dict(layer=layer,rows=m,sample=sample,mismatches=sum(int((u.reshape_as(v)!=v).sum()) for u,v in zip(ref,actual)),
                    relative_l2=max(float((u.reshape_as(v).float()-v.float()).norm()/u.float().norm()) for u,v in zip(ref,actual)),
                    samples_ms={'separate':[],'bmm':[]})
                for repeat in range(4 if not a.shape_sweep or (sample==0 and m in [257,279,303,320]) else 0):
                    for name,fn in ([('separate',separate),('bmm',bmm)] if repeat%2==0 else [('bmm',bmm),('separate',separate)]):
                        for _ in range(10):fn()
                        torch.npu.synchronize();start=time.perf_counter()
                        for _ in range(50):fn()
                        torch.npu.synchronize();row['samples_ms'][name].append(1000*(time.perf_counter()-start)/50)
                row['median_ms']={k:statistics.median(v) for k,v in row['samples_ms'].items() if v}
                if row['median_ms']:row['change_percent']=100*(row['median_ms']['bmm']/row['median_ms']['separate']-1)
                rows.append(row);print(json.dumps(row),flush=True)
            del weights
    report=dict(input_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),rows=rows,
        all_outputs_exact=all(r['mismatches']==0 for r in rows),
        summed_change_percent=100*(sum(r['median_ms']['bmm'] for r in rows if r['median_ms'])/sum(r['median_ms']['separate'] for r in rows if r['median_ms'])-1),
        scope='Diagnostic only; passing local exactness is not full-model validation')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('GATE',report['all_outputs_exact'],report['summed_change_percent'],flush=True)


if __name__=='__main__':main()
