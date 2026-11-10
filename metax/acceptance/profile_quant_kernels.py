"""Separate device trace and layout probes on captured real MLP inputs.

Does not change production quantization. One-time preparation is excluded from
steady kernel timing and reported separately. Failed optional layouts are kept.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu

from diagnose_quantization import bench, write
from experiments.robot.ascend_quantization import NPUWeightOnlyLinear


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    payload=torch.load(args.inputs,weights_only=False,map_location='cpu')
    rows=[]
    with torch.inference_mode():
        for name in ['gate_proj','down_proj']:
            weight=payload['weights'][name]
            original=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=torch.bfloat16)
            original.weight.data.copy_(weight)
            quant=NPUWeightOnlyLinear(original,{'forward_calls':0},name,128).npu()
            reference=original.npu()
            candidates=[(shape,x) for (projection,shape),x in payload['inputs'].items() if projection==name]
            shape,cpu_input=next(((s,x) for s,x in candidates if s[-2]==279),candidates[0])
            x=cpu_input.npu();ref=quant(x)
            functions={'bf16':lambda:reference(x),'int8_current':lambda:quant(x)}
            preparation={};failures={};tensors=[]
            # Materialize the currently passed transposed views once.
            start=time.perf_counter()
            qt=quant.qweight.t().contiguous();st=quant.antiquant_scale.t().contiguous()
            torch.npu.synchronize();preparation['int8_contiguous_ms']=1000*(time.perf_counter()-start)
            tensors.extend([qt,st])
            functions['int8_contiguous']=lambda:torch_npu.npu_weight_quant_batchmatmul(
                x.reshape(-1,quant.in_features),qt,st,antiquant_group_size=128).reshape(*shape[:-1],quant.out_features)
            try:
                start=time.perf_counter();nz= torch_npu.npu_format_cast(quant.qweight,29)
                torch.npu.synchronize();preparation['int8_nz_ms']=1000*(time.perf_counter()-start)
                tensors.append(nz)
                functions['int8_nz']=lambda:torch_npu.npu_weight_quant_batchmatmul(
                    x.reshape(-1,quant.in_features),nz.t(),quant.antiquant_scale.t(),
                    antiquant_group_size=128).reshape(*shape[:-1],quant.out_features)
            except Exception as exc:failures['int8_nz_prepare']=str(exc)
            measurements={}
            for mode,fn in functions.items():
                try:
                    actual=fn();torch.npu.synchronize()
                    measurements[mode]={'relative_l2_vs_current_int8':float((actual.float()-ref.float()).norm()/ref.float().norm()),
                        'finite':bool(actual.isfinite().all()),'samples':[]}
                except Exception as exc:failures[mode]=str(exc)
            live=[mode for mode in functions if mode not in failures]
            for round_id in range(4):
                for mode in live if round_id%2==0 else list(reversed(live)):
                    measurements[mode]['samples'].append(bench(functions[mode]))
            for mode,value in measurements.items():
                if value['samples']:
                    value['median_device_ms']=float(np.median([x['device_ms'] for x in value['samples']]))
            rows.append(dict(projection=name,input_shape=list(shape),weight_shape=list(weight.shape),
                preparation=preparation,measurements=measurements,failures=failures,
                current_weight_stride=list(quant.qweight.t().stride()),current_scale_stride=list(quant.antiquant_scale.t().stride())))
            write(args.output/'layout_results.json',rows)
            print(name,rows[-1],flush=True)
            for mode in live:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output/'traces'/name/mode)),
                    record_shapes=True,with_stack=False,profile_memory=False):
                    for _ in range(3):
                        with torch.profiler.record_function('QUANT_DIAG::'+mode):functions[mode]()
                    torch.npu.synchronize()
            del quant,reference,original,functions,tensors,qt,st,x,ref,actual
            if 'nz' in locals():del nz
            gc.collect();torch.npu.empty_cache()


if __name__=='__main__':main()
