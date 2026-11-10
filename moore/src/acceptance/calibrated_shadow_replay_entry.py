"""Diagnostic only: compare old/new weights and GEMMs on the SAME activation.

Instrumentation intentionally changes scheduling. These timings are invalid
for performance reporting. Returns the new-decoder path, without a new policy.
"""
import atexit
import os
from pathlib import Path
from types import MethodType
from acceptance.common import dump
from experiments.robot.libero.run_calibrated_libero import register
register()
from experiments.robot.libero import run_kerv_libero as evaluation
original_model=evaluation.get_model


def audited(cfg):
    import torch
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    from KERVRuntimeOptimization import calibrated_int8_kernels as old
    from KERVRuntimeOptimization import calibrated_w8a16_ops as new
    model=original_model(cfg);records=[];topk_records=[]
    native_topk=torch.topk
    def checked_topk(*args,**kwargs):
        first=native_topk(*args,**kwargs)
        if 'out' not in kwargs and args[0].device.type=='musa':
            repeat=native_topk(*args,**kwargs)
            topk_records.append({'shape':list(args[0].shape),
                'same_values':bool(torch.equal(first.values,repeat.values)),
                'same_indices':bool(torch.equal(first.indices,repeat.indices)),
                'adjacent_selected_ties':int((first.values[...,1:]==first.values[...,:-1]).sum())})
        return first
    torch.topk=checked_topk
    for name,module in model.named_modules():
        if not isinstance(module,CalibratedInt8Linear):continue
        new.prepare(module)
        def forward(self,x,_name=name):
            if x.requires_grad:raise RuntimeError('Inference only')
            w0=old.decode(self);w1=new.decode_rows(self,1024)
            y0=torch.nn.functional.linear(x,w0,self.bias)
            y1=torch.nn.functional.linear(x,w1,self.bias)
            yrepeat=torch.nn.functional.linear(x,w0,self.bias)
            difference=(y0.float()-y1.float())
            records.append({'module':_name,'input_shape':list(x.shape),
                'weights_bit_exact':bool(torch.equal(w0,w1)),
                'output_bit_exact':bool(torch.equal(y0,y1)),
                'native_repeat_bit_exact':bool(torch.equal(y0,yrepeat)),
                'max_abs':float(difference.abs().max()),
                'relative_l2':float(difference.norm()/y0.float().norm().clamp_min(1e-20))})
            self.forward_calls+=1
            return y1
        module.forward=MethodType(forward,module)
    def final():
        path=Path(os.environ['KERV_REPLAY_OUTPUT'])
        dump(path/'shadow_operator_checks.json',{'scope':'diagnostic instrumented; not performance evidence',
             'calls':records,'module_count':len(set(x['module'] for x in records)),
             'all_weights_exact':all(x['weights_bit_exact'] for x in records),
             'all_outputs_exact':all(x['output_bit_exact'] for x in records),
             'all_native_repeats_exact':all(x['native_repeat_bit_exact'] for x in records),
             'topk_checks':topk_records,
             'all_topk_repeats_exact':all(x['same_values'] and x['same_indices'] for x in topk_records)})
        torch.topk=native_topk
    atexit.register(final)
    return model


evaluation.get_model=audited
from acceptance.replay_entry import main
main()
