"""Two-slot decoded-weight pipeline for ordered, eager verifier projections.

Only weight decoding runs on the side stream. GEMMs retain the native BF16
implementation. Event dependencies guard both readiness and slot reuse.
The owner must call begin/end around every complete ordered verifier forward.
Concurrent or reordered forwards are rejected instead of using stale weights.
"""
from types import MethodType
import torch
import triton
from . import calibrated_w8a16_ops as ops


def decode_into(module, out):
    if (out.shape != module.qweight.shape or out.dtype != torch.bfloat16
            or out.device != module.qweight.device or not out.is_contiguous()):
        raise ValueError('Invalid decoded-weight workspace')
    ops._decode_rows[(triton.cdiv(module.in_features,1024),module.out_features)](
        module.qweight,module.scales,module.inverse_channel_scales,module._w8_group_map,
        module.protected_values,out,module.in_features,module.out_features,module.group_size,
        module.scales.shape[1],module.protected_groups.numel(),module._w8_identity_inverse,1024,
        num_warps=4,enable_fp_fusion=False)


class OrderedWeightPrefetch:
    def __init__(self, named_modules):
        self.named_modules=list(named_modules)
        if not self.named_modules:raise ValueError('Empty projection list')
        self.device=self.named_modules[0][1].qweight.device
        if self.device.type!='musa':raise ValueError('MUSA pipeline required')
        for _,module in self.named_modules:
            if module.qweight.device!=self.device:raise ValueError('Mixed projection devices')
            if not all(t.is_contiguous() for t in [module.qweight,module.scales,module.protected_values]):
                raise ValueError('Contiguous quantized buffers required')
            ops.prepare(module)
        capacity=max(m.qweight.numel() for _,m in self.named_modules)
        self.pool=torch.empty((2,capacity),device=self.device,dtype=torch.bfloat16)
        self.stream=torch.musa.Stream(device=self.device)
        self.stream.wait_stream(torch.musa.current_stream(self.device))
        self.pool.record_stream(self.stream)
        self.ready=[torch.musa.Event(),torch.musa.Event()]
        self.free=[torch.musa.Event(),torch.musa.Event()]
        self.free_valid=[False,False]
        self.views=[self.pool[i%2,:m.qweight.numel()].view(m.qweight.shape)
                    for i,(_,m) in enumerate(self.named_modules)]
        self.active=False;self.cursor=0;self.scope='inference'
        self.stats={};self.fallback_calls=0
        self.module_calls=[0]*len(self.named_modules)

    @property
    def workspace_bytes(self):return self.pool.numel()*self.pool.element_size()

    def _prefetch(self,index):
        if index>=len(self.named_modules):return
        slot=index%2;module=self.named_modules[index][1]
        with torch.musa.stream(self.stream):
            if self.free_valid[slot]:self.stream.wait_event(self.free[slot])
            decode_into(module,self.views[index])
            self.ready[slot].record(self.stream)
        self.stats[self.scope]['decodes']+=1

    def begin(self,scope='inference'):
        if self.active:raise RuntimeError('Overlapping verifier forwards are unsupported')
        self.active=True;self.cursor=0;self.scope=scope
        self.compute_stream=torch.musa.current_stream(self.device)
        self.stats.setdefault(scope,{'passes':0,'decodes':0,'linears':0})
        self.stats[scope]['passes']+=1
        self._prefetch(0)

    def linear(self,index,value):
        module=self.named_modules[index][1]
        if value.requires_grad:raise RuntimeError('Inference only')
        if value.dtype!=torch.bfloat16 or value.device!=self.device or value.shape[-1]!=module.in_features:
            raise ValueError('Expected matching colocated BF16 input')
        if not self.active:
            self.fallback_calls+=1
            return torch.nn.functional.linear(value,ops.decode_rows(module),module.bias)
        if index!=self.cursor:raise RuntimeError(f'Projection order changed: expected {self.cursor}, got {index}')
        compute=torch.musa.current_stream(self.device)
        if compute!=self.compute_stream:raise RuntimeError('Compute stream changed inside verifier forward')
        slot=index%2
        compute.wait_event(self.ready[slot])
        self._prefetch(index+1)
        result=torch.nn.functional.linear(value,self.views[index],module.bias)
        self.free[slot].record(compute);self.free_valid[slot]=True
        self.cursor+=1;self.stats[self.scope]['linears']+=1
        if self.scope=='inference':self.module_calls[index]+=1
        return result

    def end(self):
        if not self.active:raise RuntimeError('Verifier pipeline was not started')
        self.active=False
        if self.cursor!=len(self.named_modules):raise RuntimeError('Incomplete/reordered verifier projection traversal')

    def audit(self,digest):
        """Hash weights after actual side-stream decode and readiness wait."""
        proof=[];self.begin('audit')
        for i,(name,module) in enumerate(self.named_modules):
            x=torch.zeros((1,module.in_features),device=self.device,dtype=torch.bfloat16)
            self.linear(i,x)
            proof.append({'name':name,'decoded_sha256':digest(self.views[i])})
        self.end()
        return proof


def install(model):
    from .calibrated_int8 import CalibratedInt8Linear
    named=[(n,m) for n,m in model.named_modules() if isinstance(m,CalibratedInt8Linear)]
    pipeline=OrderedWeightPrefetch(named)
    verifier=model.get_submodule('base_model.language_model.model')
    for i,(_,module) in enumerate(named):
        def forward(self,value,_index=i):
            result=pipeline.linear(_index,value);self.forward_calls+=1;return result
        module.forward=MethodType(forward,module)
    verifier.register_forward_pre_hook(lambda *_:pipeline.begin())
    def completed(_module,_args,result):
        pipeline.end()
        return result
    verifier.register_forward_hook(completed)
    model.register_buffer('_calibrated_prefetch_pool',pipeline.pool,persistent=False)
    model._calibrated_prefetch=pipeline
    return pipeline
