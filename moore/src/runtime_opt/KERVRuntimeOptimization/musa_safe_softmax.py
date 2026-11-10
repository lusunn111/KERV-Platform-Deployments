"""Opt-in inference Softmax for the MUSA eager Llama attention path.

The row reduction follows FlagGems' softmax_kernel_inner ONE_TILE_PER_CTA
algorithm (Apache-2.0, third_party/FlagGems/src/flag_gems/ops/softmax.py).
No process-wide torch functional or dispatcher registration is changed.
"""
from types import FunctionType, MethodType
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows(X, Y, N:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X+row*N+col, col<N, other=-float('inf')).to(tl.float32)
    numerator = tl.exp(x-tl.max(x, 0))
    denominator = tl.sum(numerator, 0)
    tl.store(Y+row*N+col, numerator/denominator, col<N)


def softmax(value, dim=None, _stacklevel=3, dtype=None):
    """Supported inference rows use FP32 reduction; other cases use PyTorch."""
    if (value.device.type!='musa' or dim not in (-1,value.ndim-1)
            or value.ndim==0 or value.shape[-1]==0 or value.shape[-1]>32768
            or value.requires_grad or value.dtype not in (torch.float32,torch.bfloat16,torch.float16)
            or dtype not in (None,torch.float32,torch.bfloat16,torch.float16)):
        return torch.nn.functional.softmax(value,dim=dim,_stacklevel=_stacklevel,dtype=dtype)
    # torch.softmax(dtype=...) casts before the reduction.
    x=value.to(dtype=dtype or value.dtype).contiguous()
    out=torch.empty(x.shape,device=x.device,dtype=dtype or x.dtype)
    n=x.shape[-1]
    if x.numel():
        _softmax_rows[(x.numel()//n,)](x,out,n,triton.next_power_of_2(n),
                                      num_warps=4,enable_fp_fusion=False)
    return out


@triton.jit
def _repair_rows(X, Y, COUNT, N, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+col,col<N,other=-float('inf')).to(tl.float32)
    y=tl.load(Y+row*N+col,col<N,other=0).to(tl.float32)
    total=tl.sum(y,0)
    # Ordered range comparisons also reject NaN/Inf sums and avoid the
    # unsupported llvm.is.fpclass intrinsic in the current MUSA compiler.
    bad_sum=~((total>=0.99999) & (total<=1.00001))
    bad_value=tl.sum((y<0).to(tl.int32),0)>0
    bad_mask=tl.sum(((col<N) & (x < -1e30) & (y!=0)).to(tl.int32),0)>0
    if bad_sum | bad_value | bad_mask:
        numerator=tl.exp(x-tl.max(x,0))
        probability=numerator/tl.sum(numerator,0)
        tl.store(Y+row*N+col,probability,col<N)
        tl.atomic_add(COUNT,1)


def guarded_softmax(value, dim=-1, _stacklevel=3, dtype=None, *, repaired_rows):
    """Keep native bits on valid rows; recompute only probability invariant failures.

    This is an inference workaround for the reproduced native tail corruption,
    not a proof that every conceivable backend error is detected.
    """
    output=torch.nn.functional.softmax(value,dim=dim,_stacklevel=_stacklevel,dtype=dtype)
    if (value.device.type!='musa' or value.requires_grad or value.ndim==0
            or dim not in (-1,value.ndim-1) or not value.is_contiguous()
            or output.dtype!=torch.float32 or value.shape[-1]==0 or value.shape[-1]>32768):
        return output
    n=value.shape[-1]
    if value.numel():
        _repair_rows[(value.numel()//n,)](value,output,repaired_rows,n,triton.next_power_of_2(n),
                                       num_warps=4,enable_fp_fusion=False)
    return output


class _Proxy:
    def __init__(self, original, **overrides):
        self.original=original;self.overrides=overrides
    def __getattr__(self,name):
        return self.overrides[name] if name in self.overrides else getattr(self.original,name)


def patch_llama_attention_softmax(model, mode='guarded'):
    """Give each eager Llama attention a private functional namespace.

    Reusing its original forward code preserves KV cache, RoPE, matmul,
    dropout, output projection and return conventions in Transformers 4.40.
    """
    from transformers.models.llama.modeling_llama import LlamaAttention
    if mode not in ('guarded','reference'):raise ValueError('Unknown Softmax mode')
    patched=[];compatible=0
    for name,module in model.named_modules():
        if type(module) is not LlamaAttention:continue
        compatible+=1
        if getattr(module,'_musa_safe_softmax_enabled',False):continue
        original=module.forward.__func__
        if original is not LlamaAttention.forward:
            raise RuntimeError('Unexpected pre-existing Llama attention forward patch: '+name)
        namespace=dict(original.__globals__)
        if 'nn' not in original.__code__.co_names:
            raise RuntimeError('Unsupported Transformers attention implementation')
        module._musa_safe_softmax_calls=0
        module.register_buffer('_musa_softmax_repaired_rows',torch.zeros((),dtype=torch.int32,
                               device=module.q_proj.weight.device),persistent=False)
        def counted(value,*args,_owner=module,**kwargs):
            _owner._musa_safe_softmax_calls+=1
            if mode=='reference':return softmax(value,*args,**kwargs)
            return guarded_softmax(value,*args,repaired_rows=_owner._musa_softmax_repaired_rows,**kwargs)
        namespace['nn']=_Proxy(namespace['nn'],functional=_Proxy(torch.nn.functional,softmax=counted))
        function=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
        function.__kwdefaults__=original.__kwdefaults__
        module.forward=MethodType(function,module)
        module._musa_safe_softmax_enabled=True
        patched.append(name)
    if compatible!=32:
        raise RuntimeError('Expected all 32 eager verifier attention modules, got '+str(compatible))
    return patched
