"""Lossless BF16 byte-plane storage for MetaX, restored before vendor BF16 GEMM.

Store low bytes verbatim; dictionary-code the 16 common high bytes in nibbles;
restore all remaining BF16 words from explicit sparse exceptions. No rounding,
clipping, quantization, or non-BF16 arithmetic is applied to model weights.
"""
import numpy as np
import torch
from torch import nn
import triton
import triton.language as tl
from ports.metax.weight_only import Workspace


@triton.jit
def _restore(L,H,D,W,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    low=tl.load(L+i,i<N,other=0).to(tl.uint16)
    pair=tl.load(H+i//2,i<N,other=0).to(tl.uint32)
    code=(pair>>(4*(i%2)))&15
    high=tl.load(D+code).to(tl.uint16)
    tl.store(W+i,(high<<8)|low,i<N)


@triton.jit
def _exceptions(I,V,W,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    position=tl.load(I+i,i<N,other=0)
    value=tl.load(V+i,i<N,other=0)
    tl.store(W+position,value,i<N)


class LosslessStorage(nn.Module):
    def __init__(self,encodings):
        super().__init__()
        for key in ['low','high','dictionary','indices','values']:
            self.register_buffer(key,torch.cat([e[key] for e in encodings]))


def encode(weight):
    assert weight.device.type=='cpu' and weight.dtype==torch.bfloat16
    bits=weight.detach().contiguous().view(torch.int16).numpy().reshape(-1).view(np.uint16)
    high=(bits>>8).astype(np.uint8)
    counts=np.bincount(high,minlength=256)
    dictionary=np.argsort(-counts,kind='stable')[:16].astype(np.uint8)
    covered=np.zeros(256,dtype=bool);covered[dictionary]=True
    lookup=np.zeros(256,dtype=np.uint8);lookup[dictionary]=np.arange(16,dtype=np.uint8)
    codes=lookup[high]
    assert bits.size%2==0
    packed=codes[0::2]|(codes[1::2]<<4)
    indices=np.flatnonzero(~covered[high]).astype(np.int32)
    return dict(low=torch.from_numpy((bits&255).astype(np.uint8)),high=torch.from_numpy(packed),
        dictionary=torch.from_numpy(dictionary),indices=torch.from_numpy(indices),
        values=torch.from_numpy(bits[indices].copy().view(np.int16)))


class LosslessLinear(nn.Module):
    def __init__(self,original,storage,workspace,offsets,counts,manifest):
        super().__init__()
        assert original.bias is None
        self.out_features,self.in_features=original.weight.shape
        object.__setattr__(self,'storage',storage);object.__setattr__(self,'workspace',workspace)
        object.__setattr__(self,'manifest',manifest)
        self.offsets=offsets;self.counts=counts

    def forward(self,x):
        if x.dtype!=torch.bfloat16 or x.device.type!='cuda' or not getattr(torch.version,'maca',None):
            raise ValueError('MetaX BF16 input required')
        pointer_key=tuple(getattr(self.storage,key).data_ptr() for key in self.offsets)+(self.workspace.weight.data_ptr(),)
        if getattr(self,'_pointer_key',None)!=pointer_key:
            views={key:getattr(self.storage,key).narrow(0,self.offsets[key],self.counts[key]) for key in self.offsets}
            object.__setattr__(self,'_views',views)
            count=self.in_features*self.out_features
            weight=self.workspace.weight[:count].view(self.out_features,self.in_features)
            object.__setattr__(self,'_weight',weight)
            object.__setattr__(self,'_bits',weight.view(torch.int16))
            self._pointer_key=pointer_key
        v=self._views;count=self.in_features*self.out_features
        _restore[(triton.cdiv(count,1024),)](v['low'],v['high'],v['dictionary'],self._bits,count,1024)
        if self.counts['indices']:
            _exceptions[(triton.cdiv(self.counts['indices'],512),)](v['indices'],v['values'],self._bits,self.counts['indices'],512)
        self.manifest['forward_calls']+=1
        return nn.functional.linear(x,self._weight)


def install(model,selected):
    """selected: name, original Linear; construct CPU storage before device placement."""
    encodings=[];items=[]
    for name,linear in selected:
        if not isinstance(linear,nn.Linear) or linear.bias is not None:raise ValueError(name)
        encoding=encode(linear.weight);encodings.append(encoding)
        items.append(dict(name=name,shape=list(linear.weight.shape),bf16_bytes=linear.weight.numel()*2,
            compressed_bytes=sum(x.numel()*x.element_size() for x in encoding.values()),
            exception_count=encoding['indices'].numel(),weight_count=linear.weight.numel()))
    workspace=Workspace(max(linear.weight.numel() for _,linear in selected))
    model.add_module('_metax_lossless_workspace',workspace)
    manifest=dict(profile='metax_bf16_lossless_dictionary4',lossless=True,activation_dtype='bfloat16',
        restored_weight_dtype='bfloat16',storage='low8 + dictionary4 high bytes + sparse int32/int16 exceptions',
        forward_calls=0,modules=items,workspace_bytes=workspace.weight.numel()*2,
        execution='MACA Triton exact BF16 bit restoration + vendor BF16 GEMM')
    # Bound every transfer/allocation below 1 GiB; no multi-gigabyte memcpy dependency.
    for begin in range(0,len(selected),16):
        storage=LosslessStorage(encodings[begin:begin+16])
        model.add_module(f'_metax_lossless_storage_{begin//16}',storage)
        offsets={key:0 for key in encodings[0]}
        for (name,linear),encoding in zip(selected[begin:begin+16],encodings[begin:begin+16]):
            counts={key:value.numel() for key,value in encoding.items()}
            replacement=LosslessLinear(linear,storage,workspace,offsets.copy(),counts,manifest)
            parent,child=name.rsplit('.',1);model.get_submodule(parent)._modules[child]=replacement
            for key in offsets:offsets[key]+=counts[key]
    manifest['storage_chunks']=triton.cdiv(len(selected),16)
    manifest['storage_saving_minus_workspace_bytes']=sum(m['bf16_bytes']-m['compressed_bytes'] for m in items)-manifest['workspace_bytes']
    model._metax_quant_manifest=manifest
    print(f"[MetaX lossless BF16] {len(selected)} matrices; exact storage saving minus workspace {manifest['storage_saving_minus_workspace_bytes']} bytes",flush=True)
    return manifest


def apply_cpu_lossless(model):
    selected=[]
    for name,module in model.named_modules():
        if not name.startswith('base_model.language_model.model.layers.') or not isinstance(module,nn.Linear):continue
        layer=int(name.split('.')[4])
        if '.mlp.' in name or ('.self_attn.o_proj' in name and 7<=layer<15):selected.append((name,module))
    if len(selected)!=104:raise ValueError(f'Expected 96 MLP + 8 attention output projections, got {len(selected)}')
    return install(model,selected)
