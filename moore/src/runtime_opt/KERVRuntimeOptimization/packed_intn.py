"""Packed signed W10/W12 storage, original-BF16 protected input groups.

Only weights are quantized. Native BF16 GEMM consumes a per-projection decode.
RTN calibration starts from the original BF16 weight, not a previous INT8 decode.
"""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pack_codes(values,bits):
    if bits not in (10,12):raise ValueError('Supported packed widths are10 and12')
    v=np.asarray(values,dtype=np.int16).reshape(-1)
    if np.any(v < -(1<<(bits-1))) or np.any(v >= (1<<(bits-1))):raise ValueError('Code outside signed range')
    codes=(v.astype(np.int32)+(1<<(bits-1))).astype(np.uint16)
    # Aligned independent packs avoid scatter races and ambiguous byte carries.
    block=4 if bits==10 else 2
    codes=np.pad(codes,(0,(-len(codes))%block)).reshape(-1,block).astype(np.uint64)
    packed=np.zeros(len(codes),dtype=np.uint64)
    for i in range(block):packed |= codes[:,i] << (i*bits)
    out=np.empty((len(codes),block*bits//8),dtype=np.uint8)
    for i in range(out.shape[1]):out[:,i]=(packed>>(8*i)).astype(np.uint8)
    return out.reshape(-1)


def unpack_codes(packed,bits,count):
    if bits not in (10,12):raise ValueError('Supported packed widths are10 and12')
    block=4 if bits==10 else 2;width=block*bits//8
    raw=np.asarray(packed,dtype=np.uint8).reshape(-1,width)
    if len(raw)*block<count:raise ValueError('Truncated packed weights')
    words=np.zeros(len(raw),dtype=np.uint64)
    for i in range(width):words |= raw[:,i].astype(np.uint64)<<(8*i)
    codes=np.empty((len(raw),block),dtype=np.int16)
    for i in range(block):codes[:,i]=((words>>(i*bits))&((1<<bits)-1)).astype(np.int16)-(1<<(bits-1))
    return codes.reshape(-1)[:count].copy()


class PackedIntLinear(nn.Module):
    def __init__(self,k,n,bits=10,group_size=32,protected_count=0):
        super().__init__()
        if bits not in (10,12) or min(k,n,group_size)<=0:raise ValueError('Invalid packed quantization shape/width')
        ng=math.ceil(k/group_size)
        if not 0<=protected_count<=ng:raise ValueError('Invalid protected count')
        self.in_features=int(k);self.out_features=int(n);self.bits=int(bits);self.group_size=int(group_size);self.forward_calls=0
        block=4 if bits==10 else 2
        self.register_buffer('packed',torch.zeros(math.ceil(n*k/block)*(block*bits//8),dtype=torch.uint8))
        self.register_buffer('scales',torch.ones(n,ng,dtype=torch.float32))
        self.register_buffer('protected_groups',torch.empty(protected_count,dtype=torch.int32))
        self.register_buffer('protected_values',torch.zeros(n,protected_count,group_size,dtype=torch.bfloat16))
        self.register_buffer('group_map',torch.full((ng,),-1,dtype=torch.int32),persistent=False)
        self.register_buffer('bias',None)
        self.calibration={}

    @property
    def storage_bytes(self):return sum(x.numel()*x.element_size() for x in self.buffers())

    def prepare(self):
        ids=self.protected_groups.detach().cpu().long();ng=len(self.group_map)
        if len(ids)!=len(torch.unique(ids)) or bool(((ids<0)|(ids>=ng)).any()):raise ValueError('Invalid protected groups')
        mapping=torch.full((ng,),-1,dtype=torch.int32);mapping[ids]=torch.arange(len(ids),dtype=torch.int32)
        self.group_map.copy_(mapping.to(self.group_map.device))
        return self

    def dequantize(self):
        if self.packed.device.type=='musa':
            from .packed_intn_kernels import decode
            return decode(self)
        if self.packed.device.type!='cpu':raise ValueError('Expected CPU or MUSA packed buffers')
        q=torch.from_numpy(unpack_codes(self.packed.numpy(),self.bits,self.in_features*self.out_features)).reshape(self.out_features,self.in_features)
        columns=torch.arange(self.in_features)
        w=(q.float()*self.scales[:,columns//self.group_size]).to(torch.bfloat16)
        for slot,g in enumerate(self.protected_groups.tolist()):
            start=g*self.group_size;end=min(start+self.group_size,self.in_features)
            w[:,start:end]=self.protected_values[:,slot,:end-start]
        return w

    def forward(self,x):
        if x.requires_grad:raise RuntimeError('Packed weight inference only')
        if x.dtype!=torch.bfloat16 or x.device!=self.packed.device or x.shape[-1]!=self.in_features:
            raise ValueError('Expected matching BF16 input on the weight device')
        result=F.linear(x,self.dequantize(),self.bias);self.forward_calls+=1;return result

    @classmethod
    def from_linear(cls,linear,bits=10,group_size=32,protected_groups=()):
        if linear.weight.device.type!='cpu' or linear.weight.dtype!=torch.bfloat16:raise ValueError('Prepare original CPU BF16 weights')
        w=linear.weight.detach();n,k=w.shape;ids=sorted(set(int(x) for x in protected_groups))
        q=cls(k,n,bits,group_size,len(ids));q.protected_groups.copy_(torch.tensor(ids,dtype=torch.int32));q.prepare()
        limit=(1<<(bits-1))-1;ng=math.ceil(k/group_size)
        groups=F.pad(w.float(),(0,ng*group_size-k)).reshape(n,ng,group_size)
        scales=groups.abs().amax(-1)/limit;scales=torch.where(scales==0,torch.ones_like(scales),scales)
        codes=(groups/scales[:,:,None]).round().clamp(-limit,limit).to(torch.int16).reshape(n,-1)[:,:k]
        q.packed.copy_(torch.from_numpy(pack_codes(codes.numpy(),bits)));q.scales.copy_(scales)
        for slot,g in enumerate(ids):
            start=g*group_size;end=min(start+group_size,k);q.protected_values[:,slot,:end-start].copy_(w[:,start:end])
        if linear.bias is not None:q.bias=linear.bias.detach().clone()
        q.calibration={'method':'original_BF16_RTN','bits':bits,'group_size':group_size,
                       'protected_groups':len(ids),'target_decode':'FP32 scale product rounded to BF16',
                       'original_weight_sha256':hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        return q.eval()

    def payload(self):
        return {'schema':'kerv_packed_intn_v1','shape':[self.out_features,self.in_features],
                'bits':self.bits,'group_size':self.group_size,'protected_count':self.protected_groups.numel(),
                'state_dict':{n:x.detach().cpu() for n,x in self.state_dict().items()},'calibration':self.calibration}

    @classmethod
    def from_payload(cls,p):
        if p.get('schema')!='kerv_packed_intn_v1':raise ValueError('Invalid packed payload schema')
        n,k=p['shape'];q=cls(k,n,p['bits'],p['group_size'],p['protected_count'])
        if 'bias' in p['state_dict']:q.bias=torch.empty_like(p['state_dict']['bias'])
        q.load_state_dict(p['state_dict'],strict=True);q.calibration=p['calibration'];q.prepare();return q.eval()
