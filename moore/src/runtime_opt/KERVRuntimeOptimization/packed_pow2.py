"""W10 block floating storage: signed codes and one int8 scale exponent/group."""
import math
import torch
from torch.nn import functional as F
from .packed_intn import PackedIntLinear,pack_codes,unpack_codes


class PackedPow2Linear(PackedIntLinear):
    def __init__(self,k,n,bits=10,group_size=32,protected_count=0):
        super().__init__(k,n,bits,group_size,protected_count)
        self.scales=torch.zeros_like(self.scales,dtype=torch.int8)

    @classmethod
    def from_linear(cls,linear,bits=10,group_size=32,protected_groups=()):
        import hashlib
        if linear.weight.device.type!='cpu' or linear.weight.dtype!=torch.bfloat16:raise ValueError('Expected original CPU BF16 weights')
        w=linear.weight.detach();n,k=w.shape;ids=sorted(set(int(x) for x in protected_groups))
        q=cls(k,n,bits,group_size,len(ids));q.protected_groups.copy_(torch.tensor(ids,dtype=torch.int32));q.prepare()
        ng=math.ceil(k/group_size);limit=(1<<(bits-1))-1
        values=F.pad(w.float(),(0,ng*group_size-k)).reshape(n,ng,group_size)
        maximum=values.abs().amax(-1)
        exp=torch.ceil(torch.log2((maximum/limit).clamp_min(torch.finfo(torch.float32).tiny)))
        exp=torch.where(maximum==0,torch.zeros_like(exp),exp)
        if not bool(torch.isfinite(exp).all()) or bool(((exp < -126)|(exp > 127)).any()):
            raise ValueError('Scale exponent outside normal FP32 range')
        scale=torch.ldexp(torch.ones_like(exp),exp.to(torch.int32))
        codes=(values/scale[:,:,None]).round().clamp(-limit,limit).to(torch.int16).reshape(n,-1)[:,:k]
        q.packed.copy_(torch.from_numpy(pack_codes(codes.numpy(),bits)));q.scales.copy_(exp.to(torch.int8))
        for slot,g in enumerate(ids):
            start=g*group_size;end=min(start+group_size,k);q.protected_values[:,slot,:end-start].copy_(w[:,start:end])
        if linear.bias is not None:q.bias=linear.bias.detach().clone()
        q.calibration={'method':'original_BF16_RTN_power_of_two_scale','bits':bits,'group_size':group_size,
                       'scale_storage':'int8 exponent; scale=2**exponent','protected_groups':len(ids),
                       'original_weight_sha256':hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        return q.eval()

    def dequantize(self):
        if self.packed.device.type=='musa':
            from .packed_pow2_kernels import decode
            return decode(self)
        if self.packed.device.type!='cpu':raise ValueError('Expected CPU or MUSA buffers')
        q=torch.from_numpy(unpack_codes(self.packed.numpy(),self.bits,self.in_features*self.out_features)).reshape(self.out_features,self.in_features)
        scale=torch.ldexp(torch.ones_like(self.scales,dtype=torch.float32),self.scales.to(torch.int32))
        columns=torch.arange(self.in_features);w=(q.float()*scale[:,columns//self.group_size]).to(torch.bfloat16)
        for slot,g in enumerate(self.protected_groups.tolist()):
            start=g*self.group_size;end=min(start+self.group_size,self.in_features)
            w[:,start:end]=self.protected_values[:,slot,:end-start]
        return w

    def payload(self):
        p=super().payload();p['schema']='kerv_packed_pow2_v1';return p

    @classmethod
    def from_payload(cls,p):
        if p.get('schema')!='kerv_packed_pow2_v1':raise ValueError('Invalid power-of-two payload')
        n,k=p['shape'];q=cls(k,n,p['bits'],p['group_size'],p['protected_count'])
        if 'bias' in p['state_dict']:q.bias=torch.empty_like(p['state_dict']['bias'])
        q.load_state_dict(p['state_dict'],strict=True);q.calibration=p['calibration'];q.prepare();return q.eval()
