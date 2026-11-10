"""Minimal diagnostic for MUSA dot-layout lowering; not a deployment path."""
import torch
import torch_musa
import triton
import triton.language as tl

@triton.jit
def dot(A,B,O,K:tl.constexpr,M:tl.constexpr,N:tl.constexpr,BK:tl.constexpr):
    r=tl.arange(0,16);n=tl.arange(0,64);k=tl.arange(0,BK)
    acc=tl.zeros((16,64),tl.float32)
    for i in range(tl.cdiv(K,BK)):
        kk=i*BK+k
        a=tl.load(A+r[:,None]*K+kk[None,:],(r[:,None]<M)&(kk[None,:]<K),other=0)
        b=tl.load(B+n[:,None]*K+kk[None,:],(n[:,None]<N)&(kk[None,:]<K),other=0)
        acc=tl.dot(a,tl.trans(b),acc)
    tl.store(O+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))

def main():
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    from KERVRuntimeOptimization import calibrated_w8a16_ops as ops
    torch.manual_seed(1)
    for k in [64,128,4096]:
        for random in [False,True]:
            q=CalibratedInt8Linear(k,64,32,1).to('musa')
            q.protected_groups.fill_(1)
            q.qweight.random_(-127,128);q.scales.fill_(.01)
            q.protected_values.fill_(.25);ops.prepare(q)
            a=(torch.randn(1,k) if random else torch.ones(1,k)).to('musa',torch.bfloat16)
            b=q.dequantize();expected=torch.nn.functional.linear(a,b)
            o=torch.empty(1,64,device='musa',dtype=torch.bfloat16)
            dot[(1,)](a,b,o,k,1,64,64,num_warps=4)
            z=ops.linear(q,a)
            torch.musa.synchronize()
            print(k,random,'plain',float((o.float()-expected.float()).abs().max()),
                  'w8',float((z.float()-expected.float()).abs().max()),
                  'expect',expected[0,:5].tolist(),'got',z[0,:5].tolist(),flush=True)

if __name__=='__main__':main()
