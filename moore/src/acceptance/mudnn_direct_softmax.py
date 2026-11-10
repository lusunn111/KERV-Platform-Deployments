"""Call muDNN C Softmax API directly; no GPU PyTorch Softmax dispatch."""
import argparse
import ctypes as C
import json
from pathlib import Path
import torch
import torch_musa


class DirectSoftmax:
    def __init__(self,x,y):
        self.lib=C.CDLL('/usr/local/musa/lib/libmudnn.so')
        v=C.c_void_p;i=C.c_int;z=C.c_size_t
        signatures={
            'mudnnCreate':[C.POINTER(v)],'mudnnDestroy':[v],'mudnnSetStream':[v,v],
            'mudnnCreateTensorDescriptor':[C.POINTER(v)],'mudnnDestroyTensorDescriptor':[v],
            'mudnnSetTensorNdDescriptor':[v,i,i,C.POINTER(i),C.POINTER(i)],
            'mudnnCreateSoftmaxDescriptor':[C.POINTER(v)],'mudnnDestroySoftmaxDescriptor':[v],
            'mudnnSetSoftmaxDescriptor':[v,i,i,i],
            'mudnnGetSoftmaxForwardWorkspaceSize':[v,v,v,v,C.POINTER(z)],
            'mudnnSoftmaxForwardEx':[v,v,v,v,v,z,v,v]}
        for name,args in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=i
        self.handle=v();self.xdesc=v();self.ydesc=v();self.softmax=v()
        self.check('mudnnCreate',C.byref(self.handle))
        self.check('mudnnSetStream',self.handle,torch.musa.current_stream()._as_parameter_)
        for desc,tensor in [(self.xdesc,x),(self.ydesc,y)]:
            self.check('mudnnCreateTensorDescriptor',C.byref(desc))
            dims=(i*tensor.ndim)(*tensor.shape);strides=(i*tensor.ndim)(*tensor.stride())
            self.check('mudnnSetTensorNdDescriptor',desc,0,tensor.ndim,dims,strides)
        self.check('mudnnCreateSoftmaxDescriptor',C.byref(self.softmax))
        self.check('mudnnSetSoftmaxDescriptor',self.softmax,x.ndim-1,1,2)
        self.ws_size=z()
        self.check('mudnnGetSoftmaxForwardWorkspaceSize',self.handle,self.softmax,self.xdesc,
                   self.ydesc,C.byref(self.ws_size))
        self.workspace=torch.empty(self.ws_size.value,dtype=torch.uint8,device=x.device)
        self.x=x;self.y=y
    def check(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:raise RuntimeError(name+' status '+str(status))
    def run(self):
        self.check('mudnnSoftmaxForwardEx',self.handle,self.softmax,self.xdesc,self.x.data_ptr(),
            self.workspace.data_ptr() if self.ws_size.value else None,self.ws_size.value,
            self.ydesc,self.y.data_ptr())
        return self.y
    def close(self):
        torch.musa.synchronize()
        for name,value in [('mudnnDestroySoftmaxDescriptor',self.softmax),
                           ('mudnnDestroyTensorDescriptor',self.xdesc),
                           ('mudnnDestroyTensorDescriptor',self.ydesc),('mudnnDestroy',self.handle)]:
            self.check(name,value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3000)
    p.add_argument('--rows',type=int,default=8128)
    p.add_argument('--columns',type=int,default=537)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.input:cpu=torch.load(a.input,weights_only=False)['old'][0][1].float()
    else:
        torch.manual_seed(117)
        cpu=torch.randn(a.rows,a.columns)*.5
        cpu[:,min(256,a.columns//2):]=-10000.
        cpu[:,0]=1.
    expected=cpu.softmax(-1);x=cpu.to('musa');out=torch.full_like(x,float('nan'))
    runner=DirectSoftmax(x,out);failures=[]
    try:
        for step in range(a.repeats):
            y=runner.run()
            invalid=((y.sum(-1)-1).abs()>1e-5).any() | (~y.isfinite()).any()
            if bool(invalid.cpu()):
                host=y.cpu();error=(host-expected).abs()
                failures.append({'iteration':step,'max_abs_error':float(error.max()),
                                 'bad_rows':(error>1e-5).any(-1).nonzero().tolist()})
                if len(failures)<=2:torch.save({'input':cpu,'output':host,'reference':expected},
                                              a.output/('failure_%d.pt'%step))
            if (step+1)%500==0:print(step+1,'failures',len(failures),flush=True)
        assert torch.equal(x.cpu(),cpu)
        report={'api':'mudnnSoftmaxForwardEx called via ctypes, no GPU Torch Softmax',
                'shape':list(cpu.shape),'workspace_bytes':runner.ws_size.value,'repeats':a.repeats,
                'input_unchanged':True,'failures':failures}
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1]) as prof:
                for _ in range(3):runner.run()
                torch.musa.synchronize()
            prof.export_chrome_trace(str(a.output/'trace.json'))
    finally:runner.close()


if __name__=='__main__':main()
