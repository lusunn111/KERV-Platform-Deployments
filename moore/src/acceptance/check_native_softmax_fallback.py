"""Build and test the already-shipped native ATen MUSA Softmax kernel."""
import argparse
import json
from pathlib import Path
import torch
import torch_musa
from torch.utils.cpp_extension import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    build=a.output/'build';build.mkdir()
    module=load(name='softmax_native_fallback_probe',sources=['/workspace/src/acceptance/native_softmax_fallback.cpp'],
        extra_cflags=['-O2'],extra_ldflags=['-L/usr/local/lib/python3.10/dist-packages/torch_musa/lib',
        '-Wl,-rpath,/usr/local/lib/python3.10/dist-packages/torch_musa/lib','-lmusa_kernels'],
        build_directory=str(build),verbose=True)
    cpu=torch.load(a.input,weights_only=False)['old'][0][1].float();expected=cpu.softmax(-1)
    x=cpu.to('musa');failures=[];max_error=0
    for i in range(a.repeats):
        y=module.forward(x,-1)
        invalid=((y.sum(-1)-1).abs()>1e-5).any() | (~y.isfinite()).any()
        if bool(invalid.cpu()):failures.append(i)
        if i<3:max_error=max(max_error,float((y.cpu()-expected).abs().max()))
    (a.output/'results.json').write_text(json.dumps({'iterations':a.repeats,'failures':failures,
        'checked_max_abs_error':max_error,'kernel':'existing structured_softmax_cuda_out, musified ATen kernel'},indent=2)+'\n')
    assert not failures


if __name__=='__main__':main()
