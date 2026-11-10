"""Compile the candidate upstream SoftMax.cpp in an isolated namespace.

No installed library or operator registration is changed. This exercises the
actual patched dispatcher, including the native branch and muDNN branch.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch_musa
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3000)
    parser.add_argument('--upstream-tests', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = args.output / 'build'
    build.mkdir()
    source = Path('/workspace/third_party/torch_musa_upstream/torch_musa/csrc/aten/ops/SoftMax.cpp').read_text()
    source = source.replace('namespace musa {',
        'namespace softmax_patch_probe {\nusing namespace at::musa;', 1)
    source += '''
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", [](const at::Tensor& x, int64_t dim) {
    return at::softmax_patch_probe::Softmax(x, dim, false);
  });
  m.def("out", [](const at::Tensor& x, int64_t dim, at::Tensor out) {
    return at::softmax_patch_probe::SoftmaxOut(x, dim, false, out);
  });
}
'''
    candidate = build / 'candidate.cpp'
    candidate.write_text(source)
    libdir = '/usr/local/lib/python3.10/dist-packages/torch_musa/lib'
    module = load(name='softmax_upstream_patch_probe', sources=[str(candidate)],
        extra_include_paths=['/home/torch_musa', '/usr/local/musa/include',
            '/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible/include',
            '/usr/local/lib/python3.10/dist-packages/torch_musa/share/generated_cuda_compatible'],
        extra_cflags=['-O2', '-DTORCH_MUSA_USE_MUDNN_C_API'],
        extra_ldflags=[f'-L{libdir}', f'-Wl,-rpath,{libdir}', '-lmusa_kernels',
            '-lmusa_python', '-L/usr/local/musa/lib', '-lmudnn', '-lmusart'],
        build_directory=str(build), verbose=True)

    if args.upstream_tests:
        # Override only within this disposable worker. Autograd still uses the
        # framework's existing derivative and the original backward operator.
        import pytest
        library = torch.library.Library('aten', 'IMPL', 'PrivateUse1')
        def forward(x, dim, half_to_float):
            assert not half_to_float
            return module.forward(x, dim)
        def out(x, dim, half_to_float, *, out):
            assert not half_to_float
            return module.out(x, dim, out)
        library.impl('_softmax', forward)
        library.impl('_softmax.out', out)
        status = pytest.main([
            '/workspace/third_party/torch_musa_upstream/tests/unittest/operator/test_softmax.py',
            '-q', '-k', 'wide_row_boundaries or unaligned_masked_rows_repeated',
            '--junitxml='+str(args.output/'junit.xml')])
        (args.output/'pytest_result.json').write_text(json.dumps({'exit_code':int(status)})+'\n')
        raise SystemExit(status)

    cases = []
    for columns in [511, 512, 513, 516, 537, 538, 539, 540, 1024]:
        cpu = torch.sin(torch.arange(32*columns).reshape(32, columns).float())*.5
        cpu[:, columns//2:] = -10000.
        expected = cpu.softmax(-1)
        for noncontiguous in [False, True]:
            host = cpu.t().contiguous().t() if noncontiguous else cpu
            x = host.to('musa')
            with torch.musa.stream(torch.musa.Stream()):
                # Establish input dependency from the stream that made x.
                torch.musa.current_stream().wait_stream(torch.musa.default_stream())
                y = module.forward(x, -1)
                out = torch.empty_like(y)
                returned = module.out(x, 1, out)
                grad_cpu = torch.sin(torch.arange(cpu.numel()).reshape_as(cpu).float())
                grad = torch.ops.aten._softmax_backward_data(grad_cpu.to('musa'), y, -1, torch.float32)
                torch.musa.current_stream().synchronize()
            torch.testing.assert_close(y.cpu(), expected, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(out.cpu(), expected, atol=1e-6, rtol=1e-5)
            expected_grad = expected*(grad_cpu-(grad_cpu*expected).sum(-1, keepdim=True))
            torch.testing.assert_close(grad.cpu(), expected_grad, atol=1e-6, rtol=1e-5)
            assert returned.data_ptr() == out.data_ptr()
            cases.append({'columns':columns, 'noncontiguous':noncontiguous, 'passed':True})
    # A non-last dimension continues through muDNN.
    cpu = torch.randn(537, 32)
    torch.testing.assert_close(module.forward(cpu.to('musa'), 0).cpu(), cpu.softmax(0),
                               atol=1e-6, rtol=1e-5)
    cpu = torch.sin(torch.arange(8128*537).reshape(8128, 537).float())*.5
    cpu[:, 256:] = -10000.
    expected = cpu.softmax(-1)
    x = cpu.to('musa')
    expected_device = expected.to('musa')
    failures = []
    max_error = 0.
    for step in range(args.repeats):
        y = module.forward(x, -1)
        # Every lane is checked against CPU, rather than only checking row sums.
        bad = (~y.isfinite()).any() | ((y - expected_device).abs() > 1e-6).any()
        if bool(bad.cpu()): failures.append(step)
        if step < 3: max_error = max(max_error, float((y.cpu()-expected).abs().max()))
        if (step+1) % 500 == 0: print(step+1, 'failures', len(failures), flush=True)
    # Timing excludes CPU comparisons, allocations for reference, and compilation.
    timings = {}
    for name, fn in [('native_mudnn', lambda: torch.softmax(x, -1)),
                     ('patched', lambda: module.forward(x, -1))]:
        for _ in range(20): fn()
        torch.musa.synchronize()
        start = time.perf_counter()
        for _ in range(500): fn()
        torch.musa.synchronize()
        timings[name] = (time.perf_counter()-start)*1000/500
    result = {'cases':cases, 'non_last_dim_passed':True, 'iterations':args.repeats,
        'full_probability_failures':failures, 'checked_max_abs_error':max_error,
        'operator_ms':timings, 'scope':'isolated namespace, actual patched SoftMax.cpp'}
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    assert not failures


if __name__ == '__main__': main()
