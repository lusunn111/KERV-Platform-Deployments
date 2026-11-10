"""Exact reconstruction across tile boundaries, empty and dense exceptions."""
import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.inline_exact_bf16 import InlineExactBF16Linear


class InlineExactCpuTests(unittest.TestCase):
    def make_linear(self,k=137,n=67):
        torch.manual_seed(454);linear=nn.Linear(k,n,bias=True,dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.flatten()[::17]=1e-12
            linear.weight.flatten()[1::31]=0
            linear.weight.flatten()[2::79]=-1e-15
        return linear

    def test_sparse_restore_and_gemm(self):
        linear=self.make_linear();q=InlineExactBF16Linear(linear)
        self.assertTrue(torch.equal(q.dequantize().view(torch.int16),linear.weight.view(torch.int16)))
        self.assertEqual(q.original_sha256,q.decoded_sha256)
        self.assertEqual(q.exception_starts.dtype,torch.int16)
        self.assertEqual(int(q.exception_starts[-1]),q.exception_values.numel())
        self.assertGreater(q.exception_values.numel(),0)
        self.assertEqual(q.quantization['relative_weight_rmse'],0.)
        x=torch.randn(8,linear.in_features,dtype=torch.bfloat16)
        with torch.inference_mode():self.assertTrue(torch.equal(q(x),linear(x)))

    def test_zero_signed_boundary_and_wide_prefix(self):
        linear=nn.Linear(9,1,bias=False,dtype=torch.bfloat16)
        bits=torch.tensor([0,-32768,1,-32767,16256,14336,-18432,14335,32639],dtype=torch.int16)
        with torch.no_grad():linear.weight.copy_(bits.view(torch.bfloat16).reshape(1,-1))
        q=InlineExactBF16Linear(linear,8)
        self.assertTrue(torch.equal(q.dequantize().view(torch.int16),linear.weight.view(torch.int16)))
        zeros=nn.Linear(5,1,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():zeros.weight.zero_()
        q=InlineExactBF16Linear(zeros)
        self.assertEqual(q.exception_values.numel(),0)
        self.assertTrue(torch.equal(q.dequantize().view(torch.int16),zeros.weight.view(torch.int16)))
        large=nn.Linear(256,256,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():
            large.weight.fill_(1e-12);large.weight.flatten()[::128]=1.
        q=InlineExactBF16Linear(large)
        self.assertEqual(q.exception_starts.dtype,torch.int32)
        self.assertGreater(q.exception_values.numel(),32767)
        self.assertTrue(torch.equal(q.dequantize().view(torch.int16),large.weight.view(torch.int16)))


class InlineExactMusaTests(unittest.TestCase):
    def test_cpu_musa_bits_and_native_gemm(self):
        import torch_musa
        linear=InlineExactCpuTests().make_linear();q=InlineExactBF16Linear(linear)
        expected=linear.weight.detach().to('musa');bias=linear.bias.detach().to('musa');q.to('musa')
        with torch.inference_mode():
            self.assertTrue(torch.equal(q.dequantize().view(torch.int16),expected.view(torch.int16)))
            for m in (1,8,279):
                x=torch.randn(m,linear.in_features,device='musa',dtype=torch.bfloat16)
                self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,expected,bias)))

    def test_musa_unrolled_sparse_and_empty_paths(self):
        import torch_musa
        torch.manual_seed(458)
        for tiny in (False,True):
            linear=nn.Linear(128,128,bias=False,dtype=torch.bfloat16)
            with torch.no_grad():
                linear.weight.fill_(.125)
                if tiny:linear.weight.flatten()[[500,2000,8000]]=1e-12
            q=InlineExactBF16Linear(linear)
            self.assertLessEqual(q.maximum_exceptions_per_block,1)
            expected=linear.weight.detach().to('musa');q.to('musa')
            self.assertTrue(torch.equal(q.dequantize().view(torch.int16),expected.view(torch.int16)))


if __name__=='__main__':unittest.main()
