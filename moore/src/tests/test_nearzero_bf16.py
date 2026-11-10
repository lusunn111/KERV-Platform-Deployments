"""Check bounded perturbation, exact retained values, and real MUSA decode."""
import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.nearzero_bf16 import NearZeroBF16Linear


class NearZeroCpuTests(unittest.TestCase):
    def make_linear(self,k=65,n=67):
        torch.manual_seed(451);linear=nn.Linear(k,n,bias=True,dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.flatten()[::17]=1e-8
            linear.weight.flatten()[1::31]=-1e-9
        return linear

    def test_error_bound_and_unchanged_nonzero_values(self):
        linear=self.make_linear();q=NearZeroBF16Linear(linear)
        w=linear.weight.detach().flatten();actual=q.dequantize().flatten();error=(w.float()-actual.float()).abs()
        groups=(len(w)+127)//128
        maximum=torch.nn.functional.pad(w.float().abs(),(0,groups*128-len(w))).reshape(groups,128).amax(1)
        bound=maximum.repeat_interleave(128)[:len(w)]/32768
        self.assertTrue(bool((error<=bound).all()))
        keep=actual!=0
        self.assertTrue(torch.equal(w[keep].view(torch.int16),actual[keep].view(torch.int16)))
        self.assertGreater(q.quantization['zeroed_nonzero_weights'],0)
        self.assertLess(q.quantization['relative_weight_rmse'],1e-4)
        self.assertLess(q.storage_bytes,w.numel()*2+linear.bias.numel()*2)
        x=torch.randn(8,linear.in_features,dtype=torch.bfloat16)
        with torch.inference_mode():self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,q.dequantize(),linear.bias)))

    def test_bf16_extremes_zero_sign_and_boundary(self):
        linear=nn.Linear(9,1,bias=False,dtype=torch.bfloat16)
        bits=torch.tensor([0, -32768, 1, -32767, 16256, 14336, -18432, 14335, 32639],dtype=torch.int16)
        with torch.no_grad():linear.weight.copy_(bits.view(torch.bfloat16).reshape(1,-1))
        q=NearZeroBF16Linear(linear,8)
        self.assertTrue(torch.isfinite(q.dequantize()).all())
        # Group0 has max1, so its floor is2**-15. Both signed floor codes represent signed zero.
        self.assertEqual(float(q.dequantize()[0,5]),0.)
        self.assertEqual(int(q.dequantize().view(torch.int16)[0,6]),-32768)
        self.assertEqual(float(q.dequantize()[0,7]),0.)
        zeros=nn.Linear(5,1,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():zeros.weight.zero_()
        self.assertTrue(torch.equal(NearZeroBF16Linear(zeros).dequantize(),zeros.weight))
        with torch.no_grad():zeros.weight[0,0]=float('inf')
        with self.assertRaises(ValueError):NearZeroBF16Linear(zeros)


class NearZeroMusaTests(unittest.TestCase):
    def test_musa_bits_and_native_gemm(self):
        import torch_musa
        linear=NearZeroCpuTests().make_linear();q=NearZeroBF16Linear(linear)
        expected=q.dequantize().to('musa');bias=linear.bias.detach().to('musa');q.to('musa')
        with torch.inference_mode():
            self.assertTrue(torch.equal(q.dequantize().view(torch.int16),expected.view(torch.int16)))
            for m in (1,8,279):
                x=torch.randn(m,linear.in_features,device='musa',dtype=torch.bfloat16)
                self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,expected,bias)))


if __name__=='__main__':unittest.main()
