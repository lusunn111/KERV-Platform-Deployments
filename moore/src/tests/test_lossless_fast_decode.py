"""Real MUSA bit preservation for two-byte extraction, tails and exceptions."""
import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.lossless_bf16 import LosslessBF16Linear


class FastDecodeMusaTests(unittest.TestCase):
    def test_bits_across_shapes_and_launch_configs(self):
        import torch_musa
        from KERVRuntimeOptimization.lossless_bf16_fast_kernels import decode
        torch.manual_seed(465)
        for n,k in ((1,1),(7,9),(67,137)):
            linear=nn.Linear(k,n,bias=False,dtype=torch.bfloat16)
            with torch.no_grad():
                linear.weight.flatten()[::17]=1e-12;linear.weight.flatten()[1::31]=-0.
            q=LosslessBF16Linear(linear).to('musa');expected=linear.weight.detach().to('musa')
            for block,warps in ((256,4),(512,4),(1024,4),(2048,4),(4096,4),(512,8),(1024,8),(2048,8)):
                self.assertTrue(torch.equal(decode(q,block,warps).view(torch.int16),expected.view(torch.int16)))
            with torch.inference_mode():
                for m in (1,8,279):
                    x=torch.randn(m,k,device='musa',dtype=torch.bfloat16)
                    actual=torch.nn.functional.linear(x,decode(q))
                    self.assertTrue(torch.equal(actual.view(torch.int16),torch.nn.functional.linear(x,expected).view(torch.int16)))


if __name__=='__main__':unittest.main()
