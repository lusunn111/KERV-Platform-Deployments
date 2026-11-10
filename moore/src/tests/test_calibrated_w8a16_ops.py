"""Operator gates against identical decoded weights, independent of model SR."""
import unittest
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_w8a16_ops as ops


class W8A16MusaTests(unittest.TestCase):
    def make(self, k=137, n=67, protection='partial', inverse=True, bias=True):
        torch.manual_seed(349)
        ng=(k+31)//32
        ids={'partial':[0,ng-1], 'all':list(range(ng)), 'none':[]}[protection]
        ids=sorted(set(ids))
        q=CalibratedInt8Linear(k,n,32,len(ids))
        q.qweight.random_(-127,128)
        q.scales.uniform_(.0001,.01)
        if inverse:q.inverse_channel_scales.uniform_(.5,1.5)
        q.protected_groups.copy_(torch.tensor(ids,dtype=torch.int32))
        q.protected_values.normal_(std=.3)
        if bias:q.bias=torch.randn(n,dtype=torch.bfloat16)
        cpu=q.dequantize();q.to('musa');ops.prepare(q)
        return q,cpu.to('musa')

    def check(self,q,w,x):
        with torch.inference_mode():
            expected=torch.nn.functional.linear(x,w,q.bias)
            for kind in ['decode','decode_groups','decode_rows','gemm','gemv']:
                actual=ops.linear(q,x,kind=kind)
                self.assertEqual(actual.shape,expected.shape)
                self.assertTrue(bool(torch.isfinite(actual).all()),kind)
                if kind.startswith('decode'):
                    self.assertTrue(torch.equal(actual,expected),kind)
                else:
                    relative=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-20)
                    self.assertLess(float(relative),.002,kind)
        self.assertTrue(torch.equal(ops.decode(q),w))
        self.assertTrue(torch.equal(ops.decode_groups(q),w))

    def test_protected_tail_bias_inverse_and_noncontiguous_input(self):
        q,w=self.make(bias=False)
        x=torch.randn(2,7,274,device='musa',dtype=torch.bfloat16)[...,::2]
        self.assertFalse(x.is_contiguous());self.check(q,w,x)

    def test_bias_uses_native_epilogue_rounding(self):
        q,w=self.make()
        x=torch.randn(7,137,device='musa',dtype=torch.bfloat16)
        with torch.inference_mode():
            expected=torch.nn.functional.linear(x,w,q.bias)
            for kind in ['gemm','gemv']:
                self.assertTrue(torch.equal(ops.linear(q,x,kind=kind),expected))

    def test_all_protected_weights_original_bits(self):
        q,w=self.make(k=65,n=33,protection='all')
        self.check(q,w,torch.randn(1,65,device='musa',dtype=torch.bfloat16))

    def test_no_protection_identity_scale_and_zero_rows(self):
        q,w=self.make(protection='none',inverse=False,bias=False)
        self.check(q,w,torch.randn(49,137,device='musa',dtype=torch.bfloat16))
        for mode in ['gemm','gemv','decode','decode_groups']:
            out=ops.linear(q,torch.empty(0,137,device='musa',dtype=torch.bfloat16),kind=mode)
            self.assertEqual(tuple(out.shape),(0,67))

    def test_invalid_input_and_inference_only(self):
        q,_=self.make()
        with self.assertRaises(ValueError):ops.linear(q,torch.ones(1,137,device='musa'))
        with self.assertRaises(ValueError):ops.linear(q,torch.ones(1,138,device='musa',dtype=torch.bfloat16))
        with self.assertRaises(RuntimeError):ops.linear(q,torch.ones(1,137,device='musa',dtype=torch.bfloat16,requires_grad=True))
        q.protected_groups[0]=-1
        with self.assertRaises(ValueError):ops.prepare(q)


if __name__=='__main__':unittest.main()
