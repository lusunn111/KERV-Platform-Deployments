"""Packed signed-code roundtrip and actual BF16 decode precision gates."""
import unittest
import numpy as np
import torch
from torch import nn
from KERVRuntimeOptimization.packed_intn import PackedIntLinear,pack_codes,unpack_codes


class PackedCpuTests(unittest.TestCase):
    def test_all_codes_and_partial_pack_lengths(self):
        for bits in (10,12):
            v=np.arange(-(1<<(bits-1)),1<<(bits-1),dtype=np.int16)
            for count in (1,3,5,len(v)-1,len(v)):
                actual=unpack_codes(pack_codes(v[:count],bits),bits,count)
                self.assertTrue(np.array_equal(actual,v[:count]))

    def make(self,bits=10):
        torch.manual_seed(441);linear=nn.Linear(137,67,bias=True,dtype=torch.bfloat16)
        q=PackedIntLinear.from_linear(linear,bits,32,[0,4]);return linear,q

    def test_protected_tail_and_payload(self):
        for bits in (10,12):
            linear,q=self.make(bits);decoded=q.dequantize()
            self.assertTrue(torch.equal(decoded[:,:32],linear.weight[:,:32]))
            self.assertTrue(torch.equal(decoded[:,128:],linear.weight[:,128:]))
            restored=PackedIntLinear.from_payload(q.payload())
            self.assertTrue(torch.equal(restored.dequantize(),decoded))
            x=torch.randn(3,137,dtype=torch.bfloat16)
            with torch.inference_mode():self.assertTrue(torch.equal(restored(x),torch.nn.functional.linear(x,decoded,q.bias)))

    def test_higher_width_reduces_weight_rmse(self):
        torch.manual_seed(442);linear=nn.Linear(512,256,bias=False,dtype=torch.bfloat16)
        q10=PackedIntLinear.from_linear(linear,10);q12=PackedIntLinear.from_linear(linear,12)
        a=(q10.dequantize().float()-linear.weight.float()).square().sum()
        b=(q12.dequantize().float()-linear.weight.float()).square().sum()
        self.assertLess(float(b),float(a));self.assertLess(q10.storage_bytes,linear.weight.numel()*2)

    def test_invalid_codes_groups_and_width(self):
        with self.assertRaises(ValueError):pack_codes([512],10)
        with self.assertRaises(ValueError):PackedIntLinear(8,8,bits=8)
        linear=nn.Linear(32,16,dtype=torch.bfloat16)
        with self.assertRaises(ValueError):PackedIntLinear.from_linear(linear,protected_groups=[2])


class PackedMusaTests(unittest.TestCase):
    def test_cpu_gpu_decoded_bits_and_native_linear(self):
        import torch_musa
        torch.manual_seed(443)
        for bits in (10,12):
            for protection in ([],[0,4],[0,1,2,3,4]):
                linear=nn.Linear(137,67,bias=True,dtype=torch.bfloat16)
                q=PackedIntLinear.from_linear(linear,bits,32,protection);reference=q.dequantize().to('musa');q.to('musa')
                self.assertTrue(torch.equal(q.dequantize(),reference))
                for rows in (1,8,279):
                    x=torch.randn(rows,137,device='musa',dtype=torch.bfloat16)
                    with torch.inference_mode():self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,reference,q.bias)))


if __name__=='__main__':unittest.main()
