import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.packed_pow2 import PackedPow2Linear
from KERVRuntimeOptimization.packed_intn import PackedIntLinear


class Pow2CpuTests(unittest.TestCase):
    def test_payload_roundtrip_and_scale_storage(self):
        torch.manual_seed(444);linear=nn.Linear(137,67,bias=True,dtype=torch.bfloat16)
        q=PackedPow2Linear.from_linear(linear,10,32,[0,4]);w=q.dequantize()
        restored=PackedPow2Linear.from_payload(q.payload())
        self.assertTrue(torch.equal(w,restored.dequantize()))
        self.assertTrue(torch.equal(w[:,:32],linear.weight[:,:32]))
        self.assertTrue(torch.equal(w[:,128:],linear.weight[:,128:]))
        self.assertEqual(q.scales.dtype,torch.int8)
        other=PackedIntLinear.from_linear(linear,10,32,[0,4])
        self.assertEqual(other.storage_bytes-q.storage_bytes,q.scales.numel()*3)

    def test_zero_weights_and_nonfinite_rejection(self):
        linear=nn.Linear(65,19,bias=False,dtype=torch.bfloat16);linear.weight.data.zero_()
        q=PackedPow2Linear.from_linear(linear)
        self.assertTrue(torch.equal(q.dequantize(),linear.weight))
        linear.weight.data[0,0]=float('nan')
        with self.assertRaises(ValueError):PackedPow2Linear.from_linear(linear)


class Pow2MusaTests(unittest.TestCase):
    def test_exact_cpu_gpu_decode_and_native_gemm(self):
        import torch_musa
        torch.manual_seed(445)
        for protection in ([],[0,4],[0,1,2,3,4]):
            linear=nn.Linear(137,67,bias=True,dtype=torch.bfloat16)
            q=PackedPow2Linear.from_linear(linear,10,32,protection);reference=q.dequantize().to('musa');q.to('musa')
            self.assertTrue(torch.equal(q.dequantize(),reference))
            for rows in (1,8,279):
                x=torch.randn(rows,137,device='musa',dtype=torch.bfloat16)
                with torch.inference_mode():self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,reference,q.bias)))


if __name__=='__main__':unittest.main()
