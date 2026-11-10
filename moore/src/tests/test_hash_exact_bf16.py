"""Direct exception dictionary collision checks and exact CPU/MUSA weights."""
import unittest
import numpy as np
import torch
from torch import nn
from KERVRuntimeOptimization.hash_exact_bf16 import HashExactBF16Linear,_tables,_hash


class HashExactCpuTests(unittest.TestCase):
    def test_two_level_tables_cover_collisions(self):
        ids=np.random.default_rng(463).choice(45000000,1000,replace=False).astype(np.int64);bits=np.arange(len(ids),dtype=np.uint16)
        tables,config=_tables(ids,bits);k1,v1,k2,v2=tables;a1,s1,a2,s2=config
        first=_hash(ids,a1,s1);second=_hash(ids,a2,s2)
        actual=np.where(k1[first]==ids,v1[first],v2[second])
        self.assertTrue(np.array_equal(actual,bits))
        self.assertTrue(np.all((k1[first]==ids)|(k2[second]==ids)))
        self.assertGreater(np.count_nonzero(k2>=0),0)

    def make_linear(self):
        torch.manual_seed(462);linear=nn.Linear(137,67,bias=True,dtype=torch.bfloat16)
        with torch.no_grad():
            indices=torch.randperm(linear.weight.numel())
            linear.weight.flatten()[indices[:900]]=1e-12
            linear.weight.flatten()[indices[900:1100]]=-1e-15
            linear.weight.flatten()[indices[1100:1300]]=0.
        return linear

    def test_all_original_bits_and_empty_exceptions(self):
        linear=self.make_linear();q=HashExactBF16Linear(linear)
        self.assertTrue(torch.equal(q.dequantize().view(torch.int16),linear.weight.view(torch.int16)))
        self.assertEqual(q.original_sha256,q.decoded_sha256)
        linear=nn.Linear(9,1,bias=False,dtype=torch.bfloat16)
        bits=torch.tensor([0,-32768,1,-32767,16256,14336,-18432,14335,32639],dtype=torch.int16)
        with torch.no_grad():linear.weight.copy_(bits.view(torch.bfloat16).reshape(1,-1))
        self.assertTrue(torch.equal(HashExactBF16Linear(linear,8).dequantize().view(torch.int16),linear.weight.view(torch.int16)))
        with torch.no_grad():linear.weight.zero_()
        q=HashExactBF16Linear(linear)
        self.assertEqual(q.exception_count,0)
        self.assertTrue(torch.equal(q.dequantize(),linear.weight))


class HashExactMusaTests(unittest.TestCase):
    def test_gpu_shapes_and_launch_configs(self):
        import torch_musa
        from KERVRuntimeOptimization.hash_exact_bf16_kernels import decode
        linear=HashExactCpuTests().make_linear();q=HashExactBF16Linear(linear)
        # Force the secondary probe even when this small fixture has a perfect primary hash.
        if int((q.exception_keys2>=0).sum())==0:
            q.exception_keys1,q.exception_keys2=q.exception_keys2,q.exception_keys1
            q.exception_values1,q.exception_values2=q.exception_values2,q.exception_values1
            q.hash_a1,q.hash_a2=q.hash_a2,q.hash_a1
            q.hash_shift1,q.hash_shift2=q.hash_shift2,q.hash_shift1
        self.assertGreater(int((q.exception_keys2>=0).sum()),0)
        expected=linear.weight.detach().to('musa');bias=linear.bias.detach().to('musa');q.to('musa')
        with torch.inference_mode():
            for block,warps in ((256,4),(512,4),(1024,4),(2048,4),(1024,8)):
                self.assertTrue(torch.equal(decode(q,block,warps).view(torch.int16),expected.view(torch.int16)))
            for m in (1,8,279):
                x=torch.randn(m,linear.in_features,device='musa',dtype=torch.bfloat16)
                self.assertTrue(torch.equal(q(x),torch.nn.functional.linear(x,expected,bias)))

    def test_gpu_primary_and_no_exception_paths(self):
        import torch_musa
        for tiny in (False,True):
            linear=nn.Linear(128,128,bias=False,dtype=torch.bfloat16)
            with torch.no_grad():
                linear.weight.fill_(.125)
                if tiny:linear.weight.flatten()[[500,2000,8000]]=1e-12
            q=HashExactBF16Linear(linear)
            expected=linear.weight.detach().to('musa');q.to('musa')
            self.assertTrue(torch.equal(q.dequantize().view(torch.int16),expected.view(torch.int16)))


if __name__=='__main__':unittest.main()
