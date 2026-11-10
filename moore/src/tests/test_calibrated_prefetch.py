"""Stress readiness/reuse events with changing inputs and partial groups."""
import unittest
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization.calibrated_prefetch import OrderedWeightPrefetch

class PrefetchTests(unittest.TestCase):
    def make(self):
        torch.manual_seed(352);modules=[]
        for i,(k,n) in enumerate([(137,67),(65,33),(128,96),(37,99),(137,67)]):
            q=CalibratedInt8Linear(k,n,32,2)
            q.qweight.random_(-127,128);q.scales.uniform_(.0001,.01)
            q.inverse_channel_scales.uniform_(.5,1.5)
            q.protected_groups.copy_(torch.tensor([0,(k+31)//32-1],dtype=torch.int32))
            q.protected_values.normal_(std=.3)
            if i%2:q.bias=torch.randn(n,dtype=torch.bfloat16)
            modules.append((str(i),q.to('musa')))
        return OrderedWeightPrefetch(modules)

    def test_repeated_passes_do_not_overwrite_inflight_gemm_weights(self):
        p=self.make();weights=[q.dequantize() for _,q in p.named_modules]
        with torch.inference_mode():
            for repeat in range(32):
                p.begin()
                results=[]
                for i,(_,q) in enumerate(p.named_modules):
                    x=torch.randn(1+repeat%7,q.in_features,device='musa',dtype=torch.bfloat16)
                    y=p.linear(i,x)
                    expected=torch.nn.functional.linear(x,weights[i],q.bias)
                    results.append((y,expected))
                p.end()
                for y,expected in results:self.assertTrue(torch.equal(y,expected))
        self.assertEqual(p.stats['inference']['decodes'],160)
        self.assertEqual(p.stats['inference']['linears'],160)
        self.assertEqual(p.fallback_calls,0)
        self.assertEqual(p.workspace_bytes,2*max(w.numel() for w in weights)*2)

    def test_order_and_overlapping_forward_fail_closed(self):
        p=self.make();p.begin()
        with self.assertRaises(RuntimeError):p.begin()
        with self.assertRaises(RuntimeError):p.linear(1,torch.ones(1,65,device='musa',dtype=torch.bfloat16))
        with self.assertRaises(RuntimeError):p.end()
        torch.musa.synchronize()

if __name__=='__main__':unittest.main()
