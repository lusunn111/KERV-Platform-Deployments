"""Native vector loads, BF16 rounding, protected copies and scalar tails."""
import os
import unittest
import torch
import torch_musa
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization import calibrated_w8a16_ops as reference
from KERVRuntimeOptimization import calibrated_native_decode as native


class NativeDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native.load_library(os.environ['KERV_NATIVE_TEST_BUILD'])

    def make(self,k=136,group=32,protection='partial',inverse=True):
        torch.manual_seed(389);ng=(k+group-1)//group
        ids={'partial':sorted(set([0,ng-1])),'all':list(range(ng)),'none':[]}[protection]
        q=CalibratedInt8Linear(k,19,group,len(ids))
        q.qweight.random_(-128,128);q.scales.uniform_(1e-6,.8)
        if inverse:q.inverse_channel_scales.uniform_(.5,1.5)
        q.protected_groups.copy_(torch.tensor(ids,dtype=torch.int32))
        q.protected_values.normal_()
        oracle=q.dequantize().to('musa');q=q.to('musa');reference.prepare(q)
        return q,oracle

    def test_vectorized_rounding_and_protected_groups(self):
        for protection in ['partial','all','none']:
            for inverse in [True,False]:
                q,w=self.make(protection=protection,inverse=inverse)
                for threads in (64,128,256):
                    for _ in range(4):self.assertTrue(torch.equal(native.decode(q,threads),w))

    def test_partial_groups_and_scalar_fallback(self):
        for k,group in [(137,32),(136,24),(128,7),(8,32)]:
            q,w=self.make(k=k,group=group)
            self.assertTrue(torch.equal(native.decode(q),w))

    def test_current_stream_and_repeated_workspace_reuse(self):
        q,w=self.make();pool=torch.empty_like(w)
        stream=torch.musa.Stream();stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            values=[]
            for i in range(32):
                native.decode_into(q,pool)
                values.append(pool.clone())
        torch.musa.current_stream().wait_stream(stream)
        for value in values:self.assertTrue(torch.equal(value,w))

    def test_invalid_output_and_launch_configuration(self):
        q,w=self.make()
        with self.assertRaises(ValueError):native.decode_into(q,w.float())
        with self.assertRaises(ValueError):native.decode_into(q,w,32)
        with self.assertRaises(ValueError):native.decode(q.cpu())


if __name__=='__main__':unittest.main()
