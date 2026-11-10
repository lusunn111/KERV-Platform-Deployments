import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.calibrated_int8 import calibrate_linear
from KERVRuntimeOptimization.calibrated_packed import pack_calibrated


class WideCalibrationCpuTests(unittest.TestCase):
    def test_gptq_export_retains_actual_bf16_weights(self):
        torch.manual_seed(446);linear=nn.Linear(65,29,bias=True,dtype=torch.bfloat16)
        x=torch.randn(192,65,dtype=torch.bfloat16)
        for kind in ['fp32','pow2']:
            q=calibrate_linear(linear,x,32,.2,precision_bits=10,scale_kind=kind)
            self.assertEqual(q.qweight.dtype,torch.int16)
            packed=pack_calibrated(q)
            self.assertTrue(torch.equal(q.dequantize().view(torch.int16),packed.dequantize().view(torch.int16)))
            self.assertEqual(packed.calibration['method'],'gptq')

    def test_full_protection_and_invalid_scaling(self):
        linear=nn.Linear(37,19,bias=False,dtype=torch.bfloat16);x=torch.randn(128,37,dtype=torch.bfloat16)
        q=calibrate_linear(linear,x,16,1.,block_size=32,precision_bits=10,scale_kind='pow2')
        self.assertTrue(torch.equal(pack_calibrated(q).dequantize(),linear.weight))
        with self.assertRaises(ValueError):calibrate_linear(linear,x,precision_bits=10,alpha=.5)


class WideCalibrationMusaTests(unittest.TestCase):
    def test_musa_feedback_and_packed_decode(self):
        import torch_musa
        torch.manual_seed(447);linear=nn.Linear(137,67,bias=False,dtype=torch.bfloat16).to('musa')
        x=torch.randn(256,137,device='musa',dtype=torch.bfloat16)
        q=calibrate_linear(linear,x,32,.1,precision_bits=10,scale_kind='pow2',execution_device='musa')
        cpu=q.dequantize().to('musa');q.to('musa')
        self.assertTrue(torch.equal(q.dequantize(),cpu));q.cpu();packed=pack_calibrated(q).to('musa')
        self.assertTrue(torch.equal(packed.dequantize(),cpu))


if __name__=='__main__':unittest.main()
