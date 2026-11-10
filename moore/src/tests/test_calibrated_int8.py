import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear, calibrate_linear


class CalibrationCpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(340)

    def test_full_protection_exact_odd_shape_and_bias(self):
        linear=nn.Linear(37,19).to(torch.bfloat16)
        x=torch.randn(96,37,dtype=torch.bfloat16)
        quant=calibrate_linear(linear,x,group_size=16,protected_fraction=1,block_size=32)
        self.assertTrue(torch.equal(quant.dequantize().view(torch.int16),linear.weight.view(torch.int16)))
        with torch.inference_mode():self.assertTrue(torch.equal(quant(x),linear(x)))

    def test_feedback_reduces_correlated_input_error(self):
        linear=nn.Linear(64,48,bias=False).to(torch.bfloat16)
        x=torch.randn(512,32).repeat_interleave(2,dim=1)+.01*torch.randn(512,64)
        train,test=x[:256].to(torch.bfloat16),x[256:].to(torch.bfloat16)
        rtn=calibrate_linear(linear,train,32,0,method='rtn')
        gptq=calibrate_linear(linear,train,32,0)
        with torch.inference_mode():
            target=linear(test).float()
            a=(rtn(test).float()-target).square().mean()
            b=(gptq(test).float()-target).square().mean()
        self.assertLess(float(b),float(a))
        self.assertEqual(gptq.calibration['factorization'],'CPU FP64 Cholesky; FP32 error feedback')

    def test_protection_original_bits_dead_channels_and_serialization(self):
        linear=nn.Linear(65,29,bias=False).to(torch.bfloat16)
        x=torch.randn(200,65,dtype=torch.bfloat16);x[:,32:64]=0
        quant=calibrate_linear(linear,x,32,.05,alpha=.5)
        for group in quant.protected_groups.tolist():
            begin=group*32;end=min(begin+32,65)
            self.assertTrue(torch.equal(quant.dequantize()[:,begin:end].view(torch.int16),linear.weight[:,begin:end].view(torch.int16)))
        self.assertIn(1,quant.protected_groups.tolist())
        restored=CalibratedInt8Linear.from_payload(quant.payload())
        self.assertTrue(torch.equal(restored.dequantize(),quant.dequantize()))
        self.assertEqual(restored.storage_bytes,quant.storage_bytes)

    def test_invalid_calibration_fails_before_mutation(self):
        linear=nn.Linear(32,16,bias=False).to(torch.bfloat16);before=linear.weight.clone()
        for x in [torch.empty(0,32),torch.full((8,32),float('nan')),torch.ones(4,33)]:
            with self.assertRaises(ValueError):calibrate_linear(linear,x)
        self.assertTrue(torch.equal(before,linear.weight))


class CalibrationMusaTests(unittest.TestCase):
    def test_cpu_gpu_decode_and_native_gemm(self):
        from openvla.experiments.robot.device_runtime import get_device
        device=get_device()
        if device.type!='musa':self.skipTest('MUSA-only execution audit')
        torch.manual_seed(341)
        linear=nn.Linear(137,67).to(torch.bfloat16)
        inputs=torch.randn(256,137,dtype=torch.bfloat16)
        quant=calibrate_linear(linear,inputs,32,.2,alpha=.5)
        cpu=quant.dequantize().to(device);quant.to(device)
        self.assertTrue(torch.equal(cpu,quant.dequantize()))
        x=torch.randn(2,7,137,device=device,dtype=torch.bfloat16)
        with torch.inference_mode():
            self.assertTrue(torch.equal(quant(x),torch.nn.functional.linear(x,cpu,quant.bias)))
        self.assertEqual(quant.forward_calls,1)

    def test_transformer_cache_with_calibrated_mlp(self):
        import copy
        from openvla.experiments.robot.device_runtime import get_device
        from transformers import LlamaConfig
        from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM
        device=get_device()
        if device.type!='musa':self.skipTest('MUSA-only execution audit')
        torch.manual_seed(342)
        cfg=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4)
        reference=LlamaSpecForCausalLM(cfg,attn_implementation='eager').to(torch.bfloat16).eval()
        candidate=copy.deepcopy(reference);mlp=candidate.base_model.layers[0].mlp
        for name in ['gate_proj','up_proj','down_proj']:
            linear=getattr(mlp,name);data=torch.randn(256,linear.in_features,dtype=torch.bfloat16)
            setattr(mlp,name,calibrate_linear(linear,data,32,.05))
        reference.to(device);candidate.to(device);ids=torch.tensor([[1,3,5,7]],device=device)
        with torch.inference_mode():
            a=reference(input_ids=ids,use_cache=True);b=candidate(input_ids=ids,use_cache=True)
            relative=(a.logits.float()-b.logits.float()).norm()/a.logits.float().norm()
            self.assertLess(float(relative),.02)
            c=candidate(input_ids=ids[:,:1],past_key_values=b.past_key_values,use_cache=True)
            self.assertTrue(torch.isfinite(c.logits).all())

if __name__=='__main__':unittest.main()
