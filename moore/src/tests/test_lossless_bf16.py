import copy
import unittest
import torch
from torch import nn
from KERVRuntimeOptimization.lossless_bf16 import LosslessBF16Linear, compress_model

class LosslessTests(unittest.TestCase):
    def test_all_bits_and_odd_lengths(self):
        for n in (37, 65536, 65537):
            bits=torch.arange(n,dtype=torch.int32).to(torch.int16)
            linear=nn.Linear(n,1,bias=False).to(torch.bfloat16)
            with torch.no_grad():linear.weight.copy_(bits.view(torch.bfloat16).reshape(1,n))
            packed=LosslessBF16Linear(linear,128)
            self.assertTrue(torch.equal(packed.dequantize().view(torch.int16),linear.weight.view(torch.int16)))

    def test_selection_and_cpu_forward(self):
        from test_weight_only_int8 import WeightOnlyInt8Tests
        model=WeightOnlyInt8Tests.model(); reference=copy.deepcopy(model)
        report=compress_model(model)
        self.assertEqual(report['module_count'],3)
        self.assertIsInstance(model.lm_head,nn.Linear)
        self.assertIsInstance(model.ea_layer,nn.Linear)
        value=torch.randn(2,256,dtype=torch.bfloat16)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model.base_model.mlp(value),reference.base_model.mlp(value)))

    def test_musa_bit_exact_and_linear(self):
        from openvla.experiments.robot.device_runtime import get_device
        device=get_device()
        if device.type!='musa':self.skipTest('requires MUSA')
        bits=torch.arange(65536,dtype=torch.int32).to(torch.int16)
        linear=nn.Linear(256,256,bias=False).to(torch.bfloat16)
        with torch.no_grad():linear.weight.copy_(bits.view(torch.bfloat16).reshape(256,256))
        packed=LosslessBF16Linear(linear).to(device)
        self.assertTrue(torch.equal(packed.dequantize().cpu().view(torch.int16),linear.weight.view(torch.int16)))
        linear=nn.Linear(137,67).to(torch.bfloat16)
        packed=LosslessBF16Linear(linear).to(device);linear.to(device)
        value=torch.randn(2,7,137,device=device,dtype=torch.bfloat16)
        with torch.inference_mode():self.assertTrue(torch.equal(packed(value),linear(value)))

    def test_musa_transformer_and_cache(self):
        from openvla.experiments.robot.device_runtime import get_device
        from transformers import LlamaConfig
        from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM
        device=get_device()
        if device.type!='musa':self.skipTest('requires MUSA')
        cfg=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4)
        original=LlamaSpecForCausalLM(cfg,attn_implementation='eager').to(torch.bfloat16).eval()
        packed=copy.deepcopy(original);report=compress_model(packed)
        self.assertEqual(report['module_count'],6)
        self.assertIsInstance(packed.base_model.layers[0].self_attn.q_proj,nn.Linear)
        original.to(device);packed.to(device);ids=torch.tensor([[1,4,7,9]],device=device)
        with torch.inference_mode():
            a=original(input_ids=ids,use_cache=True);b=packed(input_ids=ids,use_cache=True)
            self.assertTrue(torch.equal(a.logits,b.logits))
            a=original(input_ids=ids[:,:1],past_key_values=a.past_key_values,use_cache=True)
            b=packed(input_ids=ids[:,:1],past_key_values=b.past_key_values,use_cache=True)
            self.assertTrue(torch.equal(a.logits,b.logits))

if __name__=='__main__':unittest.main()
