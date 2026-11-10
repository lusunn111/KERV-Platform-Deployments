"""Validate real W8 storage, edge groups, CPU preparation and MUSA execution."""
import copy
import unittest

import torch
from torch import nn
from KERVRuntimeOptimization.weight_only_int8 import Int8WeightOnlyLinear, quantize_linears


class WeightOnlyInt8Tests(unittest.TestCase):
    def test_partial_group_bias_and_error(self):
        torch.manual_seed(26)
        linear = nn.Linear(37, 19).to(torch.bfloat16).eval()
        quant = Int8WeightOnlyLinear.from_linear(linear, 16)
        value = torch.randn(2, 5, 37, dtype=torch.bfloat16)
        expected = linear(value).detach().float()
        actual = quant(value).float()
        self.assertLess(float((actual - expected).norm() / expected.norm()), .015)
        self.assertTrue(torch.equal(quant.bias, linear.bias))
        self.assertEqual(quant.qweight.dtype, torch.int8)
        self.assertEqual(tuple(quant.scales.shape), (19, 3))
        self.assertFalse(hasattr(quant, 'weight'))

    def test_zero_weights_are_finite(self):
        linear = nn.Linear(33, 7, bias=False).to(torch.bfloat16)
        with torch.no_grad(): linear.weight.zero_()
        quant = Int8WeightOnlyLinear.from_linear(linear, 16)
        out = quant(torch.ones(4, 33, dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(out.abs().max()), 0.)

    def test_state_dict_roundtrip_and_no_bf16_cache(self):
        quant = Int8WeightOnlyLinear.from_linear(nn.Linear(256, 128, bias=False).to(torch.bfloat16))
        restored = Int8WeightOnlyLinear(256, 128)
        restored.load_state_dict(quant.state_dict())
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(restored(x), quant(x), rtol=0, atol=0)
        self.assertEqual(set(dict(quant.named_buffers())), {'qweight', 'scales'})
        self.assertEqual(list(quant.parameters()), [])
        self.assertLess(sum(t.numel() * t.element_size() for t in quant.buffers()), 256 * 128 * 2)

    @staticmethod
    def model():
        model = nn.Module()
        model.base_model = nn.Module()
        model.base_model.mlp = nn.Sequential(nn.Linear(256, 512, bias=False), nn.Linear(512, 256, bias=False))
        model.base_model.self_attn = nn.Linear(256, 256, bias=False)
        model.ea_layer = nn.Linear(256, 256, bias=False)
        model.lm_head = nn.Linear(256, 256, bias=False)
        return model.to(torch.bfloat16).eval()

    def test_only_requested_groups_and_storage_accounting(self):
        model = self.model()
        before = copy.deepcopy(model.state_dict())
        report = quantize_linears(model, {'verify_mlp'}, 128)
        self.assertEqual(report['module_count'], 2)
        self.assertEqual(report['selected_groups'], ['verify_mlp'])
        self.assertEqual(report['estimated_weight_storage_saving_bytes'], report['weight_bytes_before'] - report['weight_bytes_after'])
        self.assertTrue(torch.equal(model.base_model.self_attn.weight, before['base_model.self_attn.weight']))
        self.assertTrue(torch.equal(model.ea_layer.weight, before['ea_layer.weight']))
        self.assertTrue(torch.equal(model.lm_head.weight, before['lm_head.weight']))

    def test_invalid_group_is_atomic(self):
        model = self.model()
        before = list(model.named_modules())
        with self.assertRaises(ValueError): quantize_linears(model, {'verify_mlp', 'typo'})
        self.assertEqual(before, list(model.named_modules()))

    def test_mixed_policy_retains_exact_bf16_weights(self):
        model = self.model()
        retained = model.base_model.mlp[1]
        before = retained.weight.detach().clone()
        name = 'base_model.mlp.0'
        policy = {'schema': 'kerv_mixed_w8_bf16_v1', 'group_size': 128,
                  'int8_modules': [name], 'weight_shapes': {name: list(model.base_model.mlp[0].weight.shape)}}
        report = quantize_linears(model, {'verify_mlp'}, 128, policy=policy)
        self.assertEqual(report['module_count'], 1)
        self.assertEqual(report['retained_bf16_module_count'], 1)
        self.assertIs(model.base_model.mlp[1], retained)
        self.assertTrue(torch.equal(model.base_model.mlp[1].weight, before))
        self.assertIsInstance(model.base_model.mlp[0], Int8WeightOnlyLinear)

    def test_invalid_mixed_policy_is_atomic(self):
        for selected, shapes in [(['missing'], {}), (['base_model.mlp.0']*2, {}),
                                  (['base_model.mlp.0'], {'base_model.mlp.0': [1, 1]})]:
            model = self.model(); before = list(model.named_modules())
            policy = {'schema': 'kerv_mixed_w8_bf16_v1', 'group_size': 128,
                      'int8_modules': selected, 'weight_shapes': shapes}
            with self.assertRaises(ValueError): quantize_linears(model, {'verify_mlp'}, 128, policy=policy)
            self.assertEqual(before, list(model.named_modules()))

    def test_mixed_policy_rejects_group_size_mismatch(self):
        model = self.model()
        policy = {'schema': 'kerv_mixed_w8_bf16_v1', 'group_size': 32,
                  'int8_modules': ['base_model.mlp.0'],
                  'weight_shapes': {'base_model.mlp.0': list(model.base_model.mlp[0].weight.shape)}}
        with self.assertRaises(ValueError): quantize_linears(model, {'verify_mlp'}, 128, policy=policy)

    def test_disabled_bridge_preserves_bf16(self):
        from openvla.experiments.robot.flagos_ops import enable_w8a16_quantization
        model = self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        report = enable_w8a16_quantization(model, False, backend='musa')
        self.assertTrue(report['fallback'])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_musa_dequant_and_linear(self):
        from openvla.experiments.robot.device_runtime import get_device
        device = get_device()
        if device.type != 'musa': self.skipTest('requires MUSA')
        torch.manual_seed(27)
        linear = nn.Linear(137, 67, bias=True).to(torch.bfloat16)
        quant = Int8WeightOnlyLinear.from_linear(linear, 128)
        expected_weight = quant.dequantize(torch.bfloat16).to(device)
        quant.to(device)
        actual_weight = quant.dequantize(torch.bfloat16)
        torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)
        x = torch.randn(2, 7, 137, device=device, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, expected_weight, quant.bias)
        torch.testing.assert_close(quant(x), expected, rtol=0, atol=0)
        self.assertEqual(quant.forward_calls, 1)

    def test_musa_transformer_forward_and_cache(self):
        from openvla.experiments.robot.device_runtime import get_device
        from transformers import LlamaConfig
        from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM
        device = get_device()
        if device.type != 'musa': self.skipTest('requires MUSA')
        torch.manual_seed(28)
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
        original = LlamaSpecForCausalLM(cfg, attn_implementation='eager').to(torch.bfloat16).eval()
        quant = copy.deepcopy(original)
        report = quantize_linears(quant, {'verify_mlp'}, 32)
        self.assertEqual(report['module_count'], 3)
        original.to(device); quant.to(device)
        ids = torch.tensor([[1, 4, 7, 9]], device=device)
        with torch.inference_mode():
            expected = original(input_ids=ids, use_cache=True)
            actual = quant(input_ids=ids, use_cache=True)
            error = (expected.logits.float() - actual.logits.float()).norm() / expected.logits.float().norm()
            self.assertLess(float(error), .02)
            next_output = quant(input_ids=ids[:, :1], past_key_values=actual.past_key_values, use_cache=True)
        self.assertTrue(torch.isfinite(next_output.logits).all())


if __name__ == '__main__': unittest.main()
