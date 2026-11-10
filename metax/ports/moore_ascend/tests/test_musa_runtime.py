"""Compare the deployed MUSA native operators with CPU references."""

import unittest

import torch
import torch.nn.functional as F

from openvla.experiments.robot.device_runtime import get_device, synchronize
from KERVRuntimeOptimization.embodied_ops.runtime import configure_kerv_ops


class MusaRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        if cls.device.type != "musa":
            raise unittest.SkipTest("requires a visible Moore Threads MUSA device")
        configure_kerv_ops(enabled=True, backend="native")

    def test_bf16_swiglu(self):
        torch.manual_seed(7)
        gate = torch.randn(8, 64).bfloat16()
        up = torch.randn_like(gate)
        actual = torch.ops.flagos_embodied.kerv_silu_mul(gate.to(self.device), up.to(self.device))
        torch.testing.assert_close(actual.cpu(), F.silu(gate) * up, rtol=0.02, atol=0.02)

    def test_residual_rms_norm(self):
        torch.manual_seed(7)
        hidden = torch.randn(8, 64).bfloat16()
        residual = torch.randn_like(hidden)
        weight = torch.randn(64).bfloat16()
        combined = (hidden + residual).float()
        expected = (combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-6)).bfloat16() * weight
        actual = torch.ops.flagos_embodied.kerv_add_rms_norm(
            hidden.to(self.device), residual.to(self.device), weight.to(self.device), 1e-6)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.03)

    def test_verify_accept_threshold_and_tie(self):
        logits = torch.zeros(3, 4, 8)
        logits[:, :, 2] = 10
        candidates = torch.tensor([[100, 102, 103, 110], [100, 102, 102, 109], [100, 102, 102, 102]])
        for threshold, expected in [(0.0, (2, 3)), (1.0, (2, 3)), (8.0, (0, 3))]:
            best, length = torch.ops.flagos_embodied.kerv_verify_accept_control(
                logits.to(self.device), candidates.to(self.device), threshold, 100)
            self.assertEqual((best.item(), length.item()), expected)

    def test_accepted_kv_path_gather_and_commit(self):
        source = torch.arange(128).reshape(1, 2, 1, 2, 8, 4).float()
        indices = torch.tensor([5, 2, 6], device=self.device)
        actual = source.to(self.device)
        torch.ops.flagos_embodied.kerv_kv_accept_commit(actual, indices, 1)
        expected = source.clone()
        expected[..., 1:4, :] = source[..., [6, 3, 7], :]
        torch.testing.assert_close(actual.cpu(), expected)

    def test_drafter_and_verifier_forward(self):
        from transformers import LlamaConfig
        from openvla.specdecoding.model.cnets import EConfig, MMModel
        from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM

        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
        verifier = LlamaSpecForCausalLM(config, attn_implementation="eager").eval()
        draft = MMModel(EConfig(**config.to_dict()), load_emb=False).eval()
        ids = torch.tensor([[1, 4, 7, 9]])
        with torch.no_grad():
            reference = verifier(input_ids=ids, output_hidden_states=True)
            verifier.to(self.device)
            actual = verifier(input_ids=ids.to(self.device), output_hidden_states=True)
            torch.testing.assert_close(actual.logits.cpu(), reference.logits, rtol=0.001, atol=0.001)
            hidden = reference.hidden_states[-1]
            expected = draft(hidden, input_ids=ids, input_embeddings=draft.embed_tokens(ids))
            draft.to(self.device)
            output = draft(hidden.to(self.device), input_ids=ids.to(self.device),
                           input_embeddings=draft.embed_tokens(ids.to(self.device)))
            torch.testing.assert_close(output.cpu(), expected, rtol=0.001, atol=0.001)
        synchronize(self.device)


if __name__ == "__main__":
    unittest.main()
