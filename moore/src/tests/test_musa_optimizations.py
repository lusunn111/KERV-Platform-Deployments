"""Check numerical equivalence and actual execution of MUSA optimizations."""

from types import SimpleNamespace
import unittest

import torch

from openvla.experiments.robot.device_runtime import get_device, synchronize


class MusaOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        if cls.device.type != "musa":
            raise unittest.SkipTest("requires MUSA")

    def test_grouped_linear_and_swiglu_verifier(self):
        from transformers import LlamaConfig
        from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM
        from KERVRuntimeOptimization.adaptive_linear_fusion import fuse_transformer_linears

        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4)
        model = LlamaSpecForCausalLM(config, attn_implementation="eager").to(
            device=self.device, dtype=torch.bfloat16).eval()
        ids = torch.tensor([[1, 4, 7, 9]], device=self.device)
        with torch.no_grad():
            expected = model(input_ids=ids).logits.clone()
            report = fuse_transformer_linears(
                model, qkv_rows=(4,), gate_up_rows=(4,), qkv_input_sizes=(64,),
                gate_up_input_sizes=(64,), swiglu_rows=(4,), swiglu_input_sizes=(64,),
                swiglu_backend="native_inplace")
            actual = model(input_ids=ids).logits
        self.assertEqual(len(report["qkv_modules"]), 1)
        self.assertEqual(len(report["gate_up_modules"]), 1)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)

    def test_musa_graph_replays_changed_inputs(self):
        from openvla.prismatic.extern.hf.flagos_stage_graph import StageGraphCache

        owner = SimpleNamespace(_flagos_shared_graph_pool=False)
        cache = StageGraphCache(owner, "on", "draft", max_entries=1)
        weight = torch.randn(16, 16, device=self.device)
        x = torch.randn(16, 16, device=self.device)
        def forward(x):
            return (x @ weight).relu()
        with torch.inference_mode():
            initial, _ = cache.run(forward, {"x": x})
            torch.testing.assert_close(initial, forward(x))
            changed = x + 3
            actual, mode = cache.run(forward, {"x": changed})
            torch.testing.assert_close(actual, forward(changed))
        if mode == "eager_fallback" and "API call is not supported" in str(getattr(owner, "_flagos_draft_graph_last_error", "")):
            self.skipTest("Installed MUSA driver rejects graph capture API; optimized profile disables graphs")
        self.assertEqual(mode, "replay", getattr(owner, "_flagos_draft_graph_last_error", ""))
        self.assertEqual(len(cache.entries), 1)
        synchronize(self.device)

    def test_triton_resident_kv_commit(self):
        from openvla.prismatic.extern.hf.flagos_persistent_ops import commit_tree_rows_

        storage = torch.arange(128).reshape(1, 2, 1, 2, 8, 4).float().to(self.device)
        expected = storage.clone()
        expected[..., 1:4, :] = storage[..., [6, 3, 7], :]
        indices = torch.tensor([5, 2, 6], device=self.device)
        commit_tree_rows_(storage, indices, 1)
        torch.testing.assert_close(storage, expected)

    def test_persistent_control_buffer(self):
        from openvla.prismatic.extern.hf.flagos_system_runtime import ControlBuffer

        buffer = ControlBuffer(self.device)
        for first, second in [(3, 2), (0, 7), (41, 1)]:
            values = buffer.read_pair(torch.tensor(first, device=self.device),
                                      torch.tensor(second, device=self.device))
            self.assertEqual(values, [first, second])

    def test_batched_kalman_matches_per_branch(self):
        from types import MethodType
        from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction

        owner = SimpleNamespace(_kalman_config={"enabled": True},
                                _store_history_flag=True, _kalman_history_window=16,
                                vocab_size=32000, bin_centers=torch.zeros(256),
                                tokenizer=SimpleNamespace(pad_token_id=0), get_action_dim=lambda: 7)
        for method in ("kalman_predict_tokens", "predict_kf_chain_tokens", "predict_kf_chains_batched"):
            setattr(owner, method, MethodType(getattr(SpecVLAforActionPrediction, method), owner))
        prefixes = [[31800], [31801, 31840], [31802, 31841, 31880]]
        remaining = [6, 5, 4]
        for history in ([], [[31800 + i + t for i in range(7)] for t in range(4)]):
            owner._token_history = history
            expected = [owner.predict_kf_chain_tokens(prefix, length)
                        for prefix, length in zip(prefixes, remaining)]
            self.assertEqual(owner.predict_kf_chains_batched(prefixes, remaining), expected)

    def test_fused_rotary_matches_reference(self):
        from KERVRuntimeOptimization.adaptive_rotary_fusion import fused_rotary_qk, fused_rotary_qk_direct

        torch.manual_seed(11)
        cos = torch.randn(32, 128, device=self.device, dtype=torch.bfloat16)
        sin = torch.randn_like(cos)
        for sequence in (8, 32):
            query = torch.randn(1, sequence, 4, 128, device=self.device,
                                dtype=torch.bfloat16).transpose(1, 2)
            key = torch.randn_like(query)
            positions = torch.arange(sequence, device=self.device).unsqueeze(0)
            def reference(value):
                first, second = value.chunk(2, dim=-1)
                rotated = torch.cat((-second, first), dim=-1)
                return value * cos[positions].unsqueeze(1) + rotated * sin[positions].unsqueeze(1)
            for actual_q, actual_k in (
                fused_rotary_qk(query, key, cos, sin, positions),
                fused_rotary_qk_direct(query, key, cos[positions], sin[positions]),
            ):
                torch.testing.assert_close(actual_q, reference(query), rtol=0, atol=0)
                torch.testing.assert_close(actual_k, reference(key), rtol=0, atol=0)

    def test_cpp_static_tree_against_python_reference(self):
        from openvla.experiments.robot.flagos_tree_ops import load_flagos_kerv_tree_ops
        from openvla.prismatic.extern.hf.utils import _build_static_verification_plan

        load_flagos_kerv_tree_ops()
        generator = torch.Generator().manual_seed(19)
        for count in (1, 7, 49, 300):
            parents = [0] + [int(torch.randint(i, (), generator=generator)) for i in range(1, count)]
            parent = dict(enumerate(parents))
            children = {}
            for node in range(1, count):
                children.setdefault(parents[node], set()).add(node)
            for limit in (0, 1, 48):
                expected = _build_static_verification_plan(parent, children, count, limit, torch.float32)
                actual = torch.ops.flagos_kerv.build_verification_subtree(torch.tensor(parents), limit)
                for lhs, rhs in zip(actual[:4], expected[:4]):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                self.assertEqual(actual[4].tolist(), [expected[4][key] for key in
                                 ("full_nodes", "verified_nodes", "full_paths", "verified_paths")])
                tokens = torch.arange(count, device=self.device).reshape(1, -1)
                packed = torch.ops.flagos_kerv.pack_verification_tokens(tokens, actual[0])
                torch.testing.assert_close(packed, tokens[:, actual[0].to(self.device)])
        with self.assertRaises(RuntimeError):
            torch.ops.flagos_kerv.build_verification_subtree(torch.tensor([0, 2, 1]), 0)

    def test_bf16_tied_draft_scores_keep_ancestors(self):
        from openvla.specdecoding.model.cnets import _select_draft_tree_nodes

        # Nodes 2 and 4 tie with their respective ancestors 0 and 1.
        scores = torch.tensor([-1., -2., -1., -3., -2., -3.],
                              device=self.device, dtype=torch.bfloat16)
        parents = [-1, -1, 0, 0, 1, 2]
        for count in range(1, scores.numel() + 1):
            selected = _select_draft_tree_nodes(scores, count).cpu().tolist()
            for node in selected:
                self.assertTrue(parents[node] == -1 or parents[node] in selected)
        self.assertEqual(_select_draft_tree_nodes(scores, 3).cpu().tolist(), [0, 1, 2])

    def test_single_kernel_tree_mask_and_prefix_gap(self):
        from KERVRuntimeOptimization.tree_attention_mask import build_tree_causal_mask

        tree = torch.eye(7)
        tree[:, 0] = 1
        for dtype in (torch.float32, torch.bfloat16):
            for past, start in [(0, 0), (3, 3), (3, 5)]:
                actual = build_tree_causal_mask(tree.to(self.device).reshape(1, 1, 7, 7),
                                               past, dtype, tree_start=start)
                expected = torch.zeros(7, start + 7, dtype=dtype)
                expected[:, past:start] = torch.finfo(dtype).min
                allowed = tree.bool() & torch.ones(7, 7, dtype=torch.bool).tril()
                expected[:, start:] = torch.where(allowed, 0, torch.finfo(dtype).min)
                torch.testing.assert_close(actual.cpu().reshape(7, -1), expected, rtol=0, atol=0)

    def test_flaggems_swiglu_and_fused_rms_norm(self):
        import flag_gems
        import torch.nn.functional as F
        from KERVRuntimeOptimization.adaptive_rms_norm import rms_norm_inference, add_rms_norm_inference

        gate = torch.randn(8, 11008, device=self.device, dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        with torch.no_grad():
            actual = flag_gems.silu_and_mul(gate, up)
        torch.testing.assert_close(actual, F.silu(gate) * up, rtol=0.03, atol=0.03)
        hidden = torch.randn(8, 4096, device=self.device, dtype=torch.bfloat16)
        residual = torch.randn_like(hidden)
        weight = torch.randn(4096, device=self.device, dtype=torch.bfloat16)
        def reference(x):
            x32 = x.float()
            return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).bfloat16() * weight
        torch.testing.assert_close(rms_norm_inference(hidden, (4096,), weight), reference(hidden),
                                   rtol=0.02, atol=0.02)
        combined = hidden + residual
        actual, saved_residual = add_rms_norm_inference(hidden.clone(), residual.clone(), (4096,), weight)
        torch.testing.assert_close(actual, reference(combined), rtol=0.02, atol=0.02)
        torch.testing.assert_close(saved_residual, combined, rtol=0, atol=0)

    def test_flagos_flat_swiglu_large_width(self):
        from KERVRuntimeOptimization.embodied_ops.runtime import configure_kerv_ops
        import torch.nn.functional as F

        configure_kerv_ops(enabled=True, backend="triton")
        try:
            gate = torch.randn(4, 11008, device=self.device, dtype=torch.bfloat16)
            up = torch.randn_like(gate)
            expected = F.silu(gate) * up
            actual = torch.ops.flagos_embodied.kerv_silu_mul(gate, up)
            self.assertEqual(actual.data_ptr(), gate.data_ptr())
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        finally:
            configure_kerv_ops(enabled=True, backend="native")


if __name__ == "__main__":
    unittest.main()
