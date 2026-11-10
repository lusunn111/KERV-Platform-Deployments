"""Exercise native fused gate/up reuse, stale-input handling, and policy guards."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch_npu

from experiments.robot.ascend_dynamic_quantization import (
    apply_cpu_mlp_dynamic_int8, direct_dynamic_mlp_forward, NPUPrepackedWeightOnlyLinear)
from experiments.robot.ascend_quantization import weight_sha256
from test_quant_policy import tiny_model


class DynamicQuantization(unittest.TestCase):
    def test_batched_qkv_shape_guard_and_native_fallback(self):
        from experiments.robot.ascend_qkv_ops import pack_cpu_verifier_qkv,transfer_packed_qkv
        model=tiny_model();layer=model.base_model.language_model.model.layers[0]
        layer.self_attn=torch.nn.Module()
        weights=[]
        for name in ['q_proj','k_proj','v_proj']:
            projection=torch.nn.Linear(256,256,bias=False,dtype=torch.bfloat16)
            weights.append(projection.weight.detach().clone())
            setattr(layer.self_attn,name,projection)
        with self.assertRaisesRegex(ValueError,'257..320'):
            pack_cpu_verifier_qkv(model,'256','all','bmm')
        groups=pack_cpu_verifier_qkv(model,'279','all','bmm')
        self.assertEqual(groups[0].projection_backend,'bmm')
        model.npu();transfer_packed_qkv(groups,torch.device('npu:0'))
        self.assertIs(groups[0].weight_slice(0),groups[0].cached_weight_slices[0])
        self.assertEqual(groups[0].batched_weight.untyped_storage().data_ptr(),groups[0].weight.untyped_storage().data_ptr())
        self.assertEqual(groups[0].cached_weight_slices[1].untyped_storage().data_ptr(),groups[0].weight.untyped_storage().data_ptr())
        with torch.inference_mode():
            for rows in [279,33,279]:
                x=torch.randn(1,rows,256,dtype=torch.bfloat16).npu()
                actual=[getattr(layer.self_attn,n)(x) for n in ['q_proj','k_proj','v_proj']]
                reference=[torch.nn.functional.linear(x,w.npu()) for w in weights]
                for got,want in zip(actual,reference):
                    if rows==33:self.assertTrue(torch.equal(got,want))
                    else:self.assertLess(float((got.float()-want.float()).norm()/want.float().norm()),.001)
                self.assertIsNone(groups[0]._cached_input)
                self.assertIsNone(groups[0]._cached_outputs)

    def test_selected_verifier_qkv_does_not_fuse_other_layers_or_drafter(self):
        from experiments.robot.ascend_qkv_ops import (pack_cpu_verifier_qkv,transfer_packed_qkv,
            verifier_qkv_scope,verifier_qkv_layer_ids)
        from KERVRuntimeOptimization.adaptive_linear_fusion import fuse_transformer_linears
        model=tiny_model()
        def attention():
            module=torch.nn.Module()
            for name in ['q_proj','k_proj','v_proj']:
                setattr(module,name,torch.nn.Linear(256,256,bias=False,dtype=torch.bfloat16))
            return module
        layers=model.base_model.language_model.model.layers
        layers[0].self_attn=attention()
        for _ in range(2):
            layer=torch.nn.Module();layer.self_attn=attention();layers.append(layer)
        model.ea_layer=attention()
        self.assertEqual(verifier_qkv_layer_ids(model,'1:3'),[1,2])
        self.assertEqual(verifier_qkv_layer_ids(model,'2,0'),[0,2])
        for invalid in ['', '1:1','0:4','-1','3','1,1','all,0']:
            with self.assertRaises(ValueError):verifier_qkv_layer_ids(model,invalid)
        original=layers[0].self_attn.q_proj.weight.detach().clone()
        groups=pack_cpu_verifier_qkv(model,'33','1:3')
        self.assertEqual([g.verifier_layer_index for g in groups],[1,2])
        model.npu();transfer_packed_qkv(groups,torch.device('npu:0'))
        manifest=fuse_transformer_linears(verifier_qkv_scope(model,'1:3'),qkv_rows=[33],qkv_input_sizes=[256])
        self.assertEqual(manifest['qkv_modules'],['layer_1','layer_2'])
        self.assertEqual(manifest['cpu_prepacked_qkv_group_count'],2)
        self.assertIsInstance(layers[0].self_attn.q_proj,torch.nn.Linear)
        self.assertIsInstance(model.ea_layer.q_proj,torch.nn.Linear)
        self.assertTrue(torch.equal(layers[0].self_attn.q_proj.weight.cpu(),original))
        with torch.inference_mode():
            x=torch.randn(1,33,256,dtype=torch.bfloat16).npu()
            self.assertTrue(bool(layers[1].self_attn.q_proj(x).isfinite().all()))

    def test_cpu_qkv_pack_and_final_device_transfer(self):
        from experiments.robot.ascend_qkv_ops import pack_cpu_verifier_qkv,transfer_packed_qkv
        from KERVRuntimeOptimization.adaptive_linear_fusion import fuse_transformer_linears
        model=tiny_model()
        layer=model.base_model.language_model.model.layers[0]
        layer.self_attn=torch.nn.Module()
        weights=[]
        for name in ['q_proj','k_proj','v_proj']:
            projection=torch.nn.Linear(256,256,bias=False,dtype=torch.bfloat16)
            weights.append(projection.weight.detach().clone())
            setattr(layer.self_attn,name,projection)
        groups=pack_cpu_verifier_qkv(model,'1,33,279')
        self.assertEqual(groups[0].weight.device.type,'cpu')
        model.npu();transfer_packed_qkv(groups,torch.device('npu:0'))
        self.assertTrue(torch.equal(groups[0].weight.cpu(),torch.cat(weights)))
        installed=fuse_transformer_linears(model,qkv_rows=[1,33,279],qkv_input_sizes=[256])
        self.assertEqual(installed['cpu_prepacked_qkv_group_count'],1)
        self.assertIs(layer.self_attn.q_proj._group,groups[0])
        with torch.inference_mode():
            x=torch.randn(1,33,256,dtype=torch.bfloat16).npu()
            for name,weight in zip(['q_proj','k_proj','v_proj'],weights):
                actual=getattr(layer.self_attn,name)(x)
                reference=torch.nn.functional.linear(x,weight.npu())
                self.assertLess(float((actual.float()-reference.float()).norm()/reference.float().norm()),.001)

    def test_weight_only_fused_mlp_rounded_matches_native(self):
        torch.manual_seed(7)
        model = tiny_model()
        mlp = model.base_model.language_model.model.layers[0].mlp
        mlp.act_fn = torch.nn.SiLU()
        prefix = 'base_model.language_model.model.layers.0.mlp.'
        entries = [dict(name=prefix+name, weight_sha256=weight_sha256(getattr(mlp,name).weight),
                        alpha=None, relative_l2=0., precision='weight_only_int8', group_size=32)
                   for name in ['gate_proj','up_proj','down_proj']]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'rounded.pt'
            torch.save(dict(version=1,implementation='ascend_dynamic_w8a8',fused_gate_up=True,
                            direct_mlp=True,fused_swiglu_quant=True,silu_backend='triton_rounded',modules=entries),path)
            manifest=apply_cpu_mlp_dynamic_int8(model,path)
        self.assertIsInstance(mlp._dynamic_gate_up,NPUPrepackedWeightOnlyLinear)
        model.npu()
        with torch.inference_mode():
            for rows in [1,33,279]:
                x=torch.randn(1,rows,256,dtype=torch.bfloat16).npu()
                custom=mlp(x)
                rounded=mlp._dynamic_rounded_swiglu_fn
                mlp._dynamic_rounded_swiglu_fn=None
                native=mlp(x)
                mlp._dynamic_rounded_swiglu_fn=rounded
                self.assertTrue(torch.equal(custom,native))
                self.assertTrue(bool(custom.isfinite().all()))
        self.assertEqual(manifest['precision_counts'],dict(dynamic_int8=0,weight_only_int8=3,bf16=0))
        self.assertEqual(manifest['activation_dtype'],'bfloat16')
        self.assertFalse(list(model.parameters()))
        # FlagScale must fuse BF16 attention without re-owning quantized MLPs.
        from KERVRuntimeOptimization.adaptive_linear_fusion import fuse_transformer_linears
        model.probe_attention=torch.nn.Module()
        for name in ['q_proj','k_proj','v_proj']:
            setattr(model.probe_attention,name,torch.nn.Linear(256,256,bias=False,dtype=torch.bfloat16).npu())
        group=mlp._dynamic_gate_up
        packed=fuse_transformer_linears(model,qkv_rows=[1,33,279],gate_up_rows=[33],qkv_input_sizes=[256])
        self.assertEqual(packed['qkv_group_count'],1)
        self.assertEqual(packed['gate_up_group_count'],0)
        self.assertIs(mlp._dynamic_gate_up,group)
        with torch.inference_mode():
            self.assertTrue(torch.equal(mlp(x),custom))

    def test_mixed_w8a8_w8a16_native_mlp(self):
        torch.manual_seed(7)
        model = tiny_model()
        mlp = model.base_model.language_model.model.layers[0].mlp
        mlp.act_fn = torch.nn.SiLU()
        prefix = 'base_model.language_model.model.layers.0.mlp.'
        weights = {name:getattr(mlp,name).weight.detach().clone() for name in ['gate_proj','up_proj','down_proj']}
        entries = [dict(name=prefix+name,weight_sha256=weight_sha256(w),
                        smooth=torch.ones(256,dtype=torch.bfloat16),alpha=0.,relative_l2=0.,
                        precision='weight_only_int8' if name=='down_proj' else 'dynamic_int8',group_size=32)
                   for name,w in weights.items()]
        with TemporaryDirectory() as directory:
            path=Path(directory)/'mixed.pt'
            torch.save(dict(version=1,implementation='ascend_dynamic_w8a8',fused_gate_up=True,
                            direct_mlp=True,fused_swiglu_quant=True,modules=entries),path)
            manifest=apply_cpu_mlp_dynamic_int8(model,path)
        self.assertIsInstance(mlp.down_proj,NPUPrepackedWeightOnlyLinear)
        model.npu()
        with torch.inference_mode():
            x=torch.randn(1,33,256,dtype=torch.bfloat16).npu()
            gate=torch.nn.functional.linear(x,weights['gate_proj'].npu())
            up=torch.nn.functional.linear(x,weights['up_proj'].npu())
            reference=torch.nn.functional.linear(torch.nn.functional.silu(gate)*up,weights['down_proj'].npu())
            actual=mlp(x)
            self.assertTrue(bool(actual.isfinite().all()))
            self.assertLess(float((actual.float()-reference.float()).norm()/reference.float().norm()),.05)
        self.assertEqual(manifest['precision_counts'],dict(dynamic_int8=2,weight_only_int8=1,bf16=0))
        self.assertEqual(manifest['forward_calls'],2)
        self.assertFalse(list(model.parameters()))

    def test_native_policy_preserves_requested_down_weights(self):
        torch.manual_seed(7)
        model = tiny_model()
        mlp = model.base_model.language_model.model.layers[0].mlp
        mlp.act_fn = torch.nn.SiLU()
        prefix = 'base_model.language_model.model.layers.0.mlp.'
        original = mlp.down_proj.weight.detach().clone()
        entries = [dict(name=prefix+name, weight_sha256=weight_sha256(getattr(mlp,name).weight),
                        smooth=torch.ones(256,dtype=torch.bfloat16), alpha=0.,relative_l2=0.,
                        precision='bf16' if name=='down_proj' else 'dynamic_int8')
                   for name in ['gate_proj','up_proj','down_proj']]
        with TemporaryDirectory() as directory:
            path = Path(directory)/'hybrid.pt'
            torch.save(dict(version=1,implementation='ascend_dynamic_w8a8',fused_gate_up=True,
                            direct_mlp=True,fused_swiglu_quant=True,modules=entries),path)
            manifest = apply_cpu_mlp_dynamic_int8(model,path)
        self.assertIsInstance(mlp.down_proj,torch.nn.Linear)
        self.assertTrue(torch.equal(mlp.down_proj.weight,original))
        model.npu()
        with torch.inference_mode():
            output=mlp(torch.randn(1,33,256,dtype=torch.bfloat16).npu())
        self.assertTrue(bool(output.isfinite().all()))
        self.assertEqual(manifest['preserved_bf16_modules'],[prefix+'down_proj'])
        self.assertEqual(manifest['forward_calls'],1)
        self.assertEqual(manifest['replaced_projection_count'],2)

    def test_exported_native_policy_load_and_full_mlp(self):
        torch.manual_seed(7)
        model = tiny_model()
        mlp = model.base_model.language_model.model.layers[0].mlp
        mlp.act_fn = torch.nn.SiLU()
        prefix = 'base_model.language_model.model.layers.0.mlp.'
        weights = {name: getattr(mlp, name).weight.detach().clone()
                   for name in ['gate_proj', 'up_proj', 'down_proj']}
        entries = [dict(name=prefix+name, weight_sha256=weight_sha256(weight),
                        smooth=torch.ones(256, dtype=torch.bfloat16), alpha=0., relative_l2=0.)
                   for name, weight in weights.items()]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'native.pt'
            torch.save(dict(version=1, implementation='ascend_dynamic_w8a8', fused_gate_up=True,
                            direct_mlp=True, fused_swiglu_quant=True, modules=entries), path)
            manifest = apply_cpu_mlp_dynamic_int8(model, path)
        model.npu()
        with torch.inference_mode():
            x = torch.randn(1, 33, 256, dtype=torch.bfloat16).npu()
            gate = torch.nn.functional.linear(x, weights['gate_proj'].npu())
            up = torch.nn.functional.linear(x, weights['up_proj'].npu())
            reference = torch.nn.functional.linear(torch.nn.functional.silu(gate)*up, weights['down_proj'].npu())
            actual = mlp(x)
            self.assertTrue(bool(actual.isfinite().all()))
            self.assertLess(float((actual.float()-reference.float()).norm()/reference.float().norm()), .05)
        self.assertTrue(manifest['fused_swiglu_quant'])
        self.assertEqual(manifest['forward_calls'], 2)
        self.assertFalse(list(model.parameters()))

    def test_fused_shared_projection_and_fresh_inputs(self):
        torch.manual_seed(7)
        model = tiny_model()
        prefix = 'base_model.language_model.model.layers.0.mlp.'
        weights = {name: model.get_submodule(prefix + name).weight.detach().clone()
                   for name in ['gate_proj', 'up_proj', 'down_proj']}
        entries = [dict(name=prefix+name, weight_sha256=weight_sha256(weight),
                        smooth=torch.ones(256, dtype=torch.bfloat16), alpha=0., relative_l2=0.)
                   for name, weight in weights.items()]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            torch.save(dict(version=1, implementation='ascend_dynamic_w8a8', fused_gate_up=True,
                            modules=entries), path)
            manifest = apply_cpu_mlp_dynamic_int8(model, path)
        model.npu()
        mlp = model.base_model.language_model.model.layers[0].mlp
        with torch.inference_mode():
            for _ in range(2):
                x = torch.randn(1, 33, 256, dtype=torch.bfloat16).npu()
                for name in ['gate_proj', 'up_proj']:
                    actual = getattr(mlp, name)(x)
                    reference = torch.nn.functional.linear(x, weights[name].npu())
                    self.assertTrue(bool(actual.isfinite().all()))
                    self.assertLess(float((actual.float()-reference.float()).norm()/reference.float().norm()), .03)
                self.assertIsNone(mlp._dynamic_gate_up._slice_output)
            self.assertEqual(manifest['forward_calls'], 2)
            # Up called with another input must not consume an earlier gate cache.
            mlp.gate_proj(x)
            other = x + 1
            actual = mlp.up_proj(other)
            reference = torch.nn.functional.linear(other, weights['up_proj'].npu())
            self.assertLess(float((actual.float()-reference.float()).norm()/reference.float().norm()), .03)
        self.assertEqual(manifest['module_count'], 2)
        self.assertEqual(manifest['replaced_projection_count'], 3)
        self.assertGreater(manifest['unique_storage_saving_bytes'], 0)
        self.assertFalse(list(model.parameters()))
        # The direct MLP route removes Python slice modules and cache traffic;
        # its numerical operations must exactly match the original call order.
        mlp.act_fn = torch.nn.functional.silu
        with torch.inference_mode():
            for rows in [1, 33, 279]:
                x = torch.randn(1, rows, 256, dtype=torch.bfloat16).npu()
                expected = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
                actual = direct_dynamic_mlp_forward(mlp, x)
                self.assertTrue(torch.equal(actual, expected))
                mlp.down_proj.fused_smooth_scale = mlp.down_proj.smooth.float().reshape(1, -1)
                mlp._dynamic_fused_swiglu_quant = True
                fused = direct_dynamic_mlp_forward(mlp, x)
                self.assertTrue(bool(fused.isfinite().all()))
                self.assertLess(float((fused.float()-expected.float()).norm()/expected.float().norm()), .04)
                mlp._dynamic_fused_swiglu_quant = False

    def test_bad_identity_rejected_before_replacement(self):
        model = tiny_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            torch.save(dict(version=1, implementation='ascend_dynamic_w8a8', modules=[
                dict(name='base_model.language_model.model.layers.0.mlp.gate_proj', weight_sha256='wrong')]), path)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                apply_cpu_mlp_dynamic_int8(model, path)
        self.assertIsInstance(model.base_model.language_model.model.layers[0].mlp.gate_proj, torch.nn.Linear)


if __name__ == '__main__':
    unittest.main()
