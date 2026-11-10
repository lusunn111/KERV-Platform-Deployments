"""CPU preparation gates for exact-only and mixed precision storage policies."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from experiments.robot import flagos_ops
from experiments.robot.libero.run_precision_libero import register,digest
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from KERVRuntimeOptimization.lossless_bf16 import LosslessBF16Linear
from KERVRuntimeOptimization.packed_intn import PackedIntLinear
from KERVRuntimeOptimization.packed_pow2 import PackedPow2Linear
from KERVRuntimeOptimization.nearzero_bf16 import NearZeroBF16Linear
from KERVRuntimeOptimization.inline_exact_bf16 import InlineExactBF16Linear


class PrecisionStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):register()

    def make_model(self):
        torch.manual_seed(407)
        model=nn.Module();model.base_model=nn.Module();model.base_model.language_model=nn.Module()
        model.base_model.language_model.model=nn.Module()
        layer=nn.Module();layer.mlp=nn.Module();layer.self_attn=nn.Module()
        for name in ['gate_proj','up_proj','down_proj']:setattr(layer.mlp,name,nn.Linear(64,64,bias=False,dtype=torch.bfloat16))
        for name in ['q_proj','k_proj','v_proj','o_proj']:setattr(layer.self_attn,name,nn.Linear(64,64,bias=False,dtype=torch.bfloat16))
        model.base_model.language_model.model.layers=nn.ModuleList([layer]);return model

    def policy(self,root,model,quant=False):
        entries=[]
        if quant:
            name='base_model.language_model.model.layers.0.mlp.up_proj';linear=model.get_submodule(name)
            q=CalibratedInt8Linear(64,64,32,0);q.qweight.random_(-127,128);q.scales.fill_(.001)
            path=root/'q.pt';torch.save(q.payload(),path)
            entries=[{'name':name,'shape':[64,64],'file':'q.pt','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'original_weight_sha256':digest(linear.weight),'decoded_weight_sha256':digest(q.dequantize()),
                      'before_bytes':64*64*2,'storage_bytes':q.storage_bytes}]
        data={'schema':'kerv_precision_first_storage_v1','complete':True,'module_count':len(entries),'modules':entries}
        p=root/'policy.json';p.write_text(json.dumps(data));return p

    def apply(self,model,path):
        return flagos_ops.enable_w8a16_quantization(model,True,groups='verify_mlp,verify_attention',policy_path=str(path))

    def test_exact_only_restores_every_original_bit(self):
        model=self.make_model();weights={n:m.weight.detach().clone() for n,m in model.named_modules() if isinstance(m,nn.Linear)}
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model);manifest=self.apply(model,p)
            self.assertFalse(manifest['lossy_quantization']);self.assertEqual(manifest['int8_module_count'],0)
            self.assertEqual(manifest['exact_module_count'],6)
            for n,w in weights.items():
                m=model.get_submodule(n);actual=m.weight if isinstance(m,nn.Linear) else m.dequantize()
                self.assertTrue(torch.equal(actual.view(torch.int16),w.view(torch.int16)))
            self.assertIs(self.apply(model,p),manifest)

    def test_hybrid_keeps_exact_attention_and_native_query(self):
        model=self.make_model()
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model,True);manifest=self.apply(model,p)
            layer=model.base_model.language_model.model.layers[0]
            self.assertTrue(manifest['lossy_quantization']);self.assertEqual(manifest['int8_module_count'],1)
            self.assertEqual(manifest['exact_module_count'],5)
            self.assertIsInstance(layer.mlp.up_proj,CalibratedInt8Linear)
            self.assertIsInstance(layer.self_attn.q_proj,nn.Linear)
            self.assertIsInstance(layer.self_attn.k_proj,LosslessBF16Linear)

    def test_exact_mlp_keeps_attention_native(self):
        model=self.make_model()
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model);data=json.loads(p.read_text())
            data['exact_bf16_groups']=['verify_mlp'];p.write_text(json.dumps(data))
            manifest=self.apply(model,p);layer=model.base_model.language_model.model.layers[0]
            self.assertFalse(manifest['lossy_quantization']);self.assertEqual(manifest['exact_module_count'],3)
            self.assertIsInstance(layer.mlp.up_proj,LosslessBF16Linear)
            for name in ['q_proj','k_proj','v_proj','o_proj']:self.assertIsInstance(getattr(layer.self_attn,name),nn.Linear)

    def test_packed_w10_with_native_remaining_weights(self):
        for encoding,cls in [('packed_intn',PackedIntLinear),('packed_pow2',PackedPow2Linear)]:
            with self.subTest(encoding=encoding):
                model=self.make_model();name='base_model.language_model.model.layers.0.mlp.up_proj'
                with tempfile.TemporaryDirectory() as folder:
                    root=Path(folder);p=self.policy(root,model,True);data=json.loads(p.read_text())
                    q=cls.from_linear(model.get_submodule(name),10);torch.save(q.payload(),root/'q.pt')
                    entry=data['modules'][0];entry.update(encoding=encoding,sha256=hashlib.sha256((root/'q.pt').read_bytes()).hexdigest(),
                                                        decoded_weight_sha256=digest(q.dequantize()))
                    data['exact_bf16_groups']=[];p.write_text(json.dumps(data));manifest=self.apply(model,p)
                    self.assertEqual(manifest['int8_module_count'],0);self.assertEqual(manifest['packed_intn_module_count'],1)
                    self.assertEqual(manifest['exact_module_count'],0);self.assertTrue(manifest['lossy_quantization'])
                    self.assertIsInstance(model.get_submodule(name),cls)
                    self.assertTrue(torch.equal(model.get_submodule(name).dequantize(),q.dequantize()))
                    self.assertIsInstance(model.base_model.language_model.model.layers[0].mlp.gate_proj,nn.Linear)

    def test_nearzero_exclusive_policy_and_error_metadata(self):
        model=self.make_model()
        with torch.no_grad():model.base_model.language_model.model.layers[0].mlp.up_proj.weight[0,0]=1e-10
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model);data=json.loads(p.read_text())
            data.update(exact_bf16_groups=[],nearzero_bf16_groups=['verify_mlp']);p.write_text(json.dumps(data))
            manifest=self.apply(model,p);layer=model.base_model.language_model.model.layers[0]
            self.assertEqual(manifest['nearzero_module_count'],3);self.assertEqual(manifest['exact_module_count'],0)
            self.assertTrue(manifest['lossy_quantization']);self.assertIsInstance(layer.mlp.up_proj,NearZeroBF16Linear)
            self.assertIsInstance(layer.self_attn.k_proj,nn.Linear)
            self.assertTrue(any(x['quantization']['zeroed_nonzero_weights']>0 for x in manifest['modules']))
            self.assertTrue(all(x['quantization']['relative_weight_rmse']<1e-4 for x in manifest['modules']))

    def test_inline_exact_policy_restores_original_weights(self):
        model=self.make_model();weights={n:m.weight.detach().clone() for n,m in model.named_modules() if isinstance(m,nn.Linear)}
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model);data=json.loads(p.read_text())
            data.update(exact_bf16_groups=[],inline_exact_bf16_groups=['verify_mlp']);p.write_text(json.dumps(data))
            manifest=self.apply(model,p)
            self.assertEqual(manifest['inline_exact_module_count'],3);self.assertEqual(manifest['exact_module_count'],3)
            self.assertFalse(manifest['lossy_quantization']);self.assertEqual(manifest['lossy_module_count'],0)
            for name,w in weights.items():
                module=model.get_submodule(name)
                actual=module.dequantize() if isinstance(module,InlineExactBF16Linear) else module.weight
                self.assertTrue(torch.equal(actual.view(torch.int16),w.view(torch.int16)))

    def test_tampered_payload_rejected_before_replacement(self):
        model=self.make_model()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=self.policy(root,model,True)
            (root/'q.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):self.apply(model,p)
            self.assertIsInstance(model.base_model.language_model.model.layers[0].mlp.up_proj,nn.Linear)

    def test_incomplete_policy_rejected(self):
        model=self.make_model()
        with tempfile.TemporaryDirectory() as folder:
            p=self.policy(Path(folder),model);data=json.loads(p.read_text());data['complete']=False;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):self.apply(model,p)


if __name__=='__main__':unittest.main()
