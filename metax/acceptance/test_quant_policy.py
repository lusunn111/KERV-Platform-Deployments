"""Policy identity, partial projection selection, and actual NPU forward gates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch_npu
from experiments.robot.ascend_quantization import apply_cpu_mlp_int8, weight_sha256


def tiny_model():
    model=torch.nn.Module();model.base_model=torch.nn.Module()
    lm=model.base_model.language_model=torch.nn.Module()
    lm.config=SimpleNamespace(num_hidden_layers=1,pretraining_tp=1)
    lm.model=torch.nn.Module();layer=torch.nn.Module();layer.mlp=torch.nn.Module()
    for name in ['gate_proj','up_proj','down_proj']:
        setattr(layer.mlp,name,torch.nn.Linear(256,256,bias=False,dtype=torch.bfloat16))
    lm.model.layers=torch.nn.ModuleList([layer])
    return model


class QuantPolicy(unittest.TestCase):
    def test_policy_identity_partial_selection_and_forward(self):
        torch.manual_seed(7);model=tiny_model()
        prefix='base_model.language_model.model.layers.0.mlp.'
        items=[dict(name=prefix+name,group_size=g,clip_ratio=clip,
            weight_sha256=weight_sha256(model.get_submodule(prefix+name).weight))
            for name,g,clip in [('gate_proj',32,.99),('down_proj',128,1.)]]
        originals={x['name']:model.get_submodule(x['name']).weight.detach().clone() for x in items}
        with TemporaryDirectory() as d:
            file=Path(d)/'policy.json';file.write_text(json.dumps({'version':1,'modules':items}))
            manifest=apply_cpu_mlp_int8(model,policy_path=str(file))
        self.assertEqual(manifest['module_count'],2)
        self.assertEqual(manifest['group_counts'],{'32':1,'64':0,'128':1})
        self.assertIsInstance(model.get_submodule(prefix+'up_proj'),torch.nn.Linear)
        model.npu()
        for item in items:
            module=model.get_submodule(item['name'])
            self.assertFalse(list(module.parameters()))
            x=torch.randn(1,9,256,dtype=torch.bfloat16).npu()
            y=module(x);reference=torch.nn.functional.linear(x,originals[item['name']].npu())
            self.assertTrue(bool(y.isfinite().all()))
            self.assertLess(float((y.float()-reference.float()).norm()/reference.float().norm()),.03)
        self.assertEqual(manifest['forward_calls'],2)

    def test_weight_identity_mismatch_fails(self):
        model=tiny_model()
        with TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps({'version':1,'modules':[
                {'name':'base_model.language_model.model.layers.0.mlp.gate_proj','group_size':32,'weight_sha256':'wrong'}]}))
            with self.assertRaisesRegex(ValueError,'weight hash mismatch'):
                apply_cpu_mlp_int8(model,policy_path=str(p))


if __name__=='__main__':unittest.main()
