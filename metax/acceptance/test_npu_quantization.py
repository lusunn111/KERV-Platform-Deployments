"""Real-shape tests for the NPU-only weight-int8/activation-BF16 adapter."""
import json, unittest
import torch
from experiments.robot.ascend_quantization import NPUWeightOnlyLinear

class NPUQuantization(unittest.TestCase):
    def test_real_mlp_shapes_and_no_bf16_shadow(self):
        torch.manual_seed(7)
        results=[]
        for n,k in [(11008,4096),(4096,11008)]:
            linear=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16)
            manifest={'forward_calls':0};layer=NPUWeightOnlyLinear(linear,manifest,'unit',128).npu()
            self.assertFalse(any(True for _ in layer.parameters()))
            self.assertEqual(layer.qweight.dtype,torch.int8)
            self.assertEqual(layer.antiquant_scale.dtype,torch.bfloat16)
            self.assertLess(sum(x.numel()*x.element_size() for x in layer.buffers()),linear.weight.numel()*2)
            for m in [8,279]:
                x=torch.randn(1,m,k,dtype=torch.bfloat16).npu()
                actual=layer(x);ref=torch.nn.functional.linear(x,linear.weight.npu())
                rel=float((actual.float()-ref.float()).norm()/ref.float().norm())
                self.assertEqual(actual.dtype,torch.bfloat16);self.assertEqual(actual.shape,(1,m,n))
                self.assertTrue(bool(actual.isfinite().all()));self.assertLess(rel,.02)
                results.append({'M':m,'K':k,'N':n,'relative_l2_vs_bf16':rel})
            self.assertEqual(manifest['forward_calls'],2)
            del layer,linear;torch.npu.empty_cache()
        print(json.dumps(results),flush=True)

if __name__=='__main__':unittest.main()
