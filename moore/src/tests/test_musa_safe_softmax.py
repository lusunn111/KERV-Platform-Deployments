import unittest
import torch
import torch_musa
from KERVRuntimeOptimization.musa_safe_softmax import softmax, patch_llama_attention_softmax, _repair_rows
import triton


class SafeSoftmaxTests(unittest.TestCase):
    def test_selective_repair_preserves_valid_rows_bit_for_bit(self):
        torch.manual_seed(93)
        x=torch.randn((8,537));x[:,270:]=-3.3895313892515355e38
        expected=x.softmax(-1)
        output=expected.clone();output[3,514:536:4]=.5
        device_output=output.to('musa');count=torch.zeros((),device='musa',dtype=torch.int32)
        _repair_rows[(8,)](x.to('musa'),device_output,count,537,1024,num_warps=4,enable_fp_fusion=False)
        actual=device_output.cpu()
        self.assertEqual(int(count.cpu()),1)
        for i in range(8):
            if i!=3:self.assertTrue(torch.equal(actual[i],output[i]))
        torch.testing.assert_close(actual[3],expected[3],rtol=2e-6,atol=3e-7)
    def test_real_tail_shapes_and_masked_probabilities(self):
        generator=torch.Generator().manual_seed(19)
        for n in [1,31,511,512,537,542,1025]:
            x=torch.randn((3,7,n),generator=generator).to(torch.bfloat16)
            if n>1:x[...,n//2:]=torch.finfo(torch.bfloat16).min
            actual=softmax(x.to('musa'),-1,dtype=torch.float32).cpu()
            expected=x.float().softmax(-1)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=3e-7)
            torch.testing.assert_close(actual.sum(-1),torch.ones((3,7)),rtol=0,atol=3e-7)
            if n>1:self.assertTrue(bool((actual[...,n//2:]==0).all()))

    def test_cast_before_reduce_and_noncontiguous(self):
        x=torch.linspace(-4,4,2048).reshape(32,64).T
        for dtype in [torch.bfloat16,torch.float16,torch.float32]:
            actual=softmax(x.to('musa'),-1,dtype=dtype).cpu()
            expected=x.to(dtype).float().softmax(-1).to(dtype)
            torch.testing.assert_close(actual,expected,rtol=3e-3 if dtype!=torch.float32 else 2e-6,atol=2e-6)

    def test_empty_and_nonfinite_rows(self):
        empty=softmax(torch.empty((0,537),device='musa'),-1)
        self.assertEqual(tuple(empty.shape),(0,537))
        x=torch.tensor([[float('-inf')]*7,[float('nan'),0.,1.,2.,3.,4.,5.]])
        actual=softmax(x.to('musa'),-1).cpu()
        torch.testing.assert_close(actual,x.softmax(-1),equal_nan=True)

    def test_scoped_patch_preserves_global_functional_and_cpu_forward(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        config=LlamaConfig(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2)
        model=torch.nn.ModuleList([LlamaAttention(config,i).eval() for i in range(32)])
        x=torch.randn((1,5,32));position=torch.arange(5)[None]
        original=torch.nn.functional.softmax
        with torch.no_grad():expected=model[0](x,position_ids=position)[0]
        self.assertEqual(len(patch_llama_attention_softmax(model)),32)
        self.assertIs(torch.nn.functional.softmax,original)
        with torch.no_grad():actual=model[0](x,position_ids=position)[0]
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(model[0]._musa_safe_softmax_calls,1)
        # Idempotent registration does not replace any forward twice.
        self.assertEqual(patch_llama_attention_softmax(model),[])


if __name__=='__main__':unittest.main()
