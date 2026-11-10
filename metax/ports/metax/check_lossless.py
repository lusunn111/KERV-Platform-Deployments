"""Use real MLP and attention matrices to prove bit-exact GPU reconstruction/GEMM."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from ports.metax.lossless import install
root=Path('/workspace/checkpoints/openvla-libero-goal');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
container=torch.nn.Module();container.add_module('layers',torch.nn.ModuleDict());selected=[];references=[]
for short,suffix in [('mlp','mlp.gate_proj.weight'),('attention','self_attn.o_proj.weight')]:
    name='language_model.model.layers.7.'+suffix
    with safe_open(root/index[name],framework='pt',device='cpu') as handle:weight=handle.get_tensor(name)
    linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=torch.bfloat16)
    linear.weight=torch.nn.Parameter(weight);container.layers[short]=linear
    selected.append(('layers.'+short,linear));references.append(weight)
manifest=install(container,selected);container.cuda()
for (name,_),reference in zip(selected,references):
    module=container.get_submodule(name)
    x=torch.randn(1,8,module.in_features,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():actual=module(x)
    assert torch.equal(module._bits.cpu(),reference.view(torch.int16)),name
    torch.testing.assert_close(actual,torch.nn.functional.linear(x,reference.cuda()),rtol=0,atol=0)
print(json.dumps(dict(exact_weight_bits=True,exact_gemm=True,actual_checkpoint_matrices=2,
    exceptions=[x['exception_count'] for x in manifest['modules']],
    storage_saving_minus_workspace_bytes=manifest['storage_saving_minus_workspace_bytes']),indent=2))
