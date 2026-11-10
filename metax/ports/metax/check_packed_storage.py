"""Two real matrices prove identical quantized bytes and decoded GPU weights after packing."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from ports.metax.weight_only import Workspace,WeightOnlyLinear,pack_cpu_storage
root=Path('/workspace/checkpoints/openvla-libero-goal');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
name='language_model.model.layers.7.mlp.gate_proj.weight'
with safe_open(root/index[name],framework='pt',device='cpu') as f:weight=f.get_tensor(name)
linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=torch.bfloat16)
linear.weight=torch.nn.Parameter(weight)
container=torch.nn.Module();workspace=Workspace(weight.numel());container.add_module('workspace',workspace)
modules=[WeightOnlyLinear(linear,workspace,{'forward_calls':0}) for _ in range(2)]
for i,module in enumerate(modules):container.add_module(str(i),module)
expected=[(m.qweight.float().reshape(m.out_features,-1,32)*m.scales.float().unsqueeze(-1)).reshape_as(m.qweight).to(torch.bfloat16) for m in modules]
packed=pack_cpu_storage(container,modules)
assert sum(b.numel() for _,b in container.named_buffers() if b.dtype==torch.int8)==sum(x.numel() for x in expected)
container.cuda()
x=torch.randn(1,8,weight.shape[1],device='cuda',dtype=torch.bfloat16)
for module,reference in zip(modules,expected):
    with torch.no_grad():actual=module(x)
    torch.testing.assert_close(workspace.weight.view_as(reference).cpu(),reference,rtol=0,atol=0)
    torch.testing.assert_close(actual,torch.nn.functional.linear(x,reference.cuda()),rtol=0,atol=0)
print(json.dumps(dict(packed_modules=2,shared_int8_allocation=True,exact_reconstruction=True,exact_gemm=True),indent=2))
