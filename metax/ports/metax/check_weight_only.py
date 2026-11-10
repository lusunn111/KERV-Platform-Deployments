"""Check the reconstruction kernel using an actual checkpoint matrix."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from safetensors import safe_open
from ports.metax.weight_only import Workspace,WeightOnlyLinear
root=Path('/workspace/checkpoints/openvla-libero-goal')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
name='language_model.model.layers.7.mlp.gate_proj.weight'
with safe_open(root/index[name],framework='pt',device='cpu') as handle:
    weight=handle.get_tensor(name)
linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,dtype=torch.bfloat16)
linear.weight=torch.nn.Parameter(weight)
workspace=Workspace(weight.numel()).cuda(); manifest={'forward_calls':0}
q=WeightOnlyLinear(linear,workspace,manifest).cuda()
x=torch.randn(1,8,weight.shape[1],device='cuda',dtype=torch.bfloat16)
with torch.no_grad():
    actual=q(x)
    reconstructed=(q.qweight.float().reshape(q.out_features,-1,32)*q.scales.float().unsqueeze(-1)).reshape_as(q.qweight).to(torch.bfloat16)
    torch.testing.assert_close(workspace.weight.view_as(reconstructed),reconstructed,rtol=0,atol=0)
    expected=torch.nn.functional.linear(x,reconstructed)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    reference=torch.nn.functional.linear(x,weight.cuda())
    error=(reference.float()-actual.float())
    print(json.dumps(dict(real_weight=name,reconstruction_exact=True,gemm_exact=True,
        max_abs_error_vs_bf16=error.abs().max().item(),relative_l2_vs_bf16=(error.norm()/reference.float().norm()).item(),
        shape=list(weight.shape),forward_calls=manifest['forward_calls']),indent=2))
