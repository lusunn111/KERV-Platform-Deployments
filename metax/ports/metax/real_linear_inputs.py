"""Frozen real layer-zero MLP activations for operator screening."""
import json,torch
from pathlib import Path
from safetensors import safe_open
ROOT=Path('/workspace/checkpoints/openvla-libero-goal')
INDEX=json.loads((ROOT/'model.safetensors.index.json').read_text())['weight_map']
INPUTS=torch.load('/workspace/outputs/metax_v2_20261104/probe01/actual_inputs.pt',weights_only=True)
def weight(key):
 with safe_open(ROOT/INDEX[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
def activation(suffix,m):
 # gate_up captures are the actual normalized MLP input. Reconstruct layer-zero
 # down input using original BF16 SiLU/multiply and the real gate/up weights.
 if suffix=='mlp.gate_proj.weight':return INPUTS[f'gate_up_1x{m}x4096'].cuda(),'actual layer0 gate input'
 if suffix=='mlp.down_proj.weight':
  x=INPUTS[f'gate_up_1x{m}x4096'].cuda()
  gate=weight('language_model.model.layers.0.mlp.gate_proj.weight').cuda()
  up=weight('language_model.model.layers.0.mlp.up_proj.weight').cuda()
  with torch.inference_mode():result=torch.nn.functional.silu(torch.nn.functional.linear(x,gate))*torch.nn.functional.linear(x,up)
  return result,'reconstructed actual layer0 down input, original BF16 SiLU/multiply'
 return INPUTS[f'qkv_1x{m}x4096'].cuda(),'actual layer0 normalized attention input used as representative projection activation (not o_proj input)'
