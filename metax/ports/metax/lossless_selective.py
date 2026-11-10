"""Minimum-size compression policy, original lossless codec/vendor BF16 GEMM."""
from torch import nn
from ports.metax.lossless import install

def apply_cpu_selective(model):
 selected=[]
 for name,module in model.named_modules():
  if not name.startswith('base_model.language_model.model.layers.') or '.mlp.' not in name or not isinstance(module,nn.Linear):continue
  if name in {'base_model.language_model.model.layers.0.mlp.gate_proj','base_model.language_model.model.layers.0.mlp.up_proj'}:continue
  selected.append((name,module))
 assert len(selected)==94
 manifest=install(model,selected)
 manifest.update(profile='metax_bf16_lossless_mlp94',policy='94 MLP matrices; attention native; layer0 gate/up native',execution='Original exact BF16 bit restoration + vendor BF16 GEMM; reduced compression coverage')
 return manifest
