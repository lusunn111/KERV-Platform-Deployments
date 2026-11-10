"""Derive fixed mixed-precision candidates from frozen init0 calibration."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sensitivity',type=Path,required=True)
    p.add_argument('--scheme',choices=['mixed','weight_only'],required=True)
    p.add_argument('--silu-backend',choices=['native','triton_rounded'],default='native')
    p.add_argument('--preserve-down-layers',type=int,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    policy=torch.load(a.source,map_location='cpu',weights_only=True)
    sensitivity=json.loads(a.sensitivity.read_text())
    # Calibration schema is a list of per-module measurements.
    records=sensitivity['modules'] if isinstance(sensitivity,dict) else sensitivity
    records={r['name']:r for r in records}
    preserved=set()
    for item in policy['modules']:
        weight_only=a.scheme=='weight_only' or item['name'].endswith('down_proj')
        item['precision']='weight_only_int8' if weight_only else 'dynamic_int8'
        if weight_only:
            if records[item['name']]['weight_sha256'] != item['weight_sha256']:
                raise ValueError('Calibration weight identity mismatch')
            if records[item['name']]['best_g32']['clip_ratio'] != 1.0:
                raise ValueError('This exporter implements unclipped group32 weights')
            item.update(group_size=32,alpha=None,
                        relative_l2=records[item['name']]['best_g32']['relative_l2'])
        layer=int(item['name'].split('.layers.')[1].split('.')[0])
        if item['name'].endswith('down_proj') and layer in a.preserve_down_layers:
            item['precision']='bf16'
            preserved.add(layer)
    if preserved != set(a.preserve_down_layers):
        raise ValueError('Requested BF16 layers are outside the calibrated policy')
    policy.update(fused_gate_up=True,direct_mlp=True,fused_swiglu_quant=True,
                  silu_backend=a.silu_backend,mixed_precision_scheme=a.scheme,
                  preserved_bf16_down_layers=sorted(preserved),
                  calibration_source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                  sensitivity_sha256=hashlib.sha256(a.sensitivity.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.save(policy,a.output)
    print(json.dumps(dict(output=str(a.output),sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                          scheme=a.scheme,silu_backend=a.silu_backend),indent=2))


if __name__=='__main__':
    main()
