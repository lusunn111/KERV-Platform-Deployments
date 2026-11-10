"""Real checkpoint load/interface + representative CPU codec audit; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace
import numpy as np
import torch
from ports.metax.specvla.adapter import build_cpu_specvla
from ports.metax.lossless import encode

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    cfg=SimpleNamespace(pretrained_checkpoint=str(root/'checkpoints/openvla-libero-goal'),
        spec_checkpoint=str(root/'checkpoints/kerv-drafter'), accept_threshold=9)
    start=time.perf_counter();model,spec=build_cpu_specvla(cfg,'upstream_original')
    assert all(x.device.type=='cpu' for x in model.parameters())
    assert model.ea_layer.embed_tokens is model.base_model.language_model.model.embed_tokens
    assert model.ea_layer.fc.in_features==8192 and model.ea_layer.fc.out_features==4096
    assert len(model.base_model.language_model.model.layers)==32
    assert model.get_action_dim('libero_goal')==7
    model.ea_layer.reset_kv();assert model.ea_layer.stable_kv is None
    bits=[]
    for name in ['base_model.language_model.model.layers.0.mlp.down_proj',
                 'base_model.language_model.model.layers.31.mlp.gate_proj']:
        weight=model.get_submodule(name).weight
        encoding=encode(weight)
        low=encoding['low'].numpy().astype(np.uint16)
        high=encoding['high'].numpy()
        codes=np.empty(low.size,dtype=np.uint8)
        codes[::2]=high&15;codes[1::2]=high>>4
        restored=low|(encoding['dictionary'].numpy()[codes].astype(np.uint16)<<8)
        restored[encoding['indices'].numpy()]=encoding['values'].numpy().view(np.uint16)
        original=weight.detach().view(torch.int16).numpy().reshape(-1).view(np.uint16)
        assert np.array_equal(original,restored),name
        bits.append(dict(module=name,shape=list(weight.shape),elements=weight.numel(),
            original_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
            restored_sha256=hashlib.sha256(restored.tobytes()).hexdigest(),
            bit_differences=0,encoded_bytes=sum(v.numel()*v.element_size() for v in encoding.values())))
    assert not torch.cuda.is_initialized()
    report=dict(kind='CPU real weight loading and codec reconstruction only; no GPU execution/quality/speed claim',
        status='passed',load_keys='upstream drafter load_state_dict strict=True completed',
        verifier_layers=32,drafter_hidden_size=4096,action_dim=7,
        shared_embedding=True,kv_reset=True,all_parameters_cpu=True,
        total_unique_parameter_bytes=sum(x.numel()*x.element_size() for x in model.parameters()),
        wall_seconds=time.perf_counter()-start,cpu_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        codec_checks=bits,gpu_initialized=False,
        tree=dict(total_tokens=model.ea_layer.total_tokens,depth=model.ea_layer.depth,top_k=model.ea_layer.top_k))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
