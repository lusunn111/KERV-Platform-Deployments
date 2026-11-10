"""Separate Torch allocator counters from device memory at QKV lifecycle stages."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch_npu
from diagnose_quantization import configure
from experiments.robot.openvla_utils import get_vla,get_processor,get_vla_action
from experiments.robot.flagos_ops import enable_linear_fusion


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-pack',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];stages=[]
    def record(name):
        torch.npu.synchronize();free,total=torch.npu.mem_get_info()
        row=dict(stage=name,allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved(),
                 max_allocated=torch.npu.max_memory_allocated(),max_reserved=torch.npu.max_memory_reserved(),
                 device_used=total-free,device_total=total)
        stages.append(row);(a.output/'stages.json').write_text(json.dumps(stages,indent=2)+'\n');print(row,flush=True)
    torch.manual_seed(7);np.random.seed(7)
    (a.output/'environment.json').write_text(json.dumps(dict(physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        allocator=os.environ.get('PYTORCH_NPU_ALLOC_CONF')),indent=2)+'\n')
    record('before_model')
    cfg=SimpleNamespace(pretrained_checkpoint=str(root/'checkpoints/openvla-libero-goal'),
        spec_checkpoint=str(root/'checkpoints/kerv-drafter'),use_spec=True,parallel_draft=False,
        accept_threshold=9,accept_threshold_start=15,load_in_8bit=False,load_in_4bit=False)
    if a.cpu_pack:
        cfg.use_flagos_linear_fusion=True;cfg.flagos_qkv_fusion_rows=','.join(map(str,range(1,513)))
    model=get_vla(cfg).eval();configure(model,a.output,'memory');record('loaded_cpu_prepacked_qkv' if a.cpu_pack else 'loaded_without_qkv')
    manifest=enable_linear_fusion(model=model.base_model.language_model,enabled=True,
        qkv_rows=','.join(map(str,range(1,513))),qkv_input_sizes='4096',gate_up_rows='',gate_up_input_sizes='',
        swiglu_rows='',swiglu_input_sizes='',swiglu_backend='native_inplace',add_rms_norm_rows='',
        add_rms_norm_input_sizes='',add_rms_norm_backend='native_inplace',record=False,record_once=True,
        log_path=None,manifest_path=str(a.output/'qkv_manifest.json'))
    assert manifest['qkv_group_count']==32
    record('packed_before_forward')
    torch.npu.empty_cache();record('packed_after_empty_cache')
    processor=get_processor(cfg)
    f=sorted((root/'outputs/npu_ar_tensorflow/observations').glob('*.npz'))[0]
    with np.load(f,allow_pickle=False) as d:obs={'full_image':d['image'].copy()};task=str(d['task'])
    model.start_rollout(max_steps=220,kalman_cfg={'enabled':True,'tree_enabled':True},store_history=True)
    get_vla_action(model,processor,cfg.pretrained_checkpoint,obs,task,'libero_goal',return_time=True,
        center_crop=True,generate_mode='speculative',history=[],step_idx=0,max_steps=220,
        dynamic_threshold=15,use_kalman=True,kalman_tree_enabled=True)
    record('after_first_action')


if __name__=='__main__':main()
