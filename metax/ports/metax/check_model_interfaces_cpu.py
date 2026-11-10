"""CPU-only: actual FlagScale bridge builds parseable SpecVLA child commands."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
from omegaconf import OmegaConf
import torch
from ports.metax.gpu_queue import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    out=a.directory.resolve();bridge_path=ROOT/'ports/moore_ascend/runtime_opt/FlagScale/inference_kerv.py'
    spec=importlib.util.spec_from_file_location('cpu_flagscale_bridge',bridge_path)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    results={}
    from experiments.robot.libero.run_kerv_libero import GenerateConfig
    import draccus
    for profile in ['upstream_original','boundary_fixed','native','numerical_fixed','numerical_fixed_packed']:
        cfg=OmegaConf.load(out/('prepared_'+profile)/'metax.yaml')
        assert cfg.inference.runtime.max_episode_steps==cfg.inference.runtime.max_planning_steps==220
        assert cfg.inference.runtime.max_tasks==3 and cfg.inference.runtime.num_trials_per_task==2
        worker=out/f'{profile}-worker-cpu.yaml';OmegaConf.save(cfg.inference,worker,resolve=True)
        captured=[]
        with patch.object(sys,'argv',['inference_kerv','--config-path',str(worker)]), \
             patch.object(bridge.subprocess,'run',lambda command,**kw:captured.append(command)):
            bridge.main()
        assert len(captured)==1
        command=captured[0]
        parsed=draccus.parse(GenerateConfig,args=command[2:])
        assert parsed.max_episode_steps==parsed.max_planning_steps==220
        if profile in {'upstream_original','boundary_fixed'}:
            assert command[1]==str(ROOT/'ports/metax/specvla/eval.py')
            assert not parsed.use_kalman_fallback and not parsed.use_kalman_tree
            assert cfg.experiment.envs.KERV_METAX_V2=='' and cfg.experiment.envs.KERV_METAX_QUANT=='none'
        if profile=='numerical_fixed_packed':
            assert cfg.experiment.envs.KERV_METAX_V2=='rms_native_mean,rope,vision_graph'
            assert cfg.experiment.envs.KERV_METAX_QUANT=='bf16_lossless_selective'
        results[profile]=dict(bridge_command=command,child_arguments_parse=True,
            child_launched=False,kalman=parsed.use_kalman_fallback)
    assert not torch.cuda.is_initialized()
    (out/'interface-checks.json').write_text(json.dumps(dict(status='passed',profiles=results,
        gpu_initialized=False,actual_bridge=True,actual_child_not_launched=True),indent=2)+'\n')
    print('Five actual FlagScale bridge commands parse successfully; no child launched or GPU context initialized')
if __name__=='__main__':main()
