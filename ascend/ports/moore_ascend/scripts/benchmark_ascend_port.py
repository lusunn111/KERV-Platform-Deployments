"""Same-input test of Moore's original algorithm and its portable runtime."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.robot.device_runtime import get_device, synchronize
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from experiments.robot.flagos_ops import enable_embodied_ops, enable_tree_builder, enable_rotary_fusion
from experiments.robot.tool_utils import compute_dynamic_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rope', action='store_true')
    parser.add_argument('--profile-order', choices=['AB','BA'], default='AB')
    parser.add_argument('--input-dir', type=Path)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[3]
    input_dir = args.input_dir or project/'outputs/npu_ar_tensorflow/observations'
    files = sorted(input_dir.glob('*.npz'))[:args.samples]
    assert len(files) == args.samples
    observations = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            observations.append(({'full_image': data['image'].copy()}, str(data['task'])))
    device = get_device(); assert device.type == 'npu'
    reports = {}
    profile_order = ('native', 'runtime') if args.profile_order == 'AB' else ('runtime', 'native')
    for profile in profile_order:
        cfg = SimpleNamespace(pretrained_checkpoint=str(project/'checkpoints/openvla-libero-goal'),
            spec_checkpoint=str(project/'checkpoints/kerv-drafter'), use_spec=True,
            parallel_draft=False, accept_threshold=9, accept_threshold_start=15,
            load_in_8bit=False, load_in_4bit=False)
        torch.manual_seed(7); np.random.seed(7)
        model = get_vla(cfg).eval(); processor = get_processor(cfg)
        enabled = profile == 'runtime'
        include = 'kerv_silu_mul,kerv_add_rms_norm,kerv_verify_accept_control,kerv_static_tree_pack,kerv_tree_embed_pack,kerv_kv_commit,kerv_kv_accept_commit'
        enable_embodied_ops(model=model, enabled=True, include=include, backend='native',
            record=True, record_once=True, log_path=str(output/f'{profile}_ops.jsonl'),
            manifest_path=str(output/f'{profile}_manifest.json'))
        enable_tree_builder(model=model, enabled=enabled, static_runtime=enabled,
            verify_max_paths=0, node_buckets='', operatorized=enabled,
            draft_tree_operatorized=enabled, kalman_batch=enabled, kv_commit_batched=enabled,
            fused_control_transfer=False, compile_verifier=False, persistent_runtime=False,
            resident_kv_cache=False, static_tree_attention='off', manifest_path=None)
        enable_rotary_fusion(model=model, enabled=enabled and args.rope,
            resident_key_write=False, record=True, record_once=True,
            log_path=str(output/f'{profile}_rope.jsonl'),
            manifest_path=str(output/f'{profile}_rope_manifest.json'))
        model._flagos_system_optimization_enabled = enabled
        model._flagos_persistent_control_buffer = enabled
        model._flagos_persistent_decode_workspace = enabled
        processor._flagos_prompt_token_cache_enabled = False
        model._flagos_inference_mode_enabled = True
        model.ea_layer._flagos_persistent_draft_workspace = enabled
        model.ea_layer._flagos_persistent_draft_kv_cache = enabled
        model.enable_flagos_runtime_stats(str(output/f'{profile}_runtime.jsonl'),
            flag_gems_enabled=False, flag_gems_include='', timing_backend='cpu', async_logging=False)
        times=[]; pipelines=[]; actions=[]
        for repeat in range(args.repeats+1):
            model.start_rollout(max_steps=220, kalman_cfg={'enabled':True,'tree_enabled':True},store_history=True)
            history=[]; sequence=[]
            model._flagos_suppress_runtime_stats = repeat > 0
            for index,(obs,task) in enumerate(observations):
                synchronize(device); start=time.perf_counter()
                values,(end_model,start_model)=get_vla_action(model,processor,cfg.pretrained_checkpoint,
                    obs,task,'libero_goal',return_time=True,center_crop=True,generate_mode='speculative',
                    history=history,step_idx=index,max_steps=220,
                    dynamic_threshold=compute_dynamic_threshold(index,220,15,3),
                    use_kalman=True,kalman_tree_enabled=True)
                synchronize(device)
                if repeat:
                    pipelines.append(1000*(time.perf_counter()-start)); times.append(1000*(end_model-start_model))
                assert values.shape==(7,) and np.isfinite(values).all()
                history.append(values.copy()); sequence.append(values.tolist())
                if (index+1)%32==0:
                    print(profile,'sequence',repeat,'action',index+1,flush=True)
            if repeat: actions.append(sequence)
            print(profile,'sequence',repeat,'done',flush=True)
        assert np.array_equal(actions,np.broadcast_to(np.asarray(actions[0]),np.asarray(actions).shape))
        reports[profile]={'model_mean_ms':float(np.mean(times)), 'pipeline_mean_ms':float(np.mean(pipelines)),
            'raw_model_ms':times,'raw_pipeline_ms':pipelines,'actions':actions,
            'hits':{k:int(getattr(model,k,0)) for k in ['_flagos_tree_operator_build_hits',
                '_flagos_kalman_batch_hits','_flagos_kv_commit_batched_hits','_flagos_decode_workspace_hits']},
            'control_transfers':int(getattr(getattr(model,'_flagos_control_buffer',None),'transfers',0)),
            'draft_kv_allocations':int(getattr(model.ea_layer,'_flagos_persistent_draft_kv_allocations',0)),
            'draft_kv_fallbacks':int(getattr(model.ea_layer,'_flagos_persistent_draft_kv_fallbacks',0))}
        rope_file=output/f'{profile}_rope.jsonl'
        reports[profile]['rope_distinct_hits']=len(rope_file.read_text().splitlines()) if rope_file.exists() else 0
        if enabled and args.rope:
            assert reports[profile]['rope_distinct_hits']>0, 'No fused RoPE execution recorded'
        (output/'partial.json').write_text(json.dumps(reports,indent=2)+'\n')
        del model,processor; gc.collect(); torch.npu.empty_cache()
    a=np.asarray(reports['native']['actions']);b=np.asarray(reports['runtime']['actions'])
    report={'samples':args.samples,'repeats':args.repeats,'warmup_sequences':1,
        'profile_order':args.profile_order, 'input_dir':str(input_dir),
        'reference':'Moore source, original F.rms_norm, shared embedding, stable selection; action protocol specified separately',
        'physical_device':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'torch':torch.__version__, 'dtype':'bfloat16', 'seed':7, 'planning_horizon':220,
        'processor_token_cache':False,
        'observation_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'action_protocol':os.environ.get('KERV_ACTION_PROTOCOL','moore'),
        'drafter_rms_reference':os.environ.get('KERV_NPU_DRAFT_RMS_REFERENCE','0'),
        'graphs':False,'linear_fusion':False,'rope':args.rope,'profiles':reports,
        'actions_exact':bool(np.array_equal(a,b)), 'max_action_difference':float(np.max(np.abs(a-b))),
        'model_speedup':reports['native']['model_mean_ms']/reports['runtime']['model_mean_ms'],
        'pipeline_speedup':reports['native']['pipeline_mean_ms']/reports['runtime']['pipeline_mean_ms']}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='profiles'}),flush=True)
    assert report['actions_exact'], 'Runtime action parity failed'


if __name__=='__main__':main()
