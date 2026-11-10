"""Same-input test of Moore's original algorithm and its portable runtime."""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.robot.device_runtime import get_device, synchronize, seed_device
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
    parser.add_argument('--comparison', choices=['native_runtime','runtime_w8a16','runtime_lossless','runtime_v2','native_v2','native_runtime_v2','v2_compression','v2_prefill','v2_prefetch','numerical_fix','numerical_diagnostic'],default='native_runtime')
    parser.add_argument('--v2-features',default='compact')
    parser.add_argument('--compression-codec',default='bf16_lossless_v2',choices=['bf16_lossless_v2','bf16_lossless_prefetch','bf16_lossless_selective'])
    parser.add_argument('--allow-action-change',action='store_true',help='Record numeric candidate differences; never claim exact trajectory parity')
    parser.add_argument('--frozen-history',action='store_true',help='Replay raw action/token history from fixture; hold KF inputs fixed when numeric candidates differ')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[2]
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=project,text=True).splitlines()
    sources=sorted({name for name in names if (project/name).is_file() and Path(name).suffix in {'.py','.cpp','.h','.yaml','.yml','.sh','.json','.toml','.txt'}})
    (output/'source-state.json').write_text(json.dumps(dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),files={name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in sources}),indent=2)+'\n')
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for name in sources:archive.add(project/name,arcname=name,recursive=False)
    input_dir = args.input_dir
    if input_dir is None: raise SystemExit('--input-dir with real captured observations is required')
    files = sorted(input_dir.glob('*.npz'))[:args.samples]
    assert len(files) == args.samples
    frozen=[];history_path=input_dir.parent/'raw/steps.jsonl'
    if args.frozen_history:
        frozen=[json.loads(line) for line in history_path.read_text().splitlines()][:args.samples]
        assert len(frozen)==args.samples and len({x['episode_id'] for x in frozen})==1
        assert [x['step_id'] for x in frozen]==list(range(args.samples))
    observations = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            observations.append(({'full_image': data['image'].copy()}, str(data['task'])))
    device = get_device(); assert device.type == 'cuda' and getattr(torch.version, 'maca', None)
    reports = {}
    reference_profile,test_profile={'native_runtime':('native','runtime'),'runtime_w8a16':('runtime','quantized'),'runtime_lossless':('runtime','lossless'),'runtime_v2':('runtime','v2'),'native_v2':('native','v2'),'native_runtime_v2':('native','v2'),'v2_compression':('v2','v2_lossless_new'),'v2_prefill':('v2_base','v2'),'v2_prefetch':('v2','v2_prefetch'),'numerical_fix':('native','v2'),'numerical_diagnostic':('v2','v2')}[args.comparison]
    order=('native','runtime','v2') if args.comparison=='native_runtime_v2' else (('v2','v2_lossless_old',test_profile) if args.comparison in {'v2_compression','v2_prefetch'} else (reference_profile,test_profile))
    if args.comparison=='numerical_fix':order=('native','runtime','old_v2','v2')
    if args.comparison=='numerical_diagnostic':order=('v2',)
    profile_order=order if args.profile_order=='AB' else order[::-1]
    for profile in profile_order:
        os.environ['KERV_METAX_V2']=args.v2_features if profile.startswith('v2') else ''
        if profile=='old_v2':os.environ['KERV_METAX_V2']='rms,rope,vision_graph'
        if profile=='v2_base':os.environ['KERV_METAX_V2']=','.join(x for x in args.v2_features.split(',') if x!='prefill_graph')
        cfg = SimpleNamespace(pretrained_checkpoint=str(project/'checkpoints/openvla-libero-goal'),
            spec_checkpoint=str(project/'checkpoints/kerv-drafter'), use_spec=True,
            parallel_draft=False, accept_threshold=9, accept_threshold_start=15,
            load_in_8bit=False, load_in_4bit=False)
        os.environ['KERV_METAX_QUANT']={'quantized':'w8a16_group32','lossless':'bf16_lossless','v2_lossless_old':'bf16_lossless','v2_lossless_new':args.compression_codec,'v2_prefetch':'bf16_lossless_prefetch'}.get(profile,'none')
        pre_load_allocated=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(7); seed_device(7); np.random.seed(7)
        model = get_vla(cfg).eval(); processor = get_processor(cfg)
        enabled = profile != 'native'
        include = 'kerv_silu_mul,kerv_add_rms_norm,kerv_verify_accept_control,kerv_static_tree_pack,kerv_static_tree_attention,kerv_action_projection_select,kerv_kv_commit,kerv_tree_embed_pack,kerv_rope_kv_store,kerv_action_verify_accept,kerv_draft_action_topk,kerv_kv_accept_commit,kerv_vision_add_layer_norm,kerv_vision_bias_gelu'
        enable_embodied_ops(model=model, enabled=True, include=include, backend='native',
            record=False, record_once=True, log_path=str(output/f'{profile}_ops.jsonl'),
            manifest_path=str(output/f'{profile}_manifest.json'))
        enable_tree_builder(model=model, enabled=enabled, static_runtime=enabled,
            verify_max_paths=0, node_buckets='', operatorized=enabled,
            draft_tree_operatorized=enabled, kalman_batch=enabled, kv_commit_batched=enabled,
            fused_control_transfer=enabled, compile_verifier=False, persistent_runtime=False,
            resident_kv_cache=False, static_tree_attention='off', manifest_path=None)
        enable_rotary_fusion(model=model, enabled=enabled and args.rope,
            resident_key_write=False, record=False, record_once=True,
            log_path=str(output/f'{profile}_rope.jsonl'),
            manifest_path=str(output/f'{profile}_rope_manifest.json'))
        model._flagos_system_optimization_enabled = enabled
        model._flagos_persistent_control_buffer = enabled
        model._flagos_persistent_decode_workspace = enabled
        processor._flagos_prompt_token_cache_enabled = enabled
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
                if args.frozen_history:
                    model._token_history=[row['action_tokens'].copy() for row in frozen[:index]]
                    history=[np.asarray(row['raw_action']) for row in frozen[:index]]
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
                if args.frozen_history and (profile in {'native','runtime'} or (profile=='v2' and args.comparison=='numerical_fix')):
                    assert model.get_history()[-1]==frozen[index]['action_tokens'],'Frozen baseline token history does not match original algorithm'
                history.append(values.copy()); sequence.append(values.tolist())
                if (index+1)%32==0:
                    print(profile,'sequence',repeat,'action',index+1,flush=True)
            if repeat: actions.append(sequence)
            print(profile,'sequence',repeat,'done',flush=True)
        assert np.array_equal(actions,np.broadcast_to(np.asarray(actions[0]),np.asarray(actions).shape))
        reports[profile]={'model_mean_ms':float(np.mean(times)), 'pipeline_mean_ms':float(np.mean(pipelines)),
            'raw_model_ms':times,'raw_pipeline_ms':pipelines,'actions':actions,
            'allocated_peak_bytes':torch.cuda.max_memory_allocated(),
            'pre_load_allocated_bytes':pre_load_allocated,
            'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
            'quantization':getattr(model,'_metax_quant_manifest',None),
            'v2_features':getattr(model,'_metax_v2_features',[]),'v2_hits':getattr(model,'_metax_v2_stats',{}),
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
        from ports.metax.v2 import release
        release(model)
        reports[profile]['restore_audit']=getattr(model,'_metax_restore_audit',None)
        del model,processor; gc.collect();torch.cuda.synchronize()
        clear_blas=getattr(torch._C,'_cuda_clearCublasWorkspaces',None)
        if clear_blas is not None:clear_blas()
        torch.cuda.empty_cache()
        reports[profile]['post_release_allocated_bytes']=torch.cuda.memory_allocated()
        reports[profile]['cleared_process_blas_workspaces']=clear_blas is not None
    a=np.asarray(reports[reference_profile]['actions']);b=np.asarray(reports[test_profile]['actions'])
    report={'samples':args.samples,'repeats':args.repeats,'warmup_sequences':1,
        'profile_order':args.profile_order,'comparison':args.comparison, 'input_dir':str(input_dir),
        'history_mode':'frozen reference raw actions and seven token prefix' if args.frozen_history else 'profile own generated actions/tokens; equal observations may have differing KF history',
        'frozen_history_sha256':hashlib.sha256(history_path.read_bytes()).hexdigest() if args.frozen_history else None,
        'reference':('Selected v2 BF16 vs old/new lossless weight storage' if args.comparison=='v2_compression' else ('Native/round-one/v2 BF16, unchanged seven protocol' if 'v2' in reports else 'MetaX BF16 runtime or storage comparison')),
        'physical_device':os.environ.get('MACA_VISIBLE_DEVICES'),
        'torch':torch.__version__, 'dtype':'bfloat16','weight_precision':('mixed W8A16/BF16' if args.comparison=='runtime_w8a16' else 'BF16 weights; runtime numeric parity measured separately'), 'seed':7, 'planning_horizon':220,
        'processor_token_cache':'optimized only','cudnn_deterministic':torch.backends.cudnn.deterministic,
        'observation_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'action_protocol':os.environ.get('KERV_ACTION_PROTOCOL','moore'),
        'drafter_rms_reference':os.environ.get('KERV_NPU_DRAFT_RMS_REFERENCE','0'),
        'graphs':('vision_graph' in args.v2_features and 'v2' in reports),'linear_fusion':bool({'qkv','gate_up'} & set(args.v2_features.split(',')) and 'v2' in reports),'rope':args.rope,'v2_rope':('rope' in args.v2_features.split(',') and 'v2' in reports),'profiles':reports,
        'actions_exact':bool(np.array_equal(a,b)), 'max_action_difference':float(np.max(np.abs(a-b))),
        'model_speedup':reports[reference_profile]['model_mean_ms']/reports[test_profile]['model_mean_ms'],
        'pipeline_speedup':reports[reference_profile]['pipeline_mean_ms']/reports[test_profile]['pipeline_mean_ms']}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'profiles','observation_sha256'}}),flush=True)
    if not args.allow_action_change and args.comparison in {'native_runtime','runtime_lossless','runtime_v2','native_v2','native_runtime_v2','v2_prefill','v2_compression','v2_prefetch','numerical_fix'}:assert report['actions_exact'], 'Runtime action parity failed'


if __name__=='__main__':main()
