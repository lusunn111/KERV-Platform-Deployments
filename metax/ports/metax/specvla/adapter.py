"""Observation/timing adapter; the loaded upstream algorithm owns generation."""
import json
import os
from pathlib import Path
import torch
from .loader import load, verify_original

def install_observer(model, speculation, variant):
    original_update = speculation.update_inference_inputs
    def update(*args, **kwargs):
        # Signature is pinned and asserted by the CPU semantic check.
        prompt, ids, input_len, candidates, best, accepted = args[:6]
        before = ids.shape[1] - input_len - 1
        result = original_update(*args, **kwargs)
        after = result[0].shape[1] - input_len - 1
        eos = int(result[0][0, -1]) == model.tokenizer.eos_token_id
        appended = after - before - int(eos)
        selected = candidates[int(best)].detach().cpu().tolist()
        model._spec_events.append(dict(proposed=int(candidates.shape[1])-1,
            accepted=int(accepted), appended=appended, before=before,
            candidate=selected, shape=list(candidates.shape), eos=eos))
        return result
    if getattr(speculation, '_observer_installed', False):
        raise RuntimeError('Load a fresh process/model for each SpecVLA profile')
    speculation.update_inference_inputs = update
    speculation._observer_installed = True
    native_generate = model.eagenerate
    def generate(*args, **kwargs):
        model._spec_events = []
        output = native_generate(*args, **kwargs)
        if not isinstance(output, torch.Tensor) or output.shape[0] != 1:
            raise RuntimeError('SpecVLA action runner requires the upstream tensor return')
        full = output[0].detach().cpu().tolist()
        model._last_action_token_ids_cpu = full[-7:]
        model._last_specvla_generated = full
        recorder = getattr(model, '_acceptance_recorder', None)
        if recorder is not None:
            begin = max(len(full)-7, 0); end = len(full)
            for event in model._spec_events:
                recorder.verify(event['proposed'], event['accepted'], model.accept_threshold, False,
                    event['shape'], candidate_tokens=event['candidate'], action_tokens_before=event['before'])
                row = recorder.buffers['verify'][-1]
                root = int(begin <= event['before'] < end)
                candidate_start = event['before'] + 1
                candidate_end = event['before'] + event['appended']
                kept = max(0, min(candidate_end, end)-max(candidate_start, begin))
                row.update(committed_prefix_length=kept, root_length=root, tree_root_length=1,
                    initial_action_root_length=root*int(event['before']==0),
                    committed_verifier_bonus_length=root*int(event['before']>0),
                    compensation_length=0, padding_length=0,
                    discarded_root_length=1-root,
                    discarded_candidate_length=event['appended']-1-kept,
                    raw_committed_candidate_length=event['appended']-1,
                    generated_tokens_length=len(full), decoded_tail_start=begin,
                    retained_candidate_is_prefix=begin <= candidate_start or kept==0,
                    upstream_variant=variant, generated_action_tokens=full,
                    note='True upstream event, after end predicate and original last-seven decoding. Retained draft tokens exclude initial/root/bonus, KF and padding. Discarded tokens recorded separately.')
        return output
    model.eagenerate = generate

def build_cpu_specvla(cfg, variant):
    """Original checkpoint constructors; separate CPU load compatibility audit."""
    prismatic, speculation, _ = load(variant)
    base = prismatic.OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_spec=True, trust_remote_code=False,
        attn_implementation='eager')
    # Upstream comparison uses the upstream fixed relaxation, not KERV's dynamic
    # schedule or Kalman completion. 9 is the pinned upstream evaluation default.
    model = speculation.SpecVLAforActionPrediction(base, cfg.pretrained_checkpoint,
        cfg.spec_checkpoint, accept_threshold=cfg.accept_threshold, parallel_draft=False)
    return model.eval(), speculation

def get_specvla(cfg, variant):
    if not getattr(torch.version, 'maca', None) or not torch.cuda.is_available():
        raise RuntimeError('Real SpecVLA deployment requires vendor MACA GPU')
    if os.environ.get('KERV_METAX_QUANT', 'none') != 'none' or os.environ.get('KERV_METAX_V2', ''):
        raise RuntimeError('SpecVLA baseline must not inherit KERV numerical/compression features')
    model, speculation = build_cpu_specvla(cfg, variant)
    model = model.to('cuda:0').eval()
    stats = Path(cfg.pretrained_checkpoint) / 'dataset_statistics.json'
    model.norm_stats = json.loads(stats.read_text())
    # Bound get_action_stats reads base.norm_stats, as in the upstream constructor.
    model.base_model.norm_stats = model.norm_stats
    model._flagos_inference_mode_enabled = True
    model._last_action_token_ids_cpu = []
    install_observer(model, speculation, variant)
    provenance = verify_original()
    manifest = dict(variant=variant, upstream_commit=provenance['upstream_commit'],
        upstream_files=provenance['files'], model_class=type(model).__module__,
        verifier_class=type(model.base_model.language_model).__module__,
        drafter_class=type(model.ea_layer).__module__,
        tree=dict(total_tokens=model.ea_layer.total_tokens, depth=model.ea_layer.depth,
                  top_k=model.ea_layer.top_k),
        acceptance='upstream fixed absolute-token-distance threshold', threshold=model.accept_threshold,
        kalman=False, dynamic_threshold=False, action_decode='upstream last-seven tokens',
        boundary_change=None if variant=='upstream_original' else '>6 -> >=6, one predicate only',
        embedding_shared=model.ea_layer.embed_tokens is model.base_model.language_model.model.embed_tokens,
        weight_paths=dict(verifier=cfg.pretrained_checkpoint, drafter=cfg.spec_checkpoint),
        compatibility='isolated absolute import routing; torch.load map_location=cpu for CUDA-serialized draft checkpoint; vendor CUDA API; no math substitutions')
    Path(os.environ['KERV_OUTPUT_ROOT'], 'specvla-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return model
