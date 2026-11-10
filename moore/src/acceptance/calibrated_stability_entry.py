"""Targeted fixed-input diagnosis. Instrumented results are not speed benchmarks."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from acceptance.common import dump, rows
from experiments.robot.libero.run_calibrated_kernel_libero import register
register()
import acceptance.replay_entry as replay


def diagnostic_worker(cfg, model, processor, profile):
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, set_seed_everywhere
    from experiments.robot.tool_utils import compute_dynamic_threshold
    root = Path(os.environ['KERV_REPLAY_OUTPUT'])
    if os.getenv('KERV_STABILITY_SAFE_SOFTMAX','off')=='on':
        from KERVRuntimeOptimization.musa_safe_softmax import patch_llama_attention_softmax
        dump(root/'safe_softmax_patch.json',{'modules':patch_llama_attention_softmax(model)})
    capture = Path(os.environ['KERV_REPLAY_CAPTURE'])
    steps = rows(capture/'steps.jsonl')
    stats = rows(capture/'runtime_stats.jsonl')
    observations = np.load(capture/'observations.npz')
    selected = [int(x) for x in os.getenv('KERV_STABILITY_FRAMES', '61,122').split(',')]
    repeats = int(os.getenv('KERV_STABILITY_REPEATS', '20'))
    tracing = os.getenv('KERV_STABILITY_TRACE', 'on') == 'on'
    attention_probe = os.getenv('KERV_STABILITY_ATTENTION_PROBE', 'off') == 'on'
    processor._flagos_prompt_token_cache_enabled = False
    processor._flagos_pinned_image_buffer_enabled = False
    model._flagos_suppress_runtime_stats = True
    trace = []
    current = {}
    progress = (root/'operations.jsonl').open('a', buffering=1)
    def note(operation, **extra):
        progress.write(json.dumps({**current, 'operation': operation, **extra})+'\n')
    def tensor(label, value):
        if tracing and isinstance(value, torch.Tensor):
            trace.append((label, value.detach().clone()))
    handles = []
    attention_stack = []
    dump(root/'module_types.json', {name:type(module).__module__+'.'+type(module).__name__
         for name,module in model.named_modules() if name.endswith(('self_attn','mlp','input_layernorm'))})
    for name, module in model.named_modules():
        if attention_probe and '.language_model.model.layers.' in name and name.endswith('self_attn'):
            def attention_pre(owner, args, name=name):
                attention_stack.append(name)
            def attention_post(owner, args, output):
                attention_stack.pop()
            handles.append(module.register_forward_pre_hook(attention_pre))
            handles.append(module.register_forward_hook(attention_post))
        if isinstance(module, torch.nn.Embedding):
            def embedding_pre(owner, args, name=name):
                ids = args[0]
                note('embedding', name=name, shape=list(ids.shape), stride=list(ids.stride()),
                     index_ptr=ids.data_ptr(), weight_ptr=owner.weight.data_ptr(), vocab=owner.num_embeddings)
                tensor(name+'.indices', ids)
            handles.append(module.register_forward_pre_hook(embedding_pre))
        if name.endswith('lm_head') or ('.language_model.model.layers.' in name and
                (name.rsplit('.',1)[-1].isdigit() or name.endswith(('self_attn','mlp','input_layernorm','post_attention_layernorm')))):
            def output_hook(owner, args, output, name=name):
                tensor(name+'.output', output[0] if isinstance(output, tuple) else output)
            handles.append(module.register_forward_hook(output_hook))
    native_topk = torch.topk
    native_sdpa = torch.nn.functional.scaled_dot_product_attention
    native_matmul = torch.matmul
    native_softmax = torch.nn.functional.softmax
    def matmul(left, right, *args, **kwargs):
        output = native_matmul(left, right, *args, **kwargs)
        if attention_stack:
            tensor(attention_stack[-1]+'.matmul.left', left)
            tensor(attention_stack[-1]+'.matmul.right', right)
            tensor(attention_stack[-1]+'.matmul.output', output)
        return output
    def softmax(value, *args, **kwargs):
        output = native_softmax(value, *args, **kwargs)
        if attention_stack:
            tensor(attention_stack[-1]+'.softmax.input', value)
            tensor(attention_stack[-1]+'.softmax.output', output)
        return output
    def sdpa(*args, **kwargs):
        output = native_sdpa(*args, **kwargs)
        tensor('sdpa.output', output)
        return output
    def topk(value, *args, **kwargs):
        result = native_topk(value, *args, **kwargs)
        tensor('topk.input', value)
        tensor('topk.indices', result.indices)
        tensor('topk.values', result.values)
        return result
    if tracing:
        torch.topk = topk
        torch.nn.functional.scaled_dot_product_attention = sdpa
        if attention_probe:
            torch.matmul = matmul
            torch.nn.functional.softmax = softmax
    reference = {}
    results = []
    try:
        for repeat in range(repeats):
            set_seed_everywhere(int(cfg.seed))
            model.start_rollout(max_steps=220, kalman_cfg={'enabled':True, 'tree_enabled':True,
                                'process_var':1., 'measurement_var':.001}, store_history=True)
            for frame in selected:
                current.update(repeat=repeat, frame=frame)
                model._acceptance_replay_frame = frame
                trace.clear()
                obs = {key:np.array(observations[key][frame],copy=True) for key in observations.files}
                model._token_history = [list(x['kalman_history_token_ids']) for x in stats[:frame]]
                history = [np.array(x['raw_action']) for x in steps[:frame]]
                observation = {'full_image':get_libero_image(obs,224), 'state':np.concatenate((
                    obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))}
                note('predict_begin')
                action, ts = get_action(cfg,model,observation,'open the middle drawer of the cabinet',
                    processor=processor,return_time=True,generate_mode='speculative',history=history,
                    step_idx=frame,max_steps=220,dynamic_threshold=compute_dynamic_threshold(frame,220,15,3),
                    use_kalman=True,kalman_process_var=1.,kalman_measurement_var=.001,
                    kalman_history_window=None,kalman_tree_enabled=True)
                tokens = list(model._last_action_token_ids_cpu)
                cpu_trace = [(name,value.cpu()) for name,value in trace]
                for name,value in cpu_trace:
                    if name.endswith('.indices') and name!='topk.indices':
                        if value.numel() and (int(value.min())<0 or int(value.max())>=32064):
                            torch.save({'name':name,'indices':value},root/'invalid_indices.pt')
                            raise RuntimeError('Invalid embedding indices captured without pre-launch CPU synchronization')
                note('predict_end', tokens=tokens, trace_tensors=len(cpu_trace))
                row = {**current, 'tokens':tokens, 'raw_action':np.asarray(action).tolist()}
                if frame not in reference:
                    reference[frame] = (tokens, cpu_trace)
                    torch.save([(name,value) for name,value in cpu_trace if name.startswith('topk.')],
                               root/('topk_reference_f%d.pt'%frame))
                    row['reference'] = True
                else:
                    previous_tokens, previous_trace = reference[frame]
                    row['tokens_exact'] = tokens == previous_tokens
                    row['first_difference'] = None
                    for index, ((name, old),(other, new)) in enumerate(zip(previous_trace,cpu_trace)):
                        if name != other or old.shape != new.shape or not torch.equal(old,new):
                            detail={'index':index,'name':name,'other_name':other,
                                    'old_shape':list(old.shape),'new_shape':list(new.shape)}
                            if old.shape==new.shape:
                                detail['unequal_elements']=int((old!=new).sum())
                                detail['max_abs_error']=float((old.float()-new.float()).abs().max())
                            row['first_difference']=detail
                            # Keep the first differing tensor and its immediate predecessor.
                            artifact=root/('difference_r%d_f%d.pt'%(repeat,frame))
                            torch.save({'difference':detail,'old':previous_trace[max(0,index-1):index+1],
                                        'new':cpu_trace[max(0,index-1):index+1]},artifact)
                            break
                    row['trace_lengths']=[len(previous_trace),len(cpu_trace)]
                results.append(row)
                with (root/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('[stability]',json.dumps(row),flush=True)
        dump(root/'summary.json', {'scope':'instrumented selected-frame diagnostic', 'frames':selected,
            'repeats':repeats,'tracing':tracing,'results':results})
        dump(root/'ready.json', {'pid':os.getpid()})
        while not (root/'request_0.json').exists():time.sleep(.1)
        dump(root/'done_0.json', {'completed':True})
    finally:
        torch.topk = native_topk
        torch.nn.functional.scaled_dot_product_attention = native_sdpa
        torch.matmul = native_matmul
        torch.nn.functional.softmax = native_softmax
        for handle in handles:handle.remove()
        progress.close()


if __name__=='__main__':
    replay.worker = diagnostic_worker
    replay.main()
