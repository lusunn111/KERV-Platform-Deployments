"""CPU-only upstream semantics, metric and queue safety checks. No deployment claim."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from ports.metax.gpu_queue import parse_device, idle
from ports.metax.specvla.loader import load, adapted_source, verify_original, PREDICATE, FIXED
from ports.metax.specvla.adapter import install_observer
from acceptance.collect_metrics import Recorder


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not torch.cuda.is_initialized(), 'CPU test started with a GPU context'
    provenance=verify_original(); result={'kind':'CPU checks, not GPU/model quality or performance',
        'upstream_commit':provenance['upstream_commit']}
    modules={variant:load(variant) for variant in ['upstream_original','boundary_fixed']}
    for variant,(prismatic,spec,utils) in modules.items():
        prefix=f'_kerv_specvla_{variant}'
        assert spec.MMModel.__module__.startswith(prefix)
        assert spec.LlamaSpecForCausalLM.__module__.startswith(prefix)
        assert prismatic.LlamaSpecForCausalLM is spec.LlamaSpecForCausalLM
        assert spec.update_inference_inputs is utils.update_inference_inputs
    original=adapted_source('openvla/prismatic/extern/hf/utils.py','upstream_original')
    fixed=adapted_source('openvla/prismatic/extern/hf/utils.py','boundary_fixed')
    assert original.replace(PREDICATE,FIXED)==fixed
    # The only AST algorithm difference is the comparison operator.
    af=ast.parse(fixed); ao=ast.parse(original)
    function=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update_inference_inputs')
    assert ast.dump(function(ao)).replace('Gt()', 'GtE()',1)==ast.dump(function(af))
    traces={}; checks=0
    def exercise(fn, first_accept, observed=False):
        embedding=torch.nn.Embedding(64,2)
        model=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2),
            ea_layer=SimpleNamespace(embed_tokens=embedding,topK_genrate=lambda *args:(None,None,None,None)),
            base_model=SimpleNamespace(language_model=SimpleNamespace(lm_head=None)),
            accept_threshold=9)
        fake_spec=SimpleNamespace(update_inference_inputs=fn)
        events=[]
        def generation():
            ids=torch.tensor([[1,3]]);prompt=torch.zeros(1,2,2);new=0
            for i in range(12):
                candidates=(torch.arange(8)+8+i*3)[None,:]
                accepted=torch.tensor(first_accept if i==0 else 0)
                before=ids.shape[1]-2
                sample=torch.zeros(64);sample[20+i]=1
                kv=[(torch.arange(10)[None,None,:,None].float(), torch.arange(10)[None,None,:,None].float()+100)]
                # Cache selection addresses start at prompt length, including root.
                idx=torch.arange(8)[None,:]
                out=fake_spec.update_inference_inputs(prompt,ids,1,candidates,0,accepted,idx,None,new,
                    kv,model,torch.zeros(1,8,2),sample,None)
                ids,new,prompt=out[0],out[5],out[6]
                eos=int(ids[0,-1])==2
                appended=ids.shape[1]-2-before-int(eos)
                if out[7] is not None:
                    assert out[7][0].shape[-2]==2+before+appended
                events.append(dict(before=before,appended=appended,eos=eos))
                if eos or new>7:
                    break
            else:
                raise AssertionError('Upstream generation did not stop')
            seq=ids[0,2:];seq=seq[seq!=2]
            return seq[None,:]
        model.eagenerate=generation
        recorder=None
        if observed:
            recorder=object.__new__(Recorder);recorder.buffers={'verify':[]}
            recorder.run_id='cpu';recorder.episode_id='unit';recorder.step_id=0;recorder.verify_id=0
            model._acceptance_recorder=recorder
            install_observer(model,fake_spec,'unit')
        generated=model.eagenerate()
        return generated, events, recorder
    for name,(_,spec,utils) in modules.items():
        variants=[]
        for accepted in range(8):
            plain,events,_=exercise(utils.update_inference_inputs,accepted)
            observed,_,recorder=exercise(utils.update_inference_inputs,accepted,True)
            assert torch.equal(plain,observed), 'Observer changed upstream output'
            assert sum(row['root_length']+row['committed_prefix_length'] for row in recorder.buffers['verify'])==7
            assert all(row['compensation_length']==0 and row['padding_length']==0 for row in recorder.buffers['verify'])
            assert all(row['retained_candidate_is_prefix'] for row in recorder.buffers['verify'])
            if name=='boundary_fixed':assert plain.shape[1]==7
            if accepted==6:
                assert plain.shape[1]==(8 if name=='upstream_original' else 7)
            variants.append(dict(first_accepted=accepted,generated=plain.tolist(),events=events,recorded=recorder.buffers['verify']))
            checks+=1
        traces[name]=variants
    # Full vendor output parse, including no-process and failure-to-parse cases.
    raw='''| Process: |\n| Usage(MiB) |\n| 0 1063812 python 55154 |\nGPU#0  MXC550\n    vram used : 57371148 KB\n'''
    busy=parse_device(raw);assert busy['pids']==[1063812] and not idle(busy,950*1024)
    quiet=raw.replace('| 0 1063812 python 55154 |\n','').replace('57371148','892928')
    assert idle(parse_device(quiet),950*1024)
    assert not idle(parse_device(quiet.replace('892928','1000000')),950*1024)
    for bad in ['',quiet.replace('vram used','broken'),quiet.replace('| Usage(MiB) |',''),raw.replace('55154 |','unknown |')]:
        assert not idle(parse_device(bad),950*1024)
    base=Path(__file__).resolve().parent
    repair=OmegaConf.load(base/'numerical_fixed.yaml')
    packed=OmegaConf.merge(repair,OmegaConf.load(base/'numerical_fixed_packed.yaml'))
    assert repair.experiment.envs.KERV_METAX_V2==packed.experiment.envs.KERV_METAX_V2=='rms_native_mean,rope,vision_graph'
    assert repair.experiment.envs.KERV_METAX_QUANT=='none'
    assert packed.experiment.envs.KERV_METAX_QUANT=='bf16_lossless_selective'
    assert not torch.cuda.is_initialized()
    result.update(status='passed',semantic_cases=checks,namespace_isolation=True,one_predicate_only=True,
        observer_output_unchanged=True,action_conservation=True,queue_parse_fail_closed=True,
        repaired_features_identical=True,gpu_context_initialized=False,traces=traces)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='traces'},indent=2))

if __name__=='__main__':main()
