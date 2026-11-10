"""CPU-only counterexample against the pinned upstream function, not a model run."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=(a.reference/'openvla/prismatic/extern/hf/utils.py').read_text()
tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='update_inference_inputs')
function_source=ast.get_source_segment(source,function)
assert 'if (input_ids.shape[1]-input_len-1+accept_length)>6:' in function_source

def run(text):
    namespace={'torch':torch};exec(compile(text,'upstream_update_function','exec'),namespace)
    update=namespace['update_inference_inputs']
    embedding=torch.nn.Embedding(32000,2)
    drafter=SimpleNamespace(embed_tokens=embedding,topK_genrate=lambda *args:(None,None,None,None))
    model=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2),ea_layer=drafter,
        base_model=SimpleNamespace(language_model=SimpleNamespace(lm_head=None)))
    input_ids=torch.tensor([[1,29871]]);input_len=1;prompt=torch.zeros(1,2,2)
    candidates=torch.tensor([[31800,31801,31802,31803,31804,31805,31806,31807]])
    sample=torch.zeros(32000);sample[31807]=1
    def cycle(ids,prompt_embeds,accepted,new_token):
        return update(prompt_embeds,ids,input_len,candidates,0,accepted,torch.arange(8)[None],
            None,new_token,[],model,torch.zeros(1,8,2),sample,None)
    first=cycle(input_ids,prompt,torch.tensor(6),0)
    ids=first[0];generated=first[5]
    stopped=bool((ids==2).any())
    if not stopped:
        candidates[0,0]=31807  # Next root is the verifier argmax from the first cycle.
        second=cycle(ids,first[6],torch.tensor(0),generated);ids=second[0]
    tokens=[int(x) for x in ids[0,2:] if x!=2]
    return dict(first_cycle_stopped=stopped,generated_tokens=tokens,decoded_last_seven=tokens[-7:],
                root_preserved_in_decoded_action=tokens[-7]==31800)
original=run(function_source)
proposal=run(function_source.replace('+accept_length)>6:', '+accept_length)>=6:'))
report=dict(kind='synthetic CPU protocol unit, not GPU/model deployment',
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),original=original,
    proposed_single_predicate_fix=proposal,decision='pending user choice; no corrected B0 model run authorized by this test')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert not original['root_preserved_in_decoded_action'] and proposal['root_preserved_in_decoded_action']
