"""Validate actual upstream B0 modules on CPU and MUSA before full episodes."""
import argparse
from pathlib import Path
from acceptance.common import dump, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from transformers import LlamaConfig
    from openvla.prismatic.extern.hf.utils import evaluate_posterior
    from openvla.prismatic.extern.hf.modeling_speculation import LlamaSpecForCausalLM
    from openvla.specdecoding.model.cnets import EConfig, MMModel
    import inspect
    module=Path(inspect.getfile(evaluate_posterior))
    assert '/ports/specvla_musa/' in str(module),module
    root=Path('/workspace')
    unchanged=[]
    for rel in ('openvla/specdecoding/model/cnets.py','openvla/prismatic/extern/hf/utils.py','openvla/prismatic/extern/hf/modeling_prismatic.py'):
        assert sha256(root/'ports/specvla_musa'/rel)==sha256(root/'third_party/SpecVLA'/rel),rel
        unchanged.append(rel)
    torch.manual_seed(7)
    logits=torch.zeros(1,4,32);logits[:,:,10]=10
    cases=[]
    for tail,expected in [([20,20,20],0),([20,10,10],0),([10,20,10],1),([10,10,10],3)]:
        candidates=torch.tensor([[1]+tail])
        a=evaluate_posterior(logits,candidates,None,accept_threshold=0)
        b=evaluate_posterior(logits.to('musa'),candidates.to('musa'),None,accept_threshold=0)
        assert int(a[1])==int(b[1])==expected
        assert int(a[0])==int(b[0])==0
        cases.append({'tail':tail,'accepted_prefix':expected,'cpu_musa_equal':True})
    config=LlamaConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4)
    verifier=LlamaSpecForCausalLM(config,attn_implementation='eager').eval()
    draft=MMModel(EConfig(**config.to_dict()),load_emb=False).eval()
    ids=torch.tensor([[1,4,7,9]])
    with torch.no_grad():
        reference=verifier(input_ids=ids,output_hidden_states=True)
        expected=draft(reference.hidden_states[-1],input_ids=ids,input_embeddings=draft.embed_tokens(ids))
        verifier.to('musa');draft.to('musa')
        actual=verifier(input_ids=ids.to('musa'),output_hidden_states=True)
        output=draft(reference.hidden_states[-1].to('musa'),input_ids=ids.to('musa'),input_embeddings=draft.embed_tokens(ids.to('musa')))
        torch.testing.assert_close(actual.logits.cpu(),reference.logits,rtol=.001,atol=.001)
        torch.testing.assert_close(output.cpu(),expected,rtol=.001,atol=.001)
    dump(args.output,{'status':'PASS','scope':'upstream B0 small-model CPU/MUSA forward and rejection-boundary validation; actual 7B smoke separate',
        'unchanged_core_files':unchanged,'posterior_cases':cases,'verifier_max_abs_difference':float((actual.logits.cpu()-reference.logits).abs().max()),
        'draft_max_abs_difference':float((output.cpu()-expected).abs().max()),'tolerance':{'rtol':.001,'atol':.001}})
    print(args.output)


if __name__=='__main__':
    main()
