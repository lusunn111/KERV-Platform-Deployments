"""Isolate native TopK -> gather -> eight-token embedding, without the model."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch_musa


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=10000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'worker_start.json').write_text(json.dumps({'pid':os.getpid()}))
    reference=torch.load(a.input,weights_only=False)
    probabilities=[x.to('musa') for name,x in reference if name=='topk.input' and x.shape[-1]==32064]
    if not probabilities:raise RuntimeError('No actual drafter TopK input tensors')
    weight=torch.arange(32064,device='musa',dtype=torch.float32).to(torch.bfloat16)[:,None].expand(-1,4096).contiguous()
    pending=[];checks=0
    with (a.output/'progress.jsonl').open('a',buffering=1) as log:
        for i in range(a.repeats):
            source=probabilities[i%len(probabilities)]
            values,indices=torch.topk(source,8,dim=-1)
            if source.shape[0]==8:
                selected=torch.topk(values.reshape(-1),8).indices
                ids=indices.reshape(-1)[selected][None]
            else:ids=indices
            # No CPU bound check or sync between index generation and lookup.
            output=torch.nn.functional.embedding(ids,weight)
            pending.append(((output!=ids.to(torch.bfloat16)[...,None]).any() | (ids<0).any() | (ids>=32064).any()))
            if len(pending)==100 or i+1==a.repeats:
                valid=not bool(torch.stack(pending).any().cpu())
                checks+=len(pending);pending.clear()
                log.write(json.dumps({'iterations':checks,'valid':valid})+'\n')
                if not valid:raise RuntimeError('Index or embedding output mismatch')
    (a.output/'summary.json').write_text(json.dumps({'iterations':checks,'all_indices_and_embeddings_valid':True,
        'actual_topk_input_tensors':len(probabilities),'scope':'standalone async native index path, not a proof of full-model fault resolution'},indent=2)+'\n')


if __name__=='__main__':main()
