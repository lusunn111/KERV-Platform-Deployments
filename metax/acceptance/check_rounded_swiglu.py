"""Exhaustively check the SiLU BF16 input domain before allowing kernel use."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from experiments.robot.ascend_mlp_ops import rounded_swiglu


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    codes=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    codes=codes[torch.isfinite(codes)]
    gate=codes.npu().reshape(1,1,-1);up=torch.ones_like(gate)
    combined=torch.cat([gate,up],dim=-1)
    with torch.inference_mode():
        reference=torch.nn.functional.silu(gate)
        actual=rounded_swiglu(combined)
        mask=actual!=reference
        mismatches=int(mask.sum())
        examples=[]
        for index in mask.reshape(-1).nonzero().reshape(-1)[:16].cpu().tolist():
            examples.append(dict(input=float(gate.reshape(-1)[index]),native=float(reference.reshape(-1)[index]),
                                 triton=float(actual.reshape(-1)[index])))
        report=dict(finite_bf16_inputs=len(codes),mismatches=mismatches,exact=mismatches==0,examples=examples,
                    scope='SiLU on every finite BF16 input, multiplication by one; multiplication and real MLP separately checked')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
