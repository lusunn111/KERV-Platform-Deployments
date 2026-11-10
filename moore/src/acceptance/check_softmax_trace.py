"""CPU-only independent check of the first differing attention tensors."""
import argparse
import json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();records=[]
    for file in sorted(a.directory.glob('difference_*.pt')):
        d=torch.load(file,weights_only=False)
        if not d['difference']['name'].endswith('.softmax.output'):continue
        x=d['old'][0][1];other=d['new'][0][1]
        assert d['old'][0][0].endswith('.softmax.input') and torch.equal(x,other)
        old=d['old'][-1][1];new=d['new'][-1][1]
        reference=x.float().softmax(-1)
        bad_rows=(old!=new).any(-1).nonzero().tolist()
        for row in bad_rows:
            index=tuple(row);cols=(old[index]!=new[index]).nonzero().flatten()
            records.append({'file':str(file),'difference':d['difference'],'input_bit_exact':True,
                'input_dtype':str(x.dtype),'row':row,'changed_columns':cols.tolist(),
                'changed_input_values':x[index][cols].float().tolist(),
                'old_row_sum':float(old[index].sum()),'new_row_sum':float(new[index].sum()),
                'old_cpu_max_error':float((old[index]-reference[index]).abs().max()),
                'new_cpu_max_error':float((new[index]-reference[index]).abs().max())})
    assert records,'Missing same-input Softmax failure evidence'
    a.output.write_text(json.dumps({'scope':'CPU independent recomputation, no MUSA operator import','records':records},indent=2)+'\n')

if __name__=='__main__':main()
