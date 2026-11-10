"""Real-activation mixed-group and clipping search, using the deployed NPU op.

This bounded PTQ pilot is not GPTQ. It searches actual output reconstruction
error for each projection, then selects <=24 upgrades at equal scale-byte cost.
Calibration and held-out task success must be evaluated separately.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

from experiments.robot.openvla_utils import get_vla
from experiments.robot.ascend_quantization import NPUWeightOnlyLinear, weight_sha256


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--max-upgrades',type=int,default=24)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert os.environ.get('KERV_DEVICE_BACKEND')=='cpu','Load original weights on CPU; use NPU only for individual projections'
    assert 0<=a.max_upgrades<=24
    inputs=torch.load(a.inputs,map_location='cpu',weights_only=True)
    provenance=json.loads(a.inputs.with_name('provenance.json').read_text())
    tasks={x['episode_id'] for x in provenance['records']}
    assert tasks=={f'task{i}_init0' for i in range(10)},'Incomplete or unexpected calibration task split'
    assert len(inputs)==54
    plan={'method':'real-NPU-output group/clip reconstruction search; not GPTQ',
        'group_sizes':[128,32],'clip_ratios':[1.,.99,.975],'max_group32_upgrades':a.max_upgrades,
        'ranking':'positive decrease in squared relative output L2; equal scale-byte upgrade cost',
        'input_sha256':sha(a.inputs),'provenance_sha256':sha(a.inputs.with_name('provenance.json')),
        'calibration_episode_ids':sorted(tasks),'evaluation_initial_state_id':1,
        'physical_device':os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}
    write(a.output/'plan.json',plan)
    root=Path(__file__).resolve().parents[1]
    cfg=SimpleNamespace(pretrained_checkpoint=str(root/'checkpoints/openvla-libero-goal'),
        spec_checkpoint=str(root/'checkpoints/kerv-drafter'),use_spec=True,parallel_draft=False,
        accept_threshold=9,accept_threshold_start=15,load_in_8bit=False,load_in_4bit=False)
    model=get_vla(cfg).eval()
    assert next(model.parameters()).device.type=='cpu'
    results=[]
    with torch.inference_mode():
        for name,cpu_x in sorted(inputs.items()):
            layer=model.get_submodule(name)
            assert layer.weight.device.type=='cpu' and cpu_x.shape[-1]==layer.in_features
            x=cpu_x.npu();reference_weight=layer.weight.detach().npu()
            reference=torch.nn.functional.linear(x,reference_weight)
            norm=float(reference.float().square().sum());assert norm>0
            record={'name':name,'weight_sha256':weight_sha256(layer.weight),'rows':len(cpu_x),
                'weight_elements':layer.weight.numel(),'candidates':[]}
            for group in plan['group_sizes']:
                for clip in plan['clip_ratios']:
                    quant=NPUWeightOnlyLinear(layer,{'forward_calls':0},name,group,clip).npu()
                    actual=quant(x);assert bool(actual.isfinite().all())
                    error=float((actual.float()-reference.float()).square().sum())/norm
                    record['candidates'].append({'group_size':group,'clip_ratio':clip,'relative_squared_l2':error,
                        'relative_l2':error**.5,'storage_bytes':sum(t.numel()*t.element_size() for t in quant.buffers())})
                    del quant,actual
            for group in [128,32]:
                record[f'best_g{group}']=min((c for c in record['candidates'] if c['group_size']==group),
                    key=lambda c:c['relative_squared_l2'])
            record['group32_error_reduction']=record['best_g128']['relative_squared_l2']-record['best_g32']['relative_squared_l2']
            results.append(record);write(a.output/'sensitivity_partial.json',results)
            print(name,'rows',len(cpu_x),'g128',record['best_g128']['relative_l2'],
                'g32',record['best_g32']['relative_l2'],flush=True)
            del x,reference_weight,reference;gc.collect();torch.npu.empty_cache()
    upgraded={r['name'] for r in sorted(results,key=lambda r:-r['group32_error_reduction'])[:a.max_upgrades]
        if r['group32_error_reduction']>0}
    modules=[]
    for r in results:
        chosen=r['best_g32'] if r['name'] in upgraded else r['best_g128']
        modules.append({'name':r['name'],'group_size':chosen['group_size'],'clip_ratio':chosen['clip_ratio'],
            'weight_sha256':r['weight_sha256']})
    base_bytes=sum(r['weight_elements']*2 for r in results)
    quant_bytes=sum((r['best_g32'] if r['name'] in upgraded else r['best_g128'])['storage_bytes'] for r in results)
    policy={'version':1,'calibration':plan,'modules':modules,
        'group_counts':{'32':len(upgraded),'128':len(results)-len(upgraded)},
        'static_storage_saving_bytes':base_bytes-quant_bytes,
        'quality_claim':'calibration reconstruction only; independent init1 LIBERO task evaluation pending'}
    write(a.output/'sensitivity.json',results);write(a.output/'policy.json',policy)
    summary={'modules':len(modules),'group_counts':policy['group_counts'],'static_storage_saving_bytes':base_bytes-quant_bytes,
        'mean_rtn128_squared_l2':sum(next(c['relative_squared_l2'] for c in r['candidates'] if c['group_size']==128 and c['clip_ratio']==1.) for r in results)/len(results),
        'mean_selected_squared_l2':sum((r['best_g32'] if r['name'] in upgraded else r['best_g128'])['relative_squared_l2'] for r in results)/len(results),
        'nontrivial_clip_modules':sum(m['clip_ratio']!=1 for m in modules),'policy_sha256':sha(a.output/'policy.json')}
    assert summary['mean_selected_squared_l2']<=summary['mean_rtn128_squared_l2']
    write(a.output/'summary.json',summary);print('SUMMARY',summary,flush=True)


if __name__=='__main__':main()
