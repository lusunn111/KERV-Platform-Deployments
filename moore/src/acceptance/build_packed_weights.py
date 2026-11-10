"""Build packed W10/W12 RTN directly from frozen BF16; keep prior protected input groups."""
import argparse
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from KERVRuntimeOptimization.packed_intn import PackedIntLinear
from KERVRuntimeOptimization.packed_pow2 import PackedPow2Linear
from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
from experiments.robot.libero.run_precision_libero import digest
from acceptance.common import load,dump,sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--bits',type=int,choices=(10,12),default=10)
    p.add_argument('--scale-kind',choices=('fp32','pow2'),default='fp32')
    p.add_argument('--module-list-policy',type=Path,help='Restrict to module names listed in this frozen policy')
    p.add_argument('--output-dir',help='Unique output subdirectory within campaign')
    p.add_argument('--exact-control',action='store_true',help='Protect every group to isolate runtime from quantization error')
    a=p.parse_args()
    torch.set_num_threads(8);c=a.campaign;out=c/(a.output_dir or (f'packed_w{a.bits}' if a.scale_kind=='fp32' else f'packed_pow2_w{a.bits}'));out.mkdir(exist_ok=False)
    checkpoint=Path('/workspace/checkpoints/openvla-libero-goal')
    index=load(checkpoint/'model.safetensors.index.json')['weight_map']
    previous=Path('/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/model_refined/policy.json');old=load(previous)
    chosen=old['modules']
    if a.module_list_policy:
        names={x['name'] for x in load(a.module_list_policy)['modules']}
        chosen=[x for x in chosen if x['name'] in names]
        if not names or len(chosen)!=len(names):raise ValueError('Unknown or empty module selection')
    records=[];errors=[];total_old=total_new=total_signal=0.
    for entry in chosen:
        key=entry['name'].removeprefix('base_model.')+'.weight'
        with safe_open(str(checkpoint/index[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key)
        assert digest(w)==entry['original_weight_sha256']
        path=previous.parent/entry['file'];assert sha256(path)==entry['sha256']
        data=torch.load(path,weights_only=True);ids=data['state_dict']['protected_groups'].tolist()
        if a.exact_control:ids=list(range((w.shape[1]+31)//32))
        linear=nn.Linear(w.shape[1],w.shape[0],bias=False,device='meta',dtype=torch.bfloat16)
        linear.weight=nn.Parameter(w,requires_grad=False)
        quantizer=PackedIntLinear if a.scale_kind=='fp32' else PackedPow2Linear
        q=quantizer.from_linear(linear,a.bits,32,ids);decoded=q.dequantize()
        old_decoded=CalibratedInt8Linear.from_payload(data).dequantize()
        signal=float(w.float().square().sum());before=float((old_decoded.float()-w.float()).square().sum())
        after=float((decoded.float()-w.float()).square().sum())
        if a.exact_control and not torch.equal(decoded.view(torch.int16),w.view(torch.int16)):raise RuntimeError('Exact control changed original BF16 bits')
        total_old+=before;total_new+=after;total_signal+=signal
        name=entry['name'].split('.layers.')[1].replace('.mlp.','_')+f'_w{a.bits}.pt';torch.save(q.payload(),out/name)
        item={'name':entry['name'],'encoding':'packed_intn' if a.scale_kind=='fp32' else 'packed_pow2','bits':a.bits,'file':name,'sha256':sha256(out/name),
              'shape':list(w.shape),'original_weight_sha256':entry['original_weight_sha256'],
              'decoded_weight_sha256':digest(decoded),'before_bytes':w.numel()*2,'storage_bytes':q.storage_bytes,
              'calibration':q.calibration}
        records.append(item)
        error={'name':entry['name'],'original_energy':signal,'old_int8_squared_error':before,'new_squared_error':after,
               'old_relative_weight_rmse':(before/signal)**.5,'new_relative_weight_rmse':(after/signal)**.5}
        errors.append(error);dump(out/'weight_errors.json',errors)
        print('PACKED',len(records),len(chosen),entry['name'],error['old_relative_weight_rmse'],error['new_relative_weight_rmse'],flush=True)
        del q,data,decoded,old_decoded,linear,w
    policy={'schema':'kerv_precision_first_storage_v1','complete':True,'modules':records,'module_count':len(records),
            'exact_bf16_groups':['verify_mlp'],'quality':'EXPERIMENTAL; task success validation required',
            'calibration_scope':('DEBUG EXACT CONTROL; all groups original BF16; not a memory-optimized deployment' if a.exact_control else 'Original BF16 RTN at higher bit width; reuse protected columns from frozen C calibration. No activation-based GPTQ performed for this new width.')}
    dump(out/'policy.json',policy)
    dump(out/'summary.json',{'bits':a.bits,'scale_kind':a.scale_kind,'modules':len(records),'old_relative_weight_rmse':(total_old/total_signal)**.5,
                            'new_relative_weight_rmse':(total_new/total_signal)**.5,
                            'weight_rmse_ratio':(total_new/total_old)**.5,
                            'quantized_projection_saving_bytes':sum(x['before_bytes']-x['storage_bytes'] for x in records),
                            'scope':'Weight reconstruction diagnostic, not action accuracy or SR'})


if __name__=='__main__':main()
