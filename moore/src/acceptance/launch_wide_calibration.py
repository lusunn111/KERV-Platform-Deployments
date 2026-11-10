"""Launch activation/GPTQ W10 calibration using frozen C data and explicit guards."""
import argparse
from pathlib import Path
import subprocess
from acceptance.common import dump,load
from acceptance.run_matrix import make_command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=Path('/root/KERV');c=a.campaign
    device=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    if 'No running processes found' not in device:raise RuntimeError('Require idle physical GPU0')
    (c/'device_before_gptq.log').write_text(device)
    old=root/'outputs/calibrated_quantization/20261026T045030Z_v2'
    layers=sorted({int(x['name'].split('.layers.')[1].split('.')[0]) for x in load(old/'model_refined/policy.json')['modules']})
    case={'profile':'native','kind':'calibration','timeout_seconds':'3600','max_steps':'220','planning_steps':'220','task_id':'0','init_state_id':'0','seed':'7'}
    command=make_command(root,case,c/'gptq_launcher/context.json','kerv-musa-runtime')
    command=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
    env=['-e','KERV_CALIBRATION_ROOT=/workspace/'+str(c.relative_to(root)),
         '-e','KERV_CALIBRATION_SOURCE_ROOT=/workspace/'+str(old.relative_to(root)),
         '-e','KERV_CALIBRATION_BITS=10','-e','KERV_CALIBRATION_SCALE_KIND=pow2',
         '-e','KERV_MODEL_OUTPUT_DIR=tree_gptq_pow2_w10',
         '-e','KERV_CALIBRATION_LAYER_LIST='+','.join(str(x) for x in layers)]
    command=command[:2]+env+command[2:]
    command+=['--override','inference.runtime.entrypoint=/workspace/src/acceptance/calibrated_tree_model_entry.py',
              '--override','inference.runtime.linear_fusion.enabled=false']
    dump(c/'gptq_calibration_command.json',{'argv':command,'selected_layers':layers,
        'scope':'Actual BF16 prompt/tree activations at each selected MLP; independent per-block GPTQ including deployed BF16 rounding. Not task success measurement.'})
    with (c/'gptq_calibration_launcher.log').open('w') as log:
        subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
