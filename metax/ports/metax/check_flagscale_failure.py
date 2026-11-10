"""Exercise the actual FlagScale -> adapter -> failing child path without a GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from omegaconf import OmegaConf

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
(out/'fail.py').write_text('raise SystemExit(23)\n')
cfg=OmegaConf.load(root/'outputs/metax_native_pair_01/metax.yaml')
cfg.experiment.exp_dir=str(out/'flagscale');cfg.experiment.exp_name='metax_failure_propagation'
cfg.inference.runtime.source_root=str(out);cfg.inference.runtime.entrypoint='fail.py'
cfg.inference.runtime.work_dir=str(out);cfg.action='test'
cfg.experiment.envs.PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
OmegaConf.save(cfg,out/'failure.yaml',resolve=True)
command=[sys.executable,str(root/'third_party/FlagScale/run.py'),'--config-path',str(out),'--config-name','failure']
result=subprocess.run(command,cwd=root/'third_party/FlagScale')
(out/'result.json').write_text(json.dumps(dict(child_exit_code=23,flagscale_exit_code=result.returncode,
    passed=result.returncode!=0),indent=2)+'\n')
assert result.returncode!=0,'FlagScale swallowed child failure'
