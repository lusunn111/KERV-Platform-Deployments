"""Separate heavy NPU trace; never used for reported low-interference timings."""
import functools, json, os, runpy
from pathlib import Path
import torch, torch_npu

root=Path(__file__).resolve().parents[1]
out=Path(os.environ['KERV_ACCEPTANCE_DIR']).parent/'device_profile'
out.mkdir(parents=True,exist_ok=False)
from openvla.prismatic.extern.hf import modeling_speculation
from experiments.robot import tool_utils
observations=[]

def wrap(module,name,label):
    original=getattr(module,name)
    @functools.wraps(original)
    def call(*args,**kwargs):
        if len(observations)<100:
            observations.append({'function':name,'scope':label,'first_argument_type':type(args[0]).__name__ if args else None,
                'first_argument_tensor_device':str(args[0].device) if args and isinstance(args[0],torch.Tensor) else None})
        with torch.profiler.record_function(label):return original(*args,**kwargs)
    setattr(module,name,call)

wrap(modeling_speculation,'kalman_predict_from_history','KERV::CPU_Kalman_HistoryEstimate')
wrap(tool_utils,'compute_dynamic_threshold','KERV::CPU_ThresholdSchedule')
with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)),record_shapes=True,with_stack=False,profile_memory=False):
    runpy.run_path(str(root/'ports/moore_ascend/openvla/experiments/robot/libero/run_kerv_libero.py'),run_name='__main__')
(out/'python_placement.json').write_text(json.dumps({'diagnostic_only':True,
    'observed_calls':observations,'warning':'Contains loading/settling/video; not a formal task timing. Python KF input is model token history, not measured joint feedback.'},indent=2)+'\n')
