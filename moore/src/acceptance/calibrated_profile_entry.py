"""cProfile and MUSA event instrumentation of one warmed real action."""
import atexit
import cProfile
import os
from pathlib import Path
import pstats
from types import MethodType
from acceptance.common import dump
from experiments.robot.libero.run_calibrated_kernel_libero import register
register()
from experiments.robot.libero import run_kerv_libero as evaluation
original_model=evaluation.get_model

def audited(cfg):
    import torch
    from KERVRuntimeOptimization.calibrated_int8 import CalibratedInt8Linear
    model=original_model(cfg);state={'active':False,'seen':0};events=[]
    out=Path(os.environ['KERV_REPLAY_OUTPUT']);prof=cProfile.Profile()
    for name,module in model.named_modules():
        if not isinstance(module,CalibratedInt8Linear):continue
        decode=module.dequantize
        def traced_decode(self,_decode=decode):
            w=_decode()
            if state['active']:state['decode_end'].record()
            return w
        module.dequantize=MethodType(traced_decode,module)
        forward=module.forward
        def traced_forward(self,x,_forward=forward,_name=name):
            if not state['active']:return _forward(x)
            begin=torch.musa.Event(enable_timing=True);middle=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True)
            state['decode_end']=middle;begin.record();y=_forward(x);end.record()
            events.append((_name,list(x.shape),begin,middle,end))
            return y
        module.forward=MethodType(traced_forward,module)
    predict=model.predict_action
    def traced_predict(*args,**kwargs):
        if getattr(model,'_acceptance_replay_frame',-1)==2:
            state['seen']+=1
            if state['seen']==2:
                state['active']=True;prof.enable()
                try:return predict(*args,**kwargs)
                finally:prof.disable();state['active']=False
        return predict(*args,**kwargs)
    model.predict_action=traced_predict
    def final():
        torch.musa.synchronize();prof.dump_stats(str(out/'action.pstats'))
        stats=pstats.Stats(prof)
        functions=sorted([{'file':k[0],'line':k[1],'function':k[2],'primitive_calls':v[0],'calls':v[1],
                  'self_seconds':v[2],'cumulative_seconds':v[3]} for k,v in stats.stats.items()],key=lambda x:x['cumulative_seconds'],reverse=True)
        times=[{'module':n,'input_shape':s,'decode_ms':b.elapsed_time(d),'linear_total_ms':b.elapsed_time(e),
                'native_gemm_after_decode_ms':d.elapsed_time(e)} for n,s,b,d,e in events]
        dump(out/'profile.json',{'scope':'one warmed action frame2, events/cProfile change execution overhead; not benchmark',
             'functions':functions,'linear_events':times,'decode_sum_ms':sum(x['decode_ms'] for x in times),
             'linear_sum_ms':sum(x['linear_total_ms'] for x in times)})
    atexit.register(final)
    return model

evaluation.get_model=audited
from acceptance.replay_entry import main
main()
