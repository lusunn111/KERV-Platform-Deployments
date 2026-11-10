"""Opt-in measurements at real rollout/verify execution points.

The diagnostic TorchDispatch copy observer covers visible ATen copies and scalar
reads, not opaque CANN/kernel-internal DMA. API enqueue duration is not DMA time.
Formal transfer and placement conclusions require a separate device trace.
"""
import contextlib, datetime, hashlib, json, os, resource, threading, time
from pathlib import Path


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Recorder:
    def __init__(self, directory, torch):
        self.path=Path(directory); self.path.mkdir(parents=True,exist_ok=True)
        self.torch=torch
        from experiments.robot.device_runtime import get_device
        self.device_type=get_device().type
        self.api=getattr(torch,self.device_type)
        self.run_id=os.environ.get('KERV_ACCEPTANCE_RUN_ID',self.path.name)
        self.buffers={k:[] for k in ['steps','verify','memory','transfers','episodes','placement']}
        self.phase='loading'; self.episode_id=None; self.step_id=None
        self.stop_event=threading.Event(); self.sample_interval=.1
        self.begin=time.perf_counter(); self.load_start=self.begin
        self.thread=threading.Thread(target=self._sample,daemon=True); self.thread.start()
        import atexit
        atexit.register(self.close)
        self.closed=False

    def add(self,kind,**values):
        self.buffers[kind].append(dict(run_id=self.run_id,episode_id=self.episode_id,step_id=self.step_id,**values))

    def _sample(self):
        while not self.stop_event.wait(self.sample_interval):
            try:
                self.add('memory',timestamp=utc(),phase=self.phase,
                    allocated_bytes=int(self.api.memory_allocated()),reserved_bytes=int(self.api.memory_reserved()),
                    allocated_peak_bytes=int(self.api.max_memory_allocated()),reserved_peak_bytes=int(self.api.max_memory_reserved()),
                    process_device_bytes=None,device_total_used_bytes=None,
                    cpu_rss_bytes=self._rss(),monitor_method=f'torch.{self.device_type} + /proc/self/statm',
                    sample_interval_seconds=self.sample_interval)
            except Exception as e:
                self.add('memory',timestamp=utc(),phase=self.phase,error=str(e))

    @staticmethod
    def _rss():
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')

    def loaded(self,model):
        self.api.synchronize(); self.phase='loaded'
        self.load_seconds=time.perf_counter()-self.load_start
        self.v2_manifest={'features':getattr(model,'_metax_v2_features',[]),'hits':getattr(model,'_metax_v2_stats',{})}
        self.quantization_manifest=getattr(model,'_metax_quant_manifest',getattr(model,'_flagos_npu_quant_manifest',None))
        self.quant_calibration=None
        if os.environ.get('KERV_QUANT_CALIBRATION_DIR'):
            from acceptance.quant_calibration_capture import MLPCalibrationCapture
            self.quant_calibration=MLPCalibrationCapture(model,self,os.environ['KERV_QUANT_CALIBRATION_DIR'])
        devices={str(x.device) for x in model.parameters()}
        self.add('placement',module='model_parameters',devices=sorted(devices),
            parameter_count=sum(x.numel() for x in model.parameters()),
            note='Parameter residency, not proof of every kernel execution location')
        self.initial_peaks={'allocated':int(self.api.max_memory_allocated()),'reserved':int(self.api.max_memory_reserved())}
        self.forward_counts={}
        def observe(name):
            def hook(module,args,kwargs):
                count=self.forward_counts.get(name,0)+1;self.forward_counts[name]=count
                if count==1:
                    tensors=[x for x in list(args)+list(kwargs.values()) if isinstance(x,self.torch.Tensor)]
                    self.add('placement',module=name,event='actual_forward_pre_hook',
                        input_devices=sorted({str(x.device) for x in tensors}),
                        parameter_devices=sorted({str(x.device) for x in module.parameters()}))
            return hook
        for name,module in [('drafter',getattr(model,'ea_layer',None)),
                            ('verifier_language',getattr(getattr(model,'base_model',None),'language_model',None))]:
            if module is not None:module.register_forward_pre_hook(observe(name),with_kwargs=True)


    def begin_step(self,task_id,init_state_id,step_id,image,state):
        self.api.synchronize()
        self.episode_id=f'task{task_id}_init{init_state_id}'
        self.step_id=int(step_id)
        if step_id==0:
            self.episode_start=time.perf_counter();self.phase='episode'
            self.episode_steps=0
        self.step_start=time.perf_counter();self.cpu_start=time.process_time()
        self.observation_hash=hashlib.sha256(image.tobytes()+state.tobytes()).hexdigest()
        self.image_hash=hashlib.sha256(image.tobytes()).hexdigest()
        self.task_id=task_id;self.init_state_id=init_state_id
        self.verify_id=0

    def end_step(self,raw_action,executed_action,done,model_seconds,pipeline_seconds,env_seconds,threshold,tokens):
        self.episode_steps+=1
        transfers=[x for x in self.buffers['transfers'] if x['episode_id']==self.episode_id and x['step_id']==self.step_id]
        diagnostic=os.environ.get('KERV_ACCEPTANCE_TRANSFERS')=='1'
        self.add('steps',task_id=self.task_id,init_state_id=self.init_state_id,
            observation_hash=self.observation_hash,image_hash=self.image_hash,
            feedback_source='history of model raw output actions; proprio state is observed but not used by OpenVLA/KF',
            action_tokens=tokens,raw_action=raw_action.tolist(),executed_action=executed_action.tolist(),
            success=bool(done),done=bool(done),termination_reason='success' if done else None,
            model_seconds=float(model_seconds),pipeline_seconds=float(pipeline_seconds),
            env_seconds=float(env_seconds),wall_seconds=time.perf_counter()-self.step_start,
            cpu_seconds=time.process_time()-self.cpu_start,threshold=threshold,
            h2d_bytes=sum(x['bytes'] for x in transfers if x['direction']=='H2D') if diagnostic else None,
            d2h_bytes=sum(x['bytes'] for x in transfers if x['direction']=='D2H') if diagnostic else None,
            copy_seconds=None,transfer_coverage='ATen-visible diagnostic only' if diagnostic else 'not collected')

    def verify(self,proposed,accepted,threshold,forced,shape,*,candidate_tokens=None,action_tokens_before=0,action_dim=7):
        assert isinstance(accepted,int) and 0<=accepted<=proposed
        seven=os.environ.get('KERV_ACTION_PROTOCOL')=='seven'
        prefix=(candidate_tokens or [])[1:1+accepted]
        padding=sum(token<0 for token in prefix)
        if padding:raise RuntimeError('Accepted prefix includes padding; refuse invalid metric')
        # Every cycle commits a verifier-selected root, then consecutive draft tokens.
        remaining=max(action_dim-action_tokens_before-1,0)
        committed=min(accepted,remaining) if seven else None
        self.add('verify',verify_id=self.verify_id,proposed_length=int(proposed),
            accepted_prefix_length=accepted,committed_prefix_length=committed,
            root_length=1,bonus_length=0 if seven else None,
            tree_root_length=1,initial_action_root_length=int(action_tokens_before==0),
            committed_verifier_bonus_length=int(action_tokens_before>0),
            compensation_length=max(remaining-committed,0) if seven and forced else (0 if seven else None),
            padding_length=padding,action_tokens_before=action_tokens_before,
            selected_candidate_tokens=candidate_tokens,
            threshold=float(threshold),compensation_triggered=bool(forced),
            fallback_reason='periodic Kalman completion' if forced else None,candidate_shape=list(shape),
            note='Posterior consecutive prefix excludes root; committed prefix additionally clips at seven-action boundary. root_length is per-tree root. Later tree roots commit the preceding verifier bonus; initial_action_root_length and committed_verifier_bonus_length separate those origins, never as acceptance.')
        self.verify_id+=1

    def end_episode(self,done,limit,planning_horizon):
        self.add('episodes',task_id=self.task_id,init_state_id=self.init_state_id,
            success=bool(done),steps=self.episode_steps,step_limit=limit,planning_horizon=planning_horizon,
            termination_reason='success' if done else ('truncated_smoke' if limit<220 else 'step_limit'),
            closed_loop_wall_seconds=time.perf_counter()-self.episode_start,
            timing_scope='first policy observation ready through terminal env.step and action logging; excludes reset/settling and video',
            allocated_peak_bytes=int(self.api.max_memory_allocated()),reserved_peak_bytes=int(self.api.max_memory_reserved()))
        self.phase='between_episodes'
        self.flush()
        if getattr(self,'v2_manifest',{}).get('features'):
            (self.path/'v2.json').write_text(json.dumps(self.v2_manifest,indent=2)+'\n')

    def transfer_context(self):
        if os.environ.get('KERV_ACCEPTANCE_TRANSFERS')!='1':return contextlib.nullcontext()
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        recorder=self
        class Copies(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {}; name=str(func)
                direction=None;size=0
                if name=='aten.copy_.default' and len(args)>1:
                    dst,src=args[:2]
                    if src.device.type!=dst.device.type and {src.device.type,dst.device.type}<={'cpu',recorder.device_type}:
                        direction='H2D' if dst.device.type==recorder.device_type else 'D2H'
                        size=dst.numel()*dst.element_size()
                elif name=='aten._to_copy.default' and args:
                    src=args[0];dst=torch.device(kwargs.get('device',src.device))
                    if src.device.type!=dst.type and {src.device.type,dst.type}<={'cpu',recorder.device_type}:
                        direction='H2D' if dst.type==recorder.device_type else 'D2H'
                        size=src.numel()*torch.empty((),dtype=kwargs.get('dtype',src.dtype)).element_size()
                elif name=='aten._local_scalar_dense.default' and args and args[0].device.type==recorder.device_type:
                    direction='D2H';size=args[0].numel()*args[0].element_size()
                start=time.perf_counter() if direction else None
                result=func(*args,**kwargs)
                if direction:
                    recorder.add('transfers',operation=name,direction=direction,bytes=int(size),
                        copy_enqueue_seconds=time.perf_counter()-start,copy_seconds=None,
                        coverage='visible ATen logical transfer; opaque device kernels and DMA timing unmeasured')
                return result
        @contextlib.contextmanager
        def observed():
            # torch_npu implements some Tensor.to/tolist/item paths outside
            # Python-visible ATen dispatch. Observe those actual API calls,
            # keeping ATen records when present to avoid double counting.
            originals={name:getattr(torch.Tensor,name) for name in ['to','cpu','tolist','item']}
            def wrap(name,original):
                def call(tensor,*args,**kwargs):
                    before=len(recorder.buffers['transfers']);start=time.perf_counter()
                    result=original(tensor,*args,**kwargs)
                    src=tensor.device.type
                    dst=result.device.type if isinstance(result,torch.Tensor) else ('cpu' if name in ['tolist','item'] else src)
                    if src!=dst and {src,dst}<={'cpu',recorder.device_type}:
                        direction='H2D' if dst==recorder.device_type else 'D2H'
                        recent=recorder.buffers['transfers'][before:]
                        if not any(x['direction']==direction for x in recent):
                            size=(result.numel()*result.element_size()) if isinstance(result,torch.Tensor) else tensor.numel()*tensor.element_size()
                            recorder.add('transfers',operation='Tensor.'+name,direction=direction,bytes=int(size),
                                copy_enqueue_seconds=time.perf_counter()-start,copy_seconds=None,
                                coverage='actual Python Tensor API logical transfer; opaque backend DMA still unmeasured')
                    return result
                return call
            try:
                for name,original in originals.items():setattr(torch.Tensor,name,wrap(name,original))
                with Copies():yield
            finally:
                for name,original in originals.items():setattr(torch.Tensor,name,original)
        return observed()

    def flush(self):
        for name,rows in self.buffers.items():
            (self.path/f'{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows))

    def close(self):
        if self.closed:return
        self.closed=True;self.stop_event.set();self.thread.join(timeout=1)
        if getattr(self,'quant_calibration',None) is not None:self.quant_calibration.close()
        self.flush()
        if getattr(self,'v2_manifest',{}).get('features'):
            (self.path/'v2.json').write_text(json.dumps(self.v2_manifest,indent=2)+'\n')
        if getattr(self,'quantization_manifest',None) is not None:
            (self.path/'quantization.json').write_text(json.dumps(self.quantization_manifest,indent=2)+'\n')
        (self.path/'collector.json').write_text(json.dumps({'run_id':self.run_id,'load_seconds':getattr(self,'load_seconds',None),'forward_counts':getattr(self,'forward_counts',{}),
            'sample_interval_seconds':self.sample_interval,'heavy_transfer_diagnostic':os.environ.get('KERV_ACCEPTANCE_TRANSFERS')=='1',
            'complete_transfer_coverage':False,'actual_dma_time_collected':False,
            'missing':['vendor per-process 100ms memory','device trace placement/copy audit','steady-state warmed memory phase']},indent=2)+'\n')


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--describe',action='store_true');p.parse_args()
    print('Set KERV_ACCEPTANCE_DIR and KERV_ACCEPTANCE_RUN_ID via acceptance/run_matrix.py; diagnostic copies require KERV_ACCEPTANCE_TRANSFERS=1.')

if __name__=='__main__':main()
