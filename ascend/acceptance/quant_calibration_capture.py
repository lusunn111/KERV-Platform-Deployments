"""Opt-in, bounded real-activation capture; diagnostic timing only."""
import json
from pathlib import Path


class MLPCalibrationCapture:
    def __init__(self, model, recorder, directory):
        self.recorder=recorder;self.torch=recorder.torch;self.path=Path(directory)
        self.path.mkdir(parents=True,exist_ok=False)
        self.chunks={};self.counts={};self.provenance=[];self.hooks=[]
        prefix='base_model.language_model.model.layers.'
        self.names=[]
        for name,layer in model.named_modules():
            if name.startswith(prefix) and '.mlp.' in name and name.rsplit('.',1)[-1] in ['gate_proj','up_proj','down_proj']:
                if 7<=int(name[len(prefix):].split('.')[0])<25:
                    self.names.append(name);self.chunks[name]=[]
                    self.hooks.append(layer.register_forward_pre_hook(self.hook(name)))
        assert len(self.names)==54
        self.closed=False

    def hook(self,name):
        def capture(module,inputs):
            r=self.recorder
            if r.phase!='episode' or r.step_id not in {0,20,40,60,90,120,160,200}:return
            phase='early' if r.step_id<=20 else ('middle' if r.step_id<=60 else 'late')
            key=(name,r.episode_id,phase)
            available=8-self.counts.get(key,0)
            if available<=0:return
            x=inputs[0].detach().reshape(-1,inputs[0].shape[-1])
            n=min(4,available,x.shape[0]);indices=self.torch.linspace(0,x.shape[0]-1,n,device=x.device).long()
            sample=x.index_select(0,indices).cpu()
            self.chunks[name].append(sample);self.counts[key]=self.counts.get(key,0)+n
            self.provenance.append(dict(module=name,episode_id=r.episode_id,step_id=r.step_id,phase=phase,
                source_shape=list(inputs[0].shape),rows=n))
        return capture

    def close(self):
        if self.closed:return
        self.closed=True
        for h in self.hooks:h.remove()
        assert all(self.chunks.values()),'No real calibration activations captured'
        data={name:self.torch.cat(chunks) for name,chunks in self.chunks.items()}
        self.torch.save(data,self.path/'inputs.pt')
        (self.path/'provenance.json').write_text(json.dumps(dict(
            scope='BF16 task0..9/init0 development trajectories; bounded early/middle/late samples',
            timing_is_diagnostic=True,rows_by_module={name:len(x) for name,x in data.items()},
            records=self.provenance),indent=2)+'\n')
