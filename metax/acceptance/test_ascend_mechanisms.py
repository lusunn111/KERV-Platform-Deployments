"""Explicit acceptance boundary and seven-action checks on the deployed NPU code."""
import os, unittest
from types import SimpleNamespace
import numpy as np
import torch
from experiments.robot.device_runtime import get_device
from experiments.robot.tool_utils import compute_dynamic_threshold
from KERVRuntimeOptimization.embodied_ops.runtime import configure_kerv_ops
from prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction
from acceptance.collect_metrics import Recorder

class AscendMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=get_device(); assert cls.device.type=='npu'
        configure_kerv_ops(enabled=True,backend='native')
    def test_full_first_middle_all_rejections_and_tie(self):
        candidates=torch.tensor([[9,10,11,12,13,14,15,16]],device=self.device)
        for mismatch,expected in [(None,7),(0,0),(3,3),('all',0)]:
            predictions=list(range(10,17))
            if mismatch=='all':predictions=[0]*7
            elif mismatch is not None:predictions[mismatch]=0
            logits=torch.zeros(1,8,32,device=self.device)
            logits[0,torch.arange(7,device=self.device),torch.tensor(predictions,device=self.device)]=100
            b,n=torch.ops.flagos_embodied.kerv_verify_accept_control(logits,candidates,0.,0)
            with self.subTest(mismatch=mismatch):self.assertEqual((b.item(),n.item()),(0,expected))
        logits=torch.zeros(2,8,32,device=self.device)
        logits[:,torch.arange(7,device=self.device),torch.arange(10,17,device=self.device)]=100
        b,n=torch.ops.flagos_embodied.kerv_verify_accept_control(logits,candidates.expand(2,-1),0.,0)
        self.assertEqual((b.item(),n.item()),(0,7))
    def test_seven_dof_kf(self):
        verified=[31890,31891,31892,31893,31894,31895,31745]
        prediction=[31810,31811,31812,31813,31814,31815,31745]
        model=SimpleNamespace(vocab_size=32000,bin_centers=np.zeros(255),
            tokenizer=SimpleNamespace(pad_token_id=32000),get_action_dim=lambda:7,kelman_predict_tokens=lambda:prediction)
        for accepted in [0,2,6]:
            prompt=torch.tensor([[1,29871]],device=self.device)
            packed,count=SpecVLAforActionPrediction._finalize_action_with_kf(model,prompt,1,torch.tensor([verified],device=self.device),0,accepted)
            self.assertEqual(packed[0,prompt.shape[1]:].cpu().tolist(),verified[:accepted+1]+prediction[accepted+1:])
            self.assertEqual(count,7)
    def test_threshold_endpoints(self):
        self.assertEqual(compute_dynamic_threshold(0,220,15,3),15)
        self.assertEqual(compute_dynamic_threshold(220,220,15,3),3)
        self.assertGreaterEqual(compute_dynamic_threshold(1000,220,15,3),3)
    def test_diagnostic_copy_bytes(self):
        recorder=object.__new__(Recorder);recorder.torch=torch;recorder.run_id='copy_unit';recorder.episode_id='unit';recorder.step_id=0
        recorder.buffers={k:[] for k in ['steps','verify','memory','transfers','episodes','placement']}
        os.environ['KERV_ACCEPTANCE_TRANSFERS']='1'
        x=torch.arange(3,dtype=torch.float32)
        with torch.inference_mode(), recorder.transfer_context():
            npu=x.to(self.device);npu.cpu();npu[0].item()
            npu.to('cpu').tolist();npu.tolist()
        rows=recorder.buffers['transfers']
        self.assertEqual(sum(x['bytes'] for x in rows if x['direction']=='H2D'),12)
        self.assertEqual(sum(x['bytes'] for x in rows if x['direction']=='D2H'),40)

if __name__=='__main__':unittest.main()
