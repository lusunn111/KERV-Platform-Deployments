"""P03 fixed-input branch coverage without loading 7B weights."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from openvla.experiments.robot.device_runtime import get_device
from openvla.experiments.robot.tool_utils import compute_dynamic_threshold
from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction
from KERVRuntimeOptimization.embodied_ops.runtime import configure_kerv_ops


class AcceptanceMechanismTests(unittest.TestCase):
    def test_rejection_positions_and_all_acceptance_on_actual_device(self):
        device = get_device()
        configure_kerv_ops(enabled=True, backend="native")
        logits = torch.zeros(1, 4, 8, device=device)
        logits[:, :, 2] = 10
        for tail, expected in [([105,105,105],0), ([105,102,102],0), ([102,105,102],1), ([102,102,102],3)]:
            candidates = torch.tensor([[100]+tail], device=device)
            best,length = torch.ops.flagos_embodied.kerv_verify_accept_control(logits,candidates,0.,100)
            self.assertEqual((best.item(),length.item()),(0,expected))

    def test_kalman_completion_preserves_prefix_and_stops_at_seven(self):
        device=get_device()
        # Test the real method with deterministic Kalman outputs at each rejection boundary.
        model=SimpleNamespace(get_action_dim=lambda:7,vocab_size=1000,bin_centers=np.arange(255),
                              tokenizer=SimpleNamespace(pad_token_id=750),kelman_predict_tokens=lambda:list(range(800,807)))
        ids=torch.tensor([[1,2]],device=device)
        candidate=torch.tensor([[990]+list(range(900,907))],device=device)
        for accepted in (0,1,3,6,7):
            actual,count=SpecVLAforActionPrediction._finalize_action_with_kf(model,ids,1,candidate,0,accepted)
            expected=[1,2,990]+list(range(900,900+accepted))+list(range(800+accepted,807))
            self.assertEqual(actual.cpu().tolist(),[expected])
            self.assertEqual(count,7)
            self.assertEqual(actual.shape[-1]-ids.shape[-1]-1,7)

    def test_rollout_reset_discards_action_and_token_history(self):
        model=SimpleNamespace(_kalman_config={},_current_history=[[1]*7],_token_history=[[9]*7],_step_idx=99)
        SpecVLAforActionPrediction.start_rollout(model,max_steps=220,kalman_cfg={"enabled":True,"tree_enabled":True},store_history=True)
        self.assertEqual(model._current_history,[])
        self.assertEqual(model._token_history,[])
        self.assertEqual(model._step_idx,-1)
        self.assertEqual(model._rollout_max_steps,220)
        model._token_history.append([8]*7)
        SpecVLAforActionPrediction.start_rollout(model,max_steps=220,store_history=False)
        self.assertIsNone(model._current_history)
        self.assertEqual(model._token_history,[])

    def test_dynamic_threshold_bounds_and_horizon(self):
        values=[compute_dynamic_threshold(i,220,15,3) for i in (0,1,8,100,219,220,1000)]
        self.assertEqual(values[0],15)
        self.assertTrue(all(3<=x<=15 for x in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertEqual(values[-1],3)
        with self.assertRaises(ValueError):
            compute_dynamic_threshold(0,220,3,15)


if __name__ == '__main__':
    unittest.main()
