"""Same native Kalman predictions, different host/device buffer construction."""
from types import SimpleNamespace
import unittest
import torch
from openvla.experiments.robot.device_runtime import get_device
from openvla.prismatic.extern.hf import utils

class TreeBufferTests(unittest.TestCase):
    def test_augmented_tree_preserves_all_tokens_paths_masks_and_depths(self):
        device=get_device()
        cases=[{1:0,2:0,3:1,4:1,5:2,6:3},{1:0},{1:0,2:1,3:2,4:3,5:4,6:5,7:6}]
        for parent in cases:
            children={}
            for node,ancestor in parent.items():children.setdefault(ancestor,set()).add(node)
            count=len(parent)+1
            tokens=torch.tensor([list(range(31800,31800+count))],dtype=torch.long,device=device)
            retrieve,mask,positions=utils._rebuild_tree_buffers_batched(parent,children,count,tokens,None)
            for empty in [False,True]:
                model=SimpleNamespace(_kalman_config={'enabled':True},get_action_dim=lambda:7,
                    _flagos_tree_builder_enabled=False,_flagos_static_tree_runtime_enabled=False,_flagos_kalman_batch_enabled=False,
                    predict_kf_chain_tokens=lambda prefix,n:[] if empty else [prefix[-1]+j+1 for j in range(n)])
                inputs=[x.clone() for x in [tokens,retrieve,mask,positions]]
                original=utils.augment_tree_with_kf_branches(model,*inputs)
                model._flagos_tree_builder_enabled=True
                actual=utils.augment_tree_with_kf_branches(model,*inputs)
                for left,right in zip(original,actual):self.assertTrue(torch.equal(left,right))
                for left,right in zip(inputs,[tokens,retrieve,mask,positions]):self.assertTrue(torch.equal(left,right))

if __name__=='__main__':unittest.main()
