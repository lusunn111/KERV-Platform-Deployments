"""Shared estimates must preserve scalar tokens, including hook fallbacks."""
import json
from pathlib import Path
from types import MethodType
import unittest
import numpy as np
import torch
from openvla.prismatic.extern.hf.modeling_speculation import SpecVLAforActionPrediction as Model
from openvla.prismatic.extern.hf import utils


class KalmanSharedHistoryTests(unittest.TestCase):
    def make(self):
        model=Model.__new__(Model);torch.nn.Module.__init__(model)
        model._kalman_config={'enabled':True,'process_var':1.,'measurement_var':.001}
        model._store_history_flag=True;model._token_history=[];model._kalman_history_window=None
        model.vocab_size=32000;model.bin_centers=np.zeros(256)
        model.get_action_dim=lambda:7
        model.tokenizer=type('Tokenizer',(),{'pad_token_id':0})()
        return model

    def check(self,model):
        prefixes=[[31800+j for j in range(depth+1)] for depth in range(8)]
        lengths=list(range(8))
        actual=model.predict_kf_chains_batched(prefixes,lengths)
        expected=[model.predict_kf_chain_tokens(p,n) for p,n in zip(prefixes,lengths)]
        self.assertEqual(actual,expected)

    def test_every_captured_history_and_windows(self):
        path=Path('/workspace/outputs/calibrated_quantization/20261026T045030Z_v2/inputs/diagnostic_capture/runtime_stats.jsonl')
        histories=[json.loads(s)['kalman_history_token_ids'] for s in path.read_text().splitlines()]
        model=self.make()
        for window in (None,1,7):
            model._kalman_history_window=window
            for i in range(len(histories)+1):
                model._token_history=histories[:i];self.check(model)

    def test_missing_malformed_history_disabled_and_oversized_tail(self):
        model=self.make()
        for history in ([],[None,[],[1]],[[31800]*7,None,[],[2]]):
            model._token_history=history;self.check(model)
        model._store_history_flag=False;self.check(model)
        model._kalman_config['enabled']=False;self.check(model)
        model._kalman_config['enabled']=True;model._store_history_flag=True
        model._token_history=[[31800]*7]
        self.assertEqual(model.predict_kf_chains_batched([[31900]],[9]),
                         [model.predict_kf_chain_tokens([31900],9)])

    def test_prefix_dependent_custom_hook_uses_scalar_calls(self):
        model=self.make();calls=[]
        def custom(self,**kwargs):
            calls.append(kwargs['prefix_tokens'])
            return [kwargs['prefix_tokens'][-1]]*kwargs['remaining_dims']
        model.kalman_predict_tokens=MethodType(custom,model)
        self.assertEqual(model.predict_kf_chains_batched([[1],[2]],[2,3]),[[1,1],[2,2,2]])
        self.assertEqual(calls,[[1],[2]])

    def test_complete_augmented_tree_buffers(self):
        model=self.make();model._kalman_tree_enabled=True
        model._flagos_tree_builder_enabled=True;model._flagos_static_tree_runtime_enabled=False
        parents={1:0,2:0,3:1,4:1,5:2,6:3,7:6}
        children={}
        for node,parent in parents.items():children.setdefault(parent,set()).add(node)
        tokens=torch.tensor([[31800+i for i in range(8)]])
        retrieve,mask,positions=utils._rebuild_tree_buffers_batched(parents,children,8,tokens,None)
        for history in ([],[[31800+i for i in range(7)]],[[31800+i+j for i in range(7)] for j in range(141)]):
            model._token_history=history;model._flagos_kalman_batch_enabled=False
            expected=utils.augment_tree_with_kf_branches(model,tokens,retrieve,mask,positions)
            model._flagos_kalman_batch_enabled=True
            actual=utils.augment_tree_with_kf_branches(model,tokens,retrieve,mask,positions)
            for x,y in zip(expected,actual):self.assertTrue(torch.equal(x,y))


if __name__=='__main__':unittest.main()
