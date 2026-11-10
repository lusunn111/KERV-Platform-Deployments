"""Compare legacy and batched-host Kalman trees on identical live inputs."""
import atexit
import os
from pathlib import Path
import torch
from acceptance.common import dump
from experiments.robot.libero.run_calibrated_kernel_libero import register
register()
from experiments.robot.libero import run_kerv_libero as evaluation
original_model=evaluation.get_model

def audited(cfg):
    model=original_model(cfg);records=[];root=Path(os.environ['KERV_REPLAY_OUTPUT'])
    def log(name,value):
        import json
        with (root/name).open('a') as f:f.write(json.dumps(value)+'\n')
    def embedding_guard(module,args):
        ids=args[0].detach().cpu()
        valid=not ids.numel() or (int(ids.min())>=0 and int(ids.max())<module.num_embeddings)
        log('embedding_indices.jsonl',{'shape':list(ids.shape),'min':int(ids.min()) if ids.numel() else None,
            'max':int(ids.max()) if ids.numel() else None,'vocab':module.num_embeddings,'valid':valid})
        if not valid:
            torch.save(ids,root/'invalid_embedding_indices.pt')
            raise RuntimeError('Out-of-bounds embedding indices before MUSA launch')
    for submodule in model.modules():
        if isinstance(submodule,torch.nn.Embedding):submodule.register_forward_pre_hook(embedding_guard)
    module=__import__(type(model).__module__,fromlist=['augment_tree_with_kf_branches'])
    original=module.augment_tree_with_kf_branches
    # initialize_tree is defined in utils and resolves this global there.
    # Patching only the re-export in modeling_speculation misses that call.
    owner_module=__import__(original.__module__,fromlist=['augment_tree_with_kf_branches'])
    def checked(owner,*args,**kwargs):
        old_tree=getattr(owner,'_flagos_tree_builder_enabled',False)
        old_batch=getattr(owner,'_flagos_kalman_batch_enabled',False)
        try:
            owner._flagos_tree_builder_enabled=False
            baseline=original(owner,*args,**kwargs)
            owner._flagos_tree_builder_enabled=True;owner._flagos_kalman_batch_enabled=False
            candidate=original(owner,*args,**kwargs)
        finally:
            owner._flagos_tree_builder_enabled=old_tree;owner._flagos_kalman_batch_enabled=old_batch
        checks=[bool(torch.equal(x,y)) for x,y in zip(baseline,candidate)]
        records.append({'input_nodes':args[0].shape[1],'output_shapes':[list(x.shape) for x in candidate],
                        'exact_tokens_retrieve_mask_positions':checks})
        log('tree_checks.jsonl',records[-1])
        if not all(checks):raise RuntimeError('Batched host tree changed buffer values')
        return candidate
    module.augment_tree_with_kf_branches=checked
    owner_module.augment_tree_with_kf_branches=checked
    def final():
        dump(Path(os.environ['KERV_REPLAY_OUTPUT'])/'tree_shadow_checks.json',
             {'scope':'instrumented same-input diagnostic, no performance claim','calls':records,
              'all_buffers_bit_exact':bool(records) and all(all(x['exact_tokens_retrieve_mask_positions']) for x in records)})
    atexit.register(final)
    return model
evaluation.get_model=audited
from acceptance.replay_entry import main
main()
