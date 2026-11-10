"""Validate source snapshots, matching raw rollout evidence and index retained files."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2];INV=ROOT/'outputs/metax_inventory'
primary=['metax_native_goal3_init01_02','metax_optimized_goal3_init01_02',
         'metax_w8a16_packed_goal3_init01','metax_bf16_lossless_goal3_init01']
replays=['metax_replay_AB_01','metax_replay_BA_01','metax_quant_replay_AB_02','metax_quant_replay_BA_02',
         'metax_lossless_replay_AB_01','metax_lossless_replay_BA_01']
sha=lambda data:hashlib.sha256(data).hexdigest()
readrows=lambda path:[json.loads(line) for line in path.read_text().splitlines() if line]
checks=[]
for name in primary:
    run=ROOT/'outputs'/name
    assert (run/'exit_code.txt').read_text().strip()=='0',name
    launch=json.loads((run/'launch.json').read_text())
    assert launch['physical_gpu']=='0' and launch['logical_device']=='cuda:0'
    assert 'metax' in launch['torch'] and launch['seed']==7 and launch['planning_horizon']==220
    manifest=json.loads((run/'source-state.json').read_text())['files']
    with tarfile.open(run/'source.tar.gz') as archive:
        for member in archive:
            assert not any(member.name.endswith(suffix) for suffix in ['.safetensors','.bin','.pt','.pth']),member.name
            assert member.isfile(),member.name
            assert sha(archive.extractfile(member).read())==manifest[member.name],member.name
    checks.append(dict(run=name,source_files_checked=len(manifest),exit_zero=True,single_metax_gpu=True))
base=readrows(ROOT/'outputs'/primary[1]/'raw/steps.jsonl')
fields=['task_id','init_state_id','step_id','observation_hash','action_tokens','raw_action','executed_action','done','threshold']
for name in [primary[0],primary[3]]:
    records=readrows(ROOT/'outputs'/name/'raw/steps.jsonl')
    assert len(records)==len(base)
    assert all(all(a[key]==b[key] for key in fields) for a,b in zip(base,records)),name
for name in replays:
    report=json.loads((ROOT/'outputs'/name/'benchmark.json').read_text())
    assert report['samples']==8 and report['repeats']==3 and report['warmup_sequences']==1
    for profile in report['profiles'].values():
        assert len(profile['raw_model_ms'])==24 and len(profile['raw_pipeline_ms'])==24
    if report.get('comparison','native_runtime')!='runtime_w8a16':assert report['actions_exact']
trace=json.loads((ROOT/'outputs/metax_device_diagnostic_02/placement_and_costs.json').read_text())
assert len(trace)==4 and all(row['kernel_event_count']>0 and row['parameter_devices']==['cuda:0'] for row in trace)
paths=[]
for directory in primary+replays+['metax_device_diagnostic_02','metax_failure_probe_01']:
    paths.extend(path for path in (ROOT/'outputs'/directory).rglob('*') if path.is_file())
paths.extend(path for path in INV.iterdir() if path.is_file() and path.name not in {'evidence_index.json','checksums.sha256','validation.json','validation-stdout.txt'})
paths.extend(path for path in (ROOT/'ports/metax').iterdir() if path.is_file())
paths.extend([ROOT/'docs/METAX_TECHNICAL_REPORT.md',ROOT/'docs/METAX_PROGRESS.md',ROOT/'scripts/run_kerv_metax.sh',ROOT/'scripts/benchmark_kerv_metax.sh',ROOT/'scripts/profile_kerv_metax.sh'])
index=[]
for path in sorted(set(paths)):
    index.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path.read_bytes())))
(INV/'evidence_index.json').write_text(json.dumps(index,indent=2)+'\n')
(INV/'checksums.sha256').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in index))
validation=dict(passed=True,source_snapshots=checks,paired_raw_steps=len(base),
                exact_groups=['B1','B2','Q2 lossless BF16'],replay_rounds=len(replays),
                real_device_traces=len(trace),indexed_files=len(index),B0='pending boundary choice, not evaluated')
(INV/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
