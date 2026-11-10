"""Warm resident isolated workers, then serialize three same-input replay rounds.

Workers coexist in memory on the single 80GiB card, but never compute together.
Replay memory is not a single-model memory benchmark. No other jobs are stopped.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time
from acceptance.common import dump, load, rows, sha256, distribution
from acceptance.run_matrix import make_command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--replay-id',default='fixed_input_v1')
    p.add_argument('--profiles',nargs='+',choices=['native','optimized','specvla'],default=['native','optimized','specvla'])
    args=p.parse_args();root=Path('/root/KERV');c=args.campaign.resolve();capture=args.capture.resolve()
    experiment=c/'replays'/args.replay_id
    experiment.mkdir(parents=True,exist_ok=False)
    resource=subprocess.check_output(['mthreads-gmi','-i','0'],text=True)
    (experiment/'device_before.log').write_text(resource)
    if 'No running processes found' not in resource:
        raise RuntimeError('Replay requires an idle card before owned workers are loaded')
    assert (capture/'observations.npz').is_file()
    capture_steps=rows(capture/'steps.jsonl');n=len(capture_steps)
    protocol={'capture':str(capture.relative_to(root)),'capture_sha256':sha256(capture/'observations.npz'),
              'frames':n,'warmup_full_passes_per_profile':1,'measured_full_passes_per_profile':3,'seed':17,
              'order':[args.profiles,list(reversed(args.profiles)),args.profiles],
              'history':'identical captured raw action and Kalman-token prefixes supplied before every frame',
              'cold_model_load_included':False,'profiler_in_timed_passes':False,'runtime_stats_in_timed_passes':False,
              'resident_workers':'co-resident on same physical GPU0; one computes at a time; memory excluded from resource comparisons'}
    dump(experiment/'protocol.json',protocol)
    workers={};logs={};commands={};results=[]
    def wait_for(path,profile,seconds=600):
        start=time.monotonic()
        while not path.exists():
            code=workers[profile].poll()
            if code is not None:raise RuntimeError(profile+' worker exited '+str(code)+' before '+path.name)
            if time.monotonic()-start>seconds:raise TimeoutError(str(path))
            time.sleep(.2)
    try:
        for profile in args.profiles:
            output=experiment/profile;output.mkdir()
            inside='/workspace/'+str(output.relative_to(root));cap='/workspace/'+str(capture.relative_to(root))
            env=['-e','KERV_ACCEPTANCE_CONTEXT=','-e','KERV_REPLAY_PROFILE='+profile,
                 '-e','KERV_REPLAY_CAPTURE='+cap,'-e','KERV_REPLAY_OUTPUT='+inside]
            if profile=='specvla':
                command=['docker','exec',*env,'-e','PYTHONPATH=/workspace/ports/specvla_musa:/workspace/ports/specvla_musa/openvla:/workspace/src',
                         '-e','KERV_DEVICE_BACKEND=musa','kerv-musa-runtime','timeout','--kill-after=30s','2100s',
                         '/workspace/.venv/bin/python','/workspace/src/acceptance/replay_entry.py']
            else:
                case={'profile':profile,'kind':'replay','timeout_seconds':'2100','max_steps':'220','task_id':'0','init_state_id':'0','seed':'17','planning_steps':'220'}
                command=make_command(root,case,output/'context.json','kerv-musa-runtime')
                command=[('KERV_ACCEPTANCE_CONTEXT=' if x.startswith('KERV_ACCEPTANCE_CONTEXT=') else x) for x in command]
                command=command[:2]+env+command[2:]
                command+=['--override','inference.runtime.entrypoint=/workspace/src/acceptance/replay_entry.py']
            dump(output/'command.json',{'argv':command,'cwd':str(root),'start_timestamp':datetime.now(timezone.utc).isoformat()})
            logs[profile]=(output/'launcher.log').open('w');commands[profile]=command
            workers[profile]=subprocess.Popen(command,cwd=root,stdout=logs[profile],stderr=subprocess.STDOUT)
            print('WARM',profile,flush=True)
            wait_for(output/'ready.json',profile)
            print('READY',profile,flush=True)
        for round_id,order in enumerate(protocol['order']):
            for profile in order:
                output=experiment/profile
                print('MEASURE',round_id,profile,flush=True)
                dump(output/('request_%d.json'%round_id),{'round':round_id,'requested_at':datetime.now(timezone.utc).isoformat()})
                wait_for(output/('done_%d.json'%round_id),profile)
                results.append(load(output/('done_%d.json'%round_id)))
        comparisons={}
        for profile in args.profiles:
            data=[row for i in range(3) for row in rows(experiment/profile/('measure_%d.jsonl'%i))]
            assert len(data)==3*n
            comparisons[profile]={'actions':len(data),'model_seconds':distribution(x['model_seconds'] for x in data),
                                  'pipeline_seconds':distribution(x['pipeline_seconds'] for x in data),
                                  'matches_capture':sum(x['tokens_match_capture'] for x in data)}
            original=rows(experiment/profile/'measure_0.jsonl')
            comparisons[profile]['within_profile_repeated_token_matches']=sum(x['action_tokens']==original[x['frame']]['action_tokens'] for x in data)
        if 'native' in args.profiles and 'optimized' in args.profiles:
            native=[row for i in range(3) for row in rows(experiment/'native'/('measure_%d.jsonl'%i))]
            optimized=[row for i in range(3) for row in rows(experiment/'optimized'/('measure_%d.jsonl'%i))]
            matches=sum(a['action_tokens']==b['action_tokens'] for a,b in zip(native,optimized))
            differences=[{'pass':a['pass'],'frame':a['frame'],'native_tokens':a['action_tokens'],'optimized_tokens':b['action_tokens']}
                         for a,b in zip(native,optimized) if a['action_tokens']!=b['action_tokens']]
            pair={'total_actions_per_profile':len(native),'identical_action_token_records':matches,
                  'same_observation_hash_records':sum(a['observation_hash']==b['observation_hash'] for a,b in zip(native,optimized)),
                  'model_mean_speedup':comparisons['native']['model_seconds']['mean']/comparisons['optimized']['model_seconds']['mean'],
                  'pipeline_mean_speedup':comparisons['native']['pipeline_seconds']['mean']/comparisons['optimized']['pipeline_seconds']['mean'],
                  'correctness_status':'PASS' if matches==len(native) else 'FAIL','differences':differences}
            dump(experiment/'native_optimized_comparison.json',pair)
        dump(experiment/'results.json',{'profiles':comparisons,'rounds':results,'protocol':protocol,'scope':'fixed observations; no closed-loop SR or single-model memory claims'})
        print('REPLAY COMPLETE',experiment,flush=True)
    finally:
        for profile in workers:
            (experiment/profile/'stop').touch()
        for profile,process in workers.items():
            try:code=process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                start_file=experiment/profile/'worker_start.json'
                if start_file.exists():
                    pid=int(load(start_file)['pid'])
                    subprocess.run(['docker','exec','kerv-musa-runtime','kill','-TERM',str(pid)])
                code=process.wait(timeout=45)
            logs[profile].close()
            meta=load(experiment/profile/'command.json');meta.update(exit_code=code,end_timestamp=datetime.now(timezone.utc).isoformat())
            dump(experiment/profile/'command.json',meta)


if __name__=='__main__':
    main()
