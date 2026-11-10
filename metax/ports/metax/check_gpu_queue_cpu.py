"""Exercise the real wait/lock/launch loop using mocked device observations only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch
from ports.metax import gpu_queue as q


def scenario(checks, bad_freeze=False):
    with tempfile.TemporaryDirectory(prefix='queue-cpu-',dir=q.ROOT/'outputs') as tmp:
        root=Path(tmp);(root/'docs').mkdir();(root/'docs/METAX_PROGRESS.md').write_text('# CPU fake progress\n')
        out=root/'test';out.mkdir()
        marker=out/'payload.txt'
        job=dict(id='harmless_cpu_marker',command=[sys.executable,'-c',
            'from pathlib import Path; Path('+repr(str(marker))+').write_text("CPU gate test only")'])
        (out/'queue.json').write_text(json.dumps(dict(jobs=[job])))
        counter={'observations':0,'sleeps':0,'freeze_checks':0}
        observations=iter(checks)
        def device():
            counter['observations']+=1
            assert counter['observations']<20,'Queue test failed to terminate'
            return next(observations,dict(valid=True,pids=[],memory_kib=892928))
        def sleep(seconds):
            assert seconds==60
            counter['sleeps']+=1
            assert counter['sleeps']<20
        def freeze(plan):
            counter['freeze_checks']+=1
            assert counter['observations']>=3, 'Launched without two polls and recheck'
            if bad_freeze:raise RuntimeError('intentional frozen source mismatch')
        with patch.object(q,'ROOT',root),patch.object(q,'LOCK',root/'card0.lock'), \
             patch.object(q,'device',device),patch.object(q.time,'sleep',sleep), \
             patch.object(q,'verify_freeze',freeze), \
             patch.object(sys,'argv',['gpu_queue','--directory',str(out)]):
            q.main()
        state=json.loads((out/'queue-state.json').read_text())
        events=[json.loads(x) for x in (out/'queue-events.jsonl').read_text().splitlines()]
        assert all(x['status']!='running' for x in events[:2])
        if bad_freeze:
            assert state['status']=='failed' and not marker.exists()
        else:
            assert marker.read_text()=='CPU gate test only' and state['status']=='completed'
            assert state['jobs']['harmless_cpu_marker']['exit_code']==0
        return dict(status=state['status'],**counter,
            payload_executed=marker.exists(),waiting_events=sum(x['status']=='waiting_gpu' for x in events))

def main():
    idle=dict(valid=True,pids=[],memory_kib=892928)
    busy=dict(valid=True,pids=[12345],memory_kib=57_000_000)
    results=dict(kind='CPU mocked device states; real lock and harmless CPU child, no GPU allocation',
        busy_then_two_idle=scenario([busy,idle,idle,idle]),
        lock_recheck_race=scenario([idle,idle,busy,idle,idle,idle]),
        source_mismatch=scenario([idle,idle,idle],True))
    assert results['lock_recheck_race']['observations']>=6
    target=q.ROOT/'outputs/metax_model_performance_20261105/queue-cpu-checks.json'
    target.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

if __name__=='__main__':main()
