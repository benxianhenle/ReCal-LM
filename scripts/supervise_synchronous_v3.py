"""Enforce total wall-clock budget independent of the training process."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/recal-r-v3-20260917'
plan=json.loads((OUT/'plan.json').read_text());deadline=plan['deadline']
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/'experiment.log').open('a') as log:
    child=subprocess.Popen(['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts/run_synchronous_v3.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launcher.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),started_at=time.time(),deadline=deadline),indent=2))
    while True:
        remaining=deadline-time.time()
        if remaining<=180:
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=max(1,remaining))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            break
        try:code=child.wait(timeout=min(20,remaining-180));break
        except subprocess.TimeoutExpired:pass
    (OUT/'supervisor-exit.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time(),deadline=deadline),indent=2))
