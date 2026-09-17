"""Detached E1 launcher with a two-hour wall-clock cap."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/v3-e1-warmstart-20260917'
if (OUT/'launcher.json').exists():
    raise RuntimeError('Already launched')
plan=json.loads((OUT/'plan.json').read_text())
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/'experiment.log').open('a') as log:
    child=subprocess.Popen(['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts/run_v3_e1_warmstart.py'),'--out',str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launcher.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),started_at=time.time(),deadline=plan['deadline']),indent=2))
    sent=False
    while child.poll() is None:
        remaining=plan['deadline']-time.time()
        if remaining<=180 and not sent:
            os.killpg(child.pid,signal.SIGTERM);sent=True
        if remaining<=0:
            os.killpg(child.pid,signal.SIGKILL)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
    (OUT/'supervisor-exit.json').write_text(json.dumps(dict(returncode=child.returncode,finished_at=time.time()),indent=2))
