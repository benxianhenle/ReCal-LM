"""Independent wall-clock watchdog for the first-round experiment."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/depth-diagnostic-33291-20260916'
DEADLINE=1789578720.0  # 2026-09-16 17:12:00 UTC; includes preparation time.
OUT.mkdir(parents=True,exist_ok=True)
command=['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts/run_depth_diagnostic.py'),'--out',str(OUT),'--weights','/cloud/cloud-ssd1/recal-experiments/depth-diagnostic-33291-20260916','--deadline',str(DEADLINE)]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/'experiment.log').open('a') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launcher.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),pid=child.pid,command=command,deadline=DEADLINE,started_at=time.time()),indent=2))
    try:code=child.wait(timeout=max(1,DEADLINE-time.time()-120))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=max(1,DEADLINE-time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    (OUT/'watchdog-exit.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time(),deadline=DEADLINE),indent=2))

subprocess.run(['/workspace/ai-training/.venv/bin/python',str(ROOT/'scripts/report_depth_diagnostic.py'),str(OUT)],cwd=ROOT,timeout=90)
