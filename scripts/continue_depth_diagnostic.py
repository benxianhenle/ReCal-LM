"""Continue paired B/C within the unchanged original wall-clock deadline."""
import json,os,signal,subprocess,time
from pathlib import Path
from report_depth_diagnostic import render
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/depth-diagnostic-33291-20260916'
OUT=ROOT/'reports/depth-diagnostic-33291-20260916-continued'
DEADLINE=1789578720.0
if OUT.exists():raise RuntimeError('Refusing to overwrite continuation')
OUT.mkdir()
command=['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts/run_depth_diagnostic.py'),'--out',str(OUT),'--weights','/cloud/cloud-ssd1/recal-experiments/depth-diagnostic-33291-20260916-continued','--deadline',str(DEADLINE),'--tokens-per-arm','3899392','--control-from',str(OLD)]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/'experiment.log').open('a') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launcher.json').write_text(json.dumps(dict(supervisor_pid=os.getpid(),pid=child.pid,command=command,deadline=DEADLINE,started_at=time.time()),indent=2))
    while True:
        remaining=DEADLINE-time.time()-120
        if remaining<=0:
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=max(1,DEADLINE-time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            break
        try:code=child.wait(timeout=min(30,remaining));break
        except subprocess.TimeoutExpired:
            try:render(OUT)
            except Exception as exc:
                with (OUT/'report-errors.log').open('a') as f:f.write(repr(exc)+'\n')
    (OUT/'watchdog-exit.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time(),deadline=DEADLINE),indent=2))
render(OUT)
