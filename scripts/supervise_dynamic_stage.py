"""Independent hard wall-clock limit for one stage; caller applies the gate."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dynamic-depth-20260917'
stage=sys.argv[1];budget=json.loads((OUT/'budget.json').read_text())
if stage=='diagnostics':
    script='run_dynamic_diagnostics.py';deadline=budget['diagnostic_deadline']
elif stage=='controller':
    gate=json.loads((OUT/'G2.json').read_text())
    if not gate['g3_authorized_by_gates']:raise RuntimeError('G2 did not pass')
    script='run_dynamic_controller.py';deadline=min(time.time()+budget['g3_max_seconds'],budget['overall_deadline'])
    (OUT/'controller-budget.json').write_text(json.dumps({'started_at':time.time(),'deadline':deadline},indent=2))
else:raise ValueError(stage)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/f'{stage}.log').open('a') as log:
    child=subprocess.Popen(['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/f'{stage}-launcher.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),deadline=deadline,started_at=time.time()),indent=2))
    try:code=child.wait(timeout=max(1,deadline-time.time()-30))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=max(1,deadline-time.time()))
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    (OUT/f'{stage}-exit.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time(),deadline=deadline),indent=2))
