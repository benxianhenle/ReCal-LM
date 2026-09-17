"""Hard stage deadlines for zero-training V2 diagnostics."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dynamic-r-v2-20260917'
budget=json.loads((OUT/'budget.json').read_text());env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (OUT/'diagnostics.log').open('a') as log:
    child=subprocess.Popen(['/workspace/ai-training/.venv/bin/python','-u',str(ROOT/'scripts/run_dynamic_v2_diagnostics.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'launcher.json').write_text(json.dumps(dict(pid=child.pid,supervisor_pid=os.getpid(),started_at=time.time(),overall_deadline=budget['overall_deadline']),indent=2))
    while True:
        stage_file=OUT/'stage-deadlines.json'
        deadline=json.loads(stage_file.read_text())['v21_deadline'] if stage_file.exists() else budget['v20_deadline']
        remaining=min(deadline,budget['overall_deadline'])-time.time()
        if remaining<=30:
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=max(1,remaining))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
            break
        try:code=child.wait(timeout=min(10,remaining-30));break
        except subprocess.TimeoutExpired:pass
    (OUT/'diagnostics-exit.json').write_text(json.dumps(dict(returncode=code,finished_at=time.time(),deadline=deadline),indent=2))
