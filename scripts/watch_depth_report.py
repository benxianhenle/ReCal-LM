"""Create a final report even if the interactive client disconnects."""
import json,time
from pathlib import Path
from report_depth_diagnostic import render
out=Path(__file__).resolve().parents[1]/'reports/depth-diagnostic-33291-20260916'
deadline=1789578720
while time.time()<deadline+60:
    if (out/'baseline.json').exists():render(out)
    if (out/'watchdog-exit.json').exists() or (out/'completion.json').exists():break
    time.sleep(30)
