import sys,json,gc,fcntl
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recal.model.text_state_model import TextStateLM
from recal.model.layers import RotaryEmbedding
from recal.evaluation.state_ablation import evaluate
from recal.training.retention import atomic_json
p=ROOT/'runs/four-model-r3b-10b'
lock=(p/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=p/'checkpoints/step_000033291_late.pt'
b=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
with torch.device('meta'):m=TextStateLM(b['config'])
m.load_state_dict(b['model'],assign=True)
for module in m.modules():
 if isinstance(module,RotaryEmbedding):
  r=RotaryEmbedding(b['config']['hidden_size']//b['config']['num_heads'],b['config']['context_length'])
  module.cos_cached=r.cos_cached;module.sin_cached=r.sin_cached
m.to('cuda').requires_grad_(False).eval()
metadata={k:b[k] for k in ['step','tokens_seen','strategy_schedule','validation_hash','extra_validation_hash']}
del b;gc.collect()
fixed=torch.load(p/'validation.pt',weights_only=True)['batches']
extra=torch.load(ROOT/'reports/attention-aux/validation-extra.pt',weights_only=True)
r=evaluate(m,fixed+extra,len(fixed));r['source']=str(source);r['metadata']=metadata
atomic_json(ROOT/'reports/abc-state-33291/baseline-diagnostics.json',r)
print(json.dumps(r['groups'],indent=2),flush=True)
