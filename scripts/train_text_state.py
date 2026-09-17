"""Resume R3B with delta decoding and token-ramped, gradient-isolated supervision."""
import argparse,copy,dataclasses,fcntl,gc,hashlib,json,math,os,signal,sys,time
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recal.model.text_state_model import TextStateLM
from recal.model.layers import RotaryEmbedding
from recal.data.stateful import TextEncoder,make_stream
from recal.training.retention import CheckpointStore,atomic_json
from recal.training.text_state_schedule import TextStateSchedule
from recal.training.scheduler import cosine_lr
from train_four_models import rng_state,restore_rng


def main():
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('--config',default=str(ROOT/'configs/four_model_text_state_3b.yaml'))
 p.add_argument('--max-updates',type=int)
 p.add_argument('--log-every',type=int,default=20)
 args=p.parse_args()
 if args.log_every<1:raise ValueError('--log-every must be positive')
 config=yaml.safe_load(Path(args.config).read_text());train=config['training']
 old=ROOT/'runs/four-model-r3b-10b';out=ROOT/'runs/four-model-text-state-10b';out.mkdir(parents=True,exist_ok=True)
 lock=(old/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock2=(out/'.train.lock').open('a');fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
 store=CheckpointStore(old/'checkpoints')
 source=old/'checkpoints'/store.manifest['roles']['latest']
 base=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
 is_resume=base.get('strategy')=='text_state_v1'
 if is_resume and base['config']!=config:raise ValueError('Resume configuration mismatch')
 if not is_resume and base['step']!=12810:raise ValueError('Unexpected original strategy baseline')
 encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
 stream=make_stream(ROOT/'artifacts/four-model-v1/data_manifest.json',encoder,train['seq_len'],1337,'train')
 stream.load_state_dict(base['data_state'])
 validation_file=old/'validation.pt'
 digest=hashlib.sha256(validation_file.read_bytes()).hexdigest()
 if digest!=base['validation_hash']:raise ValueError('Original validation changed')
 fixed=torch.load(validation_file,weights_only=True)['batches']
 extra_file=ROOT/'reports/attention-aux/validation-extra.pt'
 extra=torch.load(extra_file,weights_only=True)
 extra_digest=hashlib.sha256(extra_file.read_bytes()).hexdigest()
 if is_resume and base['extra_validation_hash']!=extra_digest:raise ValueError('Extra validation changed')
 with torch.device('meta'):model=TextStateLM(config)
 model.load_state_dict(base['model'],assign=True)
 # assign=True preserves module requires_grad; ensure the legacy embedding stays frozen.
 model.core.embedding.requires_grad_(False)
 for module in model.modules():
  if isinstance(module,RotaryEmbedding):
   fresh=RotaryEmbedding(config['hidden_size']//config['num_heads'],config['context_length'])
   module.cos_cached=fresh.cos_cached;module.sin_cached=fresh.sin_cached
 model.to('cuda').train()
 metadata={k:v for k,v in base.items() if k not in ('model','optimizer','retention')}
 del base;gc.collect()
 schedule=TextStateSchedule(**metadata['strategy_schedule']) if is_resume else TextStateSchedule(metadata['tokens_seen'])
 step=metadata['step'];tokens=metadata['tokens_seen'];initial_tokens=tokens
 params=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=train['learning_rate'],betas=(.9,.95),weight_decay=train['weight_decay'],foreach=False)
 restore_rng(metadata['rng'])
 atomic_json(out/'run_config.json',dict(config=config,source=str(source),optimizer_reset=True,started_at=time.time(),pid=os.getpid(),
  parameters={name:dict(total=sum(p.numel() for p in mod.parameters()),trainable=sum(p.numel() for p in mod.parameters() if p.requires_grad)) for name,mod in model.named_children()}))
 if not is_resume:
  # The baseline weights already exist; duplicate neither them nor nonexistent Adam moments.
  torch.save(dict(metadata,optimizer=None,lr_schedule=dict(name='cosine_lr',step=step,total_tokens=metadata['target_tokens'],training=metadata['config']['training'])),out/'original-baseline-metadata.pt')
 stopped=False
 def stop(*unused):
  nonlocal stopped
  stopped=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
 def evaluate():
  saved=rng_state();model.eval();rows=[]
  with torch.no_grad(),amp():
   for x,y in fixed+extra:
    r=model(x.cuda(),y.cuda(),diagnostics=True)
    rows.append({k:float(v) for k,v in r.items() if k.startswith(('ce_depth_','cos_depth_','norm_ratio_','state_norm_','delta_norm_','token_variance_','attention_variance_','initial_attention_'))})
  model.train();restore_rng(saved)
  averaged={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
  averaged['fixed_final_ce']=sum(r['ce_depth_3'] for r in rows[:len(fixed)])/len(fixed)
  averaged['extra_final_ce']=sum(r['ce_depth_3'] for r in rows[len(fixed):])/len(extra)
  if not all(math.isfinite(v) for v in averaged.values()):raise FloatingPointError('Nonfinite validation')
  return averaged
 start=time.monotonic();last_saved=step;updates=0;last_val=None
 def save(val):
  nonlocal last_saved
  if not is_resume and 'text_state_baseline' not in store.manifest:
   # Re-rank only the new objective. Pin the original weight file as the new boundary.
   # Publish this change atomically together with the first successful new checkpoint.
   store.manifest=copy.deepcopy(store.manifest)
   store.manifest['roles']=dict(latest=source.name,best=source.name,transition_before=source.name)
   store.manifest['snapshots'][source.name]['loss']=schedule.baseline['fixed_final_ce']
   store.manifest['text_state_baseline']=source.name
   store.manifest.pop('manual_transition',None)
  payload=dict(model=model.state_dict(),optimizer=None,strategy='text_state_v1',stage='late',step=step,tokens_seen=tokens,
   target_tokens=metadata['target_tokens'],config=config,strategy_schedule=dataclasses.asdict(schedule),
   data_state=stream.state_dict(),rng=rng_state(),validation_hash=digest,extra_validation_hash=extra_digest,
   tokenizer_hash=encoder.fingerprint,optimizer_reset_on_resume=True,wall_seconds=time.monotonic()-start)
  print(f'SAVING step={step} fixed_final_ce={val["fixed_final_ce"]:.6f}',flush=True)
  path=store.save(payload,val['fixed_final_ce']);last_saved=step
  print(f'SAVED {path}',flush=True)
 pending_logs=[]
 def emit(record):
  if 'loss_lm' in record:pending_logs.append(record)
  immediate=('validation' in record or record.get('state')!='running')
  if not immediate and record['step']%args.log_every:
   return
  if pending_logs:
   record=dict(record,log_window=dict(steps=len(pending_logs),
    first_step=pending_logs[0]['step'],last_step=pending_logs[-1]['step'],
    mean_loss_lm=sum(r['loss_lm'] for r in pending_logs)/len(pending_logs),
    max_grad_norm=max(r['grad_norm'] for r in pending_logs)))
   pending_logs.clear()
  with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  atomic_json(out/'status.json',dict(record,pid=os.getpid(),updated_at=time.time()))
  print(json.dumps(record),flush=True)
 last_val=evaluate()
 if not is_resume:
  schedule.observe(last_val,tokens)
  atomic_json(out/'initial_validation.json',last_val)
 emit(dict(event='initial_validation',step=step,tokens_seen=tokens,validation=last_val,state='running'))
 token_step=train['seq_len']*train['batch_size']*train['grad_accum']
 total_steps=math.ceil(metadata['target_tokens']/token_step)
 (out/'completion.json').unlink(missing_ok=True)
 while tokens<metadata['target_tokens'] and not stopped and not schedule.stop:
  coeff,phase=schedule.coefficients(tokens)
  optimizer.zero_grad(set_to_none=True);metrics={}
  for _ in range(train['grad_accum']):
   x,y=stream.next_batch(train['batch_size'])
   with amp():r=model(x.cuda(),y.cuda(),**coeff);loss=r['loss']/train['grad_accum']
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
   loss.backward()
   for k,v in r.items():
    if k!='logits':metrics[k]=metrics.get(k,0.)+float(v.detach())/train['grad_accum']
   tokens+=x.numel();del r,loss
  grads={name:float(torch.stack([p.grad.float().norm() for p in mod.parameters() if p.grad is not None]).norm()) for name,mod in model.named_children() if any(p.grad is not None for p in mod.parameters())}
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
  lr=cosine_lr(step,train['learning_rate'],train['warmup_steps'],total_steps)
  for g in optimizer.param_groups:g['lr']=lr
  optimizer.step();optimizer.zero_grad(set_to_none=True)
  step+=1;updates+=1
  final=stopped or (args.max_updates is not None and updates>=args.max_updates) or tokens>=metadata['target_tokens']
  # Observe early adaptation more often; normal cadence resumes after 500 updates.
  strategy_updates=(tokens-schedule.start_tokens)//token_step
  validate=(strategy_updates in (1,20,100,200,300,400,500) or step%train['validation_interval']==0 or final)
  if validate:
   last_val=evaluate();schedule.observe(last_val,tokens)
   metrics['validation']=last_val;metrics['alerts']=schedule.alerts
   atomic_json(out/'strategy_schedule.json',dataclasses.asdict(schedule))
   if updates>=20 or final or schedule.stop:save(last_val)
  emit(dict(metrics,step=step,tokens_seen=tokens,phase=phase,coefficients=coeff,learning_rate=lr,
   grad_norm=float(norm),module_grad_norm=grads,tokens_per_second=(tokens-initial_tokens)/max(time.monotonic()-start,1.),
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),state='stopping' if final or schedule.stop else 'running'))
  if final:break
 if step>last_saved:save(evaluate())
 atomic_json(out/'completion.json',dict(step=step,tokens_seen=tokens,stopped_by_signal=stopped,
  stopped_by_guard=schedule.stop,target_reached=tokens>=metadata['target_tokens'],finished_at=time.time()))

if __name__=='__main__':main()
