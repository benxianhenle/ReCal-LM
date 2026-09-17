"""Paired, bounded attention-supervision experiment; no automatic long-run restart."""
import copy, gc, hashlib, json, math, sys, time, fcntl
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recal.model.four_model import FourModelLM
from recal.model.layers import RotaryEmbedding
from recal.data.stateful import TextEncoder, make_stream
from recal.training.scheduler import cosine_lr
from recal.training.retention import CheckpointStore, atomic_json
from train_four_models import rng_state, restore_rng
RUN = ROOT/'runs/four-model-r3b-10b'
OUT = ROOT/'reports/attention-aux'
OUT.mkdir(parents=True, exist_ok=True)
lock = (RUN/'.train.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
store = CheckpointStore(RUN/'checkpoints')
source = RUN/'checkpoints'/store.manifest['roles']['latest']
encoder = TextEncoder(str(ROOT/'artifacts/four-model-v1/tokenizer.json'))
data = str(ROOT/'artifacts/four-model-v1/data_manifest.json')
fixed = torch.load(RUN/'validation.pt', weights_only=True)['batches']
extra_file = OUT/'validation-extra.pt'
if extra_file.exists():
    extra = torch.load(extra_file, weights_only=True)
else:
    stream = make_stream(data, encoder, 512, 92341, 'val')
    extra = [stream.next_batch(1) for _ in range(24)]
    torch.save(extra, extra_file)
result = {'source':str(source), 'steps_per_arm':100, 'auxiliary_weight':0.1,
          'optimizer_reset_both_arms':True, 'extra_validation_batches':24,
          'extra_validation_note':'Same document-level held-out split, different sampler seed; not guaranteed disjoint from original validation.',
          'arms':{}}
amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
def emit(record):
    with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
def evaluate(model):
    saved_rng = rng_state()
    model.eval()
    records = []
    with torch.no_grad(), amp():
        for idx,(xc,yc) in enumerate(fixed+extra):
            x,y=xc.cuda(),yc.cuda()
            hops=model.config['rollout_steps']; width=x.shape[1]-hops
            a=model.attention(x); state=a[:,:width]
            for hop in range(1,hops+1):
                pos=torch.arange(hop,hop+width,device='cuda')
                state=model.core(state,x[:,hop:hop+width],pos)
            target=y[:,hops:hops+width]
            ce=lambda logits: float(F.cross_entropy(logits.float().flatten(0,1),target.flatten()))
            lm=ce(model.decoder(state,pos)); direct=ce(model.decoder(a[:,hops:hops+width],pos))
            af=a[:,hops:hops+width].float().flatten(0,1)
            unit=F.normalize(af,dim=-1); n=af.shape[0]
            cosine=float((unit.sum(0).square().sum()-n)/(n*(n-1)))
            centered=af-af.mean(0,keepdim=True); gram=centered@centered.T
            rank=float(gram.trace().square()/gram.square().sum().clamp_min(1e-12))
            energy=float(centered.square().sum()/af.square().sum().clamp_min(1e-12))
            records.append(dict(lm=lm,attention_ce=direct,cosine=cosine,rank=rank,centered_energy=energy))
    model.train(); restore_rng(saved_rng)
    return {name:{key:sum(r[key] for r in subset)/len(subset) for key in records[0]} for name,subset in [('fixed',records[:8]),('extra',records[8:])]} | {'batches':records}
for arm, weight in [('control',0.0),('auxiliary',0.1)]:
    base=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    config=copy.deepcopy(base['config']); config['lambda_attention_lm']=weight
    train=config['training']
    with torch.device('meta'): model=FourModelLM(config,stage='late')
    model.load_state_dict(base['model'],assign=True)
    for module in model.modules():
        if isinstance(module,RotaryEmbedding):
            fresh=RotaryEmbedding(config['hidden_size']//config['num_heads'],config['context_length'])
            module.cos_cached=fresh.cos_cached; module.sin_cached=fresh.sin_cached
    model.to('cuda').train()
    stream=make_stream(data,encoder,train['seq_len'],1337,'train')
    stream.load_state_dict(base['data_state'])
    restore_rng(base['rng'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],betas=(.9,.95),weight_decay=train['weight_decay'],foreach=False)
    start_step=base['step']; token_step=train['seq_len']*train['batch_size']*train['grad_accum']
    result['arms'][arm]={'evaluations':{},'batch_hashes':[]}
    ar=result['arms'][arm]
    ar['evaluations']['0']=evaluate(model)
    emit({'arm':arm,'step':0,'validation':ar['evaluations']['0']})
    start=time.monotonic()
    for local in range(1,101):
        optimizer.zero_grad(set_to_none=True)
        sums={}; digest=hashlib.sha256()
        for _ in range(train['grad_accum']):
            x,y=stream.next_batch(train['batch_size'])
            digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
            with amp(): out=model(x.cuda(),y.cuda()); loss=out['loss']/train['grad_accum']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            for k in ['loss','loss_lm','loss_attention_lm','loss_drift']:
                sums[k]=sums.get(k,0)+float(out[k].detach())/train['grad_accum']
            del out,loss
        grads={}
        for name in ['attention','decoder','drift']:
            norms=[p.grad.float().norm() for p in getattr(model,name).parameters() if p.grad is not None]
            grads[name]=float(torch.stack(norms).norm())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr=cosine_lr(start_step+local-1,train['learning_rate'],train['warmup_steps'],math.ceil(base['target_tokens']/token_step))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        ar['batch_hashes'].append(digest.hexdigest())
        emit(dict(arm=arm,step=local,global_step=start_step+local,grad_norm=float(norm),module_grad_norm=grads,elapsed=time.monotonic()-start,**sums))
        if local in [50,100]:
            ar['evaluations'][str(local)]=evaluate(model)
            emit({'arm':arm,'step':local,'validation':ar['evaluations'][str(local)]})
            atomic_json(OUT/'results.json',result)
    if arm=='auxiliary':
        result['identical_training_batches']=ar['batch_hashes']==result['arms']['control']['batch_hashes']
        end=ar['evaluations']['100']; control=result['arms']['control']['evaluations']['100']; initial=ar['evaluations']['0']
        # Only promote if BOTH validation samples improve vs control and starting weights.
        promote=result['identical_training_batches'] and all(end[k]['lm']<min(control[k]['lm'],initial[k]['lm']) for k in ['fixed','extra'])
        result['candidate_promoted']=promote
        if promote:
            payload={k:v for k,v in base.items() if k not in ['model','optimizer','retention']}
            payload.update(model=model.state_dict(),optimizer=None,config=config,step=start_step+100,
                tokens_seen=base['tokens_seen']+100*token_step,data_state=stream.state_dict(),rng=rng_state(),
                optimizer_reset_on_resume=True,attention_aux_experiment=str(OUT/'results.json'))
            del base;gc.collect()
            emit({'event':'saving_improved_auxiliary_candidate'})
            path=store.save(payload,end['fixed']['lm'])
            result['candidate_checkpoint']=str(path)
            import yaml
            (OUT/'accepted_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            # The existing launcher needs a config exactly matching the saved candidate.
            (ROOT/'configs/four_model_3b.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            del payload
        else: del base
        atomic_json(OUT/'results.json',result)
    else: del base
    del optimizer,model,stream;gc.collect();torch.cuda.empty_cache()
emit({'event':'complete','candidate_promoted':result['candidate_promoted']})
