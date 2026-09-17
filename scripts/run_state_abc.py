"""Bounded A/B/C comparison; never edits or prunes the original checkpoint store."""
import argparse,copy,dataclasses,fcntl,gc,hashlib,json,math,os,signal,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recal.model.text_state_model import TextStateLM
from recal.model.layers import RotaryEmbedding
from recal.evaluation.state_ablation import evaluate
from recal.training.retention import atomic_json
from recal.training.text_state_schedule import TextStateSchedule
from recal.training.scheduler import cosine_lr
from recal.data.stateful import TextEncoder,make_stream
from scripts.train_four_models import rng_state,restore_rng
OUT=ROOT/'reports/abc-state-33291'
SOURCE=ROOT/'runs/four-model-r3b-10b/checkpoints/step_000033291_late.pt'


def coefficients(arm,schedule,tokens,initial_lambda):
    original,_=schedule.coefficients(tokens)
    # All arms keep alpha and R identical; only A state alignment differs.
    return dict(alpha=.1,lambda_a=original['lambda_a'] if arm=='A' else 0. if arm=='B' else initial_lambda*.5,lambda_r=0.)


def schedule_metrics(report):
    m=report['groups']['all'];out={}
    for k in range(1,4):
        out[f'ce_depth_{k}']=m[f'ce_depth_{k}'];out[f'cos_depth_{k}']=m[f'cos_depth_{k}']
        for source,target in [('state_mean_norm','state_norm'),('state_token_variance','token_variance'),('delta_mean_norm','delta_norm')]:
            out[f'{target}_depth_{k}']=m[f'{source}_{k}']
    return out


def summarize_updates(updates):
    n=len(updates)
    return dict(updates=n,mean_loss_lm=sum(r['loss_lm'] for r in updates)/n,
        clip_ratio=sum(r['grad_norm']>1. for r in updates)/n,
        mean_clip_multiplier=sum(min(1.,1./max(r['grad_norm'],1e-12)) for r in updates)/n,
        max_grad_norm=max(r['grad_norm'] for r in updates),
        spikes_over_100=sum(r['grad_norm']>100 for r in updates),
        spikes_over_1000=sum(r['grad_norm']>1000 for r in updates),
        module_grad_mean={k:sum(r['module_grad_norm'][k] for r in updates)/n for k in updates[0]['module_grad_norm']},
        module_grad_max={k:max(r['module_grad_norm'][k] for r in updates) for k in updates[0]['module_grad_norm']})


def load_model():
    b=torch.load(SOURCE,map_location='cpu',weights_only=True,mmap=True)
    with torch.device('meta'):m=TextStateLM(b['config'])
    m.load_state_dict(b['model'],assign=True)
    for module in m.modules():
        if isinstance(module,RotaryEmbedding):
            fresh=RotaryEmbedding(b['config']['hidden_size']//b['config']['num_heads'],b['config']['context_length'])
            module.cos_cached=fresh.cos_cached;module.sin_cached=fresh.sin_cached
    m.to('cuda').train()
    metadata={k:v for k,v in b.items() if k not in ('model','optimizer','retention')}
    del b;gc.collect()
    return m,metadata


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--tokens-per-arm',type=int,default=5_000_000)
    args=parser.parse_args()
    if args.tokens_per_arm<=0:raise ValueError('Positive budget required')
    OUT.mkdir(parents=True,exist_ok=True)
    if any((OUT/arm/'metrics.jsonl').exists() for arm in ['A','B','C']):
        raise RuntimeError('Experiment already started; do not mix attempts in existing metrics')
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ownlock=(OUT/'.experiment.lock').open('a');fcntl.flock(ownlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Preserve a manifest/file-stat inventory. No CheckpointStore mutation is used.
    cp=SOURCE.parent
    inventory={'manifest':(cp/'manifest.json').read_text(),'files':{p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in cp.glob('*.pt')}}
    atomic_json(OUT/'protected-checkpoints.json',inventory)
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
    data=ROOT/'artifacts/four-model-v1/data_manifest.json'
    vf=ROOT/'runs/four-model-r3b-10b/validation.pt';ef=ROOT/'reports/attention-aux/validation-extra.pt'
    fixed=torch.load(vf,weights_only=True)['batches'];extra=torch.load(ef,weights_only=True);batches=fixed+extra
    validation_hash=hashlib.sha256(vf.read_bytes()).hexdigest();extra_hash=hashlib.sha256(ef.read_bytes()).hexdigest()
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    overall_start=time.monotonic();completed={};control_hashes=None;control_lrs=None
    for arm in ['A','B','C']:
        dest=OUT/arm;dest.mkdir(exist_ok=True)
        if (dest/'result.json').exists():raise RuntimeError('Existing experiment results: use a separately reviewed rerun, never silently overwrite')
        def event(record):
            record=dict(record,arm=arm,pid=os.getpid(),updated_at=time.time())
            with (dest/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(OUT/'status.json',record);print(json.dumps(record),flush=True)
        event(dict(event='loading',state='running'))
        model,base=load_model();config=base['config'];train=config['training']
        if base['step']!=33291 or base['validation_hash']!=validation_hash or base['extra_validation_hash']!=extra_hash:raise ValueError('Experiment source or validation changed')
        schedule=TextStateSchedule(**base['strategy_schedule'])
        initial_lambda=schedule.coefficients(base['tokens_seen'])[0]['lambda_a']
        step_tokens=train['seq_len']*train['batch_size']*train['grad_accum'];updates=math.ceil(args.tokens_per_arm/step_tokens)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=train['learning_rate'],betas=(.9,.95),weight_decay=train['weight_decay'],foreach=False)
        stream=make_stream(data,encoder,train['seq_len'],1337,'train');stream.load_state_dict(base['data_state']);restore_rng(base['rng'])
        initial_rng=rng_state();baseline=evaluate(model,batches,len(fixed));restore_rng(initial_rng)
        atomic_json(dest/'initial-diagnostics.json',baseline)
        expected=json.loads((OUT/'baseline-diagnostics.json').read_text())['groups']['fixed']['ce_depth_3']
        if abs(baseline['groups']['fixed']['ce_depth_3']-expected)>1e-5:raise RuntimeError('Initial model failed baseline reproduction')
        plan=dict(source=str(SOURCE),arm=arm,initial_step=base['step'],initial_tokens=base['tokens_seen'],
            updates=updates,actual_tokens_per_arm=updates*step_tokens,initial_lambda_a=initial_lambda,
            initial_coefficients=coefficients(arm,schedule,base['tokens_seen'],initial_lambda),
            optimizer_reset=True,weak_fraction=.5,config=config,validation_hash=validation_hash,extra_validation_hash=extra_hash,
            policy='Same source, RNG, data order, LR horizon and Adam reset. A retains original A ramp and existing validation guards; B zero A alignment; C fixed half starting A alignment. Alpha .1 and lambda_R 0 in all arms.')
        atomic_json(dest/'plan.json',plan)
        event(dict(event='baseline',local_step=0,state='running',coefficients=plan['initial_coefficients'],validation={k:{m:v for m,v in g.items() if m in ['ce_depth_1','ce_depth_2','ce_depth_3','depth_1_to_3_gain']} for k,g in baseline['groups'].items()}))
        start=time.monotonic();records=[];pending=[];hashes=[];lrs=[];evaluations=[];tokens=base['tokens_seen'];local=0;last_eval=0
        for local in range(1,updates+1):
            if stopped:break
            optimizer.zero_grad(set_to_none=True);sums={};digest=hashlib.sha256();coef=coefficients(arm,schedule,tokens,initial_lambda)
            for _ in range(train['grad_accum']):
                x,y=stream.next_batch(train['batch_size']);digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
                with torch.autocast('cuda',dtype=torch.bfloat16):r=model(x.cuda(),y.cuda(),**coef);loss=r['loss']/train['grad_accum']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                for k in ['loss','loss_text','loss_lm','loss_attention_text','loss_attention_follow_r','loss_drift']:
                    sums[k]=sums.get(k,0.)+float(r[k].detach())/train['grad_accum']
                del r,loss
            batch_hash=digest.hexdigest()
            if control_hashes is not None and batch_hash!=control_hashes[local-1]:raise RuntimeError('A/B/C training data diverged')
            grads={name:float(torch.stack([p.grad.float().norm() for p in module.parameters() if p.grad is not None]).norm()) for name,module in model.named_children() if any(p.grad is not None for p in module.parameters())}
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            lr=cosine_lr(base['step']+local-1,train['learning_rate'],train['warmup_steps'],math.ceil(base['target_tokens']/step_tokens))
            if control_lrs is not None and lr!=control_lrs[local-1]:raise RuntimeError('Learning rates diverged')
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            tokens+=step_tokens;hashes.append(batch_hash);lrs.append(lr)
            rec=dict(local_step=local,global_step=base['step']+local,tokens=tokens,learning_rate=lr,coefficients=coef,grad_norm=norm,module_grad_norm=grads,**sums)
            records.append(rec);pending.append(rec)
            is_final=local==updates or stopped
            if (base['step']+local)%500==0 or is_final:
                saved_rng=rng_state();report=evaluate(model,batches,len(fixed));restore_rng(saved_rng);last_eval=local
                atomic_json(dest/f'validation-{local:05d}.json',report)
                if arm=='A':schedule.observe(schedule_metrics(report),tokens)
                evaluations.append(dict(local_step=local,groups=report['groups']))
                event(dict(event='validation',local_step=local,state='running',groups=report['groups'],control_guard_alerts=schedule.alerts if arm=='A' else []))
            if local%20==0 or is_final:
                event(dict(event='training',local_step=local,global_step=base['step']+local,branch_tokens=local*step_tokens,state='stopping' if stopped else 'running',
                    coefficients=coef,learning_rate=lr,window=summarize_updates(pending),cumulative_clip_ratio=sum(r['grad_norm']>1 for r in records)/len(records),
                    tokens_per_second=local*step_tokens/max(time.monotonic()-start,1.)))
                pending.clear()
            if stopped or (arm=='A' and schedule.stop):break
        # Signal before the next update must not count an unexecuted step.
        local=len(records)
        if local and last_eval!=local:
            saved_rng=rng_state();report=evaluate(model,batches,len(fixed));restore_rng(saved_rng)
            atomic_json(dest/f'validation-{local:05d}.json',report);evaluations.append(dict(local_step=local,groups=report['groups']))
        result=dict(plan=plan,finished_updates=local,finished_tokens=local*step_tokens,complete=local==updates and not stopped,
            training_statistics=summarize_updates(records) if records else {},evaluations=evaluations,
            training_batch_hashes=hashes,learning_rates=lrs,elapsed_seconds=time.monotonic()-start,
            paired_data_verified=arm=='A' or hashes==control_hashes[:len(hashes)],weights_saved=None)
        # Storage is selected separately, without changing protected original snapshots.
        storage=json.loads((OUT/'storage.json').read_text()) if (OUT/'storage.json').exists() else {'weights_dir':None}
        if storage.get('weights_dir') and local:
            target=Path(storage['weights_dir']);target.mkdir(parents=True,exist_ok=True)
            path=target/f'{arm}_step_{base["step"]+local:09d}.pt';temp=path.with_suffix('.pt.tmp')
            import shutil
            needed=sum(t.numel()*t.element_size() for t in model.state_dict().values())+1024**3
            if shutil.disk_usage(target).free<needed:raise RuntimeError('Insufficient branch checkpoint space')
            if path.exists():raise FileExistsError(path)
            payload=dict(model=model.state_dict(),optimizer=None,source=str(SOURCE),config=config,
                step=base['step']+local,tokens_seen=tokens,data_state=stream.state_dict(),rng=rng_state(),
                strategy='abc_state_diagnostic',arm=arm,coefficients=coefficients(arm,schedule,tokens,initial_lambda),
                strategy_schedule=dataclasses.asdict(schedule),experiment_plan=plan,validation_hash=validation_hash,extra_validation_hash=extra_hash)
            event(dict(event='saving_branch_weights',local_step=local,state='saving',path=str(path)))
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path);result['weights_saved']=str(path);del payload
        atomic_json(dest/'result.json',result)
        with (dest/'per-step-metrics.jsonl').open('w') as f:
            for record in records:f.write(json.dumps(record)+'\n')
        completed[arm]=result
        if arm=='A':control_hashes=hashes;control_lrs=lrs
        event(dict(event='arm_complete' if result['complete'] else 'arm_stopped',local_step=local,state='running' if result['complete'] else 'stopped',weights_saved=result['weights_saved']))
        del optimizer,params,model,stream,base,records,pending,baseline;gc.collect();torch.cuda.empty_cache()
        if not result['complete']:break
    unchanged=(cp/'manifest.json').read_text()==inventory['manifest'] and all((cp/name).stat().st_size==size and (cp/name).stat().st_mtime_ns==mtime for name,(size,mtime) in inventory['files'].items())
    summary=dict(completed_arms=[k for k,v in completed.items() if v['complete']],protected_checkpoints_unchanged=unchanged,elapsed_seconds=time.monotonic()-overall_start,results={})
    for arm,r in completed.items():
        summary['results'][arm]=dict(complete=r['complete'],final=r['evaluations'][-1]['groups'] if r['evaluations'] else None,training_statistics=r['training_statistics'],weights_saved=r['weights_saved'])
    if len(summary['completed_arms'])==3:
        summary['paired_data_and_lr_match']=all(completed[k]['training_batch_hashes']==control_hashes and completed[k]['learning_rates']==control_lrs for k in ['B','C'])
        control=summary['results']['A']['final']
        for arm in ['B','C']:
            final=summary['results'][arm]['final'];summary['results'][arm]['difference_from_control']={group:{metric:final[group][metric]-control[group][metric] for metric in ['ce_depth_3','depth_1_to_3_gain','delta_token_variance_3']} for group in ['fixed','extra','all']}
    atomic_json(OUT/'summary.json',summary)
    atomic_json(OUT/'completion.json',dict(complete=len(summary['completed_arms'])==3,stopped_by_signal=stopped,finished_at=time.time(),protected_checkpoints_unchanged=unchanged))
    atomic_json(OUT/'status.json',dict(state='complete' if len(summary['completed_arms'])==3 else 'stopped',pid=os.getpid(),updated_at=time.time(),completed_arms=summary['completed_arms']))
    print(json.dumps({'event':'experiment_finished','completed_arms':summary['completed_arms'],'protected_checkpoints_unchanged':unchanged}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OUT.exists():atomic_json(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
