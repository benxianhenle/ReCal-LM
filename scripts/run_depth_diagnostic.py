"""Six-hour, paired A/B/C and conditional D experiment from immutable step 33291."""
import argparse, dataclasses, fcntl, gc, hashlib, json, math, os, shutil, signal, statistics, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_state_abc import load_model, SOURCE, summarize_updates, schedule_metrics
from scripts.train_four_models import rng_state, restore_rng
from recal.data.stateful import TextEncoder, make_stream
from recal.evaluation.state_ablation import evaluate
from recal.training.retention import atomic_json
from recal.training.scheduler import cosine_lr
from recal.training.text_state_schedule import TextStateSchedule


def coefficients(arm,schedule,tokens):
    c,_=schedule.coefficients(tokens)
    if arm!='A':c['lambda_a']=0.
    c['lambda_depth']=.01 if arm in ('C','D') else 0.
    c['depth_margin']=0.
    return c


def update_snapshot(model):
    return {name:[(p,p.detach().clone()) for p in module.parameters() if p.requires_grad]
            for name,module in model.named_children() if name in ('attention','core','decoder')}


@torch.no_grad()
def relative_updates(snapshot):
    result={}
    for name,pairs in snapshot.items():
        # Actual FP32 parameter difference after AdamW, clipping AND weight decay.
        numerator=torch.stack([(p-old).square().sum() for p,old in pairs]).sum()
        denominator=torch.stack([old.square().sum() for p,old in pairs]).sum()
        result[name]=float((numerator/denominator.clamp_min(1e-30)).sqrt())
    return result


def d_gate(rows):
    samples=[r['relative_update'] for r in rows if r.get('relative_update') and r['local_step']>=100]
    ratios=[r['core']/max(min(r['attention'],r['decoder']),1e-30) for r in samples]
    return dict(eligible=len(ratios)>=5 and statistics.median(ratios)<.25 and sum(r<.25 for r in ratios)/len(ratios)>=.8,
                sample_count=len(ratios),median_r_to_min_ad=statistics.median(ratios) if ratios else None,
                fraction_below_quarter=sum(r<.25 for r in ratios)/len(ratios) if ratios else None,
                definition='At least 5 post-warm-start samples, median U_R/min(U_A,U_D)<0.25, >=80% samples below 0.25.')


def reuse_control(source_dir, out, target_tokens, validation_hashes):
    old_plan=json.loads((source_dir/'plan.json').read_text())
    result=json.loads((source_dir/'A/result.json').read_text())
    rows=[json.loads(line) for line in (source_dir/'A/steps.jsonl').read_text().splitlines()]
    if old_plan['source']!=str(SOURCE) or old_plan['validation_hashes']!=validation_hashes:
        raise ValueError('Reused control source/validation mismatch')
    if result['finished_tokens']!=target_tokens or result['updates']!=len(rows):
        raise ValueError('Matched target must equal the retained control endpoint')
    if not rows or any(row['local_step']!=i+1 for i,row in enumerate(rows)):
        raise ValueError('Control updates are not contiguous')
    if rows[-1]['branch_tokens']!=target_tokens or any('batch_sha256' not in row for row in rows):
        raise ValueError('Control token/hash evidence missing')
    old_hashes=json.loads((source_dir/'source-sha256.json').read_text())
    for name in ('recal/model/text_state_model.py','recal/evaluation/state_ablation.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=old_hashes[name]:
            raise ValueError('Model or evaluation implementation changed since control')
    (out/'A').mkdir()
    for file in (source_dir/'A').glob('validation-*.json'):shutil.copy2(file,out/'A'/file.name)
    shutil.copy2(source_dir/'A/steps.jsonl',out/'A/steps.jsonl')
    shutil.copy2(source_dir/'baseline.json',out/'baseline.json')
    result.update(original_complete=result['complete'],original_stop_reason=result['stop_reason'],
                  reused_from=str(source_dir),matched_target_tokens=target_tokens,
                  complete=True,stop_reason=None,
                  completion_scope='Complete only for the reduced matched endpoint; original 5M arm remains incomplete.')
    atomic_json(out/'A/result.json',result)
    return dict(result,rows=rows),json.loads((out/'baseline.json').read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--tokens-per-arm',type=int,default=5_000_000)
    p.add_argument('--control-from',type=Path)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise RuntimeError('Refusing to overwrite an existing attempt')
    lock=(SOURCE.parent.parent/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def event(**record):
        record.update(timestamp=time.time(),pid=os.getpid(),remaining_seconds=max(0,args.deadline-time.time()))
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        atomic_json(out/'status.json',record);print(json.dumps(record),flush=True)
    inventory={p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in SOURCE.parent.glob('*') if p.is_file()}
    atomic_json(out/'protected-checkpoints.json',inventory)
    args.weights.mkdir(parents=True,exist_ok=True)
    needed=SOURCE.stat().st_size*(3 if args.control_from else 4)+2*1024**3
    if shutil.disk_usage(args.weights).free<needed:raise RuntimeError('Insufficient space for planned branch weights')
    fixed_path=SOURCE.parent.parent/'validation.pt';extra_path=ROOT/'reports/attention-aux/validation-extra.pt'
    fixed=torch.load(fixed_path,weights_only=True)['batches'];extra=torch.load(extra_path,weights_only=True)
    batches=fixed+extra
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (fixed_path,extra_path)}
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
    manifest=ROOT/'artifacts/four-model-v1/data_manifest.json'
    plan=dict(source=str(SOURCE),deadline=args.deadline,started_at=start,tokens_per_arm=args.tokens_per_arm,
        source_optimizer_missing=True,optimizer_policy='Identical empty AdamW state for all arms; cannot reproduce unavailable historical moments.',
        validation_hashes=hashes,validation_milestones_tokens=sorted({0,args.tokens_per_arm,*[n for n in (1_000_000,2_000_000,3_000_000,4_000_000,5_000_000) if n<args.tokens_per_arm]}),
        control_reused_from=str(args.control_from) if args.control_from else None,original_requested_tokens_per_arm=5_000_000,
        diagnostic='Full 32-batch baseline and endpoints; same 32-batch CE/variance validation every 1M tokens.',
        objective='Existing LM + A text + P losses retained. C/D add .01*(relu(CE2-CE1)+relu(CE3-CE2)), m=0; literal connected A/R/D gradients.',
        arms={'A':'Original strategy and validation guards','B':'lambda_A=0','C':'B + depth improvement','D':'C + R LR 1.5, conditional on measured relative updates'},
        guards={'gradient_norm_abort':1000,'validation_ce_relative_worsening':.02,'validation_worsening_patience':2,'finish_reserve_seconds':600},
        weights=str(args.weights),checkpoint_policy='FP32 model + data/RNG/scheduler; Adam omitted due to 65 GiB writable disk limit. Resume requires optimizer reset.',scope='First round only; no 10B main training or extensions')
    atomic_json(out/'plan.json',plan)
    source_files=['scripts/run_depth_diagnostic.py','recal/model/text_state_model.py','recal/evaluation/state_ablation.py','scripts/run_state_abc.py']
    atomic_json(out/'source-sha256.json',{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files})
    completed={};control_rows=None;baseline=None;save_seconds=120.;validation_seconds=90.;decision=None
    if args.control_from:
        control,baseline=reuse_control(args.control_from,out,args.tokens_per_arm,hashes)
        completed['A']=control;control_rows=control['rows']
        event(event='control_reused',arm='A',updates=len(control_rows),matched_tokens=args.tokens_per_arm)
    for arm in (('B','C','D') if args.control_from else ('A','B','C','D')):
        if stopped or time.time()>args.deadline-600:break
        if arm=='D':
            # C is the exact unscaled counterpart of D; use its measured updates.
            decision=d_gate(completed['C']['rows']) if 'C' in completed else {'eligible':False,'reason':'C incomplete'}
            atomic_json(out/'D-gate.json',decision)
            if not decision['eligible']:
                event(event='D_skipped',decision=decision);break
            if not completed['C']['complete']:
                event(event='D_skipped',reason='C failed stability or time guard');break
            required=completed['C']['train_seconds']+600+validation_seconds+save_seconds
            if args.deadline-time.time()<required:
                decision.update(budget_admitted=False,required_seconds=required,remaining_seconds=args.deadline-time.time())
                atomic_json(out/'D-gate.json',decision)
                event(event='D_skipped',reason='Insufficient time for matched D endpoint',decision=decision);break
        dest=out/arm;dest.mkdir()
        event(event='loading',arm=arm)
        model,base=load_model();config=base['config'];train=config['training']
        if base['step']!=33291 or base['validation_hash']!=hashes[str(fixed_path)] or base['extra_validation_hash']!=hashes[str(extra_path)]:raise ValueError('Source/validation mismatch')
        schedule=TextStateSchedule(**base['strategy_schedule']);tokens=base['tokens_seen']
        step_tokens=train['seq_len']*train['batch_size']*train['grad_accum'];steps=math.ceil(args.tokens_per_arm/step_tokens)
        milestones={math.ceil(n/step_tokens) for n in plan['validation_milestones_tokens'] if n>0}
        groups=[dict(params=[p for p in module.parameters() if p.requires_grad],name=name) for name,module in model.named_children() if any(p.requires_grad for p in module.parameters())]
        optimizer=torch.optim.AdamW(groups,lr=train['learning_rate'],betas=(.9,.95),weight_decay=train['weight_decay'],foreach=False)
        params=[p for group in optimizer.param_groups for p in group['params']]
        stream=make_stream(manifest,encoder,train['seq_len'],1337,'train');stream.load_state_dict(base['data_state']);restore_rng(base['rng'])
        saved_rng=rng_state();eval_start=time.time()
        initial=evaluate(model,batches,len(fixed),full=baseline is None);restore_rng(saved_rng)
        validation_seconds=max(validation_seconds,time.time()-eval_start)
        if baseline is None:
            baseline=initial;atomic_json(out/'baseline.json',baseline)
        else:
            for group in ('fixed','extra'):
                for depth in (1,2,3):
                    key=f'ce_depth_{depth}'
                    if abs(initial['groups'][group][key]-baseline['groups'][group][key])>1e-5:raise RuntimeError('Baseline did not reproduce')
        atomic_json(dest/'validation-00000.json',initial)
        event(event='baseline',arm=arm,groups=initial['groups'],coefficients=coefficients(arm,schedule,tokens))
        rows=[];evaluations=[dict(local_step=0,groups=initial['groups'])];train_seconds=0.;reason=None;bad_ce=0;last_eval=0;arm_start=time.time()
        with (dest/'steps.jsonl').open('a') as step_log:
            for local in range(1,steps+1):
                if stopped:reason='signal';break
                reserve=max(600.,save_seconds*1.5+validation_seconds*1.5)
                if time.time()>args.deadline-reserve:reason='wall_time_budget';break
                t=time.time();optimizer.zero_grad(set_to_none=True);coef=coefficients(arm,schedule,tokens);sums={};digest=hashlib.sha256()
                for _ in range(train['grad_accum']):
                    x,y=stream.next_batch(train['batch_size']);digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes())
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        r=model(x.cuda(),y.cuda(),**coef);loss=r['loss']/train['grad_accum']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    for key in ('loss','loss_lm','loss_text','loss_attention_text','loss_attention_follow_r','loss_drift','loss_depth_improvement'):
                        sums[key]=sums.get(key,0.)+float(r[key].detach())/train['grad_accum']
                    del r,loss
                grads={name:float(torch.stack([p.grad.float().norm() for p in module.parameters() if p.grad is not None]).norm()) for name,module in model.named_children() if any(p.grad is not None for p in module.parameters())}
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                if norm>1000:
                    reason='gradient_spike_over_1000'
                    event(event='gradient_guard',arm=arm,rejected_local_step=local,grad_norm=norm,module_grad_norm=grads,batch_sha256=digest.hexdigest(),last_completed_step=len(rows))
                    optimizer.zero_grad(set_to_none=True);break
                lr=cosine_lr(base['step']+local-1,train['learning_rate'],train['warmup_steps'],math.ceil(base['target_tokens']/step_tokens))
                batch_hash=digest.hexdigest()
                if control_rows is not None and (batch_hash!=control_rows[local-1]['batch_sha256'] or lr!=control_rows[local-1]['learning_rate']):raise RuntimeError('Paired data/LR mismatch')
                for group in optimizer.param_groups:group['lr']=lr*(1.5 if arm=='D' and group['name']=='core' else 1.)
                snapshot=update_snapshot(model) if local%100==0 or local==1 else None
                optimizer.step()
                updates=relative_updates(snapshot) if snapshot is not None else None
                del snapshot
                optimizer.zero_grad(set_to_none=True);tokens+=step_tokens;torch.cuda.synchronize();train_seconds+=time.time()-t
                row=dict(local_step=local,global_step=base['step']+local,branch_tokens=local*step_tokens,learning_rate=lr,r_learning_rate=lr*(1.5 if arm=='D' else 1.),coefficients=coef,batch_sha256=batch_hash,grad_norm=norm,module_grad_norm=grads,relative_update=updates,**sums)
                rows.append(row);step_log.write(json.dumps(row)+'\n');step_log.flush()
                if local in milestones:
                    saved_rng=rng_state();eval_start=time.time()
                    report=evaluate(model,batches,len(fixed),full=local==steps);restore_rng(saved_rng)
                    validation_seconds=max(validation_seconds,time.time()-eval_start);last_eval=local
                    atomic_json(dest/f'validation-{local:05d}.json',report);evaluations.append(dict(local_step=local,groups=report['groups']))
                    m=report['groups']['all'];b=baseline['groups']['all']
                    if not all(math.isfinite(v) for g in report['groups'].values() for v in g.values()):raise FloatingPointError('Nonfinite validation')
                    bad_ce=bad_ce+1 if m['ce_depth_3']>b['ce_depth_3']*1.02 else 0
                    if arm=='A':schedule.observe(schedule_metrics(report),tokens)
                    if bad_ce>=2:reason='validation_worsening'
                    if arm=='A' and schedule.stop:reason='original_strategy_guard'
                    if arm in ('C','D') and local==math.ceil(2_000_000/step_tokens):
                        abnormal=(m['ce_depth_3']>b['ce_depth_3']*1.02 or (m['state_token_variance_3']<b['state_token_variance_3']*.1 and m['depth_2_to_3_gain']<-.02))
                        if abnormal:reason='2M_stability_gate'
                    event(event='validation',arm=arm,local_step=local,branch_tokens=local*step_tokens,groups=report['groups'],guard=reason,train_tokens_per_second=local*step_tokens/train_seconds)
                if local%20==0 or local==1 or local==steps:
                    event(event='training',arm=arm,local_step=local,branch_tokens=local*step_tokens,loss_lm=sums['loss_lm'],grad_norm=norm,relative_update=updates,train_tokens_per_second=local*step_tokens/train_seconds,elapsed_seconds=time.time()-arm_start)
                if reason:break
        local=len(rows)
        if local and last_eval!=local:
            saved_rng=rng_state();report=evaluate(model,batches,len(fixed),full=True);restore_rng(saved_rng)
            atomic_json(dest/f'validation-{local:05d}.json',report);evaluations.append(dict(local_step=local,groups=report['groups']))
        checkpoint_path=None
        if rows:
            checkpoint_path=args.weights/f'{arm}_step_{base["step"]+local:09d}.pt'
            if checkpoint_path.exists():raise FileExistsError(checkpoint_path)
            payload=dict(model=model.state_dict(),optimizer=None,optimizer_reset_on_resume=True,config=config,source=str(SOURCE),step=base['step']+local,tokens_seen=tokens,
                         target_tokens=base['target_tokens'],data_state=stream.state_dict(),rng=rng_state(),strategy_schedule=dataclasses.asdict(schedule),
                         arm=arm,strategy='depth_diagnostic',experiment_plan=plan,validation_hash=hashes[str(fixed_path)],extra_validation_hash=hashes[str(extra_path)],
                         tokenizer_hash=encoder.fingerprint,coefficients=coefficients(arm,schedule,tokens),local_step=local)
            event(event='saving',arm=arm,path=str(checkpoint_path));save_start=time.time()
            tmp=checkpoint_path.with_suffix('.pt.tmp')
            with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(tmp,checkpoint_path);save_seconds=max(save_seconds,time.time()-save_start);del payload
        result=dict(arm=arm,complete=local==steps and reason is None,finished_tokens=local*step_tokens,updates=local,stop_reason=reason,
                    checkpoint=str(checkpoint_path) if checkpoint_path else None,train_seconds=train_seconds,elapsed_seconds=time.time()-arm_start,
                    training_statistics=summarize_updates(rows) if rows else {},relative_update_gate=d_gate(rows),evaluations=evaluations,
                    paired_data_verified=arm=='A' or all(r['batch_sha256']==control_rows[i]['batch_sha256'] for i,r in enumerate(rows)))
        atomic_json(dest/'result.json',result);completed[arm]=dict(result,rows=rows)
        if arm=='A':control_rows=rows
        event(event='arm_complete' if result['complete'] else 'arm_stopped',arm=arm,updates=local,finished_tokens=local*step_tokens,reason=reason)
        del model,optimizer,params,groups,base,stream;gc.collect();torch.cuda.empty_cache()
        if arm=='A' and not result['complete']:break
        if reason in ('wall_time_budget','signal'):break
    unchanged=all((SOURCE.parent/name).stat().st_size==size and (SOURCE.parent/name).stat().st_mtime_ns==mtime for name,(size,mtime) in inventory.items())
    summary=dict(results={arm:{k:v for k,v in r.items() if k!='rows'} for arm,r in completed.items()},d_gate=decision,
                 protected_checkpoints_unchanged=unchanged,elapsed_seconds=time.time()-start,finished_at=time.time(),deadline=args.deadline)
    atomic_json(out/'summary.json',summary)
    complete=all(completed.get(a,{}).get('complete') for a in ('A','B','C')) and (decision is not None and (not decision['eligible'] or completed.get('D',{}).get('complete')))
    atomic_json(out/'completion.json',dict(complete=complete,completion_scope='matched_token_budget',tokens_per_arm=args.tokens_per_arm,original_5M_plan_complete=complete and args.tokens_per_arm>=5_000_000,finished_at=time.time(),protected_checkpoints_unchanged=unchanged,within_deadline=time.time()<=args.deadline))
    event(event='finished',complete=complete,completed_arms=[a for a,r in completed.items() if r['complete']],protected_checkpoints_unchanged=unchanged)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print(json.dumps(dict(event='failure',error=repr(exc))),flush=True)
        raise
