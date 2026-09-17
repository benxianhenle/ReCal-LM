"""Bounded E1 -> conditional E2 -> conditional E3 experiment from immutable A."""
import fcntl, hashlib, json, math, os, shutil, signal, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_dynamic_diagnostics import load_model, SOURCE, digest
from scripts.train_four_models import rng_state, restore_rng
from recal.data.stateful import TextEncoder, make_stream
from recal.training.scheduler import cosine_lr
from recal.training.retention import atomic_json
from recal.evaluation.synchronous_v3 import objective, evaluate, gate
OUT=ROOT/'reports/recal-r-v3-20260917'


def render():
    plan=json.loads((OUT/'plan.json').read_text())
    lines=['# ReCal-R V3 最后一轮实验','',
        '仅按 E1 → E2 → E3 的 Gate 推进；总上限 4 小时、8M 训练 token，R 深度上限 3。', '',
        'A 已有多深度监督 [0.2,0.4,1.0]；E1 改为 [1/3,1/3,1/3] 并关闭 alpha、lambda_A、lambda_R、monotonic，保留 lambda_drift=0.05。原方案“此前只有 CE3 监督”的前提不符合实际代码。', '',
        '主指标沿用真实训练的 D(delta) 同步路径。E2 使用 h+g*(R_old(h)-h)，gate 初始化 1，逐步学习阻尼；直接 h+R_old(h) 不兼容旧权重。E1/E2 不额外引入 Anchor 支路。', '',
        '预注册 Gate：Fixed 和 Extra 的 CE3 相对 A 及本阶段起点恶化都不超过 0.03；两组 CE 深度范围都缩小超过 1e-5；Var(n3)/Var(n1) 与 Var(delta3) 均保留起点的至少 90%。该数值阈值是对方案定性标准的操作化约定。', '',
        '| 阶段 | 状态 | token | Fixed CE1/2/3 | Extra CE1/2/3 | Gate |', '|---|---|---:|---|---|---|']
    baseline=OUT/'baseline.json'
    if baseline.exists():
        b=json.loads(baseline.read_text())
        ce=lambda r,g:'/'.join(f"{r['groups'][g][f'ce{k}']:.5f}" for k in (1,2,3))
        lines.append(f"| A | baseline | 0 | {ce(b,'fixed')} | {ce(b,'extra')} | — |")
    for stage in ('E1','E2','E3'):
        f=OUT/stage/'result.json'
        if f.exists():
            r=json.loads(f.read_text());v=r.get('validation')
            lines.append(f"| {stage} | {r.get('stop_reason') or '完成'} | {r['training_tokens']} | {ce(v,'fixed') if v and 'ce1' in v['groups']['fixed'] else '见阶段 JSON'} | {ce(v,'extra') if v and 'ce1' in v['groups']['extra'] else '见阶段 JSON'} | {r['gate']['pass_gate']} |")
        else:lines.append(f'| {stage} | 未完成/尚未进入 | — | — | — | — |')
    for stage in ('E1','E2'):
        f=OUT/stage/'result.json'
        if f.exists():
            r=json.loads(f.read_text());lines += ['',f'## {stage} 判定','', '```json',json.dumps(r['gate'],ensure_ascii=False,indent=2),'```']
    lines += ['', '## 预算与复现','',f"开始 epoch={plan['started_at']}，截止 epoch={plan['deadline']}。每阶段 3M 使用完整优化器步向下取整（每步 4096 token，共 2,998,272）；不为凑整超出 token 上限。每约 1M 验证一次，梯度范数 >1000 / 非有限数立即停止；CE3 任一组比 A 恶化 >0.03 连续两次停止。", '',
        'A 无 Adam 状态，E1 使用新 AdamW；E2 若进入则在同一进程保留 Adam 状态、数据流和 RNG。因磁盘容量限制，阶段权重不保存 Adam；中断后不可声称精确续训。原 checkpoint 不覆盖。', '',
        '[计划](plan.json) · [状态](status.json) · [来源](manifest.json) · [测试](tests.log) · [E1 日志](E1/steps.jsonl)', '']
    if (OUT/'completion.json').exists():lines += ['```json',(OUT/'completion.json').read_text(),'```','']
    (OUT/'README.md').write_text('\n'.join(lines))


def main():
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing run; never silently restart')
    plan=json.loads((OUT/'plan.json').read_text());deadline=plan['deadline'];stop_requested=False
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop(*unused):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def event(**kw):
        row=dict(time=time.time(),pid=os.getpid(),remaining_seconds=max(0,deadline-time.time()),**kw)
        atomic_json(OUT/'status.json',row)
        with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    source_stat=(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
    event(stage='preflight',action='verify_checkpoint')
    previous=json.loads((ROOT/'reports/dynamic-r-v2-20260917/manifest.json').read_text())
    if digest(SOURCE)!=previous['checkpoint_hash']:raise RuntimeError('Source hash mismatch')
    for file,value in previous['validation_hash'].items():
        if digest(Path(file))!=value:raise RuntimeError('Validation hash mismatch')
    for path in plan['weights'].values():
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise FileExistsError(p)
        if shutil.disk_usage(p.parent).free<SOURCE.stat().st_size+1024**3:raise RuntimeError('Insufficient disk budget')
    model,base=load_model();model.requires_grad_(True);model.core.embedding.requires_grad_(False);model.train()
    config=dict(base['config']);config['depth_weights']=[1/3]*3;model.config=config;model.depth_weights=(1/3,)*3
    train=config['training'];step_tokens=train['seq_len']*train['batch_size']*train['grad_accum'];steps=3_000_000//step_tokens
    fixed=torch.load(ROOT/'runs/four-model-r3b-10b/validation.pt',weights_only=True)['batches'];extra=torch.load(ROOT/'reports/attention-aux/validation-extra.pt',weights_only=True);batches=fixed+extra
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
    stream=make_stream(ROOT/'artifacts/four-model-v1/data_manifest.json',encoder,train['seq_len'],1337,'train')
    stream.load_state_dict(base['data_state']);restore_rng(base['rng'])
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=train['learning_rate'],betas=(.9,.95),weight_decay=train['weight_decay'],foreach=False)
    atomic_json(OUT/'manifest.json',dict(source=str(SOURCE),checkpoint_hash=previous['checkpoint_hash'],validation_hashes=previous['validation_hash'],
        original_config=base['config'],new_config=config,source_coefficients=base['coefficients'],source_optimizer_missing=True,
        implementation_sha256={f:digest(ROOT/f) for f in ('scripts/run_synchronous_v3.py','recal/evaluation/synchronous_v3.py')},
        trainable_parameters=sum(p.numel() for p in params),optimizer='Fresh AdamW E1; uninterrupted state retained E2',torch=torch.__version__))
    saved=rng_state();original=evaluate(model,batches,len(fixed));restore_rng(saved)
    old=json.loads((ROOT/'reports/dynamic-r-v2-20260917/V2.1.json').read_text())['scoreboard']
    for name in ('fixed','extra'):
        for k in (1,2,3):
            if abs(original['groups'][name][f'ce{k}']-old[f'Fixed R{k}'][name]['ce'])>1e-5:raise RuntimeError('A CE reproduction failed')
    atomic_json(OUT/'baseline.json',original);reference=original;total_updates=0;total_tokens=0;results={}
    render()
    for stage in ('E1','E2'):
        if stop_requested or time.time()>deadline-600:break
        dest=OUT/stage;dest.mkdir(exist_ok=True);stage_start=time.time();stage_deadline=min(deadline,stage_start+75*60)
        if stage=='E2':
            model.register_parameter('residual_gates',torch.nn.Parameter(torch.ones(3,device='cuda')))
            optimizer.add_param_group(dict(params=[model.residual_gates],weight_decay=0.))
            params.append(model.residual_gates)
            saved=rng_state();compatible=evaluate(model,batches,len(fixed));restore_rng(saved)
            if any(abs(compatible['groups'][g][f'ce{k}']-reference['groups'][g][f'ce{k}'])>1e-4 for g in ('fixed','extra') for k in (1,2,3)):raise RuntimeError('E2 residual initialization incompatible')
            atomic_json(dest/'compatibility.json',compatible)
        event(stage=stage,action='training_start',target_tokens=steps*step_tokens,training_tokens=total_tokens,stage_deadline=stage_deadline)
        count=0;bad=0;reason=None;last_eval=0;validation=reference;train_seconds=0.
        milestones={math.ceil(n/step_tokens) for n in (1_000_000,2_000_000)}|{steps}
        with (dest/'steps.jsonl').open('a') as log:
            for local in range(1,steps+1):
                if stop_requested or time.time()>stage_deadline-180:
                    reason='signal' if stop_requested else 'time_budget';break
                t=time.time();optimizer.zero_grad(set_to_none=True);sums={};batchhash=hashlib.sha256()
                for _ in range(train['grad_accum']):
                    x,y=stream.next_batch(train['batch_size']);batchhash.update(x.numpy().tobytes());batchhash.update(y.numpy().tobytes())
                    with torch.autocast('cuda',dtype=torch.bfloat16):r=objective(model,x.cuda(),y.cuda());loss=r['loss']/train['grad_accum']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
                    loss.backward()
                    for k,v in r.items():sums[k]=sums.get(k,0.)+float(v.detach())/train['grad_accum']
                    del r,loss
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                if norm>1000:
                    reason='gradient_spike_over_1000';optimizer.zero_grad(set_to_none=True)
                    event(stage=stage,action='rejected_update',grad_norm=norm,rejected_step=local);break
                lr=cosine_lr(base['step']+total_updates,train['learning_rate'],train['warmup_steps'],math.ceil(base['target_tokens']/step_tokens))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.step()
                if hasattr(model,'residual_gates'):
                    with torch.no_grad():model.residual_gates.clamp_(0,1)
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
                count+=1;total_updates+=1;total_tokens+=step_tokens;train_seconds+=time.time()-t
                row=dict(local_step=count,global_step=base['step']+total_updates,stage_tokens=count*step_tokens,total_training_tokens=total_tokens,
                    learning_rate=lr,grad_norm=norm,batch_sha256=batchhash.hexdigest(),gates=model.residual_gates.detach().tolist() if hasattr(model,'residual_gates') else None,**sums)
                log.write(json.dumps(row)+'\n');log.flush()
                if local in milestones:
                    saved=rng_state();validation=evaluate(model,batches,len(fixed));restore_rng(saved);last_eval=count
                    if not all(math.isfinite(v) for g in validation['groups'].values() for v in g.values()):raise FloatingPointError('Nonfinite evaluation')
                    atomic_json(dest/f'validation-{count:05d}.json',validation)
                    bad=bad+1 if any(validation['groups'][g]['ce3']>original['groups'][g]['ce3']+.03 for g in ('fixed','extra')) else 0
                    if bad>=2:reason='two_validation_ce_regressions'
                    event(stage=stage,action='validation',stage_tokens=count*step_tokens,training_tokens=total_tokens,groups=validation['groups'],stop_reason=reason)
                if count==1 or count%20==0:
                    event(stage=stage,action='training',local_step=count,stage_tokens=count*step_tokens,training_tokens=total_tokens,
                        tokens_per_second=count*step_tokens/train_seconds,grad_norm=norm,loss=sums['loss'],stage_deadline=stage_deadline)
                if reason:break
        if count and last_eval!=count:
            saved=rng_state();validation=evaluate(model,batches,len(fixed));restore_rng(saved)
            atomic_json(dest/f'validation-{count:05d}.json',validation)
        decision=gate(validation,reference,original,stage)
        if count!=steps or reason:decision.update(pass_gate=False,incomplete_reason=reason or 'token_target_not_met')
        path=None
        if count:
            path=Path(plan['weights'][stage]);event(stage=stage,action='saving',training_tokens=total_tokens,path=str(path))
            payload=dict(model=model.state_dict(),config=config,step=base['step']+total_updates,tokens_seen=base['tokens_seen']+total_tokens,
                target_tokens=base['target_tokens'],data_state=stream.state_dict(),rng=rng_state(),source=str(SOURCE),stage=stage,
                optimizer=None,optimizer_reset_on_resume=True,tokenizer_hash=encoder.fingerprint,validation_hash=base['validation_hash'],extra_validation_hash=base['extra_validation_hash'],
                depth_weights=[1/3]*3,objective='equal-depth LM + 0.05 original detached drift',residual_gate=hasattr(model,'residual_gates'),experiment_plan=plan)
            temp=path.with_suffix('.pt.tmp')
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,path);del payload
        result=dict(stage=stage,training_tokens=count*step_tokens,total_training_tokens=total_tokens,complete=count==steps and reason is None,
            stop_reason=reason,gate=decision,validation=validation,checkpoint=str(path) if path else None,
            train_seconds=train_seconds,elapsed_seconds=time.time()-stage_start)
        atomic_json(dest/'result.json',result);results[stage]=result;render()
        event(stage=stage,action='stage_complete',training_tokens=total_tokens,gate=decision)
        if not decision['pass_gate']:break
        reference=validation
    if results.get('E2',{}).get('gate',{}).get('pass_gate') and not stop_requested and time.time()<deadline-600:
        from scripts.run_synchronous_v3_refresh import run_refresh
        results['E3']=run_refresh(model,base,deadline,total_tokens,event)
        total_tokens+=results['E3']['training_tokens']
    unchanged=source_stat==(SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
    atomic_json(OUT/'completion.json',dict(complete=bool(results) and (not list(results.values())[-1]['gate']['pass_gate'] or 'E3' in results),
        stages=list(results),stopped_at=list(results)[-1] if results else 'preflight',training_tokens=total_tokens,
        original_checkpoint_unchanged=unchanged,within_budget=time.time()<=deadline,finished_at=time.time(),
        elapsed_seconds=time.time()-plan['started_at'],reason=list(results.values())[-1]['stop_reason'] or ('gate_failed' if not list(results.values())[-1]['gate']['pass_gate'] else 'stages_complete') if results else 'time_budget'))
    render();event(stage='finished',action='complete',training_tokens=total_tokens)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(error=repr(exc),time=time.time()));raise
