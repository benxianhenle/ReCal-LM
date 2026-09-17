"""A-final -> E1: independent optimizer/schedule; bounded 3M-token phase."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_dynamic_diagnostics import SOURCE, digest, load_model
from scripts.train_four_models import rng_state, restore_rng
from recal.data.stateful import TextEncoder, make_stream
from recal.evaluation.synchronous_v3 import objective, evaluate, gate
from recal.training.retention import atomic_json


def phase_lr(completed_steps, peak=5e-5, warmup=300):
    return peak * min((completed_steps + 1) / warmup, 1.0)


def render(out):
    plan = json.loads((out/'plan.json').read_text())
    lines = ['# V3-E1：A-final warm-start，新训练阶段', '',
        '从 A step 34243 继承模型；AdamW 重新初始化；scheduler 从 phase step 0 开始。',
        '同步 R1/R2/R3，D(delta)；三层 CE 等权，alpha=lambda_A=lambda_R=lambda_monotonic=0；保留原 Drift 权重 0.05。',
        '旧模型实际深度权重为 [0.2,0.4,1.0]，并非仅 CE3。保留原冻结的未使用 core.embedding。',
        '学习率前 300 步线性升至 5e-5，之后恒定；732 步，共 2,998,272 token。数据游标和 RNG 从 A 继承。',
        '每约 1M token 验证 Fixed/Extra，非有限梯度/梯度范数超过 1000 停止；两次连续 CE3 恶化超过 0.03 停止。',
        '沿用前轮门槛：两组 CE3 相比 A 增幅 ≤0.03，depth range 缩小 >1e-5；两项方差指标保留至少 90%。这些是操作化门槛，不代表统计显著性。',
        '仅启动 E1。通过后保留一份阶段终点供 E2 继承；失败则只归档数据、指标和游标，不保留实验模型。A 永不覆盖。无中间模型副本，不保存 optimizer；中断后须重新开阶段。',
        '', '[计划](plan.json) · [状态](status.json) · [日志](steps.jsonl) · [基线](baseline.json)', '']
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        lines += ['```json', json.dumps(result, ensure_ascii=False, indent=2), '```']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main(out):
    if (out/'manifest.json').exists():
        raise RuntimeError('Run already initialized; refusing silent restart')
    plan = json.loads((out/'plan.json').read_text())
    lock = (ROOT/'runs/four-model-r3b-10b/.train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False
    def stop(*args):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def event(action, **fields):
        row = dict(action=action, time=time.time(), pid=os.getpid(), **fields)
        atomic_json(out/'status.json', row)
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    event('verify_A')
    previous = json.loads((ROOT/'reports/dynamic-r-v2-20260917/manifest.json').read_text())
    if digest(SOURCE) != previous['checkpoint_hash']:
        raise RuntimeError('A checkpoint hash mismatch')
    for name, expected in previous['validation_hash'].items():
        if digest(Path(name)) != expected:
            raise RuntimeError('Validation data changed')
    source_stat = (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns)
    checkpoint = Path(plan['checkpoint'])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or shutil.disk_usage(checkpoint.parent).free < SOURCE.stat().st_size + 1024**3:
        raise RuntimeError('Checkpoint path occupied or insufficient space')
    model, base = load_model()
    config = copy.deepcopy(base['config'])
    config.update(depth_weights=[1/3]*3, strategy='v3_e1_warmstart', alpha=0., lambda_A=0., lambda_R=0., lambda_monotonic=0.)
    config['training'].update(learning_rate=plan['peak_lr'], warmup_steps=plan['warmup_steps'], scheduler='phase_linear_warmup_constant')
    model.config = config
    model.depth_weights = (1/3,)*3
    model.requires_grad_(True)
    model.core.embedding.requires_grad_(False)
    model.train()
    train = config['training']
    step_tokens = train['seq_len']*train['batch_size']*train['grad_accum']
    steps = plan['token_cap']//step_tokens
    encoder = TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
    stream = make_stream(ROOT/'artifacts/four-model-v1/data_manifest.json', encoder, train['seq_len'], 1337, 'train')
    stream.load_state_dict(base['data_state'])
    restore_rng(base['rng'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0., betas=(.9,.95), weight_decay=train['weight_decay'], foreach=False)
    assert not optimizer.state
    torch.save(dict(data_state=stream.state_dict(), rng=rng_state(), source=str(SOURCE), step=base['step']), out/'initial-training-state.pt')
    atomic_json(out/'manifest.json', dict(source=str(SOURCE), source_sha256=previous['checkpoint_hash'],
        validation_hashes=previous['validation_hash'], source_step=base['step'], source_tokens=base['tokens_seen'],
        original_config=base['config'], config=config, optimizer='new AdamW, no inherited state',
        scheduler='new phase: linear 300-step warmup then constant', trainable_parameters=sum(p.numel() for p in params),
        source_hashes={name:digest(ROOT/name) for name in ('scripts/run_v3_e1_warmstart.py','recal/evaluation/synchronous_v3.py','recal/model/text_state_model.py')}, torch=torch.__version__))
    fixed = torch.load(ROOT/'runs/four-model-r3b-10b/validation.pt', weights_only=True)['batches']
    extra = torch.load(ROOT/'reports/attention-aux/validation-extra.pt', weights_only=True)
    event('baseline_validation')
    saved = rng_state()
    baseline = evaluate(model, fixed+extra, len(fixed))
    restore_rng(saved)
    expected = json.loads((ROOT/'reports/recal-r-v3-20260917/baseline.json').read_text())
    for group in ('fixed','extra'):
        for k in (1,2,3):
            if abs(baseline['groups'][group][f'ce{k}']-expected['groups'][group][f'ce{k}']) > 1e-5:
                raise RuntimeError('A baseline reproduction failed')
    atomic_json(out/'baseline.json', baseline)
    render(out)
    event('training_start', target_steps=steps, target_tokens=steps*step_tokens)
    milestones = {math.ceil(n/step_tokens) for n in (1_000_000,2_000_000)} | {steps}
    count = bad = last_eval = 0
    reason = None
    validation = baseline
    training_seconds = 0.
    with (out/'steps.jsonl').open('a') as log:
        for local in range(1, steps+1):
            if stopping or time.time() > plan['deadline']-180:
                reason = 'signal' if stopping else 'time_budget'
                break
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            sums = {}
            batchhash = hashlib.sha256()
            for _ in range(train['grad_accum']):
                x,y = stream.next_batch(train['batch_size'])
                batchhash.update(x.numpy().tobytes()); batchhash.update(y.numpy().tobytes())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result = objective(model,x.cuda(),y.cuda())
                    loss = result['loss']/train['grad_accum']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite train loss')
                loss.backward()
                for key,value in result.items():
                    sums[key] = sums.get(key,0.)+float(value.detach())/train['grad_accum']
                del result,loss
            norm = float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            if norm > 1000:
                reason = 'gradient_spike_over_1000'
                break
            lr = phase_lr(count,plan['peak_lr'],plan['warmup_steps'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            count += 1
            training_seconds += time.time()-start
            row = dict(phase_step=count,global_step=base['step']+count,training_tokens=count*step_tokens,
                learning_rate=lr,grad_norm=norm,batch_sha256=batchhash.hexdigest(),**sums)
            log.write(json.dumps(row)+'\n'); log.flush()
            if count == 1 or count % 20 == 0:
                event('training',**row,tokens_per_second=count*step_tokens/training_seconds)
            if count in milestones:
                saved = rng_state()
                validation = evaluate(model,fixed+extra,len(fixed))
                restore_rng(saved)
                last_eval = count
                if not all(math.isfinite(v) for g in validation['groups'].values() for v in g.values()):
                    raise FloatingPointError('Nonfinite validation')
                atomic_json(out/f'validation-{count:05d}.json',validation)
                bad = bad+1 if any(validation['groups'][g]['ce3'] > baseline['groups'][g]['ce3']+.03 for g in ('fixed','extra')) else 0
                event('validation',phase_step=count,training_tokens=count*step_tokens,groups=validation['groups'])
                if bad >= 2:
                    reason = 'two_validation_ce_regressions'
                    break
    if count and last_eval != count:
        saved = rng_state(); validation = evaluate(model,fixed+extra,len(fixed)); restore_rng(saved)
        atomic_json(out/f'validation-{count:05d}.json',validation)
    decision = gate(validation,baseline,baseline,'E1')
    if count != steps or reason:
        decision.update(pass_gate=False,incomplete_reason=reason or 'token_target_not_met')
    state = dict(data_state=stream.state_dict(),rng=rng_state(),phase_step=count,training_tokens=count*step_tokens)
    torch.save(state,out/'final-training-state.pt')
    kept = None
    if decision['pass_gate']:
        event('saving',path=str(checkpoint),training_tokens=count*step_tokens)
        payload = dict(model=model.state_dict(),config=config,step=base['step']+count,
            tokens_seen=base['tokens_seen']+count*step_tokens,source=str(SOURCE),stage='V3-E1',
            optimizer=None,optimizer_reset_on_resume=True,coefficients=dict(alpha=0.,lambda_a=0.,lambda_r=0.,lambda_depth=0.),
            tokenizer_hash=encoder.fingerprint,validation_hash=base['validation_hash'],extra_validation_hash=base['extra_validation_hash'],
            target_tokens=base['target_tokens'],experiment_plan=plan,**state)
        temp = checkpoint.with_suffix('.pt.tmp')
        with temp.open('wb') as f:
            torch.save(payload,f); f.flush(); os.fsync(f.fileno())
        os.replace(temp,checkpoint)
        kept = str(checkpoint)
    unchanged = source_stat == (SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns)
    atomic_json(out/'result.json',dict(complete=count==steps and reason is None,training_tokens=count*step_tokens,
        phase_steps=count,stop_reason=reason,gate=decision,validation=validation,checkpoint=kept,
        original_checkpoint_unchanged=unchanged,finished_at=time.time(),train_seconds=training_seconds,
        retention='Keep passing endpoint for E2; failed experimental weights discarded'))
    render(out)
    event('finished',training_tokens=count*step_tokens,pass_gate=decision['pass_gate'],checkpoint=kept)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    try:
        main(args.out)
    except BaseException as exc:
        atomic_json(args.out/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
