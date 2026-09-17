"""Two-phase four-model training, token-time supervision and three retained snapshots."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from recal.model.four_model import FourModelLM
from recal.data.stateful import TextEncoder, make_stream
from recal.training.retention import CheckpointStore, atomic_json
from recal.training.stages import PlateauSwitch
from recal.training.scheduler import cosine_lr


def count(text):
    text = text.lower()
    scale = {'k': 1000, 'm': 10**6, 'b': 10**9}
    return int(float(text[:-1]) * scale[text[-1]]) if text[-1] in scale else int(text)


def args_parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data')
    p.add_argument('--val-data')
    p.add_argument('--tokenizer')
    p.add_argument('--output', required=True)
    p.add_argument('--target-tokens', type=count, required=True)
    p.add_argument('--max-steps', type=int, help='Bound this invocation without changing the LR token horizon')
    p.add_argument('--resume', help='Checkpoint path; only the current latest can resume in the same run')
    p.add_argument('--enter-late', action='store_true', help='Pin latest early checkpoint and manually enter teacher-free training on resume')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--weights-only', action='store_true', help='Explicitly omit Adam state; resume resets optimizer')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def main():
    args = args_parser()
    config = yaml.safe_load(Path(args.config).read_text())
    train = config['training']
    # Meta construction validates the intended core scale without allocating billions of weights.
    with torch.device('meta'):
        blueprint = FourModelLM(config)
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in blueprint.named_children()}
    del blueprint
    bytes_per_snapshot = sum(counts.values()) * 4
    if not args.weights_only:
        bytes_per_snapshot += sum(v for k, v in counts.items() if k != 'teacher') * 8
    budget = dict(parameters=counts, estimated_snapshot_bytes=bytes_per_snapshot,
                  required_space_for_three_plus_atomic_temp=bytes_per_snapshot * 4 + 2 * 1024**3)
    print(json.dumps(budget), flush=True)
    if args.dry_run:
        return
    if not all((args.data, args.val_data, args.tokenizer)):
        raise ValueError('Real data, held-out data, and tokenizer are required')
    if args.target_tokens <= 0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError('Positive token budget and steps required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.train.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Existing retained files already occupy part of the four-file capacity reservation.
    store = CheckpointStore(output / 'checkpoints')
    occupied = sum((output / 'checkpoints' / name).stat().st_size for name in store.manifest['snapshots'])
    remaining_reserve = max(bytes_per_snapshot + 2 * 1024**3,
        budget['required_space_for_three_plus_atomic_temp'] - occupied)
    if shutil.disk_usage(output).free < remaining_reserve:
        raise RuntimeError(f'Insufficient free disk for checkpoint policy: {budget}; need free {remaining_reserve}')
    if store.manifest['snapshots'] and not args.resume:
        raise ValueError('Existing run requires --resume or a new output directory')
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This trainer requires native CUDA BF16 or CPU FP32')
    encoder = TextEncoder(args.tokenizer)
    if encoder.vocab_size > config['vocab_size']:
        raise ValueError('Tokenizer larger than model vocabulary')
    seq_len = int(train['seq_len'])
    stream = make_stream(args.data, encoder, seq_len, args.seed, 'train')
    val_stream = make_stream(args.val_data, encoder, seq_len, args.seed + 1, 'val')
    val_file = output / 'validation.pt'
    if val_file.exists():
        val = torch.load(val_file, weights_only=True, map_location='cpu')
        if val['identity'] != val_stream.identity or val['tokenizer'] != encoder.fingerprint or val['seq_len'] != seq_len:
            raise ValueError('Validation source changed')
    else:
        val = dict(identity=val_stream.identity, tokenizer=encoder.fingerprint, seq_len=seq_len,
                   batches=[val_stream.next_batch(train['batch_size']) for _ in range(train['validation_batches'])])
        torch.save(val, val_file)
    if len(val['batches']) != train['validation_batches']:
        raise ValueError('Validation batch count changed')
    digest = hashlib.sha256(val_file.read_bytes()).hexdigest()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        if ckpt['config'] != config or ckpt['target_tokens'] != args.target_tokens:
            raise ValueError('Config/token horizon changed; use a separately planned run')
        if ckpt['validation_hash'] != digest:
            raise ValueError('Fixed validation data changed')
        if store.manifest['snapshots']:
            latest = output / 'checkpoints' / store.manifest['roles']['latest']
            if Path(args.resume).resolve() != latest.resolve():
                raise ValueError('Cannot rewind existing output; resume its latest checkpoint')
    if args.enter_late and (ckpt is None or ckpt['stage'] != 'early'):
        raise ValueError('--enter-late requires an early-stage resume checkpoint')
    manual_transition = bool(ckpt and ckpt['stage'] == 'early' and (args.enter_late or
        store.manifest.get('manual_transition') == Path(args.resume).name))
    stage = ckpt['stage'] if ckpt else 'early'
    model = FourModelLM(config, stage=stage)
    if ckpt:
        model.load_state_dict(ckpt['model'])
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=train['learning_rate'], betas=(0.9, 0.95), weight_decay=train['weight_decay'], foreach=False)
    plateau = PlateauSwitch(train.get('plateau_patience', 5), train.get('plateau_relative_tolerance', 0.001))
    step = tokens_seen = 0
    resumed_rng = None
    if ckpt:
        plateau = PlateauSwitch(**ckpt['plateau'])
        stream.load_state_dict(ckpt['data_state'])
        step, tokens_seen = ckpt['step'], ckpt['tokens_seen']
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print('WEIGHTS-ONLY RESUME: Adam moments reset; this is not an exact continuation.', flush=True)
        resumed_rng = ckpt['rng']
        del ckpt
    amp = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')
    if resumed_rng:
        restore_rng(resumed_rng)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t0 = time.monotonic()
    initial_tokens = tokens_seen
    (output / 'completion.json').unlink(missing_ok=True)
    atomic_json(output / 'run_config.json', dict(config=config, arguments=vars(args), budget=budget,
        tokenizer_hash=encoder.fingerprint, validation_hash=digest, started_at=time.time()))

    def evaluate():
        saved_rng = rng_state()
        model.eval()
        values = []
        with torch.no_grad(), amp():
            for x, y in val['batches']:
                out = model(x.to(device), y.to(device), decoder_source='core', decoder_step=config['rollout_steps'])
                values.append(float(out['loss_lm']))
        model.train()
        restore_rng(saved_rng)
        value = sum(values) / len(values)
        if not math.isfinite(value):
            raise FloatingPointError('Validation loss is not finite')
        return value

    last_saved_step = step if args.resume else -1

    def save(value, boundary=None):
        nonlocal last_saved_step
        optimizer.zero_grad(set_to_none=True)
        payload = dict(model=model.state_dict(), optimizer=None if args.weights_only else optimizer.state_dict(),
            stage=model.stage, step=step, tokens_seen=tokens_seen, target_tokens=args.target_tokens,
            config=config, plateau=plateau.state_dict(), data_state=stream.state_dict(), rng=rng_state(),
            validation_hash=digest, tokenizer_hash=encoder.fingerprint,
            optimizer_reset_on_resume=args.weights_only, wall_seconds=time.monotonic() - t0)
        needed = bytes_per_snapshot + 1024**3
        if shutil.disk_usage(output).free < needed:
            raise RuntimeError('Insufficient disk for atomic checkpoint; existing snapshots preserved')
        path = store.save(payload, value, boundary=boundary)
        last_saved_step = step
        print(f'SAVED {path}', flush=True)

    # A pre-transition checkpoint is deliberately saved BEFORE removing the teacher.
    # Resuming it completes the already-decided transition without another plateau observation.
    if manual_transition:
        boundary = store.mark_manual_transition()
        plateau.switched = True
        atomic_json(output / 'manual_transition.json', dict(step=step, tokens_seen=tokens_seen,
            from_stage='early', to_stage='late', boundary=str(boundary), requested_at=time.time(),
            optimizer_reset=args.weights_only))
        print(f'MANUAL TRANSITION at step={step}; boundary={boundary}', flush=True)
    if plateau.switched and model.stage == 'early':
        model.enter_late()
        if device.type == 'cuda': torch.cuda.empty_cache()
        print('TRANSITION early -> late; EMA teacher removed', flush=True)
    tokens_per_step = train['batch_size'] * seq_len * train['grad_accum']
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    model.train()
    while tokens_seen < args.target_tokens and not stopped:
        if args.max_steps is not None and step >= args.max_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        remaining_micro = math.ceil((args.target_tokens - tokens_seen) / (train['batch_size'] * seq_len))
        accum = min(train['grad_accum'], remaining_micro)
        for _ in range(accum):
            x, y = stream.next_batch(train['batch_size'])
            with amp():
                out = model(x.to(device), y.to(device))
                loss = out['loss'] / accum
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; no optimizer step committed')
            loss.backward()
            for key in ('loss', 'loss_lm', 'loss_core', 'loss_attention', 'loss_drift', 'loss_attention_lm'):
                metrics[key] = metrics.get(key, 0.0) + float(out[key].detach()) / accum
            tokens_seen += x.numel()
            del out, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        lr = cosine_lr(step, train['learning_rate'], train['warmup_steps'], total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        model.update_teacher()
        step += 1
        optimizer.zero_grad(set_to_none=True)
        metrics.update(step=step, tokens_seen=tokens_seen, stage=model.stage, learning_rate=lr,
            grad_norm=float(grad_norm), tokens_per_second=(tokens_seen - initial_tokens) / max(time.monotonic() - t0, 1e-6),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
        final = stopped or tokens_seen >= args.target_tokens or (args.max_steps is not None and step >= args.max_steps)
        if step % train['validation_interval'] == 0 or final:
            value = evaluate()
            transition = plateau.observe(value) if model.stage == 'early' else False
            metrics.update(val_loss_lm=value, stable_changes=plateau.stable_changes, transition=transition)
            print(f'VALIDATED step={step} loss={value:.6f}; saving checkpoint', flush=True)
            save(value, boundary='before' if transition else None)
            if transition:
                model.enter_late()
                if device.type == 'cuda': torch.cuda.empty_cache()
                print('TRANSITION early -> late; EMA teacher removed', flush=True)
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        atomic_json(output / 'status.json', dict(metrics, pid=os.getpid(), updated_at=time.time(), state='running'))
        print(json.dumps(metrics), flush=True)
    if step > last_saved_step:
        save(evaluate())
    atomic_json(output / 'completion.json', dict(step=step, tokens_seen=tokens_seen, stage=model.stage,
        stopped=stopped, target_reached=tokens_seen >= args.target_tokens,
        wall_seconds=time.monotonic() - t0))


if __name__ == '__main__':
    main()
