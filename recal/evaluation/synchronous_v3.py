"""Synchronous depth training; retain the trained D(delta) interface."""
import torch
from torch.nn import functional as F


def step(model, previous, positions, depth):
    proposal = model.recur(previous, positions)
    if hasattr(model, 'residual_gates'):
        # Learn a damping of the old update, not an incompatible h + old R(h).
        return previous + model.residual_gates[depth-1].clamp(0, 1) * (proposal-previous)
    return proposal


def objective(model, ids, labels):
    pos = torch.arange(ids.shape[1], device=ids.device)
    previous = model.attention(ids)
    losses, drifts = [], []
    for k in (1, 2, 3):
        state = step(model, previous, pos, k)
        logits = model.decoder(state-previous, pos)
        losses.append(F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()))
        target = (1-F.cosine_similarity(state.detach().float(), previous.detach().float(), dim=-1)).clamp(0, 1)
        drifts.append(F.mse_loss(model.drift(state.detach()).float(), target))
        previous = state
    text = torch.stack(losses).mean()
    drift = torch.stack(drifts).mean()
    return dict(loss=text+model.config.get('lambda_drift', .05)*drift, loss_text=text, loss_drift=drift,
                **{f'ce_depth_{i+1}': v for i, v in enumerate(losses)})


@torch.no_grad()
def evaluate(model, batches, fixed_count):
    training = model.training
    model.eval()
    rows = []
    try:
        for x, y in batches:
            device = next(model.parameters()).device
            x, y = x.to(device), y.to(device)
            pos = torch.arange(x.shape[1], device=device)
            row = {}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                previous = model.attention(x)
                for k in (1, 2, 3):
                    state = step(model, previous, pos, k)
                    delta = state-previous
                    logits = model.decoder(delta, pos)
                    row[f'ce{k}'] = float(F.cross_entropy(logits.float().flatten(0, 1), y.flatten()))
                    row[f'state_var{k}'] = float(state.float().var(1, unbiased=False).mean())
                    row[f'delta_var{k}'] = float(delta.float().var(1, unbiased=False).mean())
                    previous = state
            rows.append(row)
    finally:
        model.train(training)
    groups = {}
    for name, selected in [('fixed', rows[:fixed_count]), ('extra', rows[fixed_count:]), ('all', rows)]:
        g = {k: sum(r[k] for r in selected)/len(selected) for k in selected[0]}
        g['ce_range'] = max(g[f'ce{k}'] for k in (1,2,3))-min(g[f'ce{k}'] for k in (1,2,3))
        g['variance_ratio_3_1'] = g['state_var3']/max(g['state_var1'], 1e-30)
        groups[name] = g
    return dict(groups=groups, batches=rows)


def gate(current, reference, original, stage):
    decisions = {}
    for name in ('fixed', 'extra'):
        c, r, a = current['groups'][name], reference['groups'][name], original['groups'][name]
        decisions[name] = dict(ce3_delta_vs_A=c['ce3']-a['ce3'], ce3_delta_vs_previous=c['ce3']-r['ce3'],
            ce_range_before=r['ce_range'], ce_range_after=c['ce_range'],
            variance_ratio_relative=c['variance_ratio_3_1']/max(r['variance_ratio_3_1'], 1e-30),
            delta3_variance_relative=c['delta_var3']/max(r['delta_var3'], 1e-30))
        d = decisions[name]
        d['pass_gate'] = (d['ce3_delta_vs_A']<=.03 and d['ce3_delta_vs_previous']<=.03
            and c['ce_range'] < r['ce_range']-1e-5
            and d['variance_ratio_relative']>=.9 and d['delta3_variance_relative']>=.9)
    return dict(stage=stage, pass_gate=all(g['pass_gate'] for g in decisions.values()), groups=decisions,
        rule='Both heldout groups: CE3 increase <=0.03 vs A and previous stage, range shrinks >1e-5, Var(n3)/Var(n1) and Var(delta3) retain >=90% of previous stage.')
