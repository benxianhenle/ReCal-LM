"""Read-only delta-path diagnostics. Shuffles are interventions, not causal generation scores."""
import torch
from torch.nn import functional as F


def information(state):
    with torch.autocast(device_type=state.device.type, enabled=False):
        x=state.detach().float().flatten(0,1)
        centered=x-x.mean(0,keepdim=True)
        unit=F.normalize(x,dim=-1);n=x.shape[0]
        gram=centered@centered.T
        return dict(token_variance=float(centered.square().mean()),
            mean_norm=float(x.norm(dim=-1).mean()),
            mean_pair_cosine=float((unit.sum(0).square().sum()-unit.square().sum())/max(n*(n-1),1)),
            centered_energy=float(centered.square().sum()/x.square().sum().clamp_min(1e-20)),
            effective_rank=float(gram.trace().square()/gram.square().sum().clamp_min(1e-20)))


@torch.no_grad()
def evaluate(model,batches,fixed_count=8,shuffle_repeats=3,full=True):
    was_training=model.training;model.eval();rows=[]
    device=next(model.parameters()).device
    try:
        for batch_index,(xc,yc) in enumerate(batches):
            x,y=xc.to(device),yc.to(device)
            pos=torch.arange(x.shape[1],device=device)
            ce=lambda state:float(F.cross_entropy(model.decoder(state,pos).float().flatten(0,1),y.flatten()))
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                initial=model.attention(x);previous=initial;states=[];deltas=[];record={}
                for k in range(1,4):
                    state=model.recur(previous,pos);delta=state-previous
                    states.append(state);deltas.append(delta)
                    record[f'ce_depth_{k}']=ce(delta)
                    if full:record[f'absolute_state_ce_{k}']=ce(state)
                    accepted=model.attention.stack(state,pos)
                    record[f'cos_depth_{k}']=float(F.cosine_similarity(accepted.float(),state.float(),dim=-1).mean())
                    for prefix,tensor in [('state',state),('delta',delta),('accepted',accepted)]:
                        values=information(tensor) if full else dict(token_variance=float(tensor.float().var(dim=1,unbiased=False).mean()),mean_norm=float(tensor.float().norm(dim=-1).mean()))
                        for metric,value in values.items():record[f'{prefix}_{metric}_{k}']=value
                    previous=state
                record['depth_1_to_2_gain']=record['ce_depth_1']-record['ce_depth_2']
                record['depth_1_to_3_gain']=record['ce_depth_1']-record['ce_depth_3']
                record['depth_2_to_3_gain']=record['ce_depth_2']-record['ce_depth_3']
                if not full:
                    rows.append(record)
                    continue
                n2,n3=states[1:];delta3=deltas[-1]
                record['zero_delta3_ce']=ce(torch.zeros_like(delta3))
                record['mean_delta3_ce']=ce(delta3.mean(dim=1,keepdim=True).expand_as(delta3))
                # These distinguish information in the delta from NEW information supplied by R3.
                record['constant_r3_state_ce']=ce(n3[:,:1].expand_as(n3)-n2)
                record['mean_r3_state_ce']=ce(n3.mean(dim=1,keepdim=True).expand_as(n3)-n2)
                record['negative_previous_state_ce']=ce(-n2)
                lag=torch.cat([torch.zeros_like(delta3[:,:1]),delta3[:,:-1]],dim=1)
                record['causal_lag_delta3_ce']=ce(lag)
                delta_losses=[];state_losses=[]
                for repeat in range(shuffle_repeats):
                    generator=torch.Generator(device='cpu').manual_seed(73019+batch_index*31+repeat)
                    perm=torch.randperm(x.shape[1],generator=generator).to(device)
                    delta_losses.append(ce(delta3[:,perm]))
                    state_losses.append(ce(n3[:,perm]-n2))
                record['shuffle_delta3_ce']=sum(delta_losses)/len(delta_losses)
                record['shuffle_r3_state_ce']=sum(state_losses)/len(state_losses)
                record['depth_1_to_3_gain']=record['ce_depth_1']-record['ce_depth_3']
                record['depth_2_to_3_gain']=record['ce_depth_2']-record['ce_depth_3']
                rows.append(record)
    finally:model.train(was_training)
    groups={}
    for name,subset in [('fixed',rows[:fixed_count]),('extra',rows[fixed_count:]),('all',rows)]:
        if subset:groups[name]={k:sum(row[k] for row in subset)/len(subset) for k in subset[0]}
    return dict(groups=groups,batches=rows,shuffle_repeats=shuffle_repeats,
        note='Normal CE decodes deltas. Absolute-state CE is out-of-training-distribution. Token shuffles and full-sequence means can mix future positions: diagnostic interventions only. Shared batch-mean uncertainty is not document-independent.')
