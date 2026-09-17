"""Frozen depth diagnostics and causal continuation policies; no validation training."""
import torch
from torch import nn
from torch.nn import functional as F


def token_ce(model,state,positions,labels):
    logits=model.decoder(state,positions)
    return F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),reduction='none',ignore_index=-100).reshape_as(labels)


@torch.no_grad()
def sweep(model,x,y,depth=8,absolute=False,ablate=False):
    positions=torch.arange(x.shape[1],device=x.device);previous=model.attention(x)
    states=[];deltas=[];ces=[];absolute_ces=[];dynamics=[];ablations={}
    for k in range(1,depth+1):
        state=model.recur(previous,positions);delta=state-previous
        ces.append(token_ce(model,delta,positions,y));states.append(state);deltas.append(delta)
        if absolute:absolute_ces.append(token_ce(model,state,positions,y))
        s=state.float();d=delta.float();prev=previous.float()
        dynamics.append(dict(state_variance=float(s.var(dim=1,unbiased=False).mean()),delta_variance=float(d.var(dim=1,unbiased=False).mean()),
            state_norm=float(s.norm(dim=-1).mean()),delta_norm=float(d.norm(dim=-1).mean()),cos_previous=float(F.cosine_similarity(s,prev,dim=-1).mean())))
        if ablate and k in (2,3,4):
            permutation=torch.randperm(x.shape[1],generator=torch.Generator().manual_seed(20260917+k)).to(x.device)
            ablations[k]=dict(delta_zero=token_ce(model,torch.zeros_like(delta),positions,y),delta_shuffle=token_ce(model,delta[:,permutation],positions,y),
                absolute_zero=token_ce(model,previous,positions,y),absolute_shuffle=token_ce(model,previous+delta[:,permutation],positions,y))
        previous=state
    return dict(ce=torch.stack(ces,-1),absolute_ce=torch.stack(absolute_ces,-1) if absolute else None,
        states=torch.stack(states,-2),deltas=torch.stack(deltas,-2),dynamics=dynamics,ablations=ablations)


def oracle_summary(ce,mask):
    valid=ce[mask].double()
    means=valid.mean(0);minimum,best=valid.min(-1)
    histogram=torch.bincount(best,minlength=valid.shape[-1]).double()/len(best)
    # Stable rank buckets avoid empty bins caused by tied CE quantiles.
    order=torch.argsort(valid[:,0],stable=True);buckets=[]
    for i,indices in enumerate(torch.tensor_split(order,5)):
        buckets.append(dict(quantile=[i*.2,(i+1)*.2],tokens=len(indices),initial_ce=float(valid[indices,0].mean()),
            mean_optimal_depth=float((best[indices]+1).double().mean()),oracle_gain=float((valid[indices,2]-minimum[indices]).mean())))
    return dict(tokens=len(best),ce=means.tolist(),gains=(means[:-1]-means[1:]).tolist(),
        best_depth_percent=(histogram*100).tolist(),mean_optimal_depth=float((best+1).double().mean()),
        oracle_ce=float(minimum.mean()),fixed_r3_ce=float(means[2]),best_fixed_depth=int(means.argmin())+1,
        oracle_gain_r3=float(means[2]-minimum.mean()),oracle_gain_best_fixed=float(means.min()-minimum.mean()),difficulty_buckets=buckets)


def diagnostic_gates(groups):
    concentrated=[]
    for name,m in groups.items():
        if name not in ('fixed','extra'):continue
        winner=max(range(len(m['ce'])),key=lambda k:m['best_depth_percent'][k])
        if m['best_depth_percent'][winner]>80 and winner<len(m['ce'])-1 and all(g<0 for g in m['gains'][winner:]):concentrated.append(name)
    boundary=any(groups[g]['best_depth_percent'][-1]>=20 for g in ('fixed','extra'))
    minimum=min(min(groups[g]['oracle_gain_r3'],groups[g]['oracle_gain_best_fixed']) for g in ('fixed','extra'))
    return dict(g1_pass=not concentrated,g1_concentrated_groups=concentrated,boundary_extension_required=boundary,
        g2_pass=not concentrated and minimum>=.05,minimum_oracle_gain=minimum,
        oracle_category='worthwhile' if minimum>=.05 else 'gray' if minimum>=.02 else 'stop',
        rule='Both held-out groups must gain >=.05 versus R3 AND best fixed depth. No validation labels used for controller fitting.')


class ContinueHead(nn.Module):
    def __init__(self,hidden,depths=8,width=128,embedding=16):
        super().__init__();self.depth_embedding=nn.Embedding(depths,embedding)
        self.net=nn.Sequential(nn.LayerNorm(hidden*2+embedding),nn.Linear(hidden*2+embedding,width),nn.GELU(),nn.Linear(width,1))
    def forward(self,state,delta,depth):
        index=torch.as_tensor(depth,device=state.device,dtype=torch.long)-1
        e=self.depth_embedding(index).expand(*state.shape[:-1],-1)
        return self.net(torch.cat([state.float(),delta.float(),e],-1)).squeeze(-1)


def selected_depth(probabilities,threshold=.5):
    # probabilities shape [..., K-1]; fallback is max depth K.
    stops=probabilities<threshold
    return torch.cat([stops,torch.ones_like(stops[...,:1])],-1).int().argmax(-1)+1


@torch.no_grad()
def cached_policy(head,cache,threshold=.5):
    states=cache['states'];deltas=cache['deltas'];depth=states.shape[-2]
    probabilities=torch.stack([head(states[...,k,:],deltas[...,k,:],k+1).sigmoid() for k in range(depth-1)],-1)
    choices=selected_depth(probabilities,threshold)
    ce=cache['ce'].gather(-1,(choices-1).unsqueeze(-1)).squeeze(-1)
    return ce,choices,probabilities


@torch.no_grad()
def adaptive_policy(model,head,x,y,depth=8,threshold=.5,force_depth=None):
    """Causal masked-state rollout, decoded once with final mixed-depth deltas.

    Dense kernels still compute inactive rows. Mean stopping depth is a logical
    compute proxy, NOT an achieved throughput saving. Cached sweep logits have
    different cross-token context and must be reported separately.
    """
    pos=torch.arange(x.shape[1],device=x.device);previous=model.attention(x)
    active=torch.ones_like(x,dtype=torch.bool);selected=torch.zeros_like(previous);choices=torch.zeros_like(x)
    executed=0
    for k in range(1,depth+1):
        state=model.recur(previous,pos);delta=state-previous
        stop=(torch.full_like(active,k>=force_depth) if force_depth is not None else head(state,delta,k).sigmoid()<threshold)
        if k==depth:stop=torch.ones_like(active)
        halt=active&stop
        selected=torch.where(halt[...,None],delta,selected);choices=torch.where(halt,k,choices)
        previous=torch.where(active[...,None],state,previous);active=active&~halt;executed=k
        if not active.any():break
    return token_ce(model,selected,pos,y),choices,executed
