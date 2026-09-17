"""V2 real-path heuristics; routing APIs deliberately do not accept target tokens."""
import torch
from torch import nn
from torch.nn import functional as F


def distribution_features(logits):
    logp=F.log_softmax(logits.float(),dim=-1);prob=logp.exp();top=prob.topk(2,dim=-1).values
    return torch.stack([-(prob*logp).sum(-1),top[...,0],top[...,0]-top[...,1]],-1)


def future_values(ce):
    return torch.stack([ce[...,k]-ce[...,k+1:].min(-1).values for k in range(ce.shape[-1]-1)],-1).detach()


class FutureValueHead(nn.Module):
    def __init__(self,hidden,depths=3,width=128,embedding=16):
        super().__init__();self.norm_n=nn.LayerNorm(hidden);self.norm_delta=nn.LayerNorm(hidden)
        self.depth_embedding=nn.Embedding(depths,embedding)
        self.net=nn.Sequential(nn.Linear(hidden*2+3+embedding,width),nn.GELU(),nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,state,delta,features,depth):
        e=self.depth_embedding(torch.as_tensor(depth-1,device=state.device)).expand(*state.shape[:-1],-1)
        return self.net(torch.cat([self.norm_n(state.float()),self.norm_delta(delta.float()),features.float(),e],-1)).squeeze(-1)


def policy_score(kind,features,state=None,delta=None,depth=None,head=None):
    if kind=='entropy':return features[...,0]
    if kind=='confidence':return features[...,1]
    if kind=='margin':return features[...,2]
    if kind=='value':return head(state,delta,features,depth)
    raise ValueError(kind)


@torch.no_grad()
def real_path(model,ids,kind='entropy',thresholds=(0.,0.),max_depth=3,force_depth=None,head=None):
    if max_depth not in (3,4):raise ValueError('V2 locks R5-R8')
    if force_depth is not None and not 1<=force_depth<=max_depth:raise ValueError('Invalid forced depth')
    positions=torch.arange(ids.shape[1],device=ids.device);previous=model.attention(ids)
    active=torch.ones_like(ids,dtype=torch.bool);chosen_delta=torch.zeros_like(previous);depths=torch.zeros_like(ids)
    logits=None
    for k in range(1,max_depth+1):
        proposal=model.recur(previous,positions);delta=proposal-previous
        candidate=torch.where(active[...,None],delta,chosen_delta)
        logits=None
        if force_depth is not None:
            stop=torch.full_like(active,k>=force_depth)
        elif k==max_depth:
            stop=torch.ones_like(active)
        else:
            logits=model.decoder(candidate,positions)
            features=distribution_features(logits)
            score=policy_score(kind,features,proposal,delta,k,head)
            should_continue=score>thresholds[k-1] if kind in ('entropy','value') else score<thresholds[k-1]
            stop=~should_continue
        halt=active&stop;chosen_delta=torch.where(halt[...,None],delta,chosen_delta)
        depths=torch.where(halt,k,depths);previous=torch.where(active[...,None],proposal,previous);active=active&~halt
        if not active.any():break
    if logits is None:logits=model.decoder(chosen_delta,positions)
    # If all active tokens stop at a threshold, candidate == final chosen_delta,
    # so reusing these real mixed-context logits is exact (not cached depth CE).
    return logits,depths,k


@torch.no_grad()
def score_batches(model,batches,kind='entropy',thresholds=(0.,0.),force_depth=None,head=None):
    device=next(model.parameters()).device;count=0;total=0.;hist=torch.zeros(4,dtype=torch.long);calls=0
    import time
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    rows=[]
    for x,y in batches:
        x,y=x.to(device),y.to(device);mask=y!=-100
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            logits,depth,k=real_path(model,x,kind,thresholds,force_depth=force_depth,head=head)
        ce=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none',ignore_index=-100).reshape_as(y)
        if not torch.isfinite(ce[mask]).all():raise FloatingPointError('Nonfinite real-path CE')
        n=int(mask.sum());value=float(ce[mask].double().sum());count+=n;total+=value;calls+=k
        hist+=torch.bincount(depth[mask].cpu()-1,minlength=4);rows.append(dict(tokens=n,ce=value/n))
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(tokens=count,ce=total/count,average_depth=float((hist*torch.arange(1,5)).sum()/count),
        depth_percent=(hist.double()*100/count).tolist(),mean_dense_recurrence_calls=calls/len(batches),elapsed_seconds=time.perf_counter()-started,batches=rows)


@torch.no_grad()
def dense_features(model,batches,head=None):
    device=next(model.parameters()).device;collected=[]
    for x,y in batches:
        x=x.to(device);positions=torch.arange(x.shape[1],device=device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            previous=model.attention(x);per_depth=[]
            for k in (1,2):
                state=model.recur(previous,positions);delta=state-previous
                features=distribution_features(model.decoder(delta,positions))
                per_depth.append(features.cpu() if head is None else head(state,delta,features,k).cpu()[...,None])
                previous=state
        collected.append(torch.stack(per_depth,-2))
    return torch.cat(collected).flatten(0,1)


def heuristic_gate(results,baseline,tolerance=1e-5):
    winners=[name for name,r in results.items() if all(r[g]['ce']<baseline[g]['ce']-tolerance for g in ('fixed','extra'))]
    all_worse=all(all(r[g]['ce']>baseline[g]['ce']+tolerance for g in ('fixed','extra')) for r in results.values())
    return dict(pass_gate=bool(winners),passing_strategies=winners,all_worse=all_worse,
        rule='Thresholds chosen only on calibration. A policy must lower actual mixed-path CE in BOTH held-out groups; 1e-5 numerical tolerance.',
        reason='pass' if winners else 'All heuristics worse on both groups' if all_worse else 'No heuristic improves both groups; gate not passed')
