"""Causal block refresh prototype: a completed block informs the NEXT block.

R1/R2 are synchronous. At the third transition a block takes R3 or A(h2).
The decoder sees the actual selected delta context. This prototype evaluates
quality and intentionally computes both candidates; it makes no speedup claim.
"""
import torch
from torch import nn
from torch.nn import functional as F
from recal.evaluation.synchronous_v3 import step


class RefreshHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(8),nn.Linear(8,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,features):return self.net(features.float()).squeeze(-1)


def delayed_features(anchor,state,delta,logits1,logits2,block_size=128,rho=.9):
    """No feature for block b may read any position in block b or later."""
    if state.shape[1]%block_size:raise ValueError('Require complete blocks')
    log1=F.log_softmax(logits1.float(),-1);log2=F.log_softmax(logits2.float(),-1);prob2=log2.exp()
    token=torch.stack([delta.float().norm(dim=-1),delta.float().var(-1,unbiased=False),
        (prob2*(log2-log1)).sum(-1).clamp_min(0),F.cosine_similarity(state.float(),anchor.float(),dim=-1),
        -(prob2*log2).sum(-1),prob2.max(-1).values,state.float().norm(dim=-1)],-1)
    blocks=token.reshape(token.shape[0],-1,block_size,7).mean(2)
    # Stable magnitude compression for norm / variance inputs. Depth remains fixed.
    blocks=blocks.clone();blocks[...,[0,1,6]]=torch.log1p(blocks[...,[0,1,6]])
    cumulative=torch.zeros_like(blocks[:,0,0]);features=[]
    for b in range(blocks.shape[1]):
        if b==0:features.append(torch.zeros((*blocks.shape[:1],8),device=blocks.device))
        else:
            observed=blocks[:,b-1];cumulative=rho*cumulative+observed[:,2]+(1-observed[:,3]).clamp_min(0)
            features.append(torch.cat([observed,cumulative[:,None]],-1))
    return torch.stack(features,1)


@torch.no_grad()
def candidates(model,ids,block_size=128):
    pos=torch.arange(ids.shape[1],device=ids.device);anchor=model.attention(ids)
    h1=step(model,anchor,pos,1);p1=model.decoder(h1-anchor,pos)
    h2=step(model,h1,pos,2);p2=model.decoder(h2-h1,pos)
    feature=delayed_features(anchor,h2,h2-h1,p1,p2,block_size)
    h3=step(model,h2,pos,3);refreshed=model.attention.stack(h2,pos)
    return dict(delta_r=h3-h2,delta_a=refreshed-h2,features=feature,positions=pos)


def choices(head,features,threshold):
    selected=torch.sigmoid(head(features))>threshold
    selected=selected.clone();selected[:,0]=False
    return selected


@torch.no_grad()
def decode_choices(model,candidate,selected,block_size=128):
    mask=selected.repeat_interleave(block_size,dim=1)
    delta=torch.where(mask[...,None],candidate['delta_a'],candidate['delta_r'])
    return model.decoder(delta,candidate['positions'])


@torch.no_grad()
def real_path(model,head,ids,threshold,block_size=128):
    candidate=candidates(model,ids,block_size)
    selected=choices(head,candidate['features'],threshold)
    return decode_choices(model,candidate,selected,block_size),selected


@torch.no_grad()
def oracle_labels(model,candidate,labels,margin=.01,block_size=128):
    """Single-block interventions are training labels, not a cached CE gate."""
    shape=candidate['features'].shape[:2];zero=torch.zeros(shape,dtype=torch.bool,device=labels.device)
    logits=decode_choices(model,candidate,zero,block_size)
    base=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
    benefits=[]
    for b in range(1,shape[1]):
        select=zero.clone();select[:,b]=True
        logits=decode_choices(model,candidate,select,block_size)
        ce=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),reduction='none').reshape_as(labels)
        sl=slice(b*block_size,(b+1)*block_size)
        benefits.append((base[:,sl]-ce[:,sl]).mean(1))
    gain=torch.stack(benefits,1)
    return (gain>margin).float(),gain
