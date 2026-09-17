"""Depth recurrence and delta decoding, with explicitly isolated auxiliary gradients."""
import torch
from torch.nn import functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from .four_model import FourModelLM


def frozen(module, state, positions):
    # Enter functional_call INSIDE the checkpoint closure: recomputation must use
    # detached parameters too. Input gradients are preserved.
    def call(x):
        params = {name: p.detach() for name, p in module.named_parameters()}
        return functional_call(module, params, (x, positions), {'use_checkpoint': False})
    if torch.is_grad_enabled() and state.requires_grad:
        return checkpoint(call, state, use_reentrant=False)
    return call(state)


def distance(a, b):
    a, b = a.float(), b.float()
    an = F.layer_norm(a, (a.shape[-1],))
    bn = F.layer_norm(b, (b.shape[-1],))
    return (1-F.cosine_similarity(an, bn, dim=-1)).mean() + .1*F.smooth_l1_loss(a,b)


class TextStateLM(FourModelLM):
    def __init__(self, config):
        super().__init__(config, stage='late')
        self.depth_weights = tuple(config['depth_weights'])
        if not self.depth_weights or any(w <= 0 for w in self.depth_weights):
            raise ValueError('Positive depth weights required')
        # Retain the old token embedding in the checkpoint for rollback compatibility.
        # Pure state recurrence no longer consumes token IDs inside R.
        self.core.embedding.requires_grad_(False)

    def recur(self, state, positions):
        return self.core.stack(self.core.input_norm(state + self.core.input_projection(state)), positions)

    def forward(self, ids, labels=None, *, alpha=0., lambda_a=0., lambda_r=0., lambda_depth=0., depth_margin=0., diagnostics=False):
        if min(alpha,lambda_a,lambda_r,lambda_depth,depth_margin)<0: raise ValueError('Negative loss coefficient')
        positions=torch.arange(ids.shape[1],device=ids.device)
        initial=self.attention(ids)
        previous=initial
        zero=initial.new_zeros((),dtype=torch.float32)
        text=zero; atext=zero; follow=zero; drift=zero
        metrics={}; states=[]; depth_ces=[]
        for depth,w in enumerate(self.depth_weights,1):
            state=self.recur(previous,positions)
            delta=state-previous
            logits=self.decoder(delta,positions)
            ce=F.cross_entropy(logits.float().flatten(0,1),labels.flatten()) if labels is not None else zero
            text=text+w*ce
            depth_ces.append(ce)
            metrics[f'ce_depth_{depth}']=ce.detach()
            # Attention accepts hidden states through its shared Transformer stack;
            # token input A(x) uses the original embedding followed by that same stack.
            if alpha or lambda_a or diagnostics:
                accepted=self.attention.stack(state.detach(),positions)
                align=distance(accepted,state.detach())
                follow=follow+align
                if alpha and labels is not None:
                    aux_delta=accepted-previous.detach()
                    atext=atext+w*F.cross_entropy(frozen(self.decoder,aux_delta,positions).float().flatten(0,1),labels.flatten())
                if diagnostics:
                    with torch.no_grad():
                        s=state.float(); a=accepted.float(); d=delta.float()
                        metrics.update({f'cos_depth_{depth}':F.cosine_similarity(a,s,dim=-1).mean(),
                            f'norm_ratio_depth_{depth}':a.norm(dim=-1).mean()/s.norm(dim=-1).mean().clamp_min(1e-8),
                            f'state_norm_depth_{depth}':s.norm(dim=-1).mean(),
                            f'delta_norm_depth_{depth}':d.norm(dim=-1).mean(),
                            f'token_variance_depth_{depth}':s.var(dim=1,unbiased=False).mean(),
                            f'attention_variance_depth_{depth}':a.var(dim=1,unbiased=False).mean(),
                            f'initial_attention_variance':initial.float().var(dim=1,unbiased=False).mean()})
            # P remains separately supervised: predict change between consecutive R states.
            target=(1-F.cosine_similarity(state.detach().float(),previous.detach().float(),dim=-1)).clamp(0,1)
            drift=drift+F.mse_loss(self.drift(state.detach()).float(),target)/len(self.depth_weights)
            states.append(state)
            previous=state
        fixed_point=zero
        if lambda_r:
            # Numerically identical R trajectory, but isolate only its initial A state.
            # No detach between R steps; frozen A still differentiates with respect to R.
            r_previous=initial.detach()
            for _ in self.depth_weights:
                r_state=self.recur(r_previous,positions)
                accepted=frozen(self.attention.stack,r_state,positions)
                fixed_point=fixed_point+distance(accepted,r_state)
                r_previous=r_state
        # Literal objective: both CE terms retain the existing A -> R -> D gradients.
        depth_improvement=sum((F.relu(b-a+depth_margin) for a,b in zip(depth_ces,depth_ces[1:])),zero)
        total=text+alpha*atext+lambda_a*follow+lambda_r*fixed_point+self.config.get('lambda_drift',.05)*drift+lambda_depth*depth_improvement
        return dict(loss=total,loss_text=text,loss_lm=ce,loss_attention_text=atext,
                    loss_attention_follow_r=follow,loss_r_fixed_point=fixed_point,
                    loss_drift=drift,loss_depth_improvement=depth_improvement,logits=logits,**metrics)
