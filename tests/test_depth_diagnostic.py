import copy
import pytest
import torch
from test_four_model import config
from recal.model.text_state_model import TextStateLM
from recal.evaluation.state_ablation import evaluate
from scripts.run_depth_diagnostic import update_snapshot, relative_updates, d_gate, coefficients
from recal.training.text_state_schedule import TextStateSchedule


def test_literal_depth_objective_and_default_compatibility():
    torch.manual_seed(19)
    m=TextStateLM(dict(config(True),depth_weights=[.2,.4,1.]))
    x=torch.randint(0,32,(1,8));y=torch.randint(0,32,(1,8))
    a=m(x,y,alpha=.1);b=m(x,y,alpha=.1,lambda_depth=.01,depth_margin=1.)
    expected=sum(torch.relu(b[f'ce_depth_{k+1}']-b[f'ce_depth_{k}']+1.) for k in (1,2))
    torch.testing.assert_close(b['loss_depth_improvement'],expected)
    torch.testing.assert_close(b['loss'],a['loss']+.01*expected)
    # Compare autograd to a manually constructed, connected three-CE reference.
    pos=torch.arange(8);prev=m.attention(x);ces=[]
    for _ in range(3):
        state=m.recur(prev,pos)
        ces.append(torch.nn.functional.cross_entropy(m.decoder(state-prev,pos).float().flatten(0,1),y.flatten()))
        prev=state
    manual=sum(torch.relu(ces[k+1]-ces[k]+1.) for k in (0,1))
    params=[p for p in m.parameters() if p.requires_grad]
    got=torch.autograd.grad(b['loss_depth_improvement'],params,allow_unused=True)
    ref=torch.autograd.grad(manual,params,allow_unused=True)
    for g,r in zip(got,ref):
        if r is None:assert g is None
        else:torch.testing.assert_close(g,r,atol=2e-6,rtol=1e-4)


def test_relative_update_uses_actual_adamw_and_excludes_frozen_embedding():
    m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.]))
    snap=update_snapshot(m)
    with torch.no_grad():
        for name,pairs in snap.items():
            for p,old in pairs:p.mul_(1.02 if name=='core' else 1.1)
        m.core.embedding.weight.add_(100.)
    u=relative_updates(snap)
    assert u==pytest.approx(dict(core=.02,attention=.1,decoder=.1),rel=1e-5)
    rows=[dict(local_step=i*100,relative_update=u) for i in range(1,6)]
    assert d_gate(rows)['eligible']
    assert not d_gate(rows[:4])['eligible']


def test_fast_validation_matches_full_and_preserves_rng():
    m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.]))
    x=torch.randint(0,32,(1,8));rng=torch.get_rng_state()
    full=evaluate(m,[(x,x)],1)['groups']['fixed'];fast=evaluate(m,[(x,x)],1,full=False)['groups']['fixed']
    for k,v in fast.items():assert v==pytest.approx(full[k],rel=1e-5,abs=1e-8)
    assert torch.equal(rng,torch.get_rng_state())
    s=TextStateSchedule(0)
    b=coefficients('B',s,136_359_936);c=coefficients('C',s,136_359_936)
    assert b['lambda_a']==c['lambda_a']==0 and c['lambda_depth']==.01 and b['lambda_depth']==0
