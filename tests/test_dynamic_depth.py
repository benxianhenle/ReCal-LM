import torch
import pytest
from test_four_model import config
from recal.model.text_state_model import TextStateLM
from recal.evaluation.dynamic_depth import sweep,oracle_summary,diagnostic_gates,ContinueHead,selected_depth,adaptive_policy


def test_sweep_matches_training_and_keeps_everything_frozen():
    torch.manual_seed(2);m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.])).eval().requires_grad_(False)
    x=torch.randint(0,32,(1,8));before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state()
    r=sweep(m,x,x,depth=8,absolute=True,ablate=True)
    torch.testing.assert_close(r['ce'][...,2].mean(),m(x,x)['loss_lm'])
    assert r['states'].shape==(1,8,8,16)
    torch.testing.assert_close(r['ablations'][2]['delta_zero'].mean(),torch.tensor(32.).log())
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())


def test_oracle_mask_and_gate_and_earliest_stop():
    ce=torch.tensor([[[3.,2.,4.],[1.,3.,5.],[999.,999.,999.]]]);mask=torch.tensor([[True,True,False]])
    r=oracle_summary(ce,mask)
    assert r['tokens']==2 and r['oracle_ce']==1.5 and r['oracle_gain_r3']==3.
    assert r['best_depth_percent']==[50.,50.,0.]
    assert diagnostic_gates({'fixed':r,'extra':r})['g2_pass']
    assert selected_depth(torch.tensor([[.8,.2,.9],[.8,.7,.6],[.1,.9,.8]])).tolist()==[2,4,1]


def test_fixed_policy_reproduces_depth_and_is_causal():
    torch.manual_seed(3);m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.])).eval().requires_grad_(False)
    head=ContinueHead(16);x=torch.randint(0,32,(1,8));y=torch.randint(0,32,(1,8))
    ce,k,_=adaptive_policy(m,head,x,y,force_depth=3)
    torch.testing.assert_close(ce,sweep(m,x,y,depth=3)['ce'][...,2]);assert (k==3).all()
    a,ak,_=adaptive_policy(m,head,x,y)
    altered=x.clone();altered[:,5:]=(altered[:,5:]+1)%32
    b,bk,_=adaptive_policy(m,head,altered,y)
    torch.testing.assert_close(a[:,:5],b[:,:5]);assert torch.equal(ak[:,:5],bk[:,:5])


def test_controller_gradient_does_not_enter_backbone():
    m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.])).eval().requires_grad_(False);head=ContinueHead(16)
    x=torch.randint(0,32,(1,8));r=sweep(m,x,x)
    losses=[]
    for k in range(7):
        label=(r['ce'][...,k+1]<r['ce'][...,k]).float()
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(head(r['states'][...,k,:],r['deltas'][...,k,:],k+1),label))
    torch.stack(losses).mean().backward()
    assert any(p.grad is not None for p in head.parameters())
    assert all(p.grad is None for p in m.parameters())


def test_controller_gate_requires_both_splits_actual_path_and_compute_budget():
    import copy
    from scripts.run_dynamic_controller import controller_gate
    group=dict(actual_recovery=.6,cached_recovery=.7,actual_average_depth=2.5,actual_ce=6.7,fixed_r3_ce=6.9)
    groups={'fixed':dict(group),'extra':dict(group)}
    assert controller_gate(groups)['pass_gate']
    changed=copy.deepcopy(groups);changed['extra']['actual_recovery']=.1
    assert not controller_gate(changed)['pass_gate'] and controller_gate(changed)['weak_recovery']
    changed=copy.deepcopy(groups);changed['fixed']['actual_average_depth']=3.1
    assert not controller_gate(changed)['pass_gate']
    changed=copy.deepcopy(groups);changed['extra']['cached_recovery']=.4
    assert not controller_gate(changed)['pass_gate']
