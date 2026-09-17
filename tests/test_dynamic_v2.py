import torch
from test_four_model import config
from recal.model.text_state_model import TextStateLM
from recal.evaluation.dynamic_depth import sweep
from recal.evaluation.dynamic_depth_v2 import distribution_features,future_values,FutureValueHead,real_path,heuristic_gate
from recal.data.dynamic_v2 import accepts_bucket


def test_future_value_sees_beyond_locally_bad_step_and_detaches():
    ce=torch.tensor([[7.,7.2,6.7]],requires_grad=True)
    torch.testing.assert_close(future_values(ce),torch.tensor([[.3,.5]]));assert not future_values(ce).requires_grad


def test_feature_probabilities_and_entropy():
    stats=distribution_features(torch.zeros(2,4,32));torch.testing.assert_close(stats[...,0],torch.full((2,4),32.).log())
    torch.testing.assert_close(stats[...,1],torch.full((2,4),1/32));assert (stats[...,2]==0).all()


def test_real_fixed_paths_match_and_threshold_route_is_causal_without_targets():
    torch.manual_seed(11);m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.])).eval().requires_grad_(False)
    x=torch.randint(0,32,(1,8));y=torch.randint(0,32,(1,8));ce=sweep(m,x,y,3)['ce']
    for k in (1,2,3):
        logits,depth,_=real_path(m,x,force_depth=k)
        actual=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
        torch.testing.assert_close(actual,ce[...,k-1]);assert (depth==k).all()
    for name in ('entropy','confidence','margin'):
        logits,depth,_=real_path(m,x,name,(.02,.02));changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%32
        other,k,_=real_path(m,changed,name,(.02,.02))
        torch.testing.assert_close(logits[:,:5],other[:,:5]);assert torch.equal(depth[:,:5],k[:,:5])
    logits,depth,_=real_path(m,x,'entropy',(float('inf'),float('inf')));assert (depth==1).all()
    torch.testing.assert_close(logits,real_path(m,x,force_depth=1)[0])


def test_disjoint_splits_gate_and_small_value_head():
    for b in range(1000):assert sum(accepts_bucket(b,p) for p in ('validation','calibration','train'))==1
    baseline={g:{'ce':7.} for g in ('fixed','extra')}
    assert not heuristic_gate({'entropy':{'fixed':{'ce':6.9},'extra':{'ce':7.1}}},baseline)['pass_gate']
    assert heuristic_gate({'entropy':{'fixed':{'ce':6.9},'extra':{'ce':6.9}}},baseline)['pass_gate']
    assert sum(p.numel() for p in FutureValueHead(3072).parameters())<1000000
