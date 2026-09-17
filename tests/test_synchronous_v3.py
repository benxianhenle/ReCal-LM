import torch
from test_four_model import config
from recal.model.text_state_model import TextStateLM
from recal.evaluation.synchronous_v3 import objective, step, gate


def test_equal_depth_objective_and_gradients_match_existing_model():
    torch.manual_seed(51)
    model=TextStateLM(dict(config(),depth_weights=[1/3]*3)).train()
    x=torch.randint(0,32,(1,8)); y=torch.randint(0,32,(1,8))
    expected=model(x,y,alpha=0,lambda_a=0,lambda_r=0)
    actual=objective(model,x,y)
    torch.testing.assert_close(actual['loss'],expected['loss'])
    a=torch.autograd.grad(actual['loss'],[model.attention.embedding.weight,model.core.input_projection.weight,model.decoder.head.weight],retain_graph=True)
    b=torch.autograd.grad(expected['loss'],[model.attention.embedding.weight,model.core.input_projection.weight,model.decoder.head.weight])
    for p,q in zip(a,b):torch.testing.assert_close(p,q)


def test_residual_initialization_matches_trajectory_and_is_causal():
    torch.manual_seed(52)
    model=TextStateLM(dict(config(),depth_weights=[1/3]*3)).eval()
    x=torch.randint(0,32,(1,8)); pos=torch.arange(8); h=model.attention(x)
    expected=model.recur(h,pos)
    model.register_parameter('residual_gates',torch.nn.Parameter(torch.ones(3)))
    torch.testing.assert_close(step(model,h,pos,1),expected,rtol=1e-5,atol=1e-6)
    actual=objective(model,x,x)
    actual['loss'].backward();assert model.residual_gates.grad is not None
    with torch.no_grad():model.residual_gates.fill_(.7)
    later=x.clone();later[:,5:]=(later[:,5:]+1)%32
    for ids in [x,later]:
        previous=model.attention(ids)
        for k in (1,2,3):
            state=step(model,previous,pos,k);logits=model.decoder(state-previous,pos);previous=state
        if ids is x:first=logits
        else:torch.testing.assert_close(first[:,:5],logits[:,:5])


def test_gate_requires_ce_range_and_both_variances_on_both_groups():
    base={'groups':{g:dict(ce3=7.,ce_range=.02,variance_ratio_3_1=.01,delta_var3=.1) for g in ('fixed','extra')}}
    good={'groups':{g:dict(ce3=7.01,ce_range=.01,variance_ratio_3_1=.0095,delta_var3=.095) for g in ('fixed','extra')}}
    assert gate(good,base,base,'E1')['pass_gate']
    good['groups']['extra']['delta_var3']=.08
    assert not gate(good,base,base,'E1')['pass_gate']


def test_block_refresh_uses_only_previous_block_and_real_causal_decoder():
    from recal.evaluation.block_refresh_v3 import RefreshHead,candidates,choices,real_path,oracle_labels
    torch.manual_seed(61)
    model=TextStateLM(dict(config(),depth_weights=[1/3]*3)).eval().requires_grad_(False)
    head=RefreshHead().eval();x=torch.randint(0,32,(1,8));changed=x.clone();changed[:,5:]=(changed[:,5:]+1)%32
    a=candidates(model,x,block_size=4);b=candidates(model,changed,block_size=4)
    torch.testing.assert_close(a['features'],b['features'])
    assert not choices(head,a['features'],-1.)[:,0].any()
    logits,select=real_path(model,head,x,-1.,block_size=4)
    later,_=real_path(model,head,changed,-1.,block_size=4)
    torch.testing.assert_close(logits[:,:5],later[:,:5])
    assert select[:,1].all()
    labels,gain=oracle_labels(model,a,x,block_size=4)
    assert labels.shape==(1,1) and not gain.requires_grad
