import pytest
import torch
from test_four_model import config,gradient_groups
from recal.model.four_model import FourModelLM
from recal.model.text_state_model import TextStateLM
from recal.training.text_state_schedule import TextStateSchedule

@pytest.mark.parametrize('gc',[False,True])
@pytest.mark.parametrize('key,groups',[
 ('loss_text',{'attention','core','decoder'}),
 ('loss_attention_text',{'attention'}),('loss_attention_follow_r',{'attention'}),
 ('loss_r_fixed_point',{'core'}),('loss_drift',{'drift'})])
def test_gradient_ownership(gc,key,groups):
 c=dict(config(gc),depth_weights=[.2,.4,1.])
 m=TextStateLM(c).train();x=torch.randint(0,32,(1,8))
 o=m(x,x,alpha=.1,lambda_a=.05,lambda_r=.01)
 o[key].backward()
 assert gradient_groups(m)==groups
 assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_compatible_weights_causal_depth_and_bptt():
 c=dict(config(True),depth_weights=[.2,.4,1.])
 m=TextStateLM(c).train();m.load_state_dict(FourModelLM(c,stage='late').state_dict(),strict=True)
 x=torch.randint(0,32,(1,8));y=x.clone();y[:,5:]=(y[:,5:]+1)%32
 torch.testing.assert_close(m(x)['logits'][:,:5],m(y)['logits'][:,:5])
 states=[]
 def hook(module,args,out):out.retain_grad();states.append(out)
 h=m.core.stack.register_forward_hook(hook)
 o=m(x,x);torch.nn.functional.cross_entropy(o['logits'].flatten(0,1),x.flatten()).backward();h.remove()
 assert all(s.grad is not None and s.grad.abs().sum()>0 for s in states[:3])
 assert gradient_groups(m)=={'attention','core','decoder'}
 torch.testing.assert_close(o['loss_text'],sum(w*o[f'ce_depth_{i}'] for i,w in enumerate([.2,.4,1.],1)))


def metrics(ce=7.,cos=.8):
 m={'ce_depth_1':ce+.1,'ce_depth_2':ce+.05,'ce_depth_3':ce}
 for k in range(1,4):m.update({f'state_norm_depth_{k}':10.,f'token_variance_depth_{k}':1.,f'delta_norm_depth_{k}':5.,f'cos_depth_{k}':cos})
 return m


def test_schedule_ramps_and_validation_gate():
 s=TextStateSchedule(50_000_000)
 assert s.coefficients(50_000_000)[0]==dict(alpha=0.,lambda_a=0.,lambda_r=0.)
 assert s.coefficients(100_000_000)[0]==dict(alpha=.1,lambda_a=0.,lambda_r=0.)
 assert s.coefficients(200_000_000)[0]==dict(alpha=.1,lambda_a=.05,lambda_r=0.)
 s.observe(metrics(),50_000_000)
 for _ in range(3):s.observe(metrics(6.9,.85),200_000_000)
 assert s.r_start==200_000_000
 assert s.coefficients(300_000_000)[0]['lambda_r']==pytest.approx(.01)
 bad=metrics(7.5,.999);bad['state_norm_depth_1']=6.
 s.observe(bad,300_000_000)
 assert s.r_start is None and s.follow_scale<1.
 assert 'state_norm_depth_1_fell' in s.alerts


def test_persistent_validation_regression_stops():
 s=TextStateSchedule(0);s.observe(metrics(),0)
 for i in range(1,6):s.observe(metrics(7.+i*.1),i*4096)
 assert s.stop and s.alpha_scale<1.
