import torch
import pytest
from test_four_model import config
from recal.model.text_state_model import TextStateLM
from recal.evaluation.state_ablation import evaluate,information

def test_diagnostic_matches_actual_delta_path_and_preserves_model_and_rng():
 m=TextStateLM(dict(config(),depth_weights=[.2,.4,1.])).train()
 x=torch.randint(0,32,(1,8));y=torch.randint(0,32,(1,8))
 with torch.no_grad():expected=float(m(x,y)['loss_lm'])
 before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
 r=evaluate(m,[(x,y)],fixed_count=1,shuffle_repeats=2)['groups']['fixed']
 assert r['ce_depth_3']==pytest.approx(expected)
 assert r['zero_delta3_ce']==pytest.approx(float(torch.tensor(32.).log()))
 assert m.training and torch.equal(rng,torch.get_rng_state())
 assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
 assert all(p.grad is None for p in m.parameters())
 assert all(torch.isfinite(torch.tensor(v)) for v in r.values())

def test_information_separates_scale_from_token_distinction():
 x=torch.randn(1,8,16);a=information(x);b=information(x*3)
 assert b['token_variance']==pytest.approx(9*a['token_variance'],rel=1e-5)
 assert b['effective_rank']==pytest.approx(a['effective_rank'],rel=1e-5)
 c=information(torch.ones(1,8,16))
 assert c['token_variance']==0 and c['mean_pair_cosine']==pytest.approx(1.)

def test_abc_changes_only_alignment_and_records_true_clip_fraction():
 import importlib.util
 from pathlib import Path
 from recal.training.text_state_schedule import TextStateSchedule
 path=Path(__file__).resolve().parents[1]/'scripts/run_state_abc.py'
 spec=importlib.util.spec_from_file_location('abc_runner',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 s=TextStateSchedule(52_469_760,follow_scale=.125)
 tokens=136_359_936;initial=s.coefficients(tokens)[0]['lambda_a']
 a=mod.coefficients('A',s,tokens+4_096_000,initial);b=mod.coefficients('B',s,tokens+4_096_000,initial);c=mod.coefficients('C',s,tokens+4_096_000,initial)
 assert a['lambda_a']>initial and b['lambda_a']==0 and c['lambda_a']==initial*.5
 assert a['alpha']==b['alpha']==c['alpha']==.1
 assert a['lambda_r']==b['lambda_r']==c['lambda_r']==0
 rows=[dict(loss_lm=7.,grad_norm=n,module_grad_norm={'attention':n,'core':n,'decoder':n}) for n in [.5,2.,2000.]]
 r=mod.summarize_updates(rows)
 assert r['clip_ratio']==pytest.approx(2/3) and r['spikes_over_1000']==1
 assert r['max_grad_norm']==2000
