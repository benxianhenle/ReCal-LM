import copy
import json
import torch
import pytest
from recal.model.four_model import FourModelLM
from recal.training.stages import PlateauSwitch
from recal.training.retention import CheckpointStore
from recal.data.stateful import JsonlDocuments, PackedStream


def config(checkpointing=False):
    return dict(vocab_size=32, context_length=16, hidden_size=16, num_heads=2, ffn_dim=32,
        front_layers=1, recurrent_layers=2, back_layers=1, drift_hidden_size=8,
        dropout=0., rollout_steps=2, gradient_checkpointing=checkpointing, ema_decay=.9)


def gradient_groups(model):
    return {name for name, module in model.named_children()
            if any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())}


@pytest.mark.parametrize('loss,expected', [('loss_core', {'core'}), ('loss_attention', {'attention'}),
    ('loss_lm', {'decoder'}), ('loss_drift', {'drift'})])
@pytest.mark.parametrize('source', ['attention', 'core'])
def test_early_loss_gradient_ownership(loss, expected, source):
    model = FourModelLM(config())
    x = torch.randint(0, 32, (2, 8))
    out = model(x, x, decoder_source=source, decoder_step=2)
    out[loss].backward()
    assert gradient_groups(model) == expected
    all_groups = [set(map(id, mod.parameters())) for name, mod in model.named_children()]
    for i, group in enumerate(all_groups):
        assert all(not group.intersection(other) for other in all_groups[i+1:])


def test_late_removes_teacher_and_connects_three_modules():
    model = FourModelLM(config(True))
    model.enter_late()
    assert model.teacher is None
    assert not any(k.startswith('teacher.') for k in model.state_dict())
    x = torch.randint(0,32,(1,8))
    model(x,x)['loss_lm'].backward()
    assert gradient_groups(model) == {'attention','core','decoder'}


def test_ema_moves_only_teacher_and_keeps_teacher_eval():
    model = FourModelLM(config())
    before = next(model.teacher.parameters()).clone()
    with torch.no_grad(): next(model.attention.parameters()).add_(1.)
    model.update_teacher(); model.train()
    assert torch.allclose(next(model.teacher.parameters()),before+.1,atol=1e-6)
    assert not model.teacher.training
    assert all(not p.requires_grad for p in model.teacher.parameters())


@pytest.mark.parametrize('stage', ['early', 'late'])
@pytest.mark.parametrize('source', ['attention', 'core'])
def test_token_time_alignment_and_no_future_leakage(stage, source):
    if stage == 'late' and source == 'attention': return
    model = FourModelLM(config(),stage).eval()
    ids = torch.randint(0,32,(1,10))
    changed=ids.clone(); changed[:,6:]=(changed[:,6:]+1)%32
    with torch.no_grad():
        a=model(ids,ids,decoder_source=source,decoder_step=2)['logits']
        b=model(changed,changed,decoder_source=source,decoder_step=2)['logits']
    # Logit position j represents state after token j+2: first four cannot see token 6.
    torch.testing.assert_close(a[:,:4],b[:,:4])
    out=model(ids,ids,decoder_source=source,decoder_step=2)
    expected=torch.nn.functional.cross_entropy(out['logits'].reshape(-1,32),ids[:,2:10].reshape(-1))
    torch.testing.assert_close(out['loss_lm'],expected)


def test_plateau_counts_five_changes_resets_and_persists():
    p=PlateauSwitch()
    assert not p.observe(10.)
    for _ in range(4): assert not p.observe(10.)
    assert not p.observe(9.)
    assert p.stable_changes==0
    for _ in range(3): assert not p.observe(9.)
    q=PlateauSwitch(**p.state_dict())
    assert not q.observe(9.)
    assert q.observe(9.)
    assert not q.observe(9.)
    with pytest.raises(ValueError): PlateauSwitch().observe(float('nan'))


def test_three_snapshot_roles_and_reopen(tmp_path):
    store=CheckpointStore(tmp_path)
    def save(step,loss,stage='early',boundary=None):
        store.save(dict(step=step,stage=stage,model={'w':torch.tensor(step)}),loss,boundary)
        assert len(list(tmp_path.glob('*.pt'))) <= 3
        assert len(set(store.manifest['roles'].values())) <= 3
    save(1,10); save(2,9); save(3,11,boundary='before')
    save(4,12,'late'); save(5,13,'late')
    assert store.manifest['roles']['best']=='step_000000002_early.pt'
    assert store.manifest['roles']['transition_before']=='step_000000003_early.pt'
    assert store.manifest['roles']['latest']=='step_000000005_late.pt'
    save(6,8,'late')
    assert len(list(tmp_path.glob('*.pt')))==2
    assert CheckpointStore(tmp_path).manifest==store.manifest


def test_failed_write_preserves_complete_snapshots(tmp_path,monkeypatch):
    store=CheckpointStore(tmp_path)
    store.save(dict(step=1,stage='early'),10.)
    before=copy.deepcopy(store.manifest)
    def fail(*a,**k): raise OSError('disk full')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError): store.save(dict(step=2,stage='early'),9.)
    assert store.manifest==before
    assert CheckpointStore(tmp_path).manifest==before
    assert not list(tmp_path.glob('*.tmp'))


def test_stream_resume_preserves_pending_tokens(tmp_path):
    path=tmp_path/'data.jsonl'
    path.write_text(''.join(json.dumps({'text':str(i)*10})+'\n' for i in range(5)))
    class Encoder:
        fingerprint='test'
        def encode(self,s): return list(s.encode())
    def make(): return PackedStream({'a':JsonlDocuments(path)},[1.],Encoder(),7,4,'test')
    a=make(); a.next_batch(2); snapshot=a.state_dict()
    expected=[a.next_batch(2) for _ in range(6)]
    b=make(); b.load_state_dict(snapshot)
    for x,y in expected:
        bx,by=b.next_batch(2)
        assert torch.equal(x,bx) and torch.equal(y,by)


def test_parquet_cursor_roundtrip(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from recal.data.stateful import ParquetDocuments
    f=tmp_path/'sample.parquet'
    pq.write_table(pa.table({'content':[f'code example {i}' for i in range(400)]}),f,row_group_size=150)
    a=ParquetDocuments([f],'train',1)
    for _ in range(175): a.next_text()
    state=a.state_dict()
    expected=[a.next_text() for _ in range(300)]
    b=ParquetDocuments([f],'train',1); b.load_state_dict(state)
    assert [b.next_text() for _ in range(300)] == expected


def test_jsonl_with_no_text_fails_instead_of_hanging(tmp_path):
    p=tmp_path/'empty.jsonl'; p.write_text('{"text":""}\n')
    with pytest.raises(ValueError,match='no nonempty'):
        JsonlDocuments(p).next_text()


def test_manual_transition_pins_existing_latest_without_copy(tmp_path):
    store=CheckpointStore(tmp_path)
    store.save(dict(step=1,stage='early'),8.)
    store.save(dict(step=2,stage='early'),9.)
    before={p.name:p.stat().st_mtime_ns for p in tmp_path.glob('*.pt')}
    boundary=store.mark_manual_transition()
    assert boundary.name=='step_000000002_early.pt'
    assert before=={p.name:p.stat().st_mtime_ns for p in tmp_path.glob('*.pt')}
    reopened=CheckpointStore(tmp_path)
    assert reopened.manifest['manual_transition']==boundary.name
    assert reopened.manifest['roles']['transition_before']==boundary.name
    reopened.save(dict(step=3,stage='late'),8.5)
    assert len(list(tmp_path.glob('*.pt')))==3
    with pytest.raises(ValueError): reopened.mark_manual_transition()


@pytest.mark.parametrize('checkpointing',[False,True])
def test_direct_attention_supervision_updates_only_attention(checkpointing):
    c=config(checkpointing);c['lambda_attention_lm']=.1
    model=FourModelLM(c,stage='late').train()
    x=torch.randint(0,32,(2,8))
    model(x,x)['loss_attention_lm'].backward()
    assert gradient_groups(model)=={'attention'}
    assert all(p.requires_grad for p in model.decoder.parameters())
    model.zero_grad(set_to_none=True)
    model(x,x)['loss'].backward()
    assert gradient_groups(model)=={'attention','core','decoder','drift'}


def test_auxiliary_does_not_change_primary_logits_and_zero_is_backward_compatible():
    c=config();base=FourModelLM(c,stage='late').eval()
    aux=FourModelLM(dict(c,lambda_attention_lm=.1),stage='late').eval()
    aux.load_state_dict(base.state_dict())
    x=torch.randint(0,32,(1,8))
    a=base(x,x);b=aux(x,x)
    torch.testing.assert_close(a['logits'],b['logits'])
    torch.testing.assert_close(a['loss_lm'],b['loss_lm'])
    torch.testing.assert_close(b['loss'],a['loss']+.1*b['loss_attention_lm'])
    assert a['loss_attention_lm']==0
