"""G3: train only a small continuation head on the independent training stream."""
import fcntl,hashlib,json,math,os,signal,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_dynamic_diagnostics import OUT,SOURCE,load_model
from recal.evaluation.dynamic_depth import ContinueHead,sweep,cached_policy,adaptive_policy
from recal.data.stateful import TextEncoder,make_stream
from recal.training.retention import atomic_json


def controller_gate(groups):
    usable=all(groups[g]['actual_recovery']>.5 and groups[g]['cached_recovery']>.5 and groups[g]['actual_average_depth']<=3 and groups[g]['actual_ce']<groups[g]['fixed_r3_ce'] for g in ('fixed','extra'))
    weak=any(groups[g]['actual_recovery']<.2 or groups[g]['cached_recovery']<.2 for g in ('fixed','extra'))
    return dict(pass_gate=usable,weak_recovery=weak,rule='Actual mixed-context AND cached recovery >50% in Fixed and Extra, actual E[K]<=3, actual CE<R3; threshold fixed at .5.')


def evaluate(model,head,files,oracle):
    head.eval();rows=[];started=time.time()
    with torch.no_grad():
        for file in files:
            c=torch.load(file,weights_only=True,map_location='cuda');x,y=c['x'],c['y'];mask=c['mask']
            cached_ce,cached_k,prob=cached_policy(head,c)
            with torch.autocast('cuda',dtype=torch.bfloat16):actual_ce,actual_k,executed=adaptive_policy(model,head,x,y)
            labels=(c['ce'][...,1:]<c['ce'][...,:-1]).float()
            loss=F.binary_cross_entropy(prob.clamp(1e-7,1-1e-7),labels,reduction='none')[mask].mean()
            rows.append(dict(split=c['split'],count=int(mask.sum()),cached_ce=float(cached_ce[mask].sum()),actual_ce=float(actual_ce[mask].sum()),
                cached_depth=float(cached_k[mask].sum()),actual_depth=float(actual_k[mask].sum()),
                histogram=torch.bincount(actual_k[mask]-1,minlength=8).tolist(),bce=float(loss),dense_recurrence_calls=executed))
            del c
    groups={}
    for group in ('fixed','extra','all'):
        subset=[r for r in rows if group=='all' or r['split']==group];count=sum(r['count'] for r in subset)
        normal=oracle[group]['fixed_r3_ce'];reference=oracle[group]['oracle_ce'];gain=normal-reference
        actual=sum(r['actual_ce'] for r in subset)/count;cached=sum(r['cached_ce'] for r in subset)/count
        groups[group]=dict(tokens=count,fixed_r3_ce=normal,oracle_ce=reference,cached_ce=cached,actual_ce=actual,
            cached_recovery=(normal-cached)/gain,actual_recovery=(normal-actual)/gain,
            cached_average_depth=sum(r['cached_depth'] for r in subset)/count,actual_average_depth=sum(r['actual_depth'] for r in subset)/count,
            depth_percent=[100*sum(r['histogram'][k] for r in subset)/count for k in range(8)],
            bce=sum(r['bce']*r['count'] for r in subset)/count,
            mean_dense_recurrence_calls=sum(r['dense_recurrence_calls'] for r in subset)/len(subset))
    if not all(math.isfinite(v) for g in groups.values() for v in g.values() if isinstance(v,(int,float))):raise FloatingPointError('Nonfinite controller validation')
    head.train()
    return dict(groups=groups,gate=controller_gate(groups),elapsed_seconds=time.time()-started,
        note='Oracle uses dense-depth contexts. Actual policy freezes stopped states and decodes mixed deltas, so its context changes. E[K] is logical depth, not measured GPU savings.')


def main():
    gate=json.loads((OUT/'G2.json').read_text())
    if not gate['g3_authorized_by_gates']:raise RuntimeError('Prior gates did not pass')
    if (OUT/'G3.json').exists():raise RuntimeError('Controller already evaluated')
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=json.loads((OUT/'controller-budget.json').read_text())['deadline'];budget=json.loads((OUT/'budget.json').read_text())
    stopped=False
    def stop(*args):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def event(**kw):
        r=dict(stage='G3',time=time.time(),pid=os.getpid(),remaining_seconds=deadline-time.time(),**kw)
        atomic_json(OUT/'status.json',r);print(json.dumps(r),flush=True)
    event(event='loading_frozen_model')
    model,base=load_model();versions={name:p._version for name,p in model.named_parameters()}
    source_stat=[SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns]
    torch.manual_seed(20260917);torch.cuda.manual_seed_all(20260917)
    head=ContinueHead(base['config']['hidden_size']).cuda().train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json')
    stream=make_stream(ROOT/'artifacts/four-model-v1/data_manifest.json',encoder,512,1337,'train');stream.load_state_dict(base['data_state'])
    files=sorted((OUT/'validation-cache').glob('batch-*.pt'));oracle=gate['groups']
    validation_hashes=set()
    for file in files:
        c=torch.load(file,weights_only=True,map_location='cpu')
        validation_hashes.add(hashlib.sha256(c['x'].numpy().tobytes()+c['y'].numpy().tobytes()).hexdigest())
    # Force-R3 mixed-path sanity before scoring a learned head.
    c=torch.load(files[0],weights_only=True,map_location='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):actual,k,_=adaptive_policy(model,head,c['x'],c['y'],force_depth=3)
    torch.testing.assert_close(actual,c['ce'][...,2],rtol=0,atol=1e-5);del c
    plan=dict(seed=20260917,input='[n_k,delta_k,learned depth embedding(16)]',width=128,threshold=.5,learning_rate=.001,
        frozen_modules=['attention','core','decoder','original drift'],head_parameters=sum(p.numel() for p in head.parameters()),
        label='CE_(k+1)<CE_k, margin=0',loss='BCE; all depths 1..7 equally weighted',
        data='Independent document-hash train split; never fit Fixed/Extra caches',max_tokens=budget['g3_token_cap'],deadline=deadline,
        checkpoint_source=str(SOURCE),source_hash=json.loads((OUT/'manifest.json').read_text())['checkpoint_hash'],
        continuation_rule='1M first; extend by 1M only if minimum actual recovery improves by >=.02, neither group below .20, and full next block fits time; cap 3M.')
    atomic_json(OUT/'controller-plan.json',plan)
    initial=evaluate(model,head,files,oracle);atomic_json(OUT/'controller-validation-000000000.json',initial)
    event(event='initial_validation',groups=initial['groups'])
    validation_seconds=max(initial['elapsed_seconds'],30);history=[dict(tokens=0,**initial)];last_validation=0
    max_batches=budget['g3_token_cap']//512;milestones={math.ceil(n/512) for n in (250000,500000,1000000,2000000)}|{max_batches}
    stage_target=math.ceil(1000000/512);tokens=0;train_seconds=0.;reason=None;loss_sum=0.;start=time.time()
    with (OUT/'controller-training.jsonl').open('a') as log:
        for batch in range(1,max_batches+1):
            if stopped:reason='signal';break
            if time.time()>deadline-max(120,validation_seconds*1.5):reason='controller_time_budget';break
            t=time.time();x,y=stream.next_batch(1);h=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
            if h in validation_hashes:raise RuntimeError('Training/validation exact batch overlap')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):r=sweep(model,x.cuda(),y.cuda())
            optimizer.zero_grad(set_to_none=True);losses=[]
            for k in range(7):
                logits=head(r['states'][...,k,:],r['deltas'][...,k,:],k+1)
                label=(r['ce'][...,k+1]<r['ce'][...,k]).float()
                losses.append(F.binary_cross_entropy_with_logits(logits,label))
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True));optimizer.step()
            tokens+=x.numel();loss_sum+=float(loss);del r,loss,losses;torch.cuda.synchronize();train_seconds+=time.time()-t
            if batch%50==0:
                record=dict(batch=batch,tokens=tokens,mean_bce=loss_sum/50,grad_norm=norm,tokens_per_second=tokens/train_seconds,batch_hash=h)
                log.write(json.dumps(record)+'\n');log.flush();event(event='training',**record);loss_sum=0.
            if batch in milestones:
                report=evaluate(model,head,files,oracle);last_validation=tokens;validation_seconds=max(validation_seconds,report['elapsed_seconds'])
                atomic_json(OUT/f'controller-validation-{tokens:09d}.json',report);history.append(dict(tokens=tokens,**report))
                torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),plan=plan,tokens=tokens,data_state=stream.state_dict()),OUT/f'controller-{tokens:09d}.pt')
                event(event='validation',tokens=tokens,groups=report['groups'],gate=report['gate'])
                if batch>=stage_target:
                    if report['gate']['pass_gate']:reason='G3_pass';break
                    if report['gate']['weak_recovery']:reason='Recovery_below_20_percent';break
                    previous=history[-2]['groups'];minimum=min(report['groups'][g]['actual_recovery'] for g in ('fixed','extra'))
                    previous_min=min(previous[g]['actual_recovery'] for g in ('fixed','extra'))
                    if minimum-previous_min<.02:reason='No_clear_controller_improvement';break
                    needed=(1000000/(tokens/train_seconds))+2*validation_seconds+120
                    if deadline-time.time()<needed:reason='Insufficient_time_for_next_1M';break
                    stage_target=min(math.ceil((tokens//1000000+1)*1000000/512),max_batches)
        if reason is None:reason='3M_token_cap'
    if last_validation!=tokens:
        report=evaluate(model,head,files,oracle);atomic_json(OUT/f'controller-validation-{tokens:09d}.json',report);history.append(dict(tokens=tokens,**report))
    else:report=history[-1]
    unchanged=all(p._version==versions[name] and p.grad is None and not p.requires_grad for name,p in model.named_parameters())
    if not unchanged or source_stat!=[SOURCE.stat().st_size,SOURCE.stat().st_mtime_ns]:raise RuntimeError('Frozen backbone was modified')
    torch.save(dict(head=head.state_dict(),optimizer=optimizer.state_dict(),plan=plan,tokens=tokens,data_state=stream.state_dict()),OUT/'controller-final.pt')
    passed=report['gate']['pass_gate'] and reason=='G3_pass'
    result=dict(complete=True,pass_gate=passed,tokens=tokens,training_seconds=train_seconds,elapsed_seconds=time.time()-start,
        stop_reason=reason,final=report,history=history,frozen_backbone_unchanged=unchanged,threshold=.5,g4_authorized_by_gate=passed)
    atomic_json(OUT/'G3.json',result)
    event(event='complete',tokens=tokens,pass_gate=passed,reason=reason,next_stage='G4' if passed else 'STOP')
    if not passed:atomic_json(OUT/'completion.json',dict(complete=True,stopped_at='G3',reason=reason,controller_training_tokens=tokens,joint_training_tokens=0,finished_at=time.time(),within_budget=time.time()<=budget['overall_deadline']))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'controller-failure.json',dict(error=repr(exc),time=time.time()));raise
