"""Conditional E3: 2M-token block refresh head experiment, frozen E2 backbone."""
import json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from recal.data.dynamic_v2 import calibration,training_stream
from recal.data.stateful import TextEncoder
from recal.evaluation.block_refresh_v3 import RefreshHead,candidates,choices,decode_choices,oracle_labels,real_path
from recal.training.retention import atomic_json
from scripts.run_dynamic_diagnostics import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/recal-r-v3-20260917'


@torch.no_grad()
def score(model,head,batches,threshold):
    total=0.;count=0;refresh=0;blocks=0;rows=[]
    for x,y in batches:
        x,y=x.cuda(),y.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,selected=real_path(model,head,x,threshold)
        ce=F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='sum')
        if not torch.isfinite(ce):raise FloatingPointError('Nonfinite E3 CE')
        total+=float(ce);count+=y.numel();refresh+=int(selected.sum());blocks+=selected.numel()
        rows.append(dict(ce=float(ce)/y.numel(),refresh_blocks=int(selected.sum())))
    return dict(ce3=total/count,tokens=count,refresh_percent=100*refresh/blocks,batches=rows)


@torch.no_grad()
def calibrate(model,head,batches,dest,milestone):
    cached=[];prob=[]
    for x,y in batches:
        with torch.autocast('cuda',dtype=torch.bfloat16):c=candidates(model,x.cuda())
        # Candidate trajectories have no earlier final-depth feedback. Each candidate
        # threshold still runs the real decoder on its complete selected context.
        prob.append(torch.sigmoid(head(c['features']))[:,1:].flatten().cpu());cached.append((c,y.cuda()))
    p=torch.cat(prob);thresholds=[0.,1.]+[float(torch.quantile(p,q)) for q in (.2,.3,.4,.5,.6,.7,.8)]
    search=[]
    for threshold in thresholds:
        total=0.;count=0;refresh=0
        for c,y in cached:
            selected=choices(head,c['features'],threshold)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=decode_choices(model,c,selected)
            total+=float(F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='sum'));count+=y.numel();refresh+=int(selected.sum())
        search.append(dict(threshold=threshold,ce=total/count,refresh_blocks=refresh))
    chosen=min(search,key=lambda r:(r['ce'],r['refresh_blocks']))
    atomic_json(dest/f'threshold-{milestone}.json',dict(sealed_at=time.time(),chosen=chosen,search=search,selection='calibration-only real selected-context CE'))
    return chosen['threshold']


def run_refresh(model,base,deadline,previous_tokens,event):
    dest=OUT/'E3';dest.mkdir(exist_ok=True);started=time.time()
    stage_deadline=min(deadline,started+60*60)
    model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    torch.manual_seed(20260920);head=RefreshHead().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json');manifest=ROOT/'artifacts/four-model-v1/data_manifest.json'
    stream=training_stream(manifest,encoder,seed=20260920)
    cal,documents=calibration(manifest,encoder,seed=20260921)
    torch.save(dict(batches=cal,documents=documents),dest/'calibration.pt')
    fixed=torch.load(ROOT/'runs/four-model-r3b-10b/validation.pt',weights_only=True)['batches']
    extra=torch.load(ROOT/'reports/attention-aux/validation-extra.pt',weights_only=True)
    reference=json.loads((OUT/'E2/result.json').read_text())['validation']
    atomic_json(dest/'plan.json',dict(max_tokens=2_000_000,actual_token_target=(2_000_000//512)*512,block_size=128,margin=.01,rho=.9,
        loss='BCE with block oracle CE_A + 0.01 < CE_R; frozen A/R/D; no labels in routing',
        transition='After synchronous R1/R2, third transition is R3 or Attention.stack(h2); decode selected delta. Never R4.',
        decision='Previous completed block features decide next block; first block always R3.',
        oracle='One-block refresh interventions with remaining blocks R3; final gate uses real full selected-context decoder.',
        calibration_documents=documents,calibration_hash=digest(dest/'calibration.pt'),training_split='document bucket >=10; calibration 1..9; validation 0',
        gate='At final 2M: actual CE improves both Fixed and Extra >1e-5 relative to E2; stop early if either regresses >0.03.',
        head_parameters=sum(p.numel() for p in head.parameters()),deadline=stage_deadline,
        implementation_sha256={f:digest(ROOT/f) for f in ('scripts/run_synchronous_v3_refresh.py','recal/evaluation/block_refresh_v3.py')}))
    count=0;positive=0;label_count=0;reason=None;validation=None;last_eval=0
    target=2_000_000//512;milestones={math.ceil(1_000_000/512),target}
    with (dest/'steps.jsonl').open('a') as log:
        for local in range(1,target+1):
            if time.time()>stage_deadline-180:reason='time_budget';break
            x,y=stream.next_batch(1)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                c=candidates(model,x.cuda());labels,gains=oracle_labels(model,c,y.cuda())
            prediction=head(c['features'][:,1:].detach());loss=F.binary_cross_entropy_with_logits(prediction,labels)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite E3 head loss')
            opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True));opt.step()
            count+=1;positive+=int(labels.sum());label_count+=labels.numel()
            log.write(json.dumps(dict(step=count,tokens=count*512,loss=float(loss),grad_norm=norm,positive_labels=int(labels.sum()),labels=labels.numel(),mean_oracle_gain=float(gains.mean())))+'\n')
            if count%100==0 or count==1:
                log.flush();event(stage='E3',action='training',training_tokens=previous_tokens+count*512,stage_tokens=count*512,positive_fraction=positive/max(1,label_count))
            if count in milestones:
                head.eval();threshold=calibrate(model,head,cal,dest,count*512)
                validation=dict(groups={name:score(model,head,bs,threshold) for name,bs in [('fixed',fixed),('extra',extra)]})
                atomic_json(dest/f'validation-{count:05d}.json',validation);last_eval=count;head.train()
                event(stage='E3',action='validation',stage_tokens=count*512,training_tokens=previous_tokens+count*512,validation=validation)
                if any(validation['groups'][g]['ce3']>reference['groups'][g]['ce3']+.03 for g in ('fixed','extra')):reason='real_path_ce_regression';break
    if count and last_eval!=count:
        head.eval();threshold=calibrate(model,head,cal,dest,count*512)
        validation=dict(groups={name:score(model,head,bs,threshold) for name,bs in [('fixed',fixed),('extra',extra)]})
        atomic_json(dest/f'validation-{count:05d}.json',validation)
    unchanged=all(p._version==versions[n] and not p.requires_grad and p.grad is None for n,p in model.named_parameters())
    if not unchanged:raise RuntimeError('E3 frozen backbone mutated')
    decision=dict(pass_gate=bool(validation) and count==target and reason is None and all(validation['groups'][g]['ce3']<reference['groups'][g]['ce3']-1e-5 for g in ('fixed','extra')),
        ce_delta_vs_E2={g:validation['groups'][g]['ce3']-reference['groups'][g]['ce3'] for g in ('fixed','extra')} if validation else {},
        rule='Actual causal block-refresh path improves both heldout groups; cached label metrics are not success criteria.')
    torch.save(dict(head=head.state_dict(),optimizer=opt.state_dict(),data_state=stream.state_dict(),tokens=count*512,source='E2.pt',block_size=128),dest/'refresh-head.pt')
    result=dict(stage='E3',training_tokens=count*512,total_training_tokens=previous_tokens+count*512,complete=count==target and reason is None,
        stop_reason=reason,gate=decision,validation=validation,elapsed_seconds=time.time()-started,frozen_backbone_unchanged=unchanged,
        positive_label_fraction=positive/max(1,label_count),note='Dense two-candidate implementation; no compute saving claim. No continuation into formal training.')
    atomic_json(dest/'result.json',result);return result
