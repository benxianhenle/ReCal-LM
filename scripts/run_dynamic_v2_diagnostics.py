"""V2.0 tail diagnostics -> V2.1 calibration-only heuristic search, zero training."""
import csv,fcntl,hashlib,json,math,os,sys,time,unicodedata
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_dynamic_diagnostics import load_model,SOURCE,digest
from recal.data.stateful import TextEncoder
from recal.data.dynamic_v2 import calibration
from recal.evaluation.dynamic_depth_v2 import distribution_features,dense_features,score_batches,heuristic_gate
from recal.training.retention import atomic_json
OUT=ROOT/'reports/dynamic-r-v2-20260917';OLD=ROOT/'reports/dynamic-depth-20260917'


def causal_local_variance(state,width=8):
    x=state.double();c=torch.cat([torch.zeros_like(x[:,:1]),x.cumsum(1)],1);q=torch.cat([torch.zeros_like(x[:,:1]),x.square().cumsum(1)],1)
    end=torch.arange(1,x.shape[1]+1,device=x.device);start=(end-width).clamp_min(0);count=(end-start)[None,:,None]
    return (((q[:,end]-q[:,start])/count-((c[:,end]-c[:,start])/count).square()).clamp_min(0)).mean(-1).float()


def auc(feature,labels):
    order=feature.argsort(stable=True);values=feature[order];positive=labels[order].double();_,counts=values.unique_consecutive(return_counts=True)
    end=counts.cumsum(0);start=end-counts;rank=torch.repeat_interleave((start+end+1).double()/2,counts)
    n=positive.sum();neg=len(labels)-n
    if n==0 or neg==0:return None
    return float(((rank*positive).sum()-n*(n+1)/2)/(n*neg))


def token_type(encoder,token):
    raw=encoder.raw.id_to_token(token) or ''
    if raw.startswith('<') and raw.endswith('>'):return 'special'
    s=encoder.raw.decode([token],skip_special_tokens=False)
    if s.isspace():return 'whitespace'
    s=s.strip()
    if not s:return 'empty'
    if all('\u4e00'<=c<='\u9fff' for c in s):return 'cjk'
    if s.isdigit():return 'number'
    if all(c.isascii() and c.isalpha() for c in s):return 'latin'
    if all(unicodedata.category(c).startswith('P') for c in s):return 'punctuation'
    return 'mixed_or_other'


def summarize_tail(data,types):
    g=data['g34'];benefit=g>0;damage=g<0;result={}
    result['tokens']=len(g);result['mean_g34']=float(g.mean())
    result['probabilities']={name:float(test.double().mean()) for name,test in [('g>0',g>0),('g>0.01',g>.01),('g>0.1',g>.1),('g<-0.01',g<-.01),('g<-0.1',g<-.1),('g<-1',g<-1)]}
    quantiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]
    result['quantiles']={f'P{int(q*100)}':float(v) for q,v in zip(quantiles,torch.quantile(g.double(),torch.tensor(quantiles,dtype=torch.double)))}
    result['groups']={}
    for name,mask in [('benefit',benefit),('damage',damage)]:
        count=int(mask.sum());type_count={}
        for i in mask.nonzero().flatten().tolist():type_count[types[i]]=type_count.get(types[i],0)+1
        result['groups'][name]=dict(tokens=count,means={key:float(value[mask].double().mean()) for key,value in data.items() if key not in ('input_token_id','target_token_id')},input_token_types=type_count)
    result['observable_auc_higher_predicts_benefit']={key:auc(value,benefit) for key,value in data.items() if key not in ('g34','ce1','input_token_id','target_token_id')}
    return result


def main():
    if (OUT/'V2.1.json').exists():raise RuntimeError('Existing diagnostic run')
    budget=json.loads((OUT/'budget.json').read_text());start=time.time()
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def event(stage,**kw):
        row=dict(stage=stage,time=time.time(),pid=os.getpid(),training_tokens=0,**kw);atomic_json(OUT/'status.json',row);print(json.dumps(row),flush=True)
    def check(deadline):
        if time.time()>min(deadline,budget['overall_deadline']):raise TimeoutError('Stage budget exhausted')
    event('V2.0',action='verify_source')
    old_manifest=json.loads((OLD/'manifest.json').read_text())
    if digest(SOURCE)!=old_manifest['checkpoint_hash']:raise ValueError('A checkpoint changed')
    for file,expected in old_manifest['validation_hash'].items():
        if digest(Path(file))!=expected:raise ValueError('Validation changed')
    model,base=load_model();versions={name:p._version for name,p in model.named_parameters()};check(budget['v20_deadline'])
    encoder=TextEncoder(ROOT/'artifacts/four-model-v1/tokenizer.json');manifest_path=ROOT/'artifacts/four-model-v1/data_manifest.json'
    files=sorted((OLD/'validation-cache').glob('batch-*.pt'));data_rows=[];batch_groups=[];all_types=[];types_cache={};batches=[]
    for i,file in enumerate(files):
        check(budget['v20_deadline']);c=torch.load(file,map_location='cpu',weights_only=True);batches.append((c['x'],c['y']));batch_groups.append(c['split'])
        pos=torch.arange(c['x'].shape[1],device='cuda');features=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for k in (0,2):features.append(distribution_features(model.decoder(c['deltas'][...,k,:].cuda(),pos)).cpu())
        mask=c['mask'];n3=c['states'][...,2,:].float();delta3=c['deltas'][...,2,:].float()
        row=dict(g34=(c['ce'][...,2]-c['ce'][...,3])[mask],ce1=c['ce'][...,0][mask],entropy1=features[0][...,0][mask],confidence1=features[0][...,1][mask],margin1=features[0][...,2][mask],
            entropy3=features[1][...,0][mask],confidence3=features[1][...,1][mask],margin3=features[1][...,2][mask],state3_norm=n3.norm(dim=-1)[mask],delta3_norm=delta3.norm(dim=-1)[mask],
            local_state3_variance=causal_local_variance(n3)[mask],position=torch.arange(c['x'].shape[1]).expand_as(c['x'])[mask],input_token_id=c['x'][mask],target_token_id=c['y'][mask])
        types=[]
        for token in row['input_token_id'].tolist():
            if token not in types_cache:types_cache[token]=token_type(encoder,token)
            types.append(types_cache[token])
        data_rows.append(row);all_types.append(types)
        if (i+1)%8==0:event('V2.0',completed_batches=i+1,total_batches=len(files))
    analysis={}
    for group in ('fixed','extra','all'):
        indices=[i for i,g in enumerate(batch_groups) if group=='all' or g==group]
        data={k:torch.cat([data_rows[i][k] for i in indices]) for k in data_rows[0]};types=[v for i in indices for v in all_types[i]]
        analysis[group]=summarize_tail(data,types)
    with (OUT/'r4-tail-per-token.csv').open('w') as f:
        keys=['split','batch','input_token_type',*data_rows[0].keys()];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for i,row in enumerate(data_rows):
            for j in range(len(row['g34'])):writer.writerow(dict(split=batch_groups[i],batch=i,input_token_type=all_types[i][j],**{k:float(v[j]) for k,v in row.items()}))
    atomic_json(OUT/'V2.0.json',dict(complete=True,pass_gate=True,training_tokens=0,groups=analysis,r4_enabled=False,elapsed_seconds=time.time()-start,
        note='CE1 and g34 use true targets only for offline cohort analysis. Observable AUCs are exploratory held-out descriptions, never routing features or tuned thresholds. Local variance uses past <=8 positions only.'))
    event('V2.0',complete=True,r4_enabled=False)
    v21_start=time.time();v21_deadline=min(v21_start+budget['v21_max_seconds'],budget['overall_deadline'])
    atomic_json(OUT/'stage-deadlines.json',dict(v21_deadline=v21_deadline))
    fixed=[b for b,g in zip(batches,batch_groups) if g=='fixed'];extra=[b for b,g in zip(batches,batch_groups) if g=='extra']
    baseline={};scoreboard={}
    old_g1=json.loads((OLD/'G1.json').read_text())['groups']
    for k in (1,2,3):
        entry={g:score_batches(model,bs,force_depth=k) for g,bs in [('fixed',fixed),('extra',extra)]}
        for g in entry:
            if abs(entry[g]['ce']-old_g1[g]['ce'][k-1])>1e-5:raise RuntimeError('Real fixed path failed reproduction')
        scoreboard[f'Fixed R{k}']=entry
        if k==3:baseline=entry
    event('V2.1',action='fixed_paths_verified')
    cal,documents=calibration(manifest_path,encoder,budget['calibration_batches'])
    assert len(set(d['document_sha256'] for d in documents))==len(documents)
    assert all(1<=d['document_bucket']<=9 for d in documents)
    torch.save(dict(batches=cal,documents=documents),OUT/'calibration.pt')
    calibration_hash=digest(OUT/'calibration.pt')
    cal_features=dense_features(model,cal);torch.save(cal_features,OUT/'calibration-feature-quantiles-source.pt')
    atomic_json(OUT/'manifest.json',dict(checkpoint=str(SOURCE),checkpoint_hash=old_manifest['checkpoint_hash'],config_hash=old_manifest['config_hash'],validation_hash=old_manifest['validation_hash'],git_commit=old_manifest['git_commit'],
        source_hashes={n:digest(ROOT/n) for n in ['scripts/run_dynamic_v2_diagnostics.py','recal/evaluation/dynamic_depth_v2.py','recal/data/dynamic_v2.py']},
        calibration_hash=calibration_hash,calibration_documents=documents,document_split='sha256 first8 bytes mod1000: validation=0, calibration=1..9, V2 training=10..999',
        primary_path='D(delta), exactly reproduces A R1/R2/R3; all final policy CE uses real mixed execution.',max_depth=3,budget=budget))
    chosen={};search=[]
    for feature_index,kind in enumerate(('entropy','confidence','margin')):
        candidates=[]
        for q1 in budget['quantiles']:
            for q2 in budget['quantiles']:
                check(v21_deadline)
                thresholds=[float(torch.quantile(cal_features[:,k,feature_index].float(),q)) for k,q in enumerate((q1,q2))]
                result=score_batches(model,cal,kind,thresholds)
                record=dict(strategy=kind,quantiles=[q1,q2],thresholds=thresholds,calibration=result);candidates.append(record);search.append(record)
                with (OUT/'calibration-search.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if len(candidates)%7==0:event('V2.1',action='calibration_search',strategy=kind,completed_candidates=len(candidates),total_candidates=49)
        chosen[kind]=min(candidates,key=lambda r:(r['calibration']['ce'],r['calibration']['average_depth']))
    # Seal all thresholds before consulting either held-out validation set.
    atomic_json(OUT/'frozen-thresholds.json',dict(chosen=chosen,calibration_hash=calibration_hash,sealed_at=time.time(),selection='Minimum real-path calibration CE; average depth only breaks ties.'))
    heldout={}
    for kind,record in chosen.items():
        check(v21_deadline)
        heldout[kind]={g:score_batches(model,bs,kind,record['thresholds']) for g,bs in [('fixed',fixed),('extra',extra)]}
        scoreboard[kind]=heldout[kind];event('V2.1',action='heldout_evaluation',strategy=kind,results=heldout[kind])
    gate=heuristic_gate(heldout,baseline)
    for name,values in scoreboard.items():
        count=sum(r['tokens'] for r in values.values())
        values['all']=dict(ce=sum(r['ce']*r['tokens'] for r in values.values())/count,average_depth=sum(r['average_depth']*r['tokens'] for r in values.values())/count,
            depth_percent=[sum(r['depth_percent'][k]*r['tokens'] for r in values.values())/count for k in range(4)],tokens=count)
    unchanged=all(p._version==versions[n] and p.grad is None and not p.requires_grad for n,p in model.named_parameters())
    if not unchanged:raise RuntimeError('Frozen model changed')
    atomic_json(OUT/'V2.1.json',dict(complete=True,gate=gate,scoreboard=scoreboard,thresholds=chosen,calibration_hash=calibration_hash,elapsed_seconds=time.time()-v21_start,
        training_tokens=0,frozen_backbone_unchanged=unchanged,value_training_authorized=gate['pass_gate']))
    event('V2.1',complete=True,gate=gate,next_stage='V2.2' if gate['pass_gate'] else 'STOP')
    if not gate['pass_gate']:
        atomic_json(OUT/'completion.json',dict(complete=True,stopped_at='V2.1',reason=gate['reason'],training_tokens=0,finished_at=time.time(),within_budget=time.time()<=budget['overall_deadline']))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(error=repr(exc),time=time.time()));raise
