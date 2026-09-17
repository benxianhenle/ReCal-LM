"""G0 -> G1 -> G2, no trainable parameters or optimizer."""
import csv,fcntl,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from recal.model.text_state_model import TextStateLM
from recal.model.layers import RotaryEmbedding
from recal.training.retention import atomic_json
from recal.evaluation.dynamic_depth import sweep,oracle_summary,diagnostic_gates
OUT=ROOT/'reports/dynamic-depth-20260917'
SOURCE=Path('/cloud/cloud-ssd1/recal-experiments/depth-diagnostic-33291-20260916/A_step_000034243.pt')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(16*1024*1024):h.update(chunk)
    return h.hexdigest()


def load_model():
    b=torch.load(SOURCE,map_location='cpu',weights_only=True,mmap=True)
    if b['step']!=34243 or b['arm']!='A':raise ValueError('Expected final A checkpoint')
    with torch.device('meta'):model=TextStateLM(b['config'])
    model.load_state_dict(b['model'],assign=True)
    for module in model.modules():
        if isinstance(module,RotaryEmbedding):
            fresh=RotaryEmbedding(b['config']['hidden_size']//b['config']['num_heads'],b['config']['context_length'])
            module.cos_cached=fresh.cos_cached;module.sin_cached=fresh.sin_cached
    model.requires_grad_(False).to('cuda').eval()
    metadata={k:v for k,v in b.items() if k not in ('model','optimizer')}
    del b;gc.collect()
    return model,metadata


def main():
    if (OUT/'G2.json').exists():raise RuntimeError('Diagnostic already completed')
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((OUT/'budget.json').read_text());start=time.time()
    def check():
        if time.time()>budget['diagnostic_deadline']:raise TimeoutError('G0-G2 3h budget exhausted')
    def event(stage,**kw):
        r=dict(stage=stage,pid=os.getpid(),time=time.time(),training_tokens=0,**kw)
        atomic_json(OUT/'status.json',r);print(json.dumps(r),flush=True)
    event('G0',action='hash_checkpoint')
    source_hash=digest(SOURCE);check()
    fixed_file=ROOT/'runs/four-model-r3b-10b/validation.pt';extra_file=ROOT/'reports/attention-aux/validation-extra.pt'
    fixed=torch.load(fixed_file,weights_only=True)['batches'];extra=torch.load(extra_file,weights_only=True)
    batches=fixed+extra
    model,base=load_model();config_hash=hashlib.sha256(json.dumps(base['config'],sort_keys=True).encode()).hexdigest()
    hashes={str(p):digest(p) for p in (fixed_file,extra_file)}
    if hashes[str(fixed_file)]!=base['validation_hash'] or hashes[str(extra_file)]!=base['extra_validation_hash']:raise ValueError('Validation changed')
    mask_hash=hashlib.sha256()
    for x,y in batches:
        for t in (x,y,y!=-100):mask_hash.update(t.numpy().tobytes())
    manifest=dict(checkpoint=str(SOURCE),checkpoint_hash=source_hash,config_hash=config_hash,validation_hash=hashes,
        ordered_inputs_targets_mask_hash=mask_hash.hexdigest(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={n:digest(ROOT/n) for n in ['recal/evaluation/dynamic_depth.py','scripts/run_dynamic_diagnostics.py','recal/model/text_state_model.py']},
        torch=torch.__version__,fixed_batches=len(fixed),extra_batches=len(extra),batch_size=1,seq_len=512,dropout=base['config']['dropout'],
        primary_path='D(delta_k), matches existing training and baseline',secondary_path='D(n_k), requested absolute-state diagnostic; decoder not trained on this path',
        controller_training_policy='Independent train split only. Fixed/Extra caches are evaluation-only.',budget=budget)
    atomic_json(OUT/'manifest.json',manifest);event('G0',action='hashes_verified',manifest=manifest)
    if any(p.requires_grad for p in model.parameters()) or model.training:raise RuntimeError('Main model not frozen/eval')
    finite=torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
    if not finite:raise FloatingPointError('Nonfinite source parameters')
    old=json.loads((ROOT/'reports/depth-diagnostic-33291-20260916/A/validation-00952.json').read_text())['groups']
    baseline=[]
    for i,(x,y) in enumerate(batches):
        check()
        with torch.autocast('cuda',dtype=torch.bfloat16):r=sweep(model,x.cuda(),y.cuda(),depth=3)
        baseline.append(r['ce'].cpu());del r
    means={g:torch.cat(baseline[lo:hi]).mean((0,1)).tolist() for g,lo,hi in [('fixed',0,len(fixed)),('extra',len(fixed),len(batches))]}
    for g,m in means.items():
        if any(abs(m[k]-old[g][f'ce_depth_{k+1}'])>1e-5 for k in range(3)):raise RuntimeError('A baseline failed reproduction')
    atomic_json(OUT/'G0.json',dict(pass_gate=True,ce=means,elapsed_seconds=time.time()-start,manifest=manifest));event('G0',pass_gate=True,ce=means)
    cache_dir=OUT/'validation-cache';cache_dir.mkdir(exist_ok=True);ce_rows=[];abs_rows=[];dynamics=[];ablation_rows=[];g1start=time.time()
    with (OUT/'per-token-ce.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['split','batch','position','input_token_id','target_token_id','valid']+[f'ce_d{k}' for k in range(1,9)]+['best_depth'])
        for i,(x,y) in enumerate(batches):
            check()
            if time.time()-g1start>budget['g1_max_seconds']:raise TimeoutError('G1 exceeded 90min; completed batches retained')
            group='fixed' if i<len(fixed) else 'extra'
            with torch.autocast('cuda',dtype=torch.bfloat16):r=sweep(model,x.cuda(),y.cuda(),absolute=True,ablate=True)
            cache={k:r[k].cpu() for k in ('ce','absolute_ce','states','deltas')};cache.update(x=x,y=y,mask=y!=-100,split=group,batch=i)
            if not torch.isfinite(cache['ce'][cache['mask']]).all():raise FloatingPointError('Nonfinite depth CE')
            torch.save(cache,cache_dir/f'batch-{i:03d}.pt');ce_rows.append(cache['ce']);abs_rows.append(cache['absolute_ce']);dynamics.append(r['dynamics'])
            abl={str(k):{name:float(values[y.cuda()!=-100].mean()) for name,values in v.items()} for k,v in r['ablations'].items()};ablation_rows.append(abl)
            for j in range(x.shape[1]):
                losses=cache['ce'][0,j].tolist();writer.writerow([group,i,j,int(x[0,j]),int(y[0,j]),bool(y[0,j]!=-100),*losses,min(range(8),key=losses.__getitem__)+1])
            f.flush();del r,cache;event('G1',completed_batches=i+1,total_batches=len(batches))
    ce=torch.cat(ce_rows);absolute=torch.cat(abs_rows);mask=torch.cat([y!=-100 for x,y in batches]);groups={};abs_groups={}
    for g,lo,hi in [('fixed',0,len(fixed)),('extra',len(fixed),len(batches)),('all',0,len(batches))]:
        groups[g]=oracle_summary(ce[lo:hi],mask[lo:hi]);abs_groups[g]=oracle_summary(absolute[lo:hi],mask[lo:hi])
        groups[g]['dynamics']=[{key:sum(dynamics[i][k][key] for i in range(lo,hi))/(hi-lo) for key in dynamics[lo][k]} for k in range(8)]
        groups[g]['ablations']={k:{key:sum(ablation_rows[i][k][key] for i in range(lo,hi))/(hi-lo) for key in ablation_rows[lo][k]} for k in ('2','3','4')}
    gates=diagnostic_gates(groups);extension=None
    if gates['boundary_extension_required'] and gates['g1_pass']:
        extension_indices=list(range(min(2,len(fixed))))+list(range(len(fixed),min(len(fixed)+6,len(batches))))
        extended=[];masks=[]
        for i in extension_indices:
            check();x,y=batches[i]
            with torch.autocast('cuda',dtype=torch.bfloat16):r=sweep(model,x.cuda(),y.cuda(),depth=12)
            extended.append(r['ce'].cpu());masks.append(y!=-100);del r
        extension=dict(batch_indices=extension_indices,summary=oracle_summary(torch.cat(extended),torch.cat(masks)))
        atomic_json(OUT/'depth12-subset.json',extension)
        if extension['summary']['best_depth_percent'][-1]>=20:
            gates.update(g1_pass=False,g2_pass=False,stop_reason='Depth optimum still boundary-censored at 12')
    atomic_json(OUT/'G1.json',dict(groups=groups,absolute_state_groups=abs_groups,gates=gates,depth12_extension=extension,elapsed_seconds=time.time()-g1start))
    with (OUT/'depth-sweep.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['depth','fixed_ce_delta','extra_ce_delta','fixed_ce_absolute','extra_ce_absolute'])
        for k in range(8):w.writerow([k+1,groups['fixed']['ce'][k],groups['extra']['ce'][k],abs_groups['fixed']['ce'][k],abs_groups['extra']['ce'][k]])
    # G2 reuses per-token results and interventions from the same frozen sweep.
    atomic_json(OUT/'G2.json',dict(groups=groups,gates=gates,training_tokens=0,elapsed_seconds=time.time()-start,
        oracle_note='Uses true next-token labels, diagnostic bound for dense per-depth logits only; not a deployable controller or a bound for changed mixed-depth contexts.',
        ablation_note='Token shuffling may mix future positions; offline intervention only.',
        g3_authorized_by_gates=gates['g1_pass'] and gates['g2_pass']))
    event('G2',complete=True,gates=gates,next_stage='G3' if gates['g1_pass'] and gates['g2_pass'] else 'STOP')
    if not gates['g1_pass'] or not gates['g2_pass']:
        atomic_json(OUT/'completion.json',dict(complete=True,stopped_at='G2',reason=gates,training_tokens=0,finished_at=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',dict(error=repr(exc),time=time.time()));raise
