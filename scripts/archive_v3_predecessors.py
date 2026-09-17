"""Explicit old-model cleanup only after V3 has completed an optimizer update."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import torch
from recal.training.retention import atomic_json

ROOT=Path('/workspace/ReCal-LM')
OUT=ROOT/'reports/v3-e1-warmstart-20260917'
ARCH=ROOT/'reports/archive-pre-v3-20260917'
A=Path('/cloud/cloud-ssd1/recal-experiments/depth-diagnostic-33291-20260916/A_step_000034243.pt')
LAST=ROOT/'runs/four-model-r3b-10b/checkpoints/step_000033291_late.pt'


def candidates():
    result=[ROOT/'runs/four-model-r3b-10b/checkpoints/step_000012810_late.pt',ROOT/'runs/four-model-r3b-10b/checkpoints/step_000030500_late.pt']
    result += list(Path('/cloud/cloud-ssd1/recal-experiments/depth-diagnostic-33291-20260916-continued').glob('[BC]_step_*.pt'))
    result += [Path('/cloud/cloud-ssd1/recal-experiments/recal-r-v3-20260917/E1.pt')]
    result += list((ROOT/'reports/dynamic-depth-20260917').glob('controller-*.pt'))
    for folder in ('runs/manual-switch-test/checkpoints','runs/four-model-transition-smoke/checkpoints','smoke/run','smoke/resume'):
        result += list((ROOT/folder).glob('*.pt'))
    return sorted(set(p.resolve() for p in result if p.exists()))


def main():
    if (ARCH/'completion.json').exists():
        raise RuntimeError('Cleanup already completed')
    rows=(OUT/'steps.jsonl').read_text().splitlines()
    if not rows or json.loads(rows[-1])['phase_step']<1:
        raise RuntimeError('V3 has not completed an update')
    launch=json.loads((OUT/'launcher.json').read_text())
    os.kill(launch['pid'],0)
    command=Path(f"/proc/{launch['pid']}/cmdline").read_bytes()
    if b'run_v3_e1_warmstart.py' not in command:
        raise RuntimeError('Wrong active process')
    lock=(ROOT/'runs/four-model-r3b-10b/.train.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise RuntimeError('Training lock not held')
    paths=candidates()
    assert A.exists() and LAST.exists() and A.resolve() not in paths and LAST.resolve() not in paths
    ARCH.mkdir(exist_ok=True)
    before={str(p):shutil.disk_usage(p).free for p in (ROOT,A.parent)}
    records=[]
    for i,path in enumerate(paths):
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if not isinstance(payload,dict) or not any(k in payload for k in ('model','controller','state_dict','head')):
            raise RuntimeError(f'Not an identified model checkpoint: {path}; keys={list(payload)[:20]}')
        metadata={k:v for k,v in payload.items() if k not in ('model','controller','state_dict','head','optimizer','teacher','ema','ema_teacher')}
        record=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,keys=list(payload),metadata_file=f'metadata-{i:02d}.pt')
        torch.save(metadata,ARCH/record['metadata_file'])
        del metadata,payload
        records.append(record)
    atomic_json(ARCH/'deletion-manifest.json',dict(created_at=time.time(),preserved=[str(A),str(LAST)],candidates=records,bytes=sum(r['bytes'] for r in records),policy='Model weights only; preserve all datasets, validation tensors, caches, logs, metrics, code and metadata'))
    shutil.copyfile(ROOT/'runs/four-model-r3b-10b/checkpoints/manifest.json',ARCH/'original-main-checkpoints-manifest.json')
    # Freeze the complete report/source record before deleting model blobs.
    with tarfile.open(ARCH/'experiments-and-source.tar.gz','w:gz') as tar:
        for folder in ('reports','scripts','recal','configs','plans','tests','artifacts/four-model-v1'):
            for path in sorted((ROOT/folder).rglob('*')):
                if not path.is_file() or ARCH in path.parents or OUT in path.parents:
                    continue
                if path.suffix in ('.pt','.pyc') or '__pycache__' in path.parts:
                    continue
                tar.add(path,arcname=str(path.relative_to(ROOT)))
    with tarfile.open(ARCH/'experiments-and-source.tar.gz') as tar:
        if 'reports/recal-r-v3-20260917/E1/result.json' not in tar.getnames():
            raise RuntimeError('Archive verification failed')
    A.chmod(0o444)
    main_manifest=json.loads((ROOT/'runs/four-model-r3b-10b/checkpoints/manifest.json').read_text())
    main_manifest['roles']={'latest':LAST.name}
    main_manifest['snapshots']={LAST.name:main_manifest['snapshots'][LAST.name]}
    main_manifest.pop('text_state_baseline',None)
    main_manifest['retired_training_strategy']=True
    main_manifest['retention_note']='User requested old main final only; A immutable baseline and V3 successor tracked separately.'
    atomic_json(ROOT/'runs/four-model-r3b-10b/checkpoints/manifest.json',main_manifest)
    deleted=[]
    for record in records:
        path=Path(record['path'])
        if (path.stat().st_size,path.stat().st_mtime_ns)!=(record['bytes'],record['mtime_ns']):
            raise RuntimeError(f'File changed during archive: {path}')
        path.unlink()
        deleted.append(record['path'])
        atomic_json(ARCH/'progress.json',dict(deleted=deleted))
    # Retire test-store manifests without stale references to removed snapshots.
    for folder in ('runs/manual-switch-test/checkpoints','runs/four-model-transition-smoke/checkpoints'):
        manifest=ROOT/folder/'manifest.json'
        if manifest.exists():
            atomic_json(manifest,dict(version=1,roles={},snapshots={},retired=True,archive=str(ARCH)))
    after={str(p):shutil.disk_usage(p).free for p in (ROOT,A.parent)}
    atomic_json(ARCH/'completion.json',dict(deleted_files=len(deleted),deleted_bytes=sum(r['bytes'] for r in records),free_before=before,free_after=after,preserved=[str(A),str(LAST)],finished_at=time.time()))
    (ARCH/'README.md').write_text('# 历史实验归档与模型清理\n\nV3-E1 首次优化更新成功后执行。保留旧主线 step 33291、只读 A-final，以及正在运行的新分支。\n\n移除旧 best、旧阶段基线、B/C、Controller、上一轮失败 E1 和 smoke/test 模型；所有原始数据、验证数据、分析缓存、日志、指标保留。模型元数据（含数据游标和 RNG，若原 checkpoint 存在）单独保存。\n\n历史实验和实现快照：experiments-and-source.tar.gz；精确删除清单：deletion-manifest.json；执行结果：completion.json。\n\n旧 V3-E1：CE 改善但门槛失败；新分支重置 scheduler、降低 LR 后从 A 重试。旧 10B 主线不再恢复。\n')
    print((ARCH/'completion.json').read_text(),flush=True)

if __name__=='__main__':
    main()
