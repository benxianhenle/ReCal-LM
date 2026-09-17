"""Latest, best, and one pre-transition snapshot: at most three retained sets."""
import copy
import json
import math
import os
import shutil
from pathlib import Path
import torch


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(payload, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'manifest.json'
        self.manifest = json.loads(self.path.read_text()) if self.path.exists() else {
            'version': 1, 'roles': {}, 'snapshots': {}}
        for name in self.manifest['snapshots']:
            if Path(name).name != name:
                raise ValueError('Checkpoint manifest contains an invalid filename')
            if not (self.root / name).is_file():
                raise ValueError(f'Missing retained checkpoint: {name}')

    def mark_manual_transition(self):
        """Pin the latest early checkpoint before a requested phase change, without copying weights."""
        latest = self.manifest['roles'].get('latest')
        if latest is None or self.manifest['snapshots'][latest]['stage'] != 'early':
            raise ValueError('Manual transition requires a latest early checkpoint')
        existing = self.manifest['roles'].get('transition_before')
        if existing is not None and existing != latest:
            raise ValueError('A different transition boundary is already retained')
        new = copy.deepcopy(self.manifest)
        new['roles']['transition_before'] = latest
        new['manual_transition'] = latest
        assert len(set(new['roles'].values())) <= 3
        atomic_json(self.path, new)
        self.manifest = new
        return self.root / latest

    def save(self, payload, validation_loss, boundary=None):
        if not math.isfinite(validation_loss):
            raise ValueError('Cannot rank non-finite validation loss')
        if boundary not in (None, 'before'):
            raise ValueError(boundary)
        old = self.manifest
        new = copy.deepcopy(old)
        name = f"step_{payload['step']:09d}_{payload['stage']}.pt"
        if name in old['snapshots']:
            raise ValueError('Refusing to overwrite a complete snapshot at the same step/stage')
        meta = dict(step=payload['step'], stage=payload['stage'], loss=float(validation_loss))
        new['snapshots'][name] = meta
        roles = new['roles']
        roles['latest'] = name
        if 'best' not in roles or validation_loss < new['snapshots'][roles['best']]['loss']:
            roles['best'] = name
        if boundary:
            roles[f'transition_{boundary}'] = name
        retained = set(roles.values())
        assert len(retained) <= 3
        new['snapshots'] = {k: v for k, v in new['snapshots'].items() if k in retained}
        # One temporary file is required for atomic replacement; old recovery files survive failure.
        temp = self.root / (name + '.tmp')
        stored = dict(payload, retention=new)
        try:
            with temp.open('wb') as stream:
                torch.save(stored, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.root / name)
            atomic_json(self.path, new)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        self.manifest = new
        for stale in set(old['snapshots']) - retained:
            (self.root / stale).unlink(missing_ok=True)
        # A crash after publishing the file but before the manifest can leave an orphan.
        # This directory belongs exclusively to this store; only our exact filename pattern is managed.
        for orphan in self.root.glob('step_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*.pt'):
            if orphan.name not in retained:
                orphan.unlink()
        return self.root / name
