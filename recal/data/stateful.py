"""Resumable document packing, with explicit cursors and a document-level held-out split."""
import hashlib
import json
import random
from pathlib import Path


def document_is_validation(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % 1000 == 0


class TextEncoder:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.raw = Tokenizer.from_file(str(path))
        self.bos = self.raw.token_to_id('<bos>')
        self.eos = self.raw.token_to_id('<eos>')
        if self.bos is None or self.eos is None:
            raise ValueError('Tokenizer requires <bos> and <eos>')
        self.vocab_size = self.raw.get_vocab_size()
        self.fingerprint = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def encode(self, text):
        return [self.bos] + self.raw.encode(text, add_special_tokens=False).ids + [self.eos]


class JsonlDocuments:
    def __init__(self, path):
        self.path = Path(path)
        self.cursor = dict(offset=0, epoch=0)
        self.handle = None
        self.fingerprint = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def next_text(self):
        if self.handle is None:
            self.handle = self.path.open('rb')
            self.handle.seek(self.cursor['offset'])
        initial_epoch = self.cursor['epoch']
        while True:
            if self.cursor['epoch'] > initial_epoch + 1:
                raise ValueError('JSONL has no nonempty text records')
            line = self.handle.readline()
            if not line:
                if self.cursor['offset'] == 0:
                    raise ValueError('Empty JSONL data')
                self.handle.seek(0)
                self.cursor.update(offset=0, epoch=self.cursor['epoch'] + 1)
                continue
            self.cursor['offset'] = self.handle.tell()
            row = json.loads(line)
            text = row.get('text') or row.get('content')
            if isinstance(text, str) and text.strip():
                return text

    def state_dict(self):
        return dict(self.cursor)

    def load_state_dict(self, state):
        if self.handle:
            self.handle.close()
        self.handle = None
        self.cursor = dict(state)


class ParquetDocuments:
    def __init__(self, paths, split, seed):
        self.paths = list(paths)
        random.Random(seed).shuffle(self.paths)
        self.split = split
        self.cursor = dict(file=0, batch=0, row=0, epoch=0)
        self.iterator = None
        self.rows = None

    def _open(self):
        import pyarrow.parquet as pq
        file = pq.ParquetFile(self.paths[self.cursor['file']])
        names = file.schema_arrow.names
        self.column = 'text' if 'text' in names else 'content'
        self.iterator = file.iter_batches(batch_size=128, columns=[self.column])
        for _ in range(self.cursor['batch']):
            next(self.iterator)

    def next_text(self):
        initial_epoch = self.cursor['epoch']
        while True:
            if self.cursor['epoch'] > initial_epoch + 1:
                raise ValueError(f'No eligible records in Parquet split {self.split}')
            if self.iterator is None:
                self._open()
            if self.rows is None:
                try:
                    self.rows = next(self.iterator).column(0).to_pylist()
                except StopIteration:
                    self.cursor['file'] += 1
                    if self.cursor['file'] == len(self.paths):
                        self.cursor['file'] = 0
                        self.cursor['epoch'] += 1
                    self.cursor.update(batch=0, row=0)
                    self.iterator = None
                    continue
            while self.cursor['row'] < len(self.rows):
                text = self.rows[self.cursor['row']]
                self.cursor['row'] += 1
                if not isinstance(text, str) or not text.strip():
                    continue
                if document_is_validation(text) == (self.split == 'val'):
                    return text
            self.rows = None
            self.cursor['batch'] += 1
            self.cursor['row'] = 0

    def state_dict(self):
        return dict(self.cursor)

    def load_state_dict(self, state):
        self.cursor = dict(state)
        self.iterator = self.rows = None


class PackedStream:
    def __init__(self, sources, weights, encoder, seq_len, seed, identity):
        self.sources = sources
        self.weights = weights
        self.encoder = encoder
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.buffers = {key: [] for key in sources}
        self.identity = identity

    def next_batch(self, batch_size):
        import torch
        xs, ys = [], []
        for _ in range(batch_size):
            key = self.rng.choices(list(self.sources), self.weights, k=1)[0]
            buf = self.buffers[key]
            while len(buf) < self.seq_len + 1:
                buf.extend(self.encoder.encode(self.sources[key].next_text()))
            ids = buf[:self.seq_len + 1]
            # Preserve the endpoint so there is no artificial token gap between windows.
            del buf[:self.seq_len]
            xs.append(ids[:-1]); ys.append(ids[1:])
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    def state_dict(self):
        return dict(identity=self.identity, tokenizer=self.encoder.fingerprint, seq_len=self.seq_len,
                    cursors={k: v.state_dict() for k, v in self.sources.items()},
                    buffers={k: list(v) for k, v in self.buffers.items()}, rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state['identity'] != self.identity or state['tokenizer'] != self.encoder.fingerprint or state['seq_len'] != self.seq_len:
            raise ValueError('Data/tokenizer/sequence length changed since checkpoint')
        for k, v in state['cursors'].items():
            self.sources[k].load_state_dict(v)
        self.buffers = {k: list(v) for k, v in state['buffers'].items()}
        self.rng.setstate(state['rng'])


def make_stream(path, encoder, seq_len, seed, split='train'):
    path = Path(path)
    if path.suffix == '.jsonl':
        source = JsonlDocuments(path)
        return PackedStream({'text': source}, [1.0], encoder, seq_len, seed, source.fingerprint)
    manifest = json.loads(path.read_text())
    root = Path(manifest['root'])
    sources, weights, signatures = {}, [], []
    for group, item in manifest['groups'].items():
        paths = [root / name for name in item['files']]
        for f in paths:
            stat = f.stat()
            signatures.append((str(f.relative_to(root)), stat.st_size, stat.st_mtime_ns))
        sources[group] = ParquetDocuments(paths, split, seed)
        weights.append(item['weight'])
    identity = hashlib.sha256(json.dumps([signatures, weights, split, seed]).encode()).hexdigest()
    return PackedStream(sources, weights, encoder, seq_len, seed, identity)
