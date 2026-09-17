"""Disjoint document-hash buckets for V2 calibration and controller training."""
import hashlib,json,random
from pathlib import Path
import torch
from .stateful import ParquetDocuments,PackedStream


def bucket(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')%1000


def accepts_bucket(value,purpose):
    if purpose=='calibration':return 1<=value<=9
    if purpose=='train':return value>=10
    if purpose=='validation':return value==0
    raise ValueError(purpose)


class V2Documents(ParquetDocuments):
    def __init__(self,paths,purpose,seed):
        super().__init__(paths,'train',seed);self.purpose=purpose
    def next_text(self):
        for _ in range(100000):
            text=super().next_text()
            if accepts_bucket(bucket(text),self.purpose):return text
        raise RuntimeError('Document hash split search exceeded bound')


def sources(manifest,purpose,seed):
    data=json.loads(Path(manifest).read_text());root=Path(data['root'])
    return {name:V2Documents([root/f for f in group['files']],purpose,seed) for name,group in data['groups'].items()},[g['weight'] for g in data['groups'].values()]


def calibration(manifest,encoder,count=8,seq_len=512,seed=20260918):
    docs,weights=sources(manifest,'calibration',seed);rng=random.Random(seed);batches=[];records=[];seen=set()
    for _ in range(count):
        for attempt in range(10000):
            group=rng.choices(list(docs),weights,k=1)[0];text=docs[group].next_text();h=hashlib.sha256(text.encode()).hexdigest()
            if h in seen:continue
            ids=encoder.encode(text)
            if len(ids)<seq_len+1:continue
            offset=rng.randrange(len(ids)-seq_len);window=ids[offset:offset+seq_len+1];seen.add(h)
            batches.append((torch.tensor([window[:-1]]),torch.tensor([window[1:]])))
            records.append(dict(document_sha256=h,document_bucket=bucket(text),source=group,offset=offset,sequence_length=seq_len));break
        else:raise RuntimeError('Could not sample disjoint calibration document')
    return batches,records


def training_stream(manifest,encoder,seq_len=512,seed=20260919):
    docs,weights=sources(manifest,'train',seed)
    identity=hashlib.sha256(Path(manifest).read_bytes()+f'V2:bucket>=10:{seed}:{seq_len}'.encode()).hexdigest()
    return PackedStream(docs,weights,encoder,seq_len,seed,identity)
