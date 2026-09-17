"""Build a source manifest and a training-only multilingual byte-BPE tokenizer."""
import argparse,json,random,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recal.data.stateful import ParquetDocuments
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

p=argparse.ArgumentParser(__doc__)
p.add_argument('--root',required=True)
p.add_argument('--output',required=True)
p.add_argument('--sample-chars',type=int,default=64_000_000)
a=p.parse_args()
root=Path(a.root).resolve(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
if (out/'tokenizer.json').exists():
 raise SystemExit('Tokenizer exists; use a new artifact directory to avoid changing token IDs')
weights={'zh-cci3':.52,'wikipedia-zh':.08,'fineweb-edu':.16,'wikipedia-en':.04,'finemath':.10}
code=sorted(f.name for f in root.glob('code-*') if f.is_dir())
for key in code: weights[key]=.10/len(code)
groups={}
for key,w in weights.items():
 files=sorted(str(f.relative_to(root)) for f in (root/key).glob('*.parquet'))
 if not files: raise ValueError(f'No files in {key}')
 groups[key]={'weight':w,'files':files}
manifest={'root':str(root),'groups':groups,'split':'sha256(text) first 8 bytes modulo 1000 == 0 is validation','seed':1337,
 'note':'Token-window mixture; CCI3 includes unclassified articles/news. SFT excluded. Code weight split equally by language.'}
(out/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
sources={k:ParquetDocuments([root/f for f in v['files']],'train',1337) for k,v in groups.items()}
rng=random.Random(1337); chars=docs=0; group_chars={k:0 for k in groups}
# Quotas by characters approximate a balanced tokenizer sample; training mixture is by token windows.
with (out/'tokenizer_sample.jsonl').open('w') as stream:
 for key,item in groups.items():
  quota=int(a.sample_chars*item['weight'])
  while group_chars[key]<quota:
   text=sources[key].next_text()[:32768]
   stream.write(json.dumps({'text':text,'group':key},ensure_ascii=False)+'\n')
   group_chars[key]+=len(text); chars+=len(text); docs+=1
print(json.dumps({'sample_documents':docs,'sample_chars':chars,'group_chars':group_chars}),flush=True)
t=Tokenizer(models.BPE(unk_token='<unk>'))
t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder=decoders.ByteLevel()
trainer=trainers.BpeTrainer(vocab_size=32000,special_tokens=['<pad>','<bos>','<eos>','<unk>'],
 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
def texts():
 with (out/'tokenizer_sample.jsonl').open() as stream:
  for line in stream: yield json.loads(line)['text']
t.train_from_iterator(texts(),trainer=trainer)
t.post_processor=processors.TemplateProcessing(single='<bos> $A <eos>',special_tokens=[('<bos>',t.token_to_id('<bos>')),('<eos>',t.token_to_id('<eos>'))])
t.save(str(out/'tokenizer.json'))
for sample in ['你好，世界！Hello world.','def f(x):\n    return x + 1\n','数学：∑ α² = 3.14159 🧠']:
 ids=t.encode(sample,add_special_tokens=False).ids
 assert t.decode(ids)==sample
summary={'sample_documents':docs,'sample_chars':chars,'group_chars':group_chars,'vocab_size':t.get_vocab_size(),
 'tokenizer_sha256':hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(),'roundtrip_passed':True}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
