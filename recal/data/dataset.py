"""Dataset implementations for tiny demos, JSONL streams, and speed tests.

中文：用于小样例、JSONL 流式数据和速度测试的数据集实现。"""

from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, IterableDataset

from .packing import pack_token_ids


DEFAULT_TEXT = """
ReCal-LM is a recurrent calibration language model. It alternates full
Transformer calibration with cheap recurrent hidden-state updates. This tiny
text exists only to verify forward, backward, checkpoint, resume, and loss.
"""


class PackedTextDataset(Dataset):
    """In-memory next-token dataset built from a text file or fallback text.

中文：从文本文件或内置备用文本构建的内存式下一词预测数据集。"""

    def __init__(self, tokenizer, seq_len: int, text_path: str | None = None, repeat: int = 128):
        """Tokenize text and split it into fixed-length input/label pairs.

中文：将文本分词，并切分为固定长度的输入和标签对。"""

        if text_path:
            path = Path(text_path)
            if path.suffix.lower() == ".jsonl":
                parts = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    parts.append(str(payload.get("text", "")))
                text = "\n".join(parts)
            else:
                text = path.read_text(encoding="utf-8")
        else:
            text = DEFAULT_TEXT * repeat
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        examples = pack_token_ids(token_ids, seq_len)
        if not examples:
            raise ValueError(f"Not enough text to build one sequence of length {seq_len}")
        self.examples = examples

    def __len__(self) -> int:
        """Return the number of packed examples available for indexing.

中文：返回可索引的已打包样本数量。"""

        return len(self.examples)

    def __getitem__(self, index: int):
        """Return one packed sequence and its one-token-shifted labels.

中文：返回一个已打包序列及其右移一位的标签。"""

        x, y = self.examples[index % len(self.examples)]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class StreamingJsonlDataset(IterableDataset):
    """Iterable dataset that streams JSONL text rows into packed token chunks.

中文：将 JSONL 文本行流式转换为打包 token 块的可迭代数据集。"""

    def __init__(self, tokenizer, seq_len: int, jsonl_path: str, repeat: bool = True):
        """Store tokenizer, sequence length, source path, and repeat policy.

中文：保存分词器、序列长度、源文件路径和重复读取策略。"""

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.jsonl_path = jsonl_path
        self.repeat = repeat

    def _iter_once(self):
        """Read the JSONL source once and yield fixed-size training examples.

中文：单次读取 JSONL 源文件并产出固定大小的训练样本。"""

        buffer: list[int] = []
        with Path(self.jsonl_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", ""))
                if not text:
                    continue
                buffer.extend(self.tokenizer.encode(text, add_special_tokens=True))
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    del buffer[: self.seq_len + 1]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

    def __iter__(self):
        """Yield rows forever when repeat is enabled, otherwise make one pass.

中文：启用 repeat 时持续产出样本，否则只遍历一遍。"""

        while True:
            yielded = False
            for item in self._iter_once():
                yielded = True
                yield item
            if not self.repeat:
                break
            if not yielded:
                raise ValueError(f"No trainable sequences found in {self.jsonl_path}")


class RandomTokenDataset(Dataset):
    """Synthetic random-token dataset for smoke tests and throughput checks.

中文：用于冒烟测试和吞吐量检查的随机 token 合成数据集。"""

    def __init__(self, vocab_size: int, seq_len: int, size: int = 1024):
        """Configure random token range, sequence length, and virtual size.

中文：配置随机 token 范围、序列长度和虚拟数据集大小。"""

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        """Return the configured virtual dataset size.

中文：返回配置的虚拟数据集大小。"""

        return self.size

    def __getitem__(self, index: int):
        """Generate one random input sequence and shifted labels.

中文：生成一个随机输入序列和对应的移位标签。"""

        ids = torch.randint(4, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return ids[:-1], ids[1:]
