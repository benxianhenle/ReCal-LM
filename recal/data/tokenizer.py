"""Tokenizer adapters with a byte-level fallback and HuggingFace support.

中文：分词器适配层，支持字节级 fallback 和 HuggingFace tokenizer。"""

from pathlib import Path


class ByteTokenizer:
    """Small UTF-8 byte tokenizer used when no external tokenizer is provided.

中文：未提供外部分词器时使用的小型 UTF-8 字节分词器。"""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text as byte IDs offset above the reserved special tokens.

中文：将文本编码为避开保留特殊 token 的字节 ID。"""

        ids = [b + 4 for b in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode byte IDs back into UTF-8 text with replacement on errors.

中文：将字节 ID 解码回 UTF-8 文本，遇到错误字符时使用替换符。"""

        raw = bytes(max(0, min(255, i - 4)) for i in ids if i >= 4)
        return raw.decode("utf-8", errors="replace")


class HFTokenizer:
    """Thin adapter around a HuggingFace tokenizers JSON file.

中文：对 HuggingFace tokenizers JSON 文件的轻量封装。"""

    def __init__(self, path: str | Path):
        """Load a tokenizer file and expose special-token IDs.

中文：加载分词器文件，并暴露特殊 token 的 ID。"""

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))
        self.pad_token_id = self._id("<pad>", 0)
        self.bos_token_id = self._id("<bos>", 1)
        self.eos_token_id = self._id("<eos>", 2)
        self.unk_token_id = self._id("<unk>", 3)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def _id(self, token: str, fallback: int) -> int:
        """Return a tokenizer ID or the project fallback if the token is absent.

中文：返回 token ID；若不存在则使用项目默认 fallback。"""

        value = self.tokenizer.token_to_id(token)
        return fallback if value is None else value

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text and optionally wrap it in BOS/EOS tokens.

中文：编码文本，并可选地包裹 BOS/EOS token。"""

        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs with the underlying HuggingFace tokenizer.

中文：使用底层 HuggingFace tokenizer 解码 token ID。"""

        return self.tokenizer.decode(ids)


def load_tokenizer(path: str | None):
    """Load a HuggingFace tokenizer when available, otherwise use bytes.

中文：可用时加载 HuggingFace tokenizer，否则使用字节分词器。"""

    if path:
        return HFTokenizer(path)
    return ByteTokenizer()
