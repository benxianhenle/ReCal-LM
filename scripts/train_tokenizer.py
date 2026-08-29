"""Train a byte-level BPE tokenizer for ReCal-LM experiments.

中文：为 ReCal-LM 实验训练字节级 BPE tokenizer。"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main():
    """Train a tokenizer from UTF-8 text or JSONL text fields and save JSON.

中文：从 UTF-8 文本或 JSONL 文本字段训练 tokenizer 并保存 JSON。"""

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for ReCal-LM.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more UTF-8 text files.")
    parser.add_argument("--output", required=True, help="Output tokenizer JSON.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--jsonl-text-field", default="text")
    args = parser.parse_args()

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    jsonl_inputs = [path for path in args.input if Path(path).suffix.lower() == ".jsonl"]
    if jsonl_inputs:
        def iterator():
            """Yield raw text from mixed JSONL and plain-text input files.

中文：从混合的 JSONL 和纯文本输入文件中产出原始文本。"""

            for input_path in args.input:
                path = Path(input_path)
                if path.suffix.lower() == ".jsonl":
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                yield str(json.loads(line).get(args.jsonl_text_field, ""))
                else:
                    yield path.read_text(encoding="utf-8")

        tokenizer.train_from_iterator(iterator(), trainer)
    else:
        tokenizer.train(args.input, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"saved={out} vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
