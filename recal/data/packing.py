"""Token packing helpers for next-token language modeling.

中文：下一词语言建模使用的 token 打包辅助函数。"""


def pack_token_ids(token_stream: list[int], seq_len: int) -> list[tuple[list[int], list[int]]]:
    """Split token IDs into non-overlapping input/label pairs of seq_len.

中文：将 token ID 切分为不重叠的 seq_len 长度输入/标签对。"""

    examples = []
    stride = seq_len + 1
    for start in range(0, max(0, len(token_stream) - stride + 1), stride):
        chunk = token_stream[start : start + stride]
        if len(chunk) == stride:
            examples.append((chunk[:-1], chunk[1:]))
    return examples
