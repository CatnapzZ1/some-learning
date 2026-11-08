"""检查自己保存的 BPE 模型与抽样文件；不执行 2.7 的实验或全量编码。"""

import argparse
import json
import pickle
import sys
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


SEPARATOR = "<|endoftext|>"


def check_inputs(args: argparse.Namespace) -> None:
    for path in args.corpus:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"语料不存在或为空：{path}")
        print(f"语料存在：{path} ({path.stat().st_size:,} bytes；未扫描内容)")

    # 与 run_tinystories.py 保存的本地模型格式一致。
    with args.model.open("rb") as source:
        model = pickle.load(source)
    if not isinstance(model, dict):
        raise ValueError("模型应为包含 vocab 和 merges 的字典")
    vocab = model["vocab"]
    merges = model["merges"]
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("vocab 应为非空字典")
    if not all(type(key) is int and key >= 0 and isinstance(value, bytes) and value for key, value in vocab.items()):
        raise ValueError("vocab 应将非负整数 ID 映射到非空 bytes")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("vocab 中出现重复的 token 字节串")
    if not isinstance(merges, list) or not all(
        isinstance(pair, (tuple, list)) and len(pair) == 2 and all(isinstance(part, bytes) for part in pair)
        for pair in merges
    ):
        raise ValueError("merges 应为由两个 bytes 组成的配对列表")
    if len(vocab) != args.expected_vocab_size:
        raise ValueError(f"词表数量不匹配：实际 {len(vocab)}，预期 {args.expected_vocab_size}")
    if SEPARATOR.encode("utf-8") not in vocab.values():
        raise ValueError(f"词表缺少特殊 token：{SEPARATOR}")
    print(f"模型：{args.model}；词表 {len(vocab):,} 项；merges {len(merges):,} 项")

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[SEPARATOR])
    examples = ["", "Hello, world!", "你好，世界！\n", f"first{SEPARATOR}second"]
    for index, text in enumerate(examples):
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise ValueError(f"基础往返检查失败：示例 {index}")
    print(f"基础 encode/decode 往返检查：{len(examples)} 项通过")

    if args.samples is not None:
        seen_ids = set()
        with args.samples.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"样本第 {line_number} 行应为 JSON 对象")
                doc_id, text = row["doc_id"], row["text"]
                if type(doc_id) is not int or doc_id < 0 or doc_id in seen_ids:
                    raise ValueError(f"样本第 {line_number} 行的 doc_id 无效或重复")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"样本第 {line_number} 行的 text 无效或为空")
                if tokenizer.decode(tokenizer.encode(text)) != text:
                    raise ValueError(f"样本 doc_id={doc_id} 的往返检查失败")
                seen_ids.add(doc_id)
        if len(seen_ids) != args.expected_docs:
            raise ValueError(f"样本数量不匹配：实际 {len(seen_ids)}，预期 {args.expected_docs}")
        print(f"样本：{args.samples}；{len(seen_ids)} 篇文档格式与往返检查通过")

    print("检查完成。这些检查不能替代 tokenizer 单元测试，也不验证模型的训练来源。")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="自己训练并保存的 pickle 模型")
    parser.add_argument("--expected-vocab-size", type=positive_int, required=True)
    parser.add_argument("--samples", type=Path, help="已有的 doc_id/text JSONL 抽样文件")
    parser.add_argument("--expected-docs", type=positive_int, default=10)
    parser.add_argument("--corpus", type=Path, action="append", default=[], help="检查语料文件存在且非空，可重复")
    args = parser.parse_args()
    try:
        check_inputs(args)
    except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.PickleError) as error:
        print(f"检查失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
