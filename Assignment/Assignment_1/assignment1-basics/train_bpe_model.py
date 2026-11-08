"""调用现有 train_bpe，记录阶段进度并保存训练产物，不实现 BPE 算法。"""

import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
from functools import wraps
from pathlib import Path
from time import perf_counter

from cs336_basics import bpe


ORIGINAL_PRETOKENIZE_WORKER = bpe.pretokenize_worker


def logged_pretokenize_worker(task):
    started = perf_counter()
    result = ORIGINAL_PRETOKENIZE_WORKER(task)
    _, start, end, _ = task
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"[worker {os.getpid()}] 完成语料字节区间 {start}:{end}；"
        f"{perf_counter() - started:.1f}s；{len(result):,} 个不同预分词项；peak RSS {peak_mib:.0f} MiB",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pretoken-chunks", type=int, default=128)
    parser.add_argument("--max-memory-gib", type=float, default=10)
    args = parser.parse_args()
    if args.vocab_size < 257 or args.workers < 1 or args.pretoken_chunks < 1 or args.max_memory_gib <= 0:
        parser.error("vocab-size 至少为 257；workers、pretoken-chunks 和 max-memory-gib 必须为正数")
    if not args.corpus.is_file():
        parser.error(f"找不到语料：{args.corpus}")
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() or temporary_output.exists():
        parser.error("输出文件或临时输出已存在，请指定新的输出路径")

    requested_limit = int(args.max_memory_gib * 1024**3)
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    limit = requested_limit if hard_limit == resource.RLIM_INFINITY else min(requested_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    timings = {}
    originals = {}

    def report(message):
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[{perf_counter() - started:.1f}s; peak RSS {peak_mib:.0f} MiB] {message}", flush=True)

    def timed(original, name):
        @wraps(original)
        def wrapper(*positional, **keywords):
            report(f"开始 {name}")
            stage_started = perf_counter()
            if name == "pretokenize_file_in_chunks":
                keywords["num_chunks"] = args.pretoken_chunks
            result = original(*positional, **keywords)
            timings[name] = perf_counter() - stage_started
            report(f"完成 {name}，耗时 {timings[name]:.1f}s")
            return result

        return wrapper

    for name in ("pretokenize_file_in_chunks", "byte_sequence_frequency", "initialize_pair_stats"):
        originals[name] = getattr(bpe, name)
        setattr(bpe, name, timed(originals[name], name))

    originals["pretokenize_worker"] = bpe.pretokenize_worker
    bpe.pretokenize_worker = logged_pretokenize_worker

    originals["new_token_merge"] = bpe.new_token_merge

    def report_merge(vocab, pair):
        result = originals["new_token_merge"](vocab, pair)
        if len(vocab) % 1000 == 0 or len(vocab) == args.vocab_size:
            report(f"词表进度 {len(vocab):,}/{args.vocab_size:,}")
        return result

    bpe.new_token_merge = report_merge
    metadata = {
        "corpus": str(args.corpus.resolve()),
        "corpus_bytes": args.corpus.stat().st_size,
        "corpus_mtime_ns": args.corpus.stat().st_mtime_ns,
        "vocab_size_requested": args.vocab_size,
        "workers": args.workers,
        "pretoken_chunks": args.pretoken_chunks,
        "address_space_limit_bytes": limit,
        "special_tokens": ["<|endoftext|>"],
        "bpe_source_sha256": hashlib.sha256(Path(bpe.__file__).read_bytes()).hexdigest(),
    }
    report(f"PID={os.getpid()}; 配置：{json.dumps(metadata, ensure_ascii=False)}")
    temporary_created = False
    try:
        vocab, merges = bpe.train_bpe(
            input_path=args.corpus,
            vocab_size=args.vocab_size,
            special_tokens=metadata["special_tokens"],
            num_workers=args.workers,
        )
        if len(vocab) != args.vocab_size:
            raise ValueError(f"未达到目标词表大小：{len(vocab)} / {args.vocab_size}")
        metadata["training_seconds"] = perf_counter() - started
        metadata["stage_seconds"] = timings
        metadata["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        with temporary_output.open("xb") as output:
            temporary_created = True
            pickle.dump({"vocab": vocab, "merges": merges, "metadata": metadata}, output)
        temporary_output.replace(args.output)
        report(f"保存完成：{args.output}；{len(vocab):,} 个词表项，{len(merges):,} 次合并")
    except MemoryError:
        report("达到内存分配限制，训练未完成；没有生成可用于实验的模型")
        return 1
    except (OSError, ValueError) as error:
        print(f"训练失败：{error}", file=sys.stderr)
        return 1
    finally:
        for name, original in originals.items():
            setattr(bpe, name, original)
        if temporary_created and temporary_output.exists():
            temporary_output.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
