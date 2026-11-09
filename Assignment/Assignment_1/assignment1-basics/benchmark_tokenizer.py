"""Measure encoding throughput on saved documents with an existing tokenizer."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, UTC
from pathlib import Path
from time import perf_counter

from experiment_tokenizer_a import load_samples, load_tokenizer


PILE_BYTES = 825_000_000_000


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cpu_environment() -> dict:
    model_name = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "cpu_model": model_name,
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version,
        "python_executable": sys.executable,
    }


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument(
        "--min-seconds", type=positive_seconds, default=1.0, help="minimum duration of each of 3 rounds"
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite {args.output}")
    if args.num_docs <= 0:
        parser.error("--num-docs must be positive")

    tokenizer = load_tokenizer(args.model)
    documents = load_samples(args.samples, args.num_docs)
    texts = [text for _, text in documents]
    sample_bytes = sum(len(text.encode("utf-8")) for text in texts)
    root = Path(__file__).resolve().parent
    sources = [Path(__file__).resolve(), root / "experiment_tokenizer_a.py", root / "cs336_basics/tokenizer.py"]
    metadata = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "model": {"path": str(args.model), "sha256": sha256(args.model), "vocabulary_size": len(tokenizer.vocab)},
        "samples": {
            "path": str(args.samples),
            "sha256": sha256(args.samples),
            "document_ids": [doc_id for doc_id, _ in documents],
            "document_count": len(documents),
            "utf8_bytes_per_pass": sample_bytes,
        },
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sources},
        "environment": cpu_environment(),
        "measurement": {
            "timer": "time.perf_counter",
            "warmup_passes": 1,
            "round_count": 3,
            "minimum_seconds_per_round": args.min_seconds,
            "scope": "Tokenizer.encode calls, sample iteration, and returned-list length counting",
            "excluded": ["model loading", "JSON reading", "UTF-8 byte counting", "hashing", "output writing"],
        },
    }

    expected_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    if expected_tokens == 0:
        raise ValueError("The sample encoded to zero tokens")
    metadata["samples"]["tokens_per_pass"] = expected_tokens

    rounds = []
    for round_number in range(1, 4):
        repetitions = 0
        total_tokens = 0
        elapsed = 0.0
        start = perf_counter()
        while elapsed < args.min_seconds:
            for text in texts:
                total_tokens += len(tokenizer.encode(text))
            repetitions += 1
            elapsed = perf_counter() - start
        if total_tokens != expected_tokens * repetitions:
            raise ValueError("Token count changed between warmup and measurement")
        total_bytes = sample_bytes * repetitions
        rounds.append(
            {
                "round": round_number,
                "sample_passes": repetitions,
                "utf8_bytes": total_bytes,
                "tokens": total_tokens,
                "seconds": elapsed,
                "bytes_per_second": total_bytes / elapsed,
            }
        )
        print(f"Round {round_number}: {repetitions} passes, {elapsed:.3f} s, {total_bytes / elapsed:,.0f} bytes/s")

    throughput = statistics.median(row["bytes_per_second"] for row in rounds)
    pile_seconds = PILE_BYTES / throughput
    result = {
        **metadata,
        "rounds": rounds,
        "median_bytes_per_second": throughput,
        "pile_estimate": {
            "input_bytes": PILE_BYTES,
            "gb_definition": "1 GB = 1,000,000,000 bytes",
            "seconds": pile_seconds,
            "hours": pile_seconds / 3600,
            "days": pile_seconds / 86400,
            "basis": "825e9 bytes divided by the median measured bytes/second",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"Median throughput: {throughput:,.0f} bytes/s")
    print(f"825 GB estimate: {pile_seconds:,.1f} s = {pile_seconds / 3600:.2f} h = {pile_seconds / 86400:.2f} days")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
