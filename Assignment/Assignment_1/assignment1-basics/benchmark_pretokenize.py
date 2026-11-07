import argparse
from time import perf_counter

from cs336_basics.bpe import pretokenize_file_in_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    start = perf_counter()

    counts = pretokenize_file_in_chunks(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        special_tokens=["<|endoftext|>"],
        num_chunks=16,
        num_workers=args.workers,
    )

    elapsed = perf_counter() - start

    print(f"Workers: {args.workers}")
    print(f"Time: {elapsed:.2f} s")
    print(f"Unique pre-tokens: {len(counts)}")
    print(f"Total pre-tokens: {sum(counts.values())}")


if __name__ == "__main__":
    main()
