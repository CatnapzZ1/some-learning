from pathlib import Path
from time import perf_counter
import pickle

from cs336_basics.bpe import train_bpe


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_path = Path("artifacts/tinystories_bpe_parallel.pkl")

    special_tokens = ["<|endoftext|>"]

    # 1. 训练并计时
    start = perf_counter()

    vocab, merges = train_bpe(input_path=input_path, vocab_size=10000, special_tokens=special_tokens, num_workers=4)

    elapsed = perf_counter() - start

    # 2. 保存训练结果
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(
            {"vocab": vocab, "merges": merges},
            f,
        )

    # 3. 找出最长的普通 token
    special_bytes = {s.encode("utf-8") for s in special_tokens}

    normal_tokens = [token for token in vocab.values() if token not in special_bytes]

    longest = max(normal_tokens, key=len)

    # 4. 输出实验结果
    print(f"Training time: {elapsed:.2f} s")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merge count: {len(merges)}")
    print(f"Longest token: {longest!r}")
    print(f"Longest token length: {len(longest)} bytes")
    print(f"Decoded: {longest.decode('utf-8', errors='replace')}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
