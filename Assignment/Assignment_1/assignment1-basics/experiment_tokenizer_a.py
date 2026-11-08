import argparse
import json
import pickle
import random
from pathlib import Path
from collections.abc import Iterator

from cs336_basics.tokenizer import Tokenizer


SEPARATOR = "<|endoftext|>"


def iter_documents(path: Path) -> Iterator[str]:
    """按文档分隔符逐篇读取，不加载整个语料。"""

    buffer = []

    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.split(SEPARATOR)

            for part in parts[:-1]:
                buffer.append(part)
                doc = "".join(buffer)

                if doc.strip():
                    yield doc

                buffer = []

            buffer.append(parts[-1])

    # 文件末尾可能还有最后一篇文档
    doc = "".join(buffer)
    if doc.strip():
        yield doc


def sample_documents(
    path: Path,
    n: int = 10,
    seed: int = 42,
) -> list[tuple[int, str]]:
    """Reservoir Sampling：等概率抽取 n 篇文档。"""

    rng = random.Random(seed)
    reservoir = []
    total = 0

    for doc_id, doc in enumerate(iter_documents(path)):
        total += 1

        if len(reservoir) < n:
            reservoir.append((doc_id, doc))
        else:
            j = rng.randrange(total)
            if j < n:
                reservoir[j] = (doc_id, doc)

    if total < n:
        raise ValueError(f"Only {total} documents found; need {n}. Check corpus format and separator.")

    print(f"Total documents: {total}")

    return sorted(reservoir, key=lambda x: x[0])


def load_tokenizer(model_path: Path) -> Tokenizer:
    """加载自己训练并保存的 BPE 模型。"""

    with model_path.open("rb") as f:
        model = pickle.load(f)

    return Tokenizer(
        vocab=model["vocab"],
        merges=model["merges"],
        special_tokens=[SEPARATOR],
    )


def load_samples(path: Path, n: int) -> list[tuple[int, str]]:
    """读取已保存的样本，保持文档内容和顺序，以便对比不同 tokenizer。"""

    documents = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Sample line {line_number} must be a JSON object")
            doc_id = row.get("doc_id")
            text = row.get("text")
            if type(doc_id) is not int or doc_id < 0 or doc_id in seen_ids:
                raise ValueError(f"Sample line {line_number} has an invalid or duplicate doc_id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Sample line {line_number} must contain nonempty text")
            seen_ids.add(doc_id)
            documents.append((doc_id, text))

    if len(documents) != n:
        raise ValueError(f"Expected {n} saved documents, found {len(documents)}")
    print(f"Loaded saved documents: {len(documents)} from {path}")
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path)
    source.add_argument("--samples-in", type=Path, help="Reuse an existing doc_id/text JSONL sample")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-docs", type=int, default=10)

    args = parser.parse_args()

    if args.num_docs <= 0:
        parser.error("--num-docs must be positive")
    input_path = args.samples_in if args.samples_in is not None else args.corpus
    if input_path.resolve() == args.samples_out.resolve() or (
        input_path.exists() and args.samples_out.exists() and input_path.samefile(args.samples_out)
    ):
        parser.error("--samples-out must not overwrite the input file")

    if args.samples_in is not None:
        documents = load_samples(args.samples_in, args.num_docs)
    else:
        documents = sample_documents(
            args.corpus,
            n=args.num_docs,
            seed=args.seed,
        )

    tokenizer = load_tokenizer(args.model)

    args.samples_out.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    total_tokens = 0

    print(f"\nDataset: {args.dataset}")
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
    print(f"{'Doc ID':>8} {'Bytes':>12} {'Tokens':>12} {'Bytes/Token':>14}")

    with args.samples_out.open("w", encoding="utf-8") as f:
        for doc_id, text in documents:
            byte_count = len(text.encode("utf-8"))
            token_ids = tokenizer.encode(text)
            token_count = len(token_ids)

            if token_count == 0:
                raise ValueError(f"Document {doc_id} encoded to zero tokens")

            ratio = byte_count / token_count

            total_bytes += byte_count
            total_tokens += token_count

            f.write(
                json.dumps(
                    {"doc_id": doc_id, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(f"{doc_id:>8} {byte_count:>12} {token_count:>12} {ratio:>14.3f}")

    print("-" * 50)
    print(f"Total bytes: {total_bytes}")
    print(f"Total tokens: {total_tokens}")
    print(f"Overall bytes/token: {total_bytes / total_tokens:.4f}")
    print(f"Samples saved to: {args.samples_out}")


if __name__ == "__main__":
    main()
