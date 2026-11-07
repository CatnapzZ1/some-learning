from cs336_basics.bpe import pretokenize_file_in_chunks


def main():
    path = "tests/fixtures/tinystories_sample_5M.txt"
    special_tokens = ["<|endoftext|>"]

    serial = pretokenize_file_in_chunks(
        path,
        special_tokens,
        num_chunks=16,
        num_workers=1,
    )

    parallel = pretokenize_file_in_chunks(
        path,
        special_tokens,
        num_chunks=16,
        num_workers=2,
    )

    assert serial == parallel

    print("Multiprocessing: PASS")
    print(f"Unique pre-tokens: {len(parallel)}")


if __name__ == "__main__":
    main()
