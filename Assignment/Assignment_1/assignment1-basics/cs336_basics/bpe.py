import os
import re
import regex
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


# 1.初始化vocabulary
def vocab_initialize(special_tokens: list[str]) -> dict[int, bytes]:

    vocab = {x: bytes([x]) for x in range(256)}

    start = len(vocab)

    for i, item in enumerate(special_tokens, start=start):
        vocab[i] = item.encode("utf-8")

    return vocab


# vocab = vocab_initialize(special_tokens)

# 2.读取语料
# with open(input_path, "r", encoding="utf-8") as f:
# text = f.read()


# 3.隔离special tokens
def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:

    tokens = sorted((t for t in special_tokens if t), key=len, reverse=True)

    if not tokens:
        return [text]

    pattern = "(" + "|".join(re.escape(t) for t in tokens) + ")"

    parts = re.split(pattern, text)

    return [part for part in parts if part]


# chunks = split_on_special_tokens(text, special_tokens)


# 4.Pre-tokenization
def pretokenize(chunks: list[str], special_tokens: list[str]) -> dict[bytes, int]:

    PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretoken_counts: Counter[bytes] = Counter()

    special_set = set(special_tokens)

    for chunk in chunks:
        if chunk in special_set:
            continue

        for match in regex.finditer(PATTERN, chunk):
            pretoken = match.group().encode("utf-8")
            pretoken_counts[pretoken] += 1

    return dict(pretoken_counts)


# pretokenization = pretokenize(chunks, special_tokens)


def pretokenize_worker(task: tuple[str | os.PathLike, int, int, list[str]]) -> dict[bytes, int]:

    input_path, start, end, special_tokens = task

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        text = chunk.decode("utf-8")

    text_split = split_on_special_tokens(text, special_tokens)
    text_pretoken = pretokenize(text_split, special_tokens)

    return text_pretoken


def pretokenize_file_in_chunks(
    input_path: str | os.PathLike, special_tokens: list[str], num_chunks: int = 16, num_workers: int = 1
) -> dict[bytes, int]:

    with open(input_path, "rb") as f:
        split_token = next((token for token in special_tokens if token), None)

        if split_token is not None:
            boundaries = find_chunk_boundaries(f, num_chunks, split_token.encode("utf-8"))
        else:
            f.seek(0, 2)
            boundaries = [0, f.tell()]

    tasks = ((input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:]))

    total_counts = Counter()

    if num_workers == 1:
        # 串行
        for task in tasks:
            chunk_counts = pretokenize_worker(task)
            total_counts.update(chunk_counts)

    else:
        # 多进程
        with Pool(processes=num_workers) as pool:
            for chunk_counts in pool.imap_unordered(pretokenize_worker, tasks, chunksize=1):
                total_counts.update(chunk_counts)

    #    for start, end in zip(boundaries[:-1], boundaries[1:]):
    #        chunk_counts = pretokenize_worker((input_path, start, end, special_tokens))
    #        total_counts.update(chunk_counts)

    return dict(total_counts)


# 5.转成 byte sequence + 统计 frequency
def byte_sequence_frequency(pretoken_counts: dict[bytes, int]) -> dict[tuple[bytes, ...], int]:

    bytes_token_counts: dict[tuple[bytes, ...], int] = {}

    for pretoken, freq in pretoken_counts.items():
        byte_sequence = tuple(bytes([b]) for b in pretoken)
        bytes_token_counts[byte_sequence] = freq

    return bytes_token_counts


# bytes_token_frequency = byte_sequence_frequency(pretokenization)


# 6.统计 pair frequency
def pair_count(bytes_token_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:

    pair_counts = Counter()

    for tokens, freq in bytes_token_counts.items():
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += freq

    return dict(pair_counts)


# pair_frequency = pair_count(bytes_token_frequency)


# 7.找最高频 pair
def choose_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:

    items = pair_counts.items()

    best_pair, _ = max(items, key=lambda item: (item[1], item[0]))

    return best_pair


# the_best_pair = choose_best_pair(pair_frequency)


# 8.创建新 token
def new_token_merge(vocab: dict[int, bytes], pair: tuple[bytes, bytes]) -> bytes:

    new_token_id = len(vocab)  # 貌似可以维护一个全局变量，不用每次都得重新len?

    vocab[new_token_id] = pair[0] + pair[1]

    return vocab[new_token_id]


# new_token = new_token_merge(vocab, the_best_pair)


# 9.更新pretoken序列
# def update_pretokens(
#   bytes_token_counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes], new_token: bytes
# ) -> dict[tuple[bytes, ...], int]:

# new_token_counts: dict[tuple[bytes, ...], int] = {}

# for tokens, freq in bytes_token_counts.items():
#   new_tokens = []
#   if pair[0] not in tokens or pair[1] not in tokens:
#       new_token_counts[tokens] = freq
#       continue
#   i = 0
#   while i < len(tokens):
#       if tokens[i : i + 2] == pair:
#           new_tokens.append(new_token)
#           i += 2
#       else:
#           new_tokens.append(tokens[i])
#           i += 1
#   new_token_counts[tuple(new_tokens)] = freq

# return new_token_counts

# new_token_frequency = update_pretokens(bytes_token_frequency, the_best_pair, new_token)

# 提取为处理单个 pre-token 的 merge_one_word()


def merge_one_word(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
    n = len(tokens)
    new_tokens = []
    i = 0
    while i < n:
        if tokens[i : i + 2] == pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return tuple(new_tokens)


# 初始化全局频率及倒排索引


def initialize_pair_stats(
    pretoken_id: dict[int, tuple[bytes, ...]], id_freq: dict[int, int]
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[int]]]:
    pair_counts = Counter()
    pair_to_pretoken_id = defaultdict(set)

    for wid, tokens in pretoken_id.items():
        freq = id_freq[wid]

        local_counts = Counter(zip(tokens, tokens[1:]))

        for pair, count in local_counts.items():
            pair_counts[pair] += count * freq
            pair_to_pretoken_id[pair].add(wid)

    return pair_counts, pair_to_pretoken_id


# 增量增加或减少 pair 贡献


def update_pair_contribution(
    tokens: tuple[bytes, ...],
    freq: int,
    wid: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_pretoken_id: defaultdict[tuple[bytes, bytes], set[int]],
    sign: int,
):

    # 统计当前预分词项中所有相邻 pair 的出现次数
    local_counts = Counter(zip(tokens, tokens[1:]))

    for pair, count in local_counts.items():
        # 更新全局 pair 频率
        pair_counts[pair] += sign * count * freq
        # 更新反向索引
        if sign == 1:
            pair_to_pretoken_id[pair].add(wid)

        elif sign == -1:
            pair_to_pretoken_id[pair].discard(wid)

            if not pair_to_pretoken_id[pair]:
                del pair_to_pretoken_id[pair]

        if pair_counts[pair] == 0:
            del pair_counts[pair]


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_workers: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 初始化

    vocab = vocab_initialize(special_tokens)

    #    with open(input_path, "r", encoding="utf-8") as f:
    #        text = f.read()

    #    chunks = split_on_special_tokens(text, special_tokens)

    #    pretokenization = pretokenize(chunks, special_tokens)

    pretokenization = pretokenize_file_in_chunks(input_path, special_tokens, num_chunks=16, num_workers=num_workers)

    bytes_token_frequency = byte_sequence_frequency(pretokenization)

    #    pair_counts = pair_count(bytes_token_frequency)

    pretoken_id = {}
    id_freq = {}

    for wid, (tokens, freq) in enumerate(bytes_token_frequency.items()):
        pretoken_id[wid] = tokens
        id_freq[wid] = freq

    pair_counts, pair_to_pretoken_id = initialize_pair_stats(pretoken_id, id_freq)

    merges: list[tuple[bytes, bytes]] = []

    # BPE训练循环
    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        the_best_pair = choose_best_pair(pair_counts)

        new_token = new_token_merge(vocab, the_best_pair)
        merges.append(the_best_pair)

        affected_ids = set(pair_to_pretoken_id[the_best_pair])  # 创建一个独立的集合副本，而不是直接引用原来的集合。

        for wid in affected_ids:
            old_tokens = pretoken_id[wid]
            freq = id_freq[wid]

            update_pair_contribution(old_tokens, freq, wid, pair_counts, pair_to_pretoken_id, -1)

            new_tokens = merge_one_word(old_tokens, the_best_pair, new_token)
            pretoken_id[wid] = new_tokens

            update_pair_contribution(new_tokens, freq, wid, pair_counts, pair_to_pretoken_id, 1)
    # 返回值
    return vocab, merges
