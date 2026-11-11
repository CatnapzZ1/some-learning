from collections.abc import Iterable, Iterator
import re
import regex


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.byte_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {merge_pair: rank for rank, merge_pair in enumerate(self.merges)}

        self.special_to_id = {}
        next_id = max(self.vocab, default=-1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.byte_to_id:
                self.vocab[next_id] = token_bytes
                self.byte_to_id[token_bytes] = next_id
                next_id += 1

            self.special_to_id[token] = self.byte_to_id[token_bytes]

    def decode(self, ids: Iterable[int]) -> str:
        byte_sequence = b"".join(self.vocab[token_id] for token_id in ids)

        return byte_sequence.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:

        result: list[int] = []

        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)

            pattern = "(" + "|".join(re.escape(token) for token in specials) + ")"

            chunks = re.split(pattern, text)

        else:
            chunks = [text]

        special_set = set(self.special_tokens)

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in special_set:
                result.append(self.byte_to_id[chunk.encode("utf-8")])
                continue

            for match in regex.finditer(PATTERN, chunk):
                pretoken = match.group()

                ids = self._encode_pretoken(pretoken)
                result.extend(ids)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable:
            yield from self.encode(text)

    def _encode_pretoken(self, pretoken: str) -> list[int]:

        ori_tokens = pretoken.encode("utf-8")

        tokens = [bytes([id]) for id in ori_tokens]

        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]

                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            new_tokens = []
            i = 0

            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
        return [self.byte_to_id[token] for token in tokens]
