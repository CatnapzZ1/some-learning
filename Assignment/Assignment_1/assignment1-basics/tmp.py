from cs336_basics.tokenizer import Tokenizer

vocab = {i: bytes([i]) for i in range(256)}

vocab[256] = b"bc"
vocab[257] = b"ab"

merges = [
    (b"b", b"c"),
    (b"a", b"b"),
]

tokenizer = Tokenizer(vocab, merges)

assert tokenizer.encode("abc") == [97, 256]
assert tokenizer.encode("ab") == [257]
assert tokenizer.encode("") == []

assert tokenizer.decode(tokenizer.encode("abc")) == "abc"

print("Tokenizer Basic Encode: PASS")
