"""Character-level and byte-level tokenizer for autoregressive generation and text datasets."""

DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\n"


class CharacterTokenizer:
    """Character-level tokenizer with fixed vocabulary and UTF-8 byte-level fallback."""

    def __init__(self, vocab_size: int, byte_fallback: bool = True):
        self.vocab_size = max(1, int(vocab_size))
        self.byte_fallback = byte_fallback
        if self.vocab_size <= len(DEFAULT_CHARS):
            self.chars = list(DEFAULT_CHARS[: self.vocab_size])
        else:
            self.chars = list(DEFAULT_CHARS) + [
                chr(0x100 + i) for i in range(self.vocab_size - len(DEFAULT_CHARS))
            ]

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Convert string to list of character index tokens with UTF-8 byte fallback."""
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif self.byte_fallback:
                for b in ch.encode("utf-8"):
                    ids.append(b % self.vocab_size)
            else:
                ids.append(0)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert list of token IDs back into a string."""
        return "".join(self.itos.get(i, "") for i in ids)
