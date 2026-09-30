from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator
from os import PathLike

import regex as re

from cs336_basics.pretokenization import PAT, merge_pair


class Tokenizer:
    """Byte-level BPE tokenizer built from a vocabulary and ordered merges."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """Initialize tokenizer state and register any missing special tokens."""
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens: list[str] = list(special_tokens or [])

        self.rev_vocab = {token: id for id, token in vocab.items()}

        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.rev_vocab:
                self.rev_vocab[token_bytes] = len(self.vocab)
                self.vocab[len(self.vocab)] = token_bytes

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load the vocabulary and merges saved by scripts/train_bpe.py."""
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text as token IDs, preserving configured special tokens."""
        final_ids: list[int] = []
        pretoken_cache: dict[tuple[bytes, ...], list[int]] = {}

        if self.special_tokens:
            split_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            items = re.split(f"({split_pattern})", text)
        else:
            items = [text]

        for item in items:
            if not item:
                continue
            if item in self.special_tokens:
                final_ids.append(self.rev_vocab[item.encode("utf-8")])
                continue

            for match in re.finditer(PAT, item):
                pretoken = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
                if pretoken not in pretoken_cache:
                    merged = pretoken
                    for merge in self.merges:
                        if merge in zip(merged[:-1], merged[1:]):
                            merged = merge_pair(merged, merge)
                    pretoken_cache[pretoken] = [self.rev_vocab[token] for token in merged]
                final_ids.extend(pretoken_cache[pretoken])

        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of text without splitting tokens at chunk boundaries."""
        buffer = ""
        special_pattern = None
        if self.special_tokens:
            pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
            special_pattern = re.compile(pattern)

        # Keep lookahead for PAT contractions and incomplete special tokens.
        # An unfinished pre-token that reaches this suffix is retained in full.
        guard_chars = max(4, max((len(token) for token in self.special_tokens), default=0))

        for chunk in iterable:
            if not chunk:
                continue
            buffer += chunk
            limit = len(buffer) - guard_chars
            if limit <= 0:
                continue

            safe_end = 0
            segment_start = 0
            if special_pattern is not None:
                for special in special_pattern.finditer(buffer):
                    for match in re.finditer(PAT, buffer[segment_start : special.start()]):
                        candidate = segment_start + match.end()
                        if candidate <= limit:
                            safe_end = candidate
                    if special.end() <= limit:
                        safe_end = special.end()
                    segment_start = special.end()

            for match in re.finditer(PAT, buffer[segment_start:]):
                candidate = segment_start + match.end()
                if candidate <= limit:
                    safe_end = candidate

            if safe_end:
                yield from self.encode(buffer[:safe_end])
                buffer = buffer[safe_end:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text, replacing malformed UTF-8 bytes."""
        data = b"".join(self.vocab[id] for id in ids)
        return data.decode("utf-8", errors="replace")
