"""Encode the Assignment 1 corpora into uint16 NumPy arrays."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import regex as re

from cs336_basics.pretokenization import PAT, merge_pair
from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKEN = "<|endoftext|>"
READ_CHARS = 1 << 20
BATCH_CHARS = 1 << 20
DTYPE = np.dtype("<u2")

CORPORA = {
    "tinystories": {
        "model": ROOT / "outputs/bpe/tinystories-10000-mp16",
        "train": ROOT / "data/TinyStoriesV2-GPT4-train.txt",
        "valid": ROOT / "data/TinyStoriesV2-GPT4-valid.txt",
    },
    "owt": {
        "model": ROOT / "outputs/bpe/owt-c256",
        "train": ROOT / "data/owt_train.txt",
        "valid": ROOT / "data/owt_valid.txt",
    },
}

_merge_ranks: dict[tuple[bytes, bytes], int] = {}
_merges: list[tuple[bytes, bytes]] = []
_rev_vocab: dict[bytes, int] = {}
_special_id = -1


def initialize_worker(model_dir: Path) -> None:
    global _merge_ranks, _merges, _rev_vocab, _special_id
    tokenizer = Tokenizer.from_files(
        model_dir / "vocab.pkl",
        model_dir / "merges.pkl",
        special_tokens=[SPECIAL_TOKEN],
    )
    if max(tokenizer.vocab) > np.iinfo(DTYPE).max:
        raise ValueError(f"Vocabulary IDs do not fit in {DTYPE}")
    _merges = tokenizer.merges
    _merge_ranks = {pair: rank for rank, pair in enumerate(_merges)}
    _rev_vocab = tokenizer.rev_vocab
    _special_id = _rev_vocab[SPECIAL_TOKEN.encode("utf-8")]
    encode_pretoken.cache_clear()


@lru_cache(maxsize=50_000)
def encode_pretoken(word: bytes) -> tuple[int, ...]:
    pieces = tuple(bytes([byte]) for byte in word)
    while len(pieces) > 1:
        best_rank = None
        for pair in zip(pieces[:-1], pieces[1:]):
            rank = _merge_ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
        if best_rank is None:
            break
        pieces = merge_pair(pieces, _merges[best_rank])
    return tuple(_rev_vocab[piece] for piece in pieces)


def encode_batch(text: str) -> bytes:
    ids: list[int] = []
    for index, part in enumerate(text.split(SPECIAL_TOKEN)):
        if index:
            ids.append(_special_id)
        for match in re.finditer(PAT, part):
            ids.extend(encode_pretoken(match.group().encode("utf-8")))
    return np.asarray(ids, dtype=DTYPE).tobytes()


def iter_batches(path: Path, read_chars: int = READ_CHARS, batch_chars: int = BATCH_CHARS):
    """Yield text batches ending at special-token boundaries, preserving the file exactly."""
    pending = ""
    batch: list[str] = []
    batch_length = 0
    with path.open(encoding="utf-8", newline="") as source:
        while block := source.read(read_chars):
            parts = (pending + block).split(SPECIAL_TOKEN)
            pending = parts.pop()
            for document in parts:
                batch.extend((document, SPECIAL_TOKEN))
                batch_length += len(document) + len(SPECIAL_TOKEN)
                if batch_length >= batch_chars:
                    yield "".join(batch)
                    batch.clear()
                    batch_length = 0
    if pending:
        batch.append(pending)
    if batch:
        yield "".join(batch)


def write_header(file, token_count: int) -> None:
    np.lib.format.write_array_header_2_0(
        file,
        {"descr": np.lib.format.dtype_to_descr(DTYPE), "fortran_order": False, "shape": (token_count,)},
    )


def encode_file(input_path: Path, output_path: Path, model_dir: Path, workers: int) -> None:
    if output_path.exists():
        raise FileExistsError(f"Will not overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")
    if partial_path.exists():
        raise FileExistsError(f"Previous partial output exists: {partial_path}")

    started = time.perf_counter()
    token_count = 0
    input_bytes = 0
    next_report = 100 << 20
    with partial_path.open("wb+") as output:
        write_header(output, 0)
        data_offset = output.tell()

        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(model_dir,)) as pool:
            batches = iter(iter_batches(input_path))
            in_flight = deque()

            def submit_next() -> bool:
                try:
                    batch = next(batches)
                except StopIteration:
                    return False
                in_flight.append((pool.submit(encode_batch, batch), len(batch.encode("utf-8"))))
                return True

            for _ in range(workers * 2):
                if not submit_next():
                    break

            while in_flight:
                future, batch_bytes = in_flight.popleft()
                encoded = future.result()
                output.write(encoded)
                token_count += len(encoded) // DTYPE.itemsize
                input_bytes += batch_bytes
                if input_bytes >= next_report:
                    print(f"{input_path.name}: {input_bytes / 2**30:.2f} GiB read, {token_count:,} tokens", flush=True)
                    next_report += 100 << 20
                submit_next()

        if input_bytes != input_path.stat().st_size:
            raise AssertionError(f"Input byte count changed: {input_bytes} != {input_path.stat().st_size}")
        output.seek(0)
        write_header(output, token_count)
        if output.tell() != data_offset:
            raise AssertionError("NumPy header changed size while updating the token count")

    array = np.load(partial_path, mmap_mode="r", allow_pickle=False)
    if array.dtype != DTYPE or array.shape != (token_count,):
        raise AssertionError("Saved NumPy array has the wrong dtype or shape")
    del array
    partial_path.replace(output_path)
    elapsed = time.perf_counter() - started
    print(f"Saved {output_path}: {token_count:,} uint16 IDs in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *CORPORA), default="all")
    parser.add_argument("--split", choices=("all", "train", "valid"), default="all")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    datasets = CORPORA if args.dataset == "all" else {args.dataset: CORPORA[args.dataset]}
    splits = ("train", "valid") if args.split == "all" else (args.split,)
    for name, paths in datasets.items():
        for split in splits:
            output_path = ROOT / "outputs/tokenized" / f"{name}_{split}.npy"
            encode_file(paths[split], output_path, paths["model"], args.workers)


if __name__ == "__main__":
    main()
