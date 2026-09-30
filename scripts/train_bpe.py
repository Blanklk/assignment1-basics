from __future__ import annotations

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path

import psutil

from cs336_basics.pretokenization import merge_from_pretoken_counts, parallel_count_pretokens


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to the training corpus.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Final vocabulary size.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the training outputs.")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help=f"Special token to add. Repeat for multiple tokens. Defaults to {DEFAULT_SPECIAL_TOKEN!r}.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Number of pre-tokenization workers.")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Number of file chunks. Defaults to roughly 64 MiB per chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    special_tokens = args.special_tokens or [DEFAULT_SPECIAL_TOKEN]

    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if args.num_chunks is not None and args.num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if len(special_tokens) != len(set(special_tokens)):
        raise ValueError("Special tokens must be unique")

    num_workers = args.num_workers if args.num_workers is not None else min(os.cpu_count() or 1, 8)
    num_chunks = args.num_chunks
    if num_chunks is None:
        num_chunks = max(
            num_workers,
            (input_path.stat().st_size + DEFAULT_CHUNK_BYTES - 1) // DEFAULT_CHUNK_BYTES,
        )

    initial_vocab_size = 256 + len(special_tokens)
    if args.vocab_size < initial_vocab_size:
        raise ValueError(
            f"--vocab-size must be at least {initial_vocab_size} "
            f"for 256 byte tokens and {len(special_tokens)} special token(s)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    parent_process = psutil.Process()
    stop_memory_monitor = threading.Event()
    peak_rss_bytes = 0
    peak_process_count = 0
    memory_sample_count = 0

    def monitor_memory() -> None:
        nonlocal peak_rss_bytes, peak_process_count, memory_sample_count

        while True:
            rss_bytes = 0
            process_count = 0
            for process in [parent_process, *parent_process.children(recursive=True)]:
                try:
                    memory = process.memory_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                rss_bytes += memory.rss
                process_count += 1

            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            peak_process_count = max(peak_process_count, process_count)
            memory_sample_count += 1
            if stop_memory_monitor.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
                break

    memory_monitor = threading.Thread(target=monitor_memory, daemon=True)
    memory_monitor.start()
    started_at = time.perf_counter()
    try:
        pretoken_counts = parallel_count_pretokens(
            str(input_path),
            special_tokens,
            num_workers=num_workers,
            num_chunks=num_chunks,
        )
        pretokenization_seconds = time.perf_counter() - started_at

        merge_started_at = time.perf_counter()
        vocab, merges = merge_from_pretoken_counts(
            pretoken_counts,
            args.vocab_size,
            special_tokens,
        )
        merge_seconds = time.perf_counter() - merge_started_at
        total_seconds = time.perf_counter() - started_at
    finally:
        stop_memory_monitor.set()
        memory_monitor.join()

    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"
    metadata_path = output_dir / "metadata.json"

    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file, protocol=pickle.HIGHEST_PROTOCOL)

    with merges_path.open("wb") as file:
        pickle.dump(merges, file, protocol=pickle.HIGHEST_PROTOCOL)

    longest_token_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))
    metadata = {
        "input_path": str(input_path),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "num_workers": num_workers,
        "num_chunks_requested": num_chunks,
        "unique_pretokens": len(pretoken_counts),
        "pretokenization_seconds": pretokenization_seconds,
        "merge_seconds": merge_seconds,
        "total_seconds": total_seconds,
        "memory_sample_interval_seconds": MEMORY_SAMPLE_INTERVAL_SECONDS,
        "peak_sampled_process_tree_rss_bytes": peak_rss_bytes,
        "peak_process_count": peak_process_count,
        "memory_sample_count": memory_sample_count,
        "longest_token_id": longest_token_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_hex": longest_token.hex(),
        "longest_token_text": longest_token.decode("utf-8", errors="replace"),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Input: {input_path}")
    print(f"Vocabulary: {len(vocab):,} items")
    print(f"Merges: {len(merges):,}")
    print(f"Unique pre-tokens: {len(pretoken_counts):,}")
    print(f"Requested file chunks: {num_chunks:,}; workers: {num_workers:,}")
    print(f"Pre-tokenization: {pretokenization_seconds:.2f}s")
    print(f"Merging: {merge_seconds:.2f}s")
    print(f"Total: {total_seconds:.2f}s")
    print(f"Peak sampled process-tree RSS: {peak_rss_bytes / 1024**3:.2f} GiB (may count shared pages twice)")
    print(f"Peak process count: {peak_process_count}")
    print(f"Memory samples: {memory_sample_count}")
    print(f"Longest token: ID {longest_token_id}, {len(longest_token)} bytes, {metadata['longest_token_text']!r}")
    print(f"Saved vocabulary: {vocab_path}")
    print(f"Saved merges: {merges_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
