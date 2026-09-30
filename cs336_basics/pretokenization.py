import os
from typing import BinaryIO
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
import multiprocessing as mp


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open("data/test.txt", "rb") as f:
#     num_processes = 2
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         print(f"Chunk from {start} to {end}:")
#         print(chunk)
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def count_pretokens_in_range(
    task: tuple[
        str,
        int,
        int,
        tuple[str, ...],
    ],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = task

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk_bytes = file.read(end - start)

    chunk = chunk_bytes.decode("utf-8")

    if special_tokens:
        split_pattern = "|".join(
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )
        parts = re.split(split_pattern, chunk)
    else:
        parts = (chunk,)

    counts: Counter[tuple[bytes, ...]] = Counter()

    for part in parts:
        for match in re.finditer(PAT, part):
            word_bytes = match.group().encode("utf-8")
            pretoken = tuple(
                bytes([byte])
                for byte in word_bytes
            )
            counts[pretoken] += 1

    return counts


def parallel_count_pretokens(
    input_path: str,
    special_tokens: list[str],
    num_workers: int | None = None,
    num_chunks: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if num_chunks is None:
        num_chunks = num_workers
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")

    boundary_token = "<|endoftext|>"

    if boundary_token not in special_tokens:
        raise ValueError(
            "Parallel chunking requires a known "
            "special-token boundary."
        )

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            num_chunks,
            boundary_token.encode("utf-8"),
        )

    tasks = [
        (
            input_path,
            start,
            end,
            tuple(special_tokens),
        )
        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        )
    ]

    total_counts: Counter[
        tuple[bytes, ...]
    ] = Counter()

    context = mp.get_context("spawn")

    with context.Pool(
        processes=num_workers
    ) as pool:
        for local_counts in pool.imap_unordered(
            count_pretokens_in_range,
            tasks,
        ):
            total_counts.update(local_counts)

    return total_counts


def merge_from_pretoken_counts(
    token2cnt: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    vocab = {
            i: bytes([i])
            for i in range(256)
        }
    merge_list = []

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    id2token: dict[int, tuple[bytes, ...]] = {}
    id2cnt: dict[int, int] = {}
    pair_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # initialize pre_token, freq, index

    for token, cnt in token2cnt.items():
        id2token[len(id2token)] = token
        id2cnt[len(id2cnt)] = cnt

    del token2cnt

    for id, tokens in id2token.items():
        for token1, token2 in zip(tokens[:-1], tokens[1:]):
            pair_freq[(token1, token2)] += id2cnt[id]
            pair_index[(token1, token2)].add(id)

    while len(vocab) < vocab_size and pair_freq:
        merge_tokens = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        merge_token = merge_tokens[0] + merge_tokens[1]

        vocab[len(vocab)] = merge_token
        merge_list.append(merge_tokens)

        affected_ids = list(pair_index[merge_tokens])

        for pretoken_id in affected_ids:
            old_tokens = id2token[pretoken_id]

            # 记录合并前，每种相邻 pair 在该 pre-token 中出现几次
            old_pair_counts = Counter(
                zip(old_tokens[:-1], old_tokens[1:])
            )

            new_tokens = merge_pair(old_tokens, merge_tokens)
            id2token[pretoken_id] = new_tokens

            # 记录真正合并后的相邻 pair
            new_pair_counts = Counter(
                zip(new_tokens[:-1], new_tokens[1:])
            )

            # 该 pre-token 在原语料中的出现次数
            pretoken_count = id2cnt[pretoken_id]

            affected_pairs = (
                set(old_pair_counts)
                | set(new_pair_counts)
            )

            for pair in affected_pairs:
                old_count = old_pair_counts.get(pair, 0)
                new_count = new_pair_counts.get(pair, 0)

                # 乘以该 pre-token 在语料中的出现次数
                delta = (new_count - old_count) * pretoken_count
                updated_frequency = pair_freq.get(pair, 0) + delta

                # 出现负数意味着增量状态已经不一致
                if updated_frequency < 0:
                    raise AssertionError(
                        f"Negative frequency for {pair}: "
                        f"{updated_frequency}"
                    )

                if updated_frequency == 0:
                    pair_freq.pop(pair, None)
                else:
                    pair_freq[pair] = updated_frequency

                # pair_index 只记录“哪些 pre-token 包含该 pair”，
                # 不记录它在同一个 pre-token 中出现多少次。
                if new_count > 0:
                    pair_index[pair].add(pretoken_id)
                else:
                    indexed_ids = pair_index.get(pair)

                    if indexed_ids is not None:
                        indexed_ids.discard(pretoken_id)

                        if not indexed_ids:
                            del pair_index[pair]

        # merge_pair 应该已经消除所有选中的旧 pair
        assert merge_tokens not in pair_freq
        assert merge_tokens not in pair_index

    return vocab, merge_list
    


def one_chunk_merge(
    vocab_size: int,
    special_tokens: list[str],
    chunk: str,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # return vocab + merge list

    # split first
    pattern = "|".join(map(re.escape, special_tokens))
    parts = re.split(pattern, chunk)

    # PAT second

    # return vocab + merge list of the part
    vocab = {
        i: bytes([i])
        for i in range(256)
    }
    merge_list = []

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    word_list = []

    for part in parts:
        word_list.extend(re.findall(PAT, part))

    pre_tokens_list = [tuple(bytes([x]) for x in word.encode("utf-8")) for word in word_list]

    token2cnt: dict[tuple[bytes, ...], int] = defaultdict(int)
    id2token: dict[int, tuple[bytes, ...]] = {}
    id2cnt: dict[int, int] = {}
    pair_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # initialize pre_token, freq, index

    for pre_tokens in pre_tokens_list:
        token2cnt[pre_tokens] += 1

    for token, cnt in token2cnt.items():
        id2token[len(id2token)] = token
        id2cnt[len(id2cnt)] = cnt

    del token2cnt

    for id, tokens in id2token.items():
        for token1, token2 in zip(tokens[:-1], tokens[1:]):
            pair_freq[(token1, token2)] += id2cnt[id]
            pair_index[(token1, token2)].add(id)

    while len(vocab) < vocab_size and pair_freq:
        merge_tokens = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        merge_token = merge_tokens[0] + merge_tokens[1]

        vocab[len(vocab)] = merge_token
        merge_list.append(merge_tokens)

        affected_ids = list(pair_index[merge_tokens])

        for pretoken_id in affected_ids:
            old_tokens = id2token[pretoken_id]

            # 记录合并前，每种相邻 pair 在该 pre-token 中出现几次
            old_pair_counts = Counter(
                zip(old_tokens[:-1], old_tokens[1:])
            )

            new_tokens = merge_pair(old_tokens, merge_tokens)
            id2token[pretoken_id] = new_tokens

            # 记录真正合并后的相邻 pair
            new_pair_counts = Counter(
                zip(new_tokens[:-1], new_tokens[1:])
            )

            # 该 pre-token 在原语料中的出现次数
            pretoken_count = id2cnt[pretoken_id]

            affected_pairs = (
                set(old_pair_counts)
                | set(new_pair_counts)
            )

            for pair in affected_pairs:
                old_count = old_pair_counts.get(pair, 0)
                new_count = new_pair_counts.get(pair, 0)

                # 乘以该 pre-token 在语料中的出现次数
                delta = (new_count - old_count) * pretoken_count
                updated_frequency = pair_freq.get(pair, 0) + delta

                # 出现负数意味着增量状态已经不一致
                if updated_frequency < 0:
                    raise AssertionError(
                        f"Negative frequency for {pair}: "
                        f"{updated_frequency}"
                    )

                if updated_frequency == 0:
                    pair_freq.pop(pair, None)
                else:
                    pair_freq[pair] = updated_frequency

                # pair_index 只记录“哪些 pre-token 包含该 pair”，
                # 不记录它在同一个 pre-token 中出现多少次。
                if new_count > 0:
                    pair_index[pair].add(pretoken_id)
                else:
                    indexed_ids = pair_index.get(pair)

                    if indexed_ids is not None:
                        indexed_ids.discard(pretoken_id)

                        if not indexed_ids:
                            del pair_index[pair]

        # merge_pair 应该已经消除所有选中的旧 pair
        assert merge_tokens not in pair_freq
        assert merge_tokens not in pair_index

    return vocab, merge_list


def merge_pair(tokens, pair):
    result = []
    i = 0

    while i < len(tokens):
        if (
            i < len(tokens) - 1
            and (tokens[i], tokens[i + 1]) == pair
        ):
            result.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return tuple(result)
