from pathlib import Path
from random import Random
from cs336_basics.tokenizer import Tokenizer

SEP = "<|endoftext|>"
special_tokens = [SEP]

def iter_documents(path: Path):
    pending = ""
    with path.open(encoding="utf-8") as source:
        while True:
            block = source.read(1 << 20)
            if not block:
                break

            pieces = (pending + block).split(SEP)
            pending = pieces.pop()  # 可能尚未结束的文档
            for doc in pieces:
                if doc.strip():
                    yield doc

    if pending.strip():
        yield pending

def sample_documents(path: Path, k: int = 10, seed: int = 336):
    rng = Random()
    chosen = []

    for seen, doc in enumerate(iter_documents(path), start=1):
        if seen <= k:
            chosen.append(doc)
        else:
            index = rng.randrange(seen)
            if index < k:
                chosen[index] = doc

    if len(chosen) < k:
        raise ValueError(f"Fewer than {k} documents in {path}")
    return chosen

tiny_docs = sample_documents(Path("data/TinyStoriesV2-GPT4-valid.txt"))
owt_docs = sample_documents(Path("data/owt_valid.txt"))

tiny_dir = Path("outputs/bpe/tinystories-10000-mp16")
owt_dir = Path("outputs/bpe/owt-c256")

tiny_tokenizer = Tokenizer.from_files(
    vocab_filepath=tiny_dir / "vocab.pkl",
    merges_filepath=tiny_dir / "merges.pkl",
    special_tokens=special_tokens,
)
owt_tokenizer = Tokenizer.from_files(
    vocab_filepath=owt_dir / "vocab.pkl",
    merges_filepath=owt_dir / "merges.pkl",
    special_tokens=special_tokens,
)

def compression_ratio(docs: list[str], tokenizer: Tokenizer) -> float:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    return total_bytes / total_tokens

tiny_ratio = compression_ratio(tiny_docs, tiny_tokenizer)
owt_ratio = compression_ratio(owt_docs, owt_tokenizer)
owt_ratio_with_tiny_tokenizer = compression_ratio(owt_docs, tiny_tokenizer)
tiny_ratio_with_owt_tokenizer = compression_ratio(tiny_docs, owt_tokenizer)

print(f"tiny ratio: {tiny_ratio}", f"owt ratio: {owt_ratio}")
print(f"owt ratio with tiny tokenizer: {owt_ratio_with_tiny_tokenizer}")
print(f"tiny ratio with owt tokenizer: {tiny_ratio_with_owt_tokenizer}")