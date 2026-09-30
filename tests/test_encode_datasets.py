from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from scripts.encode_datasets import (
    CORPORA,
    DTYPE,
    SPECIAL_TOKEN,
    encode_batch,
    encode_file,
    initialize_worker,
    iter_batches,
)


def test_batches_preserve_text_across_split_special_tokens(tmp_path: Path) -> None:
    text = f"ab\r\n{SPECIAL_TOKEN}a'll é{SPECIAL_TOKEN}z"
    input_path = tmp_path / "input.txt"
    input_path.write_text(text, encoding="utf-8", newline="")

    assert "".join(iter_batches(input_path, read_chars=3, batch_chars=5)) == text


def test_fast_encoding_matches_tokenizer() -> None:
    text = f"a'll é {SPECIAL_TOKEN} Once upon a time!"
    for name in ("tinystories", "owt"):
        model_dir = CORPORA[name]["model"]
        initialize_worker(model_dir)
        actual = np.frombuffer(encode_batch(text), dtype=DTYPE).tolist()
        tokenizer = Tokenizer.from_files(model_dir / "vocab.pkl", model_dir / "merges.pkl", [SPECIAL_TOKEN])
        assert actual == tokenizer.encode(text)


def test_file_is_valid_uint16_npy(tmp_path: Path) -> None:
    text = f"abc{SPECIAL_TOKEN}a'll é"
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.npy"
    input_path.write_text(text, encoding="utf-8")

    model_dir = CORPORA["tinystories"]["model"]
    encode_file(input_path, output_path, model_dir, workers=1)

    saved = np.load(output_path, mmap_mode="r", allow_pickle=False)
    tokenizer = Tokenizer.from_files(model_dir / "vocab.pkl", model_dir / "merges.pkl", [SPECIAL_TOKEN])
    assert saved.dtype == DTYPE
    assert saved.tolist() == tokenizer.encode(text)
