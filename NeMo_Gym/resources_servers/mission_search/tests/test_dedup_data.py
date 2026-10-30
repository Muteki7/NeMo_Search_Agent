# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why this file exists: scripts/dedup_data.py removes the repeated questions from the train and validation files (neighbouring chunk windows made the same
question twice). A wrong rule would drop good questions or keep conflicting references silently, and the files feed run 3 and the eval. These tests check the
rule on small rows, and the script on copies of the real files (the originals must stay byte-identical)."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import dedup_data  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"


def row(question, answer):
    prompt = "You are a research assistant.\n\nQuestion: " + question + "\n"
    return {"responses_create_params": {"input": [{"role": "system", "content": "s"}, {"role": "user", "content": prompt}]}, "answer": answer, "chunk_id": 0}


def test_dedup_keeps_the_first_copy_of_a_repeated_question_with_the_same_reference() -> None:
    rows = [row("Q1?", "A"), row("Q2?", "B"), row("q1?  ", "A"), row("Q1?", "A")]
    kept, dropped = dedup_data.dedup(rows)
    assert kept == [0, 1] and dropped == {2: "repeat", 3: "repeat"}


def test_dedup_drops_all_copies_of_a_question_whose_references_differ() -> None:
    rows = [row("Q1?", "A"), row("Q2?", "B"), row("Q1?", "Another fact"), row("Q3?", "C")]
    kept, dropped = dedup_data.dedup(rows)
    assert kept == [1, 3] and dropped == {0: "conflict", 2: "conflict"}


def test_dedup_does_not_count_a_difference_in_case_punctuation_or_articles_as_a_conflict() -> None:
    kept, dropped = dedup_data.dedup([row("Q1?", "The command module."), row("Q1?", "command module")])
    assert kept == [0] and dropped == {1: "repeat"}


def test_dedup_keeps_the_order_of_the_kept_rows_and_drops_nothing_on_a_second_pass() -> None:
    rows = [row("Q3?", "C"), row("Q1?", "A"), row("Q3?", "C"), row("Q2?", "B")]
    kept, _ = dedup_data.dedup(rows)
    assert kept == [0, 1, 3]
    again, dropped = dedup_data.dedup([rows[i] for i in kept])
    assert again == [0, 1, 2] and dropped == {}


def test_question_of_returns_the_text_after_the_last_question_label() -> None:
    assert dedup_data.question_of(row("What was it?", "A")) == "What was it?"


# ---- the script on copies of the real files ----------------------------------------------------------------------
def test_the_script_on_the_real_files_removes_every_repeat_and_leaves_the_originals_alone(tmp_path, monkeypatch) -> None:
    if not (DATA / "train.jsonl").exists() or not (DATA / "validation.jsonl").exists():
        pytest.skip("data/train.jsonl or data/validation.jsonl is missing")
    for name in ("train.jsonl", "validation.jsonl"):
        shutil.copy(DATA / name, tmp_path / name)
    before = {name: (tmp_path / name).read_bytes() for name in ("train.jsonl", "validation.jsonl")}
    monkeypatch.setattr(sys, "argv", ["dedup_data.py", "--data-dir", str(tmp_path)])
    dedup_data.main()
    texts = {}
    for split, (low, high) in (("train", (1100, 1115)), ("validation", (125, 128))):
        assert (tmp_path / f"{split}.jsonl").read_bytes() == before[f"{split}.jsonl"]  # the input did not change
        original = [json.loads(line) for line in (tmp_path / f"{split}.jsonl").read_text().splitlines() if line]
        out = [json.loads(line) for line in (tmp_path / f"{split}_dedup.jsonl").read_text().splitlines() if line]
        assert low <= len(out) <= high, (split, len(out))
        texts[split] = [dedup_data.normalize_question(dedup_data.question_of(r)) for r in out]
        assert len(set(texts[split])) == len(texts[split])  # no question twice
        positions = [original.index(r) for r in out]
        assert positions == sorted(positions)  # an in-order subset of the input
        assert dedup_data.dedup(out)[1] == {}  # a second pass drops nothing
    assert not set(texts["train"]) & set(texts["validation"])  # still no question in both splits
