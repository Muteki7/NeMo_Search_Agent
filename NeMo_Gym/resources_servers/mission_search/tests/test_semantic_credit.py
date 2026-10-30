# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Why this file exists: the verifier will give extra credit for a correct answer in other words, using BGE-M3 (ColBERT, reference as the query).
A semantic score alone rewards an answer with one wrong number (on 44 answers made from correct ones, 73 to 95% still scored above the threshold).
So the credit is guarded: every number of the reference must be in the answer. These tests check the two parts of that design, the guard
(app.number_guard) and the score (m3.BGEM3.reference_coverage), the credit rule that joins them (app.semantic_credit: threshold 0.65, points 0.85,
decided 2026-10-30), and how the rule is wired into verify() and into the server. The end-to-end test replays the 148 answers of the run-2 eval
(trainer/runs, git-ignored) and checks the credit against the hand labels in data/verifier_labels.jsonl.

The `test_model_*` tests load BGE-M3 (one shared copy, tests/conftest.py). Run the rest alone with: pytest tests -k "not test_model".
"""
import asyncio
import json
import re
import statistics
from pathlib import Path

import pytest

import chunks as chunk_store
import m3

from ..app import (
    NUMBER_PATTERN,
    SEMANTIC_POINTS,
    SEMANTIC_THRESHOLD,
    MissionSearchResourcesServer,
    MissionSearchResourcesServerConfig,
    MissionSearchVerifier,
    number_guard,
    numbers_in,
    score_answer,
    semantic_credit,
)
from .test_app import _json_output, _request
from .test_retriever import FakeModel, fake_model_dir, mk_chunks

LABELS = Path(__file__).resolve().parent.parent / "data" / "verifier_labels.jsonl"


# ---- numbers_in ------------------------------------------------------------------------------------------------
def test_numbers_in_removes_thousands_separators_and_splits_times() -> None:
    assert numbers_in("35,560 feet") == ["35560"]
    assert numbers_in("35 560 feet") == ["35560"]
    assert numbers_in("at 2:35:46.4 and 0.06g") == ["2", "35", "46.4", "0.06"]


def test_numbers_in_gives_nothing_for_text_without_digits() -> None:
    assert numbers_in("a sudden loss of pressure") == []


# ---- number_guard ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "final_text, expected",
    [
        ("About 35,560 feet.", "35 560 feet"),  # another format of the same number
        ("The tank failed at 56 hours in bay 4.", "approximately 56 hours in bay 4"),
        ("<reasoning>maybe 7</reasoning>It was 0.06g", "0.06g"),
        ("Pressure was lost.", "A sudden loss of pressure"),  # no number in the reference: nothing to check
    ],
)
def test_number_guard_passes_when_every_reference_number_is_in_the_answer(final_text: str, expected: str) -> None:
    assert number_guard(final_text, expected)


@pytest.mark.parametrize(
    "final_text, expected",
    [
        ("It was 0.07g", "0.06g"),  # one digit changed
        ("About 35,580 feet.", "35 560 feet"),
        ("At 142:54:42", "142:54:41"),  # one part of a time changed
        ("The tank failed at 56 hours.", "56 hours in bay 4"),  # a number of the reference is missing
        ("<reasoning>It was 0.06g</reasoning>It was about 6 percent", "0.06g"),  # the number is only inside the reasoning
        (None, "0.06g"),  # no final message
        (None, "no number here"),
    ],
)
def test_number_guard_fails_when_a_number_is_changed_missing_or_not_in_the_final_answer(final_text, expected: str) -> None:
    assert not number_guard(final_text, expected)


def labelled_correct_rows_with_numbers() -> list:
    if not LABELS.exists():
        pytest.skip("data/verifier_labels.jsonl is missing")
    rows = [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["label"] == "correct" and numbers_in(r["reference"])]


def test_number_guard_blocks_only_the_labelled_correct_answer_that_leaves_out_a_reference_number() -> None:
    rows = labelled_correct_rows_with_numbers()
    blocked = [r["index"] for r in rows if not number_guard(r["answer"], r["reference"])]
    assert len(rows) == 22
    assert blocked == [0]  # the answer leaves out "approximately 56 hours", which the question did not ask for


def test_number_guard_fails_for_every_labelled_correct_answer_after_one_of_its_numbers_is_changed() -> None:
    rows = [r for r in labelled_correct_rows_with_numbers() if number_guard(r["answer"], r["reference"])]
    assert len(rows) == 21
    for r in rows:
        target = next(n for n in numbers_in(r["reference"]))
        changed = NUMBER_PATTERN.sub(lambda m: re.sub(r"\d$", lambda d: str((int(d.group()) + 1) % 10), m.group())
                                     if re.sub(r"[ ,]", "", m.group()) == target else m.group(), r["answer"])
        assert target not in numbers_in(changed), r["index"]  # the change really removed the number
        assert not number_guard(changed, r["reference"]), r["index"]


# ---- reference_coverage (the model) ----------------------------------------------------------------------------
REFERENCE = "A sudden loss of pressure at approximately 56 hours from one of the two service module cryogenic oxygen tanks in bay 4."
ANSWER = ("The cause of the loss of primary electrical power on the Apollo 13 mission was a sudden loss of pressure in one of the two "
          "service module cryogenic oxygen tanks in bay 4.")


def test_model_reference_coverage_of_a_text_with_itself_is_one(bge_m3) -> None:
    assert bge_m3.reference_coverage(REFERENCE, REFERENCE) >= 0.99


def test_model_reference_coverage_is_higher_for_a_long_correct_answer_than_for_an_unrelated_text(bge_m3) -> None:
    unrelated = "The recovery ship reached the splashdown area in the South Pacific and the crew was taken on board by helicopter."
    assert bge_m3.reference_coverage(REFERENCE, ANSWER) > bge_m3.reference_coverage(REFERENCE, unrelated) + 0.1


def test_model_reference_coverage_depends_on_which_text_is_the_reference(bge_m3) -> None:
    # The long answer has words that the short reference does not have: with the answer as the query the score is lower (0.743 against 0.696).
    assert bge_m3.reference_coverage(REFERENCE, ANSWER) > bge_m3.reference_coverage(ANSWER, REFERENCE) + 0.03


def test_model_reference_coverage_matches_the_value_of_an_independent_script(bge_m3) -> None:
    # 0.74281 comes from a scratch script that encoded all 72 labelled answers and references in two batches and took the mean over the
    # reference tokens of the best answer token (the numpy lines of the first run on 2026-10-29), not from this method.
    assert bge_m3.reference_coverage(REFERENCE, ANSWER) == pytest.approx(0.7428127527236938, abs=1e-3)


# ---- semantic_credit (the rule) ----------------------------------------------------------------------------------
class ScoreModel:
    """Stands in for m3.BGEM3 in the credit rule: reference_coverage gives a fixed score and every call is recorded."""

    def __init__(self, score):
        self.score, self.calls = score, []

    def reference_coverage(self, reference, answer):
        self.calls.append((reference, answer))
        return self.score


def test_semantic_credit_gives_the_points_when_the_guard_passes_and_the_score_reaches_the_threshold() -> None:
    assert (SEMANTIC_THRESHOLD, SEMANTIC_POINTS) == (0.65, 0.85)
    for score in (SEMANTIC_THRESHOLD, 0.9):  # exactly the threshold counts
        model = ScoreModel(score)
        got = semantic_credit(0.3, "<reasoning>x 5</reasoning>It was 0.06g in total.", "0.06g", model)
        assert got == (0.85, score)
        assert model.calls == [("0.06g", "It was 0.06g in total.")]  # the model sees the reference and the answer without the reasoning


def test_semantic_credit_leaves_the_reward_alone_when_the_score_is_below_the_threshold() -> None:
    model = ScoreModel(0.649)
    assert semantic_credit(0.3, "Something else", "A sudden loss of pressure", model) == (0.3, 0.649)


def test_semantic_credit_never_lowers_a_reward_and_does_not_call_the_model_for_a_reward_at_the_points() -> None:
    model = ScoreModel(0.9)
    assert semantic_credit(0.84, "answer", "ref", model) == (0.85, 0.9)  # just below the points: raised
    for reward in (0.85, 0.9, 1.0):
        assert semantic_credit(reward, "answer", "ref", model) == (reward, None)
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "final_text, expected, why",
    [
        ("It was 0.07g", "0.06g", "the number guard fails"),
        (None, "ref", "no final message"),
        ("<reasoning>only reasoning</reasoning>", "ref", "nothing left after the reasoning"),
        ("an answer", "  ", "empty reference"),
    ],
)
def test_semantic_credit_does_not_call_the_model_when_a_cheap_check_decides(final_text, expected, why) -> None:
    model = ScoreModel(0.9)
    assert semantic_credit(0.3, final_text, expected, model) == (0.3, None), why
    assert model.calls == []


def test_semantic_credit_without_a_model_is_the_old_reward() -> None:
    assert semantic_credit(0.3, "It was 0.06g", "0.06g", None) == (0.3, None)


# ---- verify() ----------------------------------------------------------------------------------------------------
REF = "A sudden loss of pressure"
FINAL = "Pressure dropped suddenly in one tank."
HIT = "Result 1:\nsudden loss of pressure\n------\n"


def verify(model):
    verifier = MissionSearchVerifier()
    if model is not None:
        verifier._semantic_model = model  # the server sets this when semantic_credit is on
    return asyncio.run(verifier.verify(_request(_json_output(("call",), ("result", HIT), ("msg", FINAL)), REF)))


def test_verify_with_the_credit_keeps_reward_answer_as_the_rules_and_builds_the_reward_from_the_credit() -> None:
    old = score_answer(FINAL, REF)
    assert old < SEMANTIC_POINTS
    got = verify(ScoreModel(0.7))
    assert got.reward_answer == old  # the yardstick between runs does not change
    assert got.reward_answer_semantic == 0.85 and got.semantic_score == 0.7
    assert got.reward == pytest.approx(0.75 * 0.85 + 0.25 * 1.0)  # the retrieval part is a full hit
    assert got.reward_components == {"answer": 0.85, "retrieval": 1.0}


def test_verify_without_a_model_has_no_credit_and_marks_the_score_as_not_worked_out() -> None:
    old = score_answer(FINAL, REF)
    got = verify(None)
    assert got.reward_answer == got.reward_answer_semantic == old and got.semantic_score == -1.0
    assert got.reward == pytest.approx(0.75 * old + 0.25 * 1.0)


# ---- the server ----------------------------------------------------------------------------------------------------
def build_server(tmp_path, monkeypatch, semantic):
    loads, model = [], FakeModel()
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(m3, "BGEM3", lambda path: loads.append(path) or model)
    chunk_store.write_chunks(tmp_path / "chunks.jsonl", mk_chunks(["engine cut off early", "battery energy at landing"]), {"chunking": {}})
    (tmp_path / "index").mkdir()
    config = MissionSearchResourcesServerConfig(
        host="127.0.0.1", port=1, entrypoint="app.py", name="x", retriever="bge_m3", chroma_dir=str(tmp_path / "index"),
        m3_dir=str(fake_model_dir(tmp_path / "model")), m3_cache_dir=str(tmp_path / "cache"), chunks_path=str(tmp_path / "chunks.jsonl"),
        semantic_credit=semantic,
    )
    server = MissionSearchResourcesServer.model_construct(config=config)
    server.setup_webserver()
    return server, loads, model


def test_the_server_loads_one_model_for_the_retriever_and_gives_it_to_the_verifier_only_when_the_credit_is_on(tmp_path, monkeypatch) -> None:
    server, loads, model = build_server(tmp_path / "on", monkeypatch, True)
    assert len(loads) == 1 and server._semantic_model is model and server._searcher.retriever.model is model
    server, loads, model = build_server(tmp_path / "off", monkeypatch, False)
    assert len(loads) == 1 and server._semantic_model is None and server._searcher.retriever.model is model


def test_the_semantic_credit_is_off_by_default() -> None:
    assert MissionSearchResourcesServerConfig(host="", port=0, entrypoint="", name="").semantic_credit is False


# ---- replay of the run-2 eval with the real model -----------------------------------------------------------------------
EVAL_DUMP = Path(__file__).resolve().parents[4] / "trainer" / "runs" / "eval_bge_m3_top2" / "eval_final.jsonl"


def test_model_credit_on_the_run_2_eval_matches_the_hand_labels(bge_m3) -> None:
    if not (EVAL_DUMP.exists() and LABELS.exists()):
        pytest.skip("the eval dump (trainer/runs, git-ignored) or data/verifier_labels.jsonl is missing")
    rows = [json.loads(line) for line in EVAL_DUMP.read_text().splitlines() if line]
    label = {r["index"]: r["label"] for r in map(json.loads, LABELS.read_text().splitlines()) if r}
    old, new, credited = [], [], []
    for r in rows:
        base = score_answer(r["final_text"], r["answer"])
        got, _ = semantic_credit(base, r["final_text"], r["answer"], bge_m3)
        old.append(base), new.append(got)
        if got > base:
            credited.append(r["index"])
    assert len(rows) == 148
    assert statistics.mean(old) == pytest.approx(0.578, abs=1e-3) and statistics.mean(new) == pytest.approx(0.677, abs=1e-3)
    assert len(credited) == 30
    assert not [i for i in credited if label.get(i) in ("wrong", "bad_question")]  # none of the 9 wrong and 7 bad-question rows
    assert {38, 144} <= set(credited)  # correct answers in other words
    assert 97 in credited  # KNOWN GAP: a wrong answer without a number gets the credit (the guard only sees numbers)
