# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.verifier_fixture import exercise_verifier_fixture

from ..app import (
    VERIFIER_FIXTURE,
    CorpusSearcher,
    MissionSearchResourcesServerConfig,
    final_message_text,
    normalize,
    score_answer,
)


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


# ---- scoring rules (pure functions, no index needed) ----------------------------------------------------------
def test_normalize_deletes_punctuation_and_articles() -> None:
    assert normalize("The  Aclar, will be!") == "aclar will be"
    assert normalize("18:24") == normalize("1824")


@pytest.mark.parametrize(
    "final_text, expected, reward",
    [
        ("1824", "1824", 1.0),  # exact
        ("The answer is 1824.", "1824", 1.0),  # containment, short answer
        ("<reasoning>maybe 1900</reasoning>1824", "1824", 1.0),  # reasoning is removed first
        ("<reasoning>The answer is 1824.</reasoning>", "1824", 0.0),  # the answer is only inside the reasoning
        ("1824 " + "word " * 40, "1824", None),  # containment but too long: F1 only, below 1.0
        ("It was reinforced with Mylar tape", "The Aclar will be reinforced with Mylar tape.", None),  # partial F1
        ("completely different", "1824", 0.0),
        # only small words in common: no reward (the old F1 gave reward for these)
        ("it was in the of that", "A sudden loss of pressure at approximately 56 hours in the tank.", 0.0),
        ("", "1824", 0.0),
        (None, "1824", 0.0),  # no final message
    ],
)
def test_score_answer(final_text, expected, reward) -> None:
    score = score_answer(final_text, expected)
    assert 0.0 <= score <= 1.0
    if reward is None:
        assert 0.0 < score < 1.0
    else:
        assert score == reward


def test_length_cap_boundary() -> None:
    # expected answer of 1 word: the cap is 5 x 1 + 15 = 20 words
    assert score_answer("1824 " + "word " * 19, "1824") == 1.0       # 20 words
    assert score_answer("1824 " + "word " * 20, "1824") < 1.0        # 21 words


def test_f1_counts_content_words_only() -> None:
    expected = "A sudden loss of pressure at approximately 56 hours from one of the two tanks in bay 4."
    # five words in common with the expected answer, but all of them are small words
    assert score_answer("it was at the end of the hours to be in a bay", expected) < score_answer(
        "a loss of pressure in the tanks", expected
    )
    assert score_answer("of the at in to was", expected) == 0.0


def test_final_message_is_the_last_message_item() -> None:
    def message(text):
        return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])

    call = SimpleNamespace(type="function_call")
    assert final_message_text([message("first"), call, message("last")]) == "last"
    assert final_message_text([call]) is None
    assert final_message_text([]) is None


# ---- search tool (needs the Chroma index from scripts/build_index.py) --------------------------------------------
INDEX_DIR = Path(__file__).resolve().parent.parent / "chroma_index"


@pytest.fixture(scope="module")
def searcher() -> CorpusSearcher:
    if not INDEX_DIR.is_dir():
        pytest.skip("chroma_index is missing: run scripts/build_index.py")
    config = MissionSearchResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return CorpusSearcher(INDEX_DIR, config.collection_name, config.embedding_model, config.top_k)


def test_search_format_and_count(searcher: CorpusSearcher) -> None:
    text = searcher.search("What time was the flotation collar inflated?")
    assert text.startswith("Result 1:\n")
    assert "\n------\nResult 2:\n" in text
    assert text.count("------\n") == 2  # top_k = 2
    assert "Result 3:" not in text


def test_parallel_searches_give_the_same_text_as_one_by_one(searcher: CorpusSearcher) -> None:
    queries = ["gimbal angle deviations", "oxygen tank pressure", "flotation collar", "lunar module battery"]
    one_by_one = [searcher.search(q) for q in queries]

    async def run_parallel():
        return await asyncio.gather(*(asyncio.to_thread(searcher.search, q) for q in queries))

    assert asyncio.run(run_parallel()) == one_by_one
