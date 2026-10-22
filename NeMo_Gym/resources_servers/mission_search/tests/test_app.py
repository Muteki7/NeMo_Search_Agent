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
    MissionSearchVerifier,
    MissionSearchVerifyRequest,
    combine,
    coverage,
    final_message_text,
    normalize,
    ramp,
    retrieval_score,
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


# ---- retrieval reward (run 2) -----------------------------------------------------------------------------------
def _output(*items):
    """A response output: ("call",), ("result", text) or ("msg", text) in order."""
    kinds = {"call": lambda a=None: SimpleNamespace(type="function_call"),
             "result": lambda text: SimpleNamespace(type="function_call_output", output=text),
             "msg": lambda text: SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])}
    return [kinds[k](*args) for k, *args in items]


REFERENCE = "The trajectory prior to the tank incident was nearly identical to that of Apollo 12."  # content words: 8


def test_coverage_and_ramp() -> None:
    assert coverage("the free return profile of the spacecraft", "To return the spacecraft to a free-return profile.") == 1.0
    assert coverage("Apollo 12 tank trajectory", REFERENCE) == 0.5                          # 4 of 8 content words
    assert coverage("nothing in common here", REFERENCE) == 0.0
    assert coverage("anything", "of the to") == 0.0                                          # a reference of only small words
    assert coverage("at 18:24 ok", "18:24") == 1.0                                           # 18 and 24 are separate words on both sides
    assert (ramp(0.0), ramp(0.3), ramp(0.6), ramp(1.0)) == (0.0, 0.0, 1.0, 1.0)
    assert abs(ramp(0.45) - 0.5) < 1e-9 and abs(ramp(0.5) - 2 / 3) < 1e-9


def test_retrieval_discount_and_no_farming() -> None:
    hit, miss = "free return profile spacecraft", "nothing useful"
    answer = "To return the spacecraft to a free-return profile."
    one = retrieval_score(_output(("call",), ("result", hit)), answer)
    assert one == (1.0, 1.0, 1)
    assert retrieval_score(_output(("call",), ("result", miss), ("call",), ("result", hit)), answer)[0] == 0.5   # a retry is worth half
    assert retrieval_score(_output(("call",), ("result", hit), ("call",), ("result", hit)), answer)[0] == 1.0   # more searches add nothing
    assert retrieval_score(_output(("call",), ("result", miss)) * 5, answer)[0] == 0.0                          # spam of misses
    assert retrieval_score(_output(("msg", "no search at all")), answer) == (0.0, 0.0, 0)


def test_combined_reward_orders_the_outcomes() -> None:
    correct_found = combine(1.0, 1.0, True)
    correct_from_memory = combine(1.0, 0.0, True)
    found_not_used = combine(0.0, 1.0, True)
    nothing = combine(0.0, 0.0, True)
    assert (correct_found, correct_from_memory, found_not_used, nothing) == (1.0, 0.75, 0.25, 0.0)
    assert correct_found > correct_from_memory > found_not_used > nothing
    assert combine(0.0, 1.0, False) == 0.0 and combine(0.3, 1.0, False) == 0.75 * 0.3        # the gate removes only the retrieval part


def _request(output, answer):
    return MissionSearchVerifyRequest.model_validate({
        "responses_create_params": {"input": "q"}, "answer": answer,
        "response": {"id": "r", "created_at": 0, "model": "m", "object": "response", "parallel_tool_calls": False,
                     "tool_choice": "auto", "tools": [], "output": output}})


def _json_output(*items):
    msg = lambda t: {"id": "m", "type": "message", "role": "assistant", "status": "completed",
                     "content": [{"type": "output_text", "text": t, "annotations": []}]}
    call = {"id": "f", "type": "function_call", "call_id": "c", "name": "search_corpus", "arguments": "{}", "status": "completed"}
    return [call if k == "call" else {"type": "function_call_output", "call_id": "c", "output": a[0]} if k == "result" else msg(a[0])
            for k, *a in items]


def test_verify_returns_the_parts_and_the_gate() -> None:
    hit = "Result 1:\nfree return profile spacecraft\n------\n"
    verifier = MissionSearchVerifier()
    answer = "To return the spacecraft to a free-return profile."
    full = asyncio.run(verifier.verify(_request(_json_output(("call",), ("result", hit), ("msg", answer)), answer)))
    assert full.reward == 1.0 and full.reward_answer == 1.0 and full.reward_retrieval == 1.0 and full.retrieval_gate == 1.0
    assert full.reward_components == {"answer": 1.0, "retrieval": 1.0} and full.n_searches == 1.0 and full.best_cov == 1.0
    spam = asyncio.run(verifier.verify(_request(_json_output(*[("call",), ("result", hit)] * 5), answer)))
    assert spam.reward == 0.0 and spam.reward_retrieval == 1.0 and spam.retrieval_gate == 0.0          # hits, but no final message
    unused = asyncio.run(verifier.verify(_request(_json_output(("call",), ("result", hit), ("msg", "I do not know.")), answer)))
    assert unused.reward == 0.25 and unused.reward_answer == 0.0                                         # found it, did not use it


def test_the_endpoint_keeps_the_extra_fields() -> None:
    """FastAPI drops fields that the response model does not list. Go through the same wrappers as the real server."""
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nemo_gym.judge import judge_failsafe
    from nemo_gym.telemetry.endpoints import traced_verify_endpoint

    app = FastAPI()
    app.post("/verify")(traced_verify_endpoint(judge_failsafe(MissionSearchVerifier().verify)))
    body = _request(_json_output(("call",), ("result", "free return profile spacecraft"), ("msg", "free return profile")),
                    "To return the spacecraft to a free-return profile.").model_dump(mode="json")
    reply = TestClient(app).post("/verify", json=body).json()
    assert {"reward", "reward_answer", "reward_retrieval", "retrieval_gate", "best_cov", "n_searches"} <= set(reply), set(reply)


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
