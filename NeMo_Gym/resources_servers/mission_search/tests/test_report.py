# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/report.py (the HTML report of eval dumps): the category rules, the join by index, escaping, the page."""
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import report  # noqa: E402

REF = "To return the spacecraft to a free-return profile."
HIT_TEXT = "The maneuver set a free return profile for the spacecraft."
OTHER_TEXT = "Totally unrelated text about thermal control."
SPAN_IN, SPAN_OUT = (10, 11), (50, 51)


def tool_text(*chunks):
    return "".join(f"Result {i}:\n{c}\n------\n" for i, c in enumerate(chunks, start=1))


def make_ctx(data_rows=(("Q?", 10),), old_chunks=None):
    table = {HIT_TEXT: SPAN_IN, OTHER_TEXT: SPAN_OUT}
    return report.Context(table, list(data_rows), old_chunks)


def make_row(chunks, final="It is X.", reward=0.0, ended="answer", index=0, question="Q?", answer=REF, with_reasoning=None):
    out = []
    if chunks:
        out = [{"type": "function_call", "call_id": "c", "name": "search_corpus", "arguments": '{"query": "q"}'},
               {"type": "function_call_output", "call_id": "c", "output": tool_text(*chunks)}]
    if final is not None:
        out.append({"type": "message", "content": [{"type": "output_text", "text": final}]})
    return {"index": index, "question": question, "answer": answer, "reward": reward, "ended": ended, "final_text": final, "output": out}


def test_parse_results_splits_the_tool_text_back_into_chunks() -> None:
    assert report.parse_results(tool_text("AAA", "B\nB")) == ["AAA", "B\nB"]
    assert report.parse_results("no results here") == []


@pytest.mark.parametrize("kwargs, expected", [
    (dict(reward=1.0, ended="answer", final_text="x", reasoning_only=False, support=0.1, prov=False, lex=False), 1),       # success beats all
    (dict(reward=0.0, ended="turn_cap", final_text=None, reasoning_only=True, support=0.0, prov=True, lex=True), 2),
    (dict(reward=0.3, ended="bad_call", final_text="x", reasoning_only=False, support=1.0, prov=True, lex=True), 2),
    (dict(reward=0.0, ended="answer", final_text="x", reasoning_only=True, support=0.0, prov=False, lex=False), 3),         # before bad reference
    (dict(reward=0.0, ended="answer", final_text="x", reasoning_only=False, support=0.4, prov=True, lex=True), 4),
    (dict(reward=0.0, ended="answer", final_text="x", reasoning_only=False, support=None, prov=False, lex=False), 5),      # no support info: rule 4 is off
    (dict(reward=0.0, ended="answer", final_text="x", reasoning_only=False, support=1.0, prov=False, lex=True), 6),
    (dict(reward=0.5, ended="answer", final_text="x", reasoning_only=False, support=1.0, prov=True, lex=False), 7),
    (dict(reward=0.2, ended="answer", final_text="x", reasoning_only=False, support=1.0, prov=True, lex=True), 7),
    (dict(reward=0.19, ended="answer", final_text="x", reasoning_only=False, support=1.0, prov=True, lex=True), 8),
])
def test_category_rules_and_their_order(kwargs, expected) -> None:
    assert report.categorize(**kwargs) == expected


def test_rows_are_joined_by_index_so_a_repeated_question_keeps_its_own_chunk() -> None:
    ctx = make_ctx(data_rows=[("Same?", 5), ("Same?", 9), ("Once?", 20)])
    assert ctx.chunk_id_of({"index": 0, "question": "Same?"}) == 5
    assert ctx.chunk_id_of({"index": 1, "question": "Same?"}) == 9
    assert ctx.chunk_id_of({"index": 7, "question": "Same?"}) is None            # no index match and the text is not unique
    assert ctx.chunk_id_of({"question": "Once?"}) == 20                          # no index, but the text occurs once
    assert ctx.chunk_id_of({"index": 0, "question": "Other?"}) is None           # the index points at another question


def test_analyse_provenance_lexical_and_reasoning_signals() -> None:
    ctx = make_ctx()
    hit = report.analyse(make_row([HIT_TEXT, OTHER_TEXT], reward=0.4), ctx)
    assert hit["prov"] and hit["lex"] and hit["category"] == 7 and hit["n_searches"] == 1
    assert [c["in_window"] for c in hit["chunks"]] == [True, False] and "free return profile" in hit["evidence"]
    wrong_place = report.analyse(make_row([OTHER_TEXT], reward=0.0), make_ctx(data_rows=[("Q?", 10)]))
    assert not wrong_place["prov"] and not wrong_place["lex"] and wrong_place["category"] == 5
    no_search = report.analyse(make_row([], final="I do not know."), ctx)
    assert no_search["category"] == 5 and no_search["n_searches"] == 0
    reasoning = report.analyse(make_row([HIT_TEXT], final=f"<reasoning>{REF}</reasoning>Done.", reward=0.1), ctx)
    assert reasoning["reasoning_only"] and reasoning["category"] == 3


def test_build_aligns_models_by_index_and_counts_every_row() -> None:
    ctx = make_ctx(data_rows=[("Same?", 10), ("Same?", 10)])
    a = [make_row([HIT_TEXT], reward=1.0, index=0, question="Same?"), make_row([OTHER_TEXT], index=1, question="Same?")]
    b = [make_row([HIT_TEXT], reward=0.5, index=0, question="Same?"), make_row([HIT_TEXT], reward=0.0, index=1, question="Same?")]
    questions, counts = report.build([("base", a), ("trained", b)], ctx)
    assert len(questions) == 2 and [q["index"] for q in questions] == [0, 1]       # a repeated question text stays two questions
    assert counts[0][1] == 1 and counts[0][5] == 1 and counts[1][7] == 1 and counts[1][8] == 1
    assert all(sum(c.values()) == 2 for c in counts)
    with pytest.raises(SystemExit):
        report.build([("a", a), ("b", [make_row([], index=0, question="Different?"), b[1]])], ctx)


class _Balance(HTMLParser):
    VOID = {"meta", "input", "br", "hr", "link", "img"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append((tag, self.stack[-3:]))
        else:
            self.stack.pop()


def test_the_page_has_the_definitions_escapes_untrusted_text_and_balanced_tags() -> None:
    ctx = make_ctx()
    evil = "<script>alert('x')</script> Q?"
    rows_a = [make_row([HIT_TEXT], reward=1.0, question=evil, final="<b>bold</b> answer")]
    rows_b = [make_row([OTHER_TEXT], reward=0.0, question=evil, final=None, ended="turn_cap")]
    ctx.data_rows[0] = (evil, 10)
    questions, counts = report.build([("base", rows_a), ("run <1>", rows_b)], ctx)
    slots = [0]
    classes = [report.model_class(lb, i, slots) for i, lb in enumerate(["base", "run <1>"])]
    text = report.page("run & name", ["base", "run <1>"], classes, questions, counts, [Path("a.jsonl"), Path("b.jsonl")])
    for _, short, title, rule, _ in report.CATEGORIES:
        assert report.esc(title) in text and report.esc(rule) in text, title                 # every definition is on the page
    assert "Lexical hit" in text and "Provenance hit" in text and "first one that matches wins" in text
    assert "<script>alert" not in text and "<b>bold</b>" not in text and "run &lt;1&gt;" in text
    assert classes == ["m0", "m1"]                                                           # base is grey, the next model is slot 1
    parser = _Balance()
    parser.feed(text)
    assert not parser.errors and not parser.stack, (parser.errors[:3], parser.stack[:5])


def test_the_chart_numbers_match_the_counts() -> None:
    labels, classes = ["a", "b"], ["m0", "m1"]
    counts = [{c[0]: 0 for c in report.CATEGORIES} for _ in labels]
    counts[0][1], counts[1][1], counts[1][5] = 20, 61, 40
    out = report.chart_html(labels, classes, counts, 148)
    assert 'data-v="61 of 148 (41%)"' in out and 'data-v="40 of 148 (27%)"' in out and "<td class='n'>20 (14%)</td>" in out
    assert out.count('class="bar m1"') == 2 and out.count('class="bar m0"') == 1               # a zero count draws no focusable bar


def test_real_dumps_if_present() -> None:
    run = Path(__file__).resolve().parents[4] / "trainer" / "runs" / "20261021-141800_run2_retrieval"
    chunks = Path(__file__).resolve().parents[4] / "Step2_Files" / "Saved_data" / "chunks.pkl"
    index = Path(__file__).resolve().parent.parent / "chroma_index"
    if not (run / "eval_base.jsonl").exists() or not index.exists():
        pytest.skip("the eval dumps or the index are missing")
    import json
    ctx = report.load_context("validation", index, chunks)
    rows = [json.loads(line) for line in (run / "eval_checkpoint-200.jsonl").read_text().splitlines()]
    questions, counts = report.build([("m", rows)], ctx)
    assert len(questions) == len(rows) == 148 and sum(counts[0].values()) == 148
    assert counts[0][1] == sum(r["reward"] >= 0.999 for r in rows)                           # category 1 is exactly "reward 1.0"
    assert all(a["chunks"] and all(c["span"] for c in a["chunks"]) for a in (q["per_model"][0] for q in questions) if a["n_searches"])
