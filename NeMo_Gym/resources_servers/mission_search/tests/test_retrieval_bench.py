# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/retrieval_bench.py: the OCR repair, the scoring functions, the metrics, and BGE-M3 against the model card."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import retrieval_bench as rb  # noqa: E402

M3_PATH = Path(__file__).resolve().parents[4] / "Embedder" / "bge-m3"


@pytest.mark.parametrize("raw, fixed", [
    ("Apollo l3", "Apollo 13"), ("l:l4 p.m.", "1:14 p.m."), ("190o", "1900"), ("9o0 psi", "900 psi"), ("l0 percent", "10 percent"),
    ("6l-1/2 hours", "61-1/2 hours"), ("O2", "O2"), ("H2O", "H2O"), ("S-IVB", "S-IVB"), ("I think", "I think"), ("Apollo 13", "Apollo 13"),
    ("l", "l"), ("the l3 engine, l2.", "the 13 engine, 12."),
])
def test_repair_ocr(raw, fixed) -> None:
    assert rb.repair_ocr(raw) == fixed


def test_sparse_scores_match_a_brute_force_sum() -> None:
    q = [{1: 0.5, 2: 0.25}, {9: 1.0}]
    d = [{1: 2.0, 3: 1.0}, {2: 4.0, 1: 1.0}, {7: 1.0}]
    out = rb.sparse_scores(q, d)
    expected = [[sum(w * dd.get(t, 0.0) for t, w in qq.items()) for dd in d] for qq in q]
    assert np.allclose(out, expected) and out[1].sum() == 0.0


def test_colbert_scores_match_a_naive_loop() -> None:
    rng = np.random.default_rng(0)
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    q = [unit(rng.normal(size=(t, 8))) for t in (3, 5)]
    d = [unit(rng.normal(size=(t, 8))) for t in (4, 1, 6)]
    out = rb.colbert_scores(q, d)
    naive = [[np.mean([max(float(qv @ dv) for dv in dd) for qv in qq]) for dd in d] for qq in q]
    assert np.allclose(out, naive, atol=1e-5)


def test_combine_uses_the_model_card_weights_and_divides_by_their_sum() -> None:
    parts = {"dense": np.array([0.6260]), "sparse": np.array([0.1956]), "colbert": np.array([0.7796])}
    assert abs(rb.combine(parts, ("dense", "sparse"))[0] - 0.4825) < 1e-3          # the card's 'sparse+dense' number
    assert abs(rb.combine(parts, ("dense", "sparse", "colbert"))[0] - 0.6014) < 1e-3  # the card's 'colbert+sparse+dense'
    assert rb.WEIGHTS == {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}


def test_metrics_on_a_tiny_case() -> None:
    # 3 chunks cover old positions 0-1, 5-6, 10-11. Query 0's window is 5..7 (chunk 1 overlaps), query 1's window is 20..22 (none).
    starts, ends = np.array([0, 5, 10]), np.array([1, 6, 11])
    windows = np.array([[5, 7], [20, 22]])
    scores = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])                           # query 0 ranks chunk 1 first
    docs = ["alpha beta gamma", "free return profile", "other words"]
    res, hits = rb.evaluate(scores, windows, starts, ends, ["return profile", "return profile"], docs)
    assert res["prov@2"] == 0.5 and res["prov@5"] == 0.5 and res["mrr@5"] == 0.5     # query 0: rank 1 -> 1.0; query 1: 0
    assert res["cov@2"] == 0.5                                                      # only query 0 retrieved the words
    assert hits[2].tolist() == [True, False]
    worse = rb.evaluate(-scores, windows, starts, ends, ["return profile"] * 2, docs)[1]
    mean, se = rb.paired(hits[2], worse[2])
    assert mean == -0.5 and se > 0


@pytest.mark.skipif(not (M3_PATH / "pytorch_model.bin").exists(), reason="BGE-M3 is not in Embedder/bge-m3")
def test_bge_m3_reproduces_the_model_card_scores(bge_m3) -> None:
    """The README of BGE-M3 prints the scores of four sentence pairs in every mode (computed with fp16: allow a small difference)."""
    m3 = bge_m3
    s1 = ["What is BGE M3?", "Defination of BM25"]
    s2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
          "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
    q, d = m3.encode(s1, max_length=128), m3.encode(s2, max_length=128)
    dense, sparse, colbert = rb.dense_scores(q[0], d[0]), rb.sparse_scores(q[1], d[1]), rb.colbert_scores(q[2], d[2])
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]                                         # the card's order: for i in s1 for j in s2
    take = lambda M: [float(M[i, j]) for i, j in pairs]
    assert np.allclose(take(dense), [0.6260, 0.3474, 0.3499, 0.6782], atol=0.01)
    assert np.allclose(take(sparse), [0.1956, 0.0088, 0.0, 0.1803], atol=0.01)
    assert np.allclose(take(colbert), [0.7796, 0.4621, 0.4524, 0.7899], atol=0.01)
    both = rb.combine({"dense": dense, "sparse": sparse, "colbert": colbert}, ("dense", "sparse", "colbert"))
    assert np.allclose(take(both), [0.6014, 0.3256, 0.3209, 0.6233], atol=0.01)
    assert np.allclose(np.linalg.norm(q[0], axis=1), 1.0, atol=1e-4) and all(t not in m3.special for t in q[1][0])
