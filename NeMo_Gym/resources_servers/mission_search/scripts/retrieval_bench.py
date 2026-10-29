"""Retrieval benchmark: does a different embedder return better chunks? CPU only, no training, no GPU.

Run it with the SERVER venv, from the repository root:

    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/retrieval_bench.py \\
        --model-queries trainer/runs/20261021-141800_run2_retrieval/eval_checkpoint-200.jsonl
    add --limit 100 for a quick check of the pipeline

What it compares (every retriever sees the same chunks and the same queries):
  NoInstruct (L2)   the embedder that is in the server today (mean pooling for queries, [CLS] for chunks, L2 distance)
  BGE-M3 dense      cosine of the normalized [CLS] vectors
  BGE-M3 sparse     lexical weights: sum over shared tokens of query weight x chunk weight
  BGE-M3 colbert    multi-vector: for each query token the best matching chunk token, averaged
  dense+sparse, all three   the weighted sum with the weights of the model card: dense 0.4, sparse 0.2, colbert 0.4
                            (divided by the sum of the weights that are used, as the model card's own numbers show)
BGE-M3 is implemented here from the model files (the encoder plus colbert_linear.pt and sparse_linear.pt), the way the model card
describes it. tests/test_retrieval_bench.py checks it against the example scores printed in the model card.

Query sets: all 1,434 questions of questions.json; the 148 validation questions; and, with --model-queries, the queries that a
trained model actually wrote for the validation questions. Each runs on the raw text and on text with the OCR digit repair
(chunks, queries and references all repaired together: repairing only the chunks would break the match).

Metrics, at top 2 / 3 / 5:
  provenance  a retrieved chunk overlaps the 3-chunk window that the question was written from (no wording involved)
  coverage    the retrieved text holds >= 60% of the reference's content words (the `retrieved` metric of eval.py)
  MRR@5       1 / rank of the first chunk that overlaps the source window (0 if none in the top 5)
  delta       paired change of provenance against NoInstruct, with its standard error
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
REPO_ROOT = SERVER_DIR.parents[2]
sys.path.insert(0, str(SERVER_DIR))
from app import coverage  # noqa: E402
from m3 import WEIGHTS, BGEM3, colbert_scores, combine, dense_scores, sparse_scores  # noqa: E402,F401  (the code that the server runs)

KS = (2, 3, 5)
HIT_COVERAGE = 0.6

# ---------------------------------------------------------------------------------------------------------------
# OCR repair: a letter that looks like a digit, inside a token that has a digit (l3 -> 13, 190o -> 1900)
# ---------------------------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"(?<![A-Za-z])[0-9oOlI][0-9oOlI.:,/\-]*(?![A-Za-z])")
_KEEP = {"O2"}                                                  # oxygen is not "02"


def _fix_token(token: str) -> str:
    core = token.rstrip(".,:;/-")
    if not re.search(r"\d", token) or not re.search(r"[oOlI]", token) or len(core) < 2 or core in _KEEP:
        return token
    return token.translate(str.maketrans("oOlI", "0011"))


def repair_ocr(text: str) -> str:
    return _TOKEN.sub(lambda m: _fix_token(m.group()), text)


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def evaluate(scores, windows, starts, ends, refs, doc_texts):
    """scores [Q, D] (higher = better). Returns metrics and the per-question provenance hits (for paired comparisons)."""
    top = np.argsort(-scores, axis=1)[:, :max(KS)]
    overlap = (starts[top] <= windows[:, 1:2]) & (ends[top] >= windows[:, 0:1])          # [Q, kmax]
    out = {}
    for k in KS:
        out[f"prov@{k}"] = float(overlap[:, :k].any(1).mean())
    first = np.where(overlap.any(1), overlap.argmax(1) + 1, 0)
    out["mrr@5"] = float(np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0).mean())
    for k in KS:
        out[f"cov@{k}"] = float(np.mean([coverage("\n".join(doc_texts[j] for j in top[i, :k]), refs[i]) >= HIT_COVERAGE for i in range(len(refs))]))
    return out, {k: overlap[:, :k].any(1) for k in KS}


def paired(base: np.ndarray, other: np.ndarray):
    d = other.astype(float) - base.astype(float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
def question_text(row):
    user = [m["content"] for m in row["responses_create_params"]["input"] if m["role"] == "user"][-1]
    return user.rsplit("Question:", 1)[-1].strip()


def load_query_sets(questions_path, split, model_queries, limit):
    sets = {}
    pool = json.load(open(questions_path))
    sets[f"all questions ({len(pool)})"] = [{"query": q["question"], "ref": q["answer"], "chunk_id": q["chunk_id"]} for q in pool]
    val_rows = [json.loads(l) for l in (SERVER_DIR / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    sets[f"{split} questions ({len(val_rows)})"] = [{"query": question_text(r), "ref": r["answer"], "chunk_id": r["chunk_id"]} for r in val_rows]
    if model_queries:
        rows = [json.loads(l) for l in Path(model_queries).read_text(encoding="utf-8").splitlines() if l]
        items = []
        for r in rows:
            calls = [c for c in r["output"] if c["type"] == "function_call"]
            i = r.get("index")
            if not calls or i is None or question_text(val_rows[i]) != r["question"]:
                continue
            items.append({"query": json.loads(calls[0]["arguments"])["query"], "ref": r["answer"], "chunk_id": val_rows[i]["chunk_id"]})
        sets[f"{split}, the trained model's own first query ({len(items)})"] = items
    if limit:
        sets = {f"{k.split(' (')[0]} (first {min(limit, len(v))})": v[:limit] for k, v in sets.items()}
    return sets


def load_chunks(chroma_dir):
    import chromadb
    got = chromadb.PersistentClient(path=str(chroma_dir)).get_collection("example_collection").get(include=["documents", "metadatas"])
    return got["documents"], np.array([m["chunk_start"] for m in got["metadatas"]]), np.array([m["chunk_end"] for m in got["metadatas"]])


# ---------------------------------------------------------------------------------------------------------------
def fmt_table(name, results, base="NoInstruct (L2)"):
    lines = [f"\n{name}", f"  {'retriever':<22}" + "".join(f"{'prov@' + str(k):>8}" for k in KS) + f"{'MRR@5':>8}" + "".join(f"{'cov@' + str(k):>8}" for k in KS) + f"{'prov@2 vs base':>20}{'prov@5 vs base':>20}"]
    for r, m in results.items():
        d2 = "" if r == base else "{:+.3f} ± {:.3f}".format(*m["paired@2"])
        d5 = "" if r == base else "{:+.3f} ± {:.3f}".format(*m["paired@5"])
        lines.append(f"  {r:<22}" + "".join(f"{m[f'prov@{k}']:>8.3f}" for k in KS) + f"{m['mrr@5']:>8.3f}" + "".join(f"{m[f'cov@{k}']:>8.3f}" for k in KS) + f"{d2:>20}{d5:>20}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m3", type=Path, default=REPO_ROOT / "Embedder" / "bge-m3")
    ap.add_argument("--questions", type=Path, default=REPO_ROOT / "Step2_Files" / "Saved_data" / "questions.json")
    ap.add_argument("--chroma-dir", type=Path, default=SERVER_DIR / "chroma_index")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--model-queries", type=Path, default=None, help="an eval dump: also test the queries the trained model wrote")
    ap.add_argument("--no-repair", action="store_true", help="skip the repaired-text variant")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N queries of each set (quick check)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from embeddings import CustomHuggingFaceEmbeddings
    docs, starts, ends = load_chunks(args.chroma_dir)
    sets = load_query_sets(args.questions, args.split, args.model_queries, args.limit)
    print(f"[bench] {len(docs)} chunks; query sets: " + ", ".join(sets))
    m3, noi = BGEM3(args.m3), CustomHuggingFaceEmbeddings()
    results, timing = {}, {}
    for variant in ("raw",) + (() if args.no_repair else ("repaired",)):
        fix = repair_ocr if variant == "repaired" else (lambda t: t)
        d_text = [fix(d) for d in docs]
        t0 = time.time(); d_m3 = m3.encode(d_text, max_length=512); timing[f"M3 chunks ({variant})"] = (time.time() - t0) / len(docs)
        t0 = time.time(); D_noi = np.array(noi.embed_documents(d_text), dtype=np.float32); timing[f"NoInstruct chunks ({variant})"] = (time.time() - t0) / len(docs)
        print(f"[bench] {variant}: chunks encoded (BGE-M3 {1000 * timing[f'M3 chunks ({variant})']:.0f} ms/chunk, NoInstruct {1000 * timing[f'NoInstruct chunks ({variant})']:.0f} ms/chunk)")
        cache = {}
        for set_name, items in sets.items():
            queries = [fix(it["query"]) for it in items]
            refs = [fix(it["ref"]) for it in items]
            windows = np.array([[it["chunk_id"] - 1, it["chunk_id"] + 1] for it in items])
            need = [q for q in dict.fromkeys(queries) if q not in cache]
            if need:
                t0 = time.time(); enc = m3.encode(need, max_length=128); timing[f"M3 queries ({variant})"] = (time.time() - t0) / len(need)
                t0 = time.time(); vec = [noi.embed_query(q) for q in need]; timing[f"NoInstruct queries ({variant})"] = (time.time() - t0) / len(need)
                for n_i, q in enumerate(need):
                    cache[q] = (enc[0][n_i], enc[1][n_i], enc[2][n_i], np.array(vec[n_i], dtype=np.float32))
            q_dense = np.stack([cache[q][0] for q in queries]); q_sparse = [cache[q][1] for q in queries]; q_col = [cache[q][2] for q in queries]
            V = np.stack([cache[q][3] for q in queries])
            parts = {"dense": dense_scores(q_dense, d_m3[0]), "sparse": sparse_scores(q_sparse, d_m3[1]), "colbert": colbert_scores(q_col, d_m3[2])}
            scorers = {"NoInstruct (L2)": -((V ** 2).sum(1)[:, None] + (D_noi ** 2).sum(1)[None, :] - 2 * V @ D_noi.T),
                       "BGE-M3 dense": parts["dense"], "BGE-M3 sparse": parts["sparse"], "BGE-M3 colbert": parts["colbert"],
                       "BGE-M3 dense+sparse": combine(parts, ("dense", "sparse")), "BGE-M3 all three": combine(parts, ("dense", "sparse", "colbert"))}
            res, hits = {}, {}
            for name, S in scorers.items():
                res[name], hits[name] = evaluate(S, windows, starts, ends, refs, d_text)
            for name in res:
                for k in (2, 5):
                    res[name][f"paired@{k}"] = paired(hits["NoInstruct (L2)"][k], hits[name][k])
            results.setdefault(set_name, {})[variant] = res
            print(fmt_table(f"{set_name} | text: {variant}", res))
    print("\n[bench] encoding cost on this CPU, per text:", {k: f"{1000 * v:.0f} ms" for k, v in timing.items()})
    out = args.out or REPO_ROOT / "trainer" / "runs" / "retrieval_bench" / f"bench_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": WEIGHTS, "ks": KS, "results": results, "timing_s_per_text": timing, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2, default=str))
    print(f"[bench] saved {out}")


if __name__ == "__main__":
    main()
