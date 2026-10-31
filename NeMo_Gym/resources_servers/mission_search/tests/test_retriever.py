# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for chunks.py, m3.py and retriever.py: the chunk store, one-query scoring, the memory-safe cache, and the BGE-M3 retriever.

The tests named test_model_* use BGE-M3 (2.3 GB of memory), loaded once for the whole run by the `bge_m3` fixture in conftest.py. The test with `real_scale` in its name checks the loader on arrays of the
real size in a child process with a hard memory limit: a loader that copies the arrays fails there, and the machine is not at risk.

How to run (from this folder). `ulimit -v` limits the virtual address space, not the resident memory:
    all tests:       (ulimit -v 25165824; .venv/bin/python -m pytest tests)                                        # measured 2026-10-29: 78 pass, 2.7 GB resident, 16.3 GB virtual
    no-model tests:  (ulimit -v 4194304;  .venv/bin/python -m pytest tests/test_retriever.py -k "not test_model")    # measured 2026-10-29: 0.5 GB resident
The cap of 24 GB is 1.5 times the measured virtual peak (16.3 GB), and well under the 31 GB of RAM that WSL gives, so a runaway stops long before the
machine is at risk. A cap of 2 times the resident peak (5.4 GB) would stop the tests with no real memory problem: the virtual size is 6 times the resident size here.
Measure again (VmPeak and VmHWM from /proc/self/status at the end of the run) when tests or imports are added."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))
import chunks as chunk_store  # noqa: E402
import m3  # noqa: E402
import retriever  # noqa: E402

M3_PATH = SERVER_DIR.parents[2] / "Embedder" / "bge-m3"
INDEX_DIR = SERVER_DIR / "chroma_index"
needs_model = pytest.mark.skipif(not (M3_PATH / "pytorch_model.bin").exists(), reason="BGE-M3 is not in Embedder/bge-m3")


def mk_chunks(texts):
    return [chunk_store.Chunk(id=f"chunk-{i}", index=i, text=t, start=i, end=i, n_merged=1) for i, t in enumerate(texts)]


def synthetic(n_docs=6, seed=0):
    rng = np.random.default_rng(seed)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    dense = unit(rng.normal(size=(n_docs, 16))).astype(np.float32)
    # real weights come from float32 tensors, so the test weights are float32 values too
    sparse = [{int(t): float(np.float32(w)) for t, w in zip(rng.choice(30, size=5, replace=False), rng.random(5))} for _ in range(n_docs)]
    colbert = [unit(rng.normal(size=(int(rng.integers(3, 9)), 16))).astype(np.float32) for _ in range(n_docs)]
    return dense, sparse, colbert


def fake_model_dir(path, revision="BAAI/bge-m3\nabc123\n", weights=b"x" * 10):
    path.mkdir(parents=True, exist_ok=True)
    (path / "pytorch_model.bin").write_bytes(weights)
    for name in ("config.json", "tokenizer.json", "colbert_linear.pt", "sparse_linear.pt"):
        (path / name).write_text(name)
    if revision is not None:
        (path / "REVISION").write_text(revision)
    return path


# ---- chunk store -----------------------------------------------------------------------------------------------
def test_format_results_is_the_autodidact_format() -> None:
    assert retriever.format_results(["a", "b\nc"]) == "Result 1:\na\n------\nResult 2:\nb\nc\n------\n"
    assert retriever.format_results(mk_chunks(["a", "b"])) == "Result 1:\na\n------\nResult 2:\nb\n------\n"
    assert retriever.format_results([]) == ""


def test_chunk_store_round_trip_and_checks(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    original = mk_chunks(["first chunk", "second chunk with ünïcode"])
    chunk_store.write_chunks(path, original, {"chunking": {"min_chars": 100}})
    chunks, meta = chunk_store.read_chunks(path)
    assert chunks == original and meta == {"chunking": {"min_chars": 100}} and not list(tmp_path.glob("*.tmp"))
    rows = path.read_text().splitlines()
    bad = {"_meta line": rows[1:], "indexes are not": [rows[0], rows[2], rows[1]],       # the text is the reason that the error must give
           "duplicate id": [rows[0], rows[1], rows[2].replace("chunk-1", "chunk-0")],
           "an empty text": [rows[0], rows[1], json.dumps({**json.loads(rows[2]), "text": "  "})]}
    for reason, lines in bad.items():
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match=reason):
            chunk_store.read_chunks(path)


def test_load_chunks_reads_the_file_and_writes_nothing(tmp_path) -> None:
    path = tmp_path / "chunks.jsonl"
    chunk_store.write_chunks(path, mk_chunks(["a", "b", "c"]), {"chunking": {"min_chars": 100}})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    chunks, meta = retriever.load_chunks(path)
    assert chunks == mk_chunks(["a", "b", "c"]) and meta == {"chunking": {"min_chars": 100}}
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_load_chunks_stops_with_the_export_command_when_the_file_is_missing_and_creates_nothing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="export_chunks.py"):
        retriever.load_chunks(tmp_path / "data" / "chunks.jsonl")
    assert list(tmp_path.iterdir()) == []


# ---- scoring ---------------------------------------------------------------------------------------------------
def queries():
    dense, sparse, colbert = synthetic(n_docs=2, seed=1)
    return dense[:1], sparse[:1], colbert[:1]


def test_one_query_scores_equal_the_batch_scores() -> None:
    dense, sparse, colbert = synthetic()
    q_dense, q_sparse, q_colbert = queries()
    index = m3.M3Index(m3.pack(dense, sparse, colbert))
    one = index.scores(q_dense[0], q_sparse[0], q_colbert[0])
    assert np.allclose(one["dense"], m3.dense_scores(q_dense, dense)[0], atol=1e-5)
    assert np.allclose(one["sparse"], m3.sparse_scores(q_sparse, sparse)[0], atol=1e-5)
    assert np.allclose(one["colbert"], m3.colbert_scores(q_colbert, colbert)[0], atol=1e-5)
    both = m3.combine({k: v[None, :] for k, v in one.items()}, ("dense", "sparse", "colbert"))[0]
    assert np.allclose(index.hybrid(q_dense[0], q_sparse[0], q_colbert[0]), both, atol=1e-5)


def test_colbert_blocks_give_the_same_scores(monkeypatch) -> None:
    dense, sparse, colbert = synthetic(n_docs=9)
    index = m3.M3Index(m3.pack(dense, sparse, colbert))
    q = queries()[2][0]
    whole = index._colbert(q)
    for tokens_per_block in (1, 7, 20):                                    # a block holds at least one chunk: very small limits still work
        monkeypatch.setattr(m3, "MAX_TEMP_BYTES", 4 * len(q) * tokens_per_block)
        assert np.allclose(index._colbert(q), whole, atol=1e-6), tokens_per_block


def test_sparse_lookup_handles_unknown_tokens_and_empty_chunks() -> None:
    dense, sparse, colbert = synthetic(n_docs=3)
    sparse[1] = {}                                                            # a chunk without sparse entries
    index = m3.M3Index(m3.pack(dense, sparse, colbert))
    out = index._sparse({10_000_000: 1.0, -5: 1.0})                           # tokens that no chunk has, and an invalid id
    assert out.shape == (3,) and not out.any() and not index._sparse({}).any()
    token = next(iter(sparse[0]))
    assert index._sparse({token: 2.0})[1] == 0.0 and index._sparse({token: 2.0})[0] == pytest.approx(2.0 * sparse[0][token])


def test_sparse_lookup_finds_the_largest_token_id_and_ignores_negative_ids() -> None:
    dense, _, colbert = synthetic(n_docs=3)
    index = m3.M3Index(m3.pack(dense, [{0: 1.0, 3: 0.5}, {3: 2.0}, {1: 1.0}], colbert))          # the largest token id is 3
    assert index._sparse({3: 1.0}).tolist() == [0.5, 2.0, 0.0]
    assert not index._sparse({-2: 1.0}).any()                                    # an index from the end of the list would give token 3 here


def test_arrays_that_do_not_fit_together_are_refused() -> None:
    arrays = m3.pack(*synthetic())
    arrays["colbert_offsets"] = arrays["colbert_offsets"][:-1]
    with pytest.raises(ValueError, match="do not fit together"):
        m3.M3Index(arrays)


# ---- the cache -------------------------------------------------------------------------------------------------
def test_cache_round_trip_is_exact_memory_mapped_and_records_the_dtypes(tmp_path) -> None:
    arrays = m3.pack(*synthetic())
    model = fake_model_dir(tmp_path / "bge-m3")
    manifest = m3.build_manifest(mk_chunks(list("abcdef")), model, {"min_chars": 100})
    directory = tmp_path / "m3-test"
    m3.save_cache(directory, arrays, manifest)
    assert not list(tmp_path.glob("*.tmp")) and sorted(p.name for p in directory.iterdir()) == sorted([f"{n}.npy" for n in arrays] + ["manifest.json"])
    loaded = m3.load_cache(directory, manifest)
    assert all(isinstance(a, np.memmap) for a in loaded.values())              # memory-mapped: nothing is copied
    assert all(np.array_equal(loaded[n], arrays[n]) for n in arrays)
    recorded = json.loads((directory / "manifest.json").read_text())["arrays"]
    assert {n: recorded[n]["dtype"] for n in ("dense", "colbert", "sparse_data", "sparse_indices", "colbert_offsets")} == {
        "dense": "float32", "colbert": "float32", "sparse_data": "float32", "sparse_indices": "int32", "colbert_offsets": "int64"}
    # chunks of one array: the first version made a full copy for each chunk (127 GB for 510 chunks)
    index = m3.M3Index(loaded)
    assert index.n == 6


def test_cache_is_not_used_when_inputs_differ_or_files_are_damaged(tmp_path) -> None:
    arrays = m3.pack(*synthetic())
    model = fake_model_dir(tmp_path / "bge-m3")
    manifest = m3.build_manifest(mk_chunks(list("abcdef")), model)
    directory = tmp_path / "m3-test"
    m3.save_cache(directory, arrays, manifest)
    assert m3.load_cache(directory, manifest) is not None
    assert m3.load_cache(directory, m3.build_manifest(mk_chunks(list("abcdeX")), model)) is None      # other text
    assert m3.load_cache(tmp_path / "does-not-exist", manifest) is None
    (directory / "manifest.json").write_text("{not json")
    assert m3.load_cache(directory, manifest) is None
    m3.save_cache(directory, arrays, manifest)
    data = (directory / "colbert.npy").read_bytes()
    (directory / "colbert.npy").write_bytes(data[: len(data) // 2])                                    # a truncated file
    assert m3.load_cache(directory, manifest) is None
    m3.save_cache(directory, arrays, manifest)
    (directory / "dense.npy").unlink()
    assert m3.load_cache(directory, manifest) is None
    m3.save_cache(directory, arrays, manifest)
    record = json.loads((directory / "manifest.json").read_text())
    record["arrays"]["dense"]["dtype"] = "float16"                                                       # the manifest says another dtype
    (directory / "manifest.json").write_text(json.dumps(record))
    assert m3.load_cache(directory, manifest) is None


def test_a_cache_over_the_size_limit_is_refused_before_anything_is_read(tmp_path, monkeypatch) -> None:
    arrays = m3.pack(*synthetic())
    manifest = m3.build_manifest(mk_chunks(list("abcdef")), fake_model_dir(tmp_path / "bge-m3"))
    m3.save_cache(tmp_path / "m3-test", arrays, manifest)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("an array was read although the cache is over the limit"))
    with pytest.raises(m3.CacheTooLargeError, match="Not loaded"):
        m3.load_cache(tmp_path / "m3-test", manifest, max_bytes=100)


def test_the_key_changes_with_every_input(tmp_path, monkeypatch) -> None:
    chunks = mk_chunks(["alpha", "beta", "gamma"])
    model = fake_model_dir(tmp_path / "bge-m3")
    base = m3.cache_key(m3.build_manifest(chunks, model, {"min_chars": 100, "max_chars": 1500}))
    assert base == m3.cache_key(m3.build_manifest(chunks, model, {"min_chars": 100, "max_chars": 1500}))
    manifest = m3.build_manifest(chunks, model)
    assert manifest["dtypes"]["colbert"] == "float32" and manifest["embedder"]["revision"] == "abc123" and manifest["doc_max_length"] == 512

    def key(**kw):
        return m3.cache_key(m3.build_manifest(kw.get("chunks", chunks), kw.get("model", model), kw.get("chunking", {"min_chars": 100, "max_chars": 1500})))

    changed = {"chunk text": key(chunks=mk_chunks(["alpha", "beta", "gamma!"])), "chunk order": key(chunks=mk_chunks(["beta", "alpha", "gamma"])),
               "chunk count": key(chunks=mk_chunks(["alpha", "beta"])), "chunk id": key(chunks=[chunk_store.Chunk("x-0", 0, "alpha", 0, 0, 1)] + chunks[1:]),
               "chunking config": key(chunking={"min_chars": 200, "max_chars": 1500}), "chunking missing": key(chunking={}),
               "embedder revision": key(model=fake_model_dir(tmp_path / "m2", revision="BAAI/bge-m3\nother\n")),
               "embedder name": key(model=fake_model_dir(tmp_path / "m3", revision="OTHER/model\nabc123\n")),
               "no revision file": key(model=fake_model_dir(tmp_path / "m4", revision=None)),
               "weights size": key(model=fake_model_dir(tmp_path / "m5", weights=b"x" * 11))}
    other_head = fake_model_dir(tmp_path / "m6")
    (other_head / "colbert_linear.pt").write_text("another head")
    changed["colbert head"] = key(model=other_head)
    for name, value in changed.items():
        assert value != base, name
    for name, attribute, value in (("max length", "DOC_MAX_LENGTH", 256), ("format version", "FORMAT", 99)):
        monkeypatch.setattr(m3, attribute, value)
        assert key() != base, name
        monkeypatch.undo()
    assert len(set(changed.values())) == len(changed)                       # no two changes give the same key


def test_a_real_scale_load_stays_small_under_a_hard_memory_cap(tmp_path) -> None:
    """The real cache: 510 chunks, about 60,000 ColBERT vectors of 1024 numbers (249 MB). A loader that makes one copy for each chunk
    needs 127 GB. The child process may use 3 GB of address space; this loader needs about a tenth of that."""
    child = textwrap.dedent(f"""
        import re, resource, sys, time
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
        sys.path.insert(0, {str(SERVER_DIR)!r})
        import numpy as np, m3
        rng = np.random.default_rng(0)
        n, tokens = 510, 60817
        colbert = rng.standard_normal((tokens, 1024), dtype=np.float32)
        colbert /= np.linalg.norm(colbert, axis=1, keepdims=True)
        arrays = {{"dense": rng.standard_normal((n, 1024), dtype=np.float32), "colbert": colbert,
                   "colbert_offsets": np.linspace(0, tokens, n + 1).astype(np.int64),
                   "sparse_indptr": (np.arange(n + 1) * 61).astype(np.int64),
                   "sparse_indices": rng.integers(0, 250002, n * 61).astype(np.int32), "sparse_data": rng.random(n * 61).astype(np.float32)}}
        manifest = {{"format": m3.FORMAT}}
        m3.save_cache({str(tmp_path / "m3-big")!r}, arrays, manifest)
        del arrays, colbert
        t = time.time()
        loaded = m3.load_cache({str(tmp_path / "m3-big")!r}, manifest)
        index = m3.M3Index(loaded)
        load_s = time.time() - t
        for _ in range(20):
            q = rng.standard_normal((15, 1024), dtype=np.float32)
            index.hybrid(rng.standard_normal(1024, dtype=np.float32), {{int(k): 0.1 for k in rng.integers(0, 250002, 12)}}, q)
        # VmHWM is the peak of this child alone. ru_maxrss is inherited from the parent at fork time, so it shows the parent's size (2 GB after a model test).
        peak_mb = int(re.search(r"^VmHWM:\\s+(\\d+) kB", open("/proc/self/status").read(), re.M).group(1)) // 1024
        print("OK", all(isinstance(a, np.memmap) for a in loaded.values()), round(load_s, 2), peak_mb)
    """)
    done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    _, mapped, load_seconds, peak_mb = done.stdout.split()
    assert mapped == "True" and float(load_seconds) < 20 and int(peak_mb) < 1500, done.stdout          # peak includes the 249 MB that the child generated


def test_missing_model_gives_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="BGE-M3 not found"):
        retriever.BGEM3Retriever(mk_chunks(["a"]), tmp_path / "nothing")


class FakeModel:
    """Stands in for m3.BGEM3: the vectors depend on the words of a text (one slot for each known word, one for all other words), and every call is recorded."""
    WORDS = "engine cut off early battery energy at landing splashdown in the ocean thermal control mode".split()

    def __init__(self):
        self.calls = []

    def encode(self, texts, max_length):
        import re
        self.calls.append((len(texts), max_length))
        dense, sparse, colbert = [], [], []
        for text in texts:
            slots = [self.WORDS.index(w) if w in self.WORDS else len(self.WORDS) for w in re.findall(r"[a-z]+", text.lower())] or [0]
            tokens = np.eye(len(self.WORDS) + 1, dtype=np.float32)[slots]
            dense.append(tokens.sum(0) / np.linalg.norm(tokens.sum(0)))
            sparse.append({int(s): 1.0 for s in slots})
            colbert.append(tokens)
        return np.array(dense, dtype=np.float32), sparse, colbert


def test_a_given_model_is_used_for_chunks_and_queries_and_no_other_model_is_loaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(m3, "BGEM3", lambda *a, **k: pytest.fail("the retriever loaded a model although it was given one"))
    model = FakeModel()
    texts = ["engine cut off early", "battery energy at landing", "splashdown in the ocean", "thermal control mode"]
    model_dir, cache = fake_model_dir(tmp_path / "model"), tmp_path / "cache"
    first = retriever.BGEM3Retriever(mk_chunks(texts), model_dir, cache, log=lambda *_: None, model=model)
    assert model.calls == [(4, m3.DOC_MAX_LENGTH)] and not first.loaded_from_cache
    assert [c.id for c in first.search("battery", 1)] == ["chunk-1"] and model.calls[-1] == (1, m3.QUERY_MAX_LENGTH)
    second = retriever.BGEM3Retriever(mk_chunks(texts), model_dir, cache, log=lambda *_: None, model=model)
    assert second.loaded_from_cache and len(model.calls) == 2                       # the cache hit encodes no chunk, and the same model is shared
    assert [c.id for c in second.search("splashdown ocean", 1)] == ["chunk-2"]


def test_corpus_searcher_gives_its_model_to_the_bge_m3_retriever(tmp_path, monkeypatch) -> None:
    sys.path.insert(0, str(SERVER_DIR.parent))
    from app import CorpusSearcher
    monkeypatch.setattr(m3, "BGEM3", lambda *a, **k: pytest.fail("the searcher loaded a model although it was given one"))
    chunk_store.write_chunks(tmp_path / "chunks.jsonl", mk_chunks(["engine cut off early", "battery energy at landing", "thermal control mode"]), {"chunking": {}})
    (tmp_path / "index").mkdir()
    model = FakeModel()
    searcher = CorpusSearcher(tmp_path / "index", "col", "unused", 2, retriever="bge_m3", m3_dir=fake_model_dir(tmp_path / "model"),
                              m3_cache_dir=tmp_path / "cache", chunks_path=tmp_path / "chunks.jsonl", m3_model=model)
    text = searcher.search("battery energy")
    assert text.startswith("Result 1:\nbattery energy at landing\n------\n") and text.count("------\n") == 2 and model.calls


def test_bge_m3_searcher_starts_and_searches_without_a_chroma_folder(tmp_path) -> None:
    # A clean clone has no chroma_index: git ignores it, and build_index.py makes it from files that are not in the repository. The BGE-M3 path never reads it.
    sys.path.insert(0, str(SERVER_DIR.parent))
    from app import CorpusSearcher
    chunk_store.write_chunks(tmp_path / "chunks.jsonl", mk_chunks(["engine cut off early", "battery energy at landing", "thermal control mode"]), {"chunking": {}})
    searcher = CorpusSearcher(tmp_path / "no_such_index", "col", "unused", 2, retriever="bge_m3", m3_dir=fake_model_dir(tmp_path / "model"),
                              m3_cache_dir=tmp_path / "cache", chunks_path=tmp_path / "chunks.jsonl", m3_model=FakeModel())
    assert searcher.search("battery energy").startswith("Result 1:\nbattery energy at landing\n------\n")
    assert not (tmp_path / "no_such_index").exists()                      # the searcher did not make the folder either


def test_chroma_searcher_still_stops_with_the_build_command_when_the_folder_is_missing(tmp_path) -> None:
    sys.path.insert(0, str(SERVER_DIR.parent))
    from app import CorpusSearcher
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        CorpusSearcher(tmp_path / "no_such_index", "col", "unused", 2, retriever="chroma")
    assert not (tmp_path / "no_such_index").exists()


def test_config_keeps_the_old_behaviour_by_default() -> None:
    sys.path.insert(0, str(SERVER_DIR.parent))
    from app import MissionSearchResourcesServerConfig
    config = MissionSearchResourcesServerConfig(host="", port=0, entrypoint="", name="")
    assert (config.retriever, config.top_k, config.chunks_path) == ("chroma", 2, "data/chunks.jsonl")
    with pytest.raises(Exception):
        MissionSearchResourcesServerConfig(host="", port=0, entrypoint="", name="", retriever="other")


# ---- with the real model ---------------------------------------------------------------------------------------
@needs_model
def test_model_bge_m3_retriever_finds_the_related_text_and_uses_its_cache(tmp_path, bge_m3) -> None:
    texts = ["The center engine of the second stage cut off about 132 seconds early.",
             "The crew used the lunar module as a lifeboat after the oxygen tank failed.",
             "The command module splashed down in the South Pacific on April 17.",
             "Battery energy at landing was 29 ampere-hours.",
             "Thermal control mode required a yaw maneuver each hour."]
    chunks, logs = mk_chunks(texts), []
    first = retriever.BGEM3Retriever(chunks, M3_PATH, tmp_path, log=logs.append, model=bge_m3)
    assert not first.loaded_from_cache and any("encoding" in line for line in logs)
    second = retriever.BGEM3Retriever(chunks, M3_PATH, tmp_path, log=logs.append, model=bge_m3)
    assert second.loaded_from_cache and len([p for p in tmp_path.iterdir() if p.name.startswith("m3-")]) == 1
    for r in (first, second):
        assert [c.text for c in r.search("How many seconds early did the center engine cut off?", 2)][0] == texts[0]
        assert [c.id for c in r.search("In which ocean did the command module splash down?", 1)] == ["chunk-2"]
    assert first.search("battery energy", 3) == second.search("battery energy", 3)           # the cache gives the same ranking
    other = retriever.BGEM3Retriever(chunks[:4], M3_PATH, tmp_path, log=logs.append, model=bge_m3)
    assert other.loaded_from_cache is False and any("other cache folders" in line for line in logs)   # other chunks: new cache, a warning about the old one


@needs_model
@pytest.mark.skipif(not INDEX_DIR.is_dir(), reason="chroma_index is missing")
def test_model_the_server_retriever_ranks_like_the_benchmark(tmp_path, bge_m3) -> None:
    """Same chunks, same queries: the one-query retriever and the batch functions of the benchmark give the same top 3."""
    chunks = chunk_store.chunks_from_chroma(INDEX_DIR, "example_collection")[:40]
    rows = [json.loads(line) for line in (SERVER_DIR / "data" / "validation.jsonl").read_text().splitlines()[:8]]
    qs = [[m["content"] for m in r["responses_create_params"]["input"] if m["role"] == "user"][-1].rsplit("Question:", 1)[-1].strip() for r in rows]
    server = retriever.BGEM3Retriever(chunks, M3_PATH, tmp_path, log=lambda *_: None, model=bge_m3)
    model, texts = bge_m3, [c.text for c in chunks]
    d, s, c = model.encode(texts, max_length=m3.DOC_MAX_LENGTH)
    qd, qs_, qc = model.encode(qs, max_length=m3.QUERY_MAX_LENGTH)
    batch = m3.combine({"dense": m3.dense_scores(qd, d), "sparse": m3.sparse_scores(qs_, s), "colbert": m3.colbert_scores(qc, c)}, ("dense", "sparse", "colbert"))
    for i, q in enumerate(qs):
        assert [x.text for x in server.search(q, 3)] == [texts[j] for j in batch[i].argsort()[::-1][:3]], q


@needs_model
def test_model_corpus_searcher_with_bge_m3_returns_the_tool_format(bge_m3) -> None:
    cache = SERVER_DIR / "m3_cache"
    chunks_file = SERVER_DIR / "data" / "chunks.jsonl"
    if not INDEX_DIR.is_dir() or not chunks_file.exists() or not list(cache.glob("m3-*/manifest.json")):
        pytest.skip("needs the index, data/chunks.jsonl (scripts/export_chunks.py) and a cache from an earlier start (the first build takes about 80 s)")
    from app import CorpusSearcher
    searcher = CorpusSearcher(INDEX_DIR, "example_collection", "avsolatorio/NoInstruct-small-Embedding-v0", 3,
                              retriever="bge_m3", m3_dir=M3_PATH, m3_cache_dir=cache, chunks_path=chunks_file, m3_model=bge_m3)
    text = searcher.search("What time was the flotation collar inflated?")
    assert text.startswith("Result 1:\n") and text.count("------\n") == 3 and "Result 4:" not in text
