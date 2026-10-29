"""BGE-M3 built from its model files: dense, sparse and ColBERT scores, an exact in-memory index and a safe disk cache.

Used by the search tool (retriever.py) and by scripts/retrieval_bench.py, so that the benchmark measures the code that the server runs.
tests/test_retrieval_bench.py checks the three scores against the example scores in the model card.

One forward pass gives three outputs for a text:
  dense    the normalized [CLS] vector                                   score = dot product
  sparse   a weight for each token id (relu of a linear layer)            score = sum over shared tokens of query weight x chunk weight
  colbert  one normalized vector for each token after [CLS]               score = mean over query tokens of the best dot product with a chunk token
The hybrid is the weighted sum with the weights of the model card (dense 0.4, sparse 0.2, colbert 0.4), divided by the sum of the
weights that are used.

Memory rules (an earlier loader made 510 copies of a 249 MB array and ran the machine out of memory):
  - The cache is a folder of .npy files. The arrays are opened with mmap: the operating system pages the file in and nothing is copied.
  - Layout: dense [N, 1024]; colbert [total tokens, 1024] with colbert_offsets [N + 1] (the rows of chunk i are offsets[i]:offsets[i + 1]);
    the sparse weights as CSR (sparse_indptr [N + 1], sparse_indices int32, sparse_data float32). All vectors are float32.
  - A cache that needs more than MAX_CACHE_BYTES is refused before anything is read.
  - One search never builds a matrix larger than MAX_TEMP_BYTES: the ColBERT product runs in blocks of chunks.
  - manifest.json in the cache folder records everything that the arrays depend on, with the dtype of each array.
    The loader compares it with what the code expects.
"""
import hashlib
import json
import os
import shutil
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np

WEIGHTS = {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}      # the model card: weights_for_different_modes=[0.4, 0.2, 0.4]
DOC_MAX_LENGTH = 512
QUERY_MAX_LENGTH = 128
FORMAT = 2                                                   # raise it when the cache layout or the encoding code changes
MAX_CACHE_BYTES = 2 * 1024 ** 3                              # the real cache is 0.25 GB
MAX_TEMP_BYTES = 256 * 1024 ** 2                             # the largest temporary matrix of one search (query tokens x chunk tokens x 4 bytes)


class CacheTooLargeError(RuntimeError):
    pass


class BGEM3:
    def __init__(self, path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path)).eval()
        h = self.model.config.hidden_size
        self.colbert = torch.nn.Linear(h, h)
        self.colbert.load_state_dict(torch.load(Path(path) / "colbert_linear.pt", map_location="cpu"))
        self.sparse = torch.nn.Linear(h, 1)
        self.sparse.load_state_dict(torch.load(Path(path) / "sparse_linear.pt", map_location="cpu"))
        self.special = {i for i in (self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id) if i is not None}
        self._lock = threading.Lock()                        # parallel searches take turns: one forward pass at a time on the CPU

    def encode(self, texts, max_length=512, batch_size=16):
        """-> dense [N, 1024] (unit length), sparse: list of {token id: weight}, colbert: list of [tokens - 1, 1024] (unit length)."""
        import torch
        import torch.nn.functional as F
        n = len(texts)
        dense, sparse, colbert = [None] * n, [None] * n, [None] * n
        order = sorted(range(n), key=lambda i: len(texts[i]))      # similar lengths together: less padding
        with self._lock, torch.no_grad():
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                x = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                h = self.model(**x).last_hidden_state
                d = F.normalize(h[:, 0], dim=-1)
                w = torch.relu(self.sparse(h)).squeeze(-1)
                cv = F.normalize(self.colbert(h[:, 1:]), dim=-1)
                for b, i in enumerate(idx):
                    n_tok = int(x["attention_mask"][b].sum())
                    dense[i] = d[b].numpy()
                    weights = {}
                    for pos in range(n_tok):
                        tid = int(x["input_ids"][b, pos])
                        wt = float(w[b, pos])
                        if tid not in self.special and wt > weights.get(tid, 0.0):
                            weights[tid] = wt                           # a repeated token keeps its largest weight
                    sparse[i] = weights
                    colbert[i] = cv[b, :n_tok - 1].numpy()              # everything after [CLS]
        return np.stack(dense), sparse, colbert


# ---------------------------------------------------------------------------------------------------------------
# Scores for a batch of queries against all chunks (used by the benchmark; the lists are small there)
# ---------------------------------------------------------------------------------------------------------------
def dense_scores(q, d):
    return q @ d.T


def sparse_scores(q_sparse, d_sparse):
    inverted = defaultdict(list)
    for j, doc in enumerate(d_sparse):
        for token, weight in doc.items():
            inverted[token].append((j, weight))
    out = np.zeros((len(q_sparse), len(d_sparse)), dtype=np.float32)
    for i, query in enumerate(q_sparse):
        for token, qw in query.items():
            for j, dw in inverted.get(token, ()):
                out[i, j] += qw * dw
    return out


def colbert_scores(q_vecs, d_vecs):
    all_d = np.concatenate(d_vecs)
    offsets = np.concatenate([[0], np.cumsum([len(v) for v in d_vecs])[:-1]])
    out = np.zeros((len(q_vecs), len(d_vecs)), dtype=np.float32)
    for i, q in enumerate(q_vecs):
        best = np.maximum.reduceat(q @ all_d.T, offsets, axis=1)     # best chunk token for each query token, per chunk
        out[i] = best.sum(0) / len(q)
    return out


def combine(parts: dict, names) -> np.ndarray:
    """Weighted sum of the named modes, divided by the sum of their weights (the model card's convention)."""
    total = sum(WEIGHTS[n] for n in names)
    return sum(WEIGHTS[n] * parts[n] for n in names) / total


# ---------------------------------------------------------------------------------------------------------------
# The arrays of an index
# ---------------------------------------------------------------------------------------------------------------
def pack(dense, sparse, colbert) -> dict:
    """The output of BGEM3.encode as six flat arrays (ragged data as one matrix plus offsets, sparse as CSR)."""
    return {
        "dense": np.ascontiguousarray(dense, dtype=np.float32),
        "colbert": np.concatenate(colbert).astype(np.float32, copy=False),
        "colbert_offsets": np.cumsum([0] + [len(v) for v in colbert]).astype(np.int64),
        "sparse_indptr": np.cumsum([0] + [len(d) for d in sparse]).astype(np.int64),
        "sparse_indices": np.array([t for d in sparse for t in d.keys()], dtype=np.int32),
        "sparse_data": np.array([w for d in sparse for w in d.values()], dtype=np.float32),
    }


class M3Index:
    """Exact search over all chunks, one query at a time. Built from the arrays of `pack` (they may be memory-mapped)."""

    def __init__(self, arrays: dict):
        self.dense, self.colbert, self.offsets = arrays["dense"], arrays["colbert"], arrays["colbert_offsets"]
        indptr, indices, data = arrays["sparse_indptr"], arrays["sparse_indices"], arrays["sparse_data"]
        self.n = len(self.dense)
        if not (len(self.offsets) == len(indptr) == self.n + 1 and self.offsets[-1] == len(self.colbert) and indptr[-1] == len(indices) == len(data)):
            raise ValueError("the arrays of the index do not fit together (a damaged or mixed cache)")
        # token -> chunks (the transpose of the CSR matrix): sort the entries by token id
        indices = np.asarray(indices)
        order = np.argsort(indices, kind="stable")
        self.tok_rows = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(indptr))[order]
        self.tok_vals = np.asarray(data)[order]
        self.vocab = int(indices.max()) + 1 if len(indices) else 0
        self.tok_ptr = np.concatenate([[0], np.cumsum(np.bincount(indices, minlength=self.vocab))])

    def _sparse(self, q_sparse: dict) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for token, qw in q_sparse.items():
            if 0 <= token < self.vocab:
                a, b = self.tok_ptr[token], self.tok_ptr[token + 1]
                if b > a:
                    out[self.tok_rows[a:b]] += qw * self.tok_vals[a:b]       # a token appears once for each chunk: the rows are unique
        return out

    def _colbert(self, q: np.ndarray) -> np.ndarray:
        """Mean over the query tokens of the best dot product with a chunk token, in blocks of chunks so that the temporary matrix
        (query tokens x block tokens x 4 bytes) stays under MAX_TEMP_BYTES."""
        tq, out = len(q), np.zeros(self.n, dtype=np.float32)
        block_tokens = max(1, MAX_TEMP_BYTES // (4 * tq))
        first = 0
        while first < self.n:
            base = int(self.offsets[first])
            last = int(np.searchsorted(self.offsets, base + block_tokens, side="right")) - 1
            last = min(max(last, first + 1), self.n)                               # at least one chunk in a block
            sims = q @ self.colbert[base:int(self.offsets[last])].T
            out[first:last] = np.maximum.reduceat(sims, self.offsets[first:last] - base, axis=1).sum(0) / tq
            first = last
        return out

    def scores(self, q_dense, q_sparse, q_colbert) -> dict:
        return {"dense": np.asarray(self.dense @ q_dense), "sparse": self._sparse(q_sparse), "colbert": self._colbert(q_colbert)}

    def hybrid(self, q_dense, q_sparse, q_colbert) -> np.ndarray:
        return combine(self.scores(q_dense, q_sparse, q_colbert), ("dense", "sparse", "colbert"))


# ---------------------------------------------------------------------------------------------------------------
# The cache: one folder for each set of inputs, with a manifest
# ---------------------------------------------------------------------------------------------------------------
def _sha256(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def embedder_identity(model_dir) -> dict:
    """What the encodings depend on in the model folder: the revision (file REVISION: repository on line 1, commit on line 2),
    the size of the weights file, and the hashes of the small files (config, tokenizer, the two heads)."""
    model_dir = Path(model_dir)
    lines = (model_dir / "REVISION").read_text().split() if (model_dir / "REVISION").exists() else []
    return {
        "repository": lines[0] if lines else "unknown", "revision": lines[1] if len(lines) > 1 else "unknown",
        "pytorch_model.bin_bytes": os.path.getsize(model_dir / "pytorch_model.bin"),
        "small_files_sha256": {n: _sha256(model_dir / n) for n in ("config.json", "tokenizer.json", "colbert_linear.pt", "sparse_linear.pt")},
    }


def build_manifest(chunks, model_dir, chunking: dict = None) -> dict:
    """Everything that decides if cached arrays are still valid. `chunks` is a list of objects with `id` and `text`."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.id}\x00{c.text}\x01".encode("utf-8"))
    return {
        "format": FORMAT,
        "embedder": embedder_identity(model_dir),
        "doc_max_length": DOC_MAX_LENGTH,
        "dtypes": {"dense": "float32", "colbert": "float32", "sparse_data": "float32", "sparse_indices": "int32", "offsets": "int64"},
        "normalization": {"dense": "l2 after [CLS]", "colbert": "l2 on the tokens after [CLS]",
                          "sparse": "relu weights, the largest weight of a repeated token, special tokens removed"},
        "chunks": {"count": len(chunks), "sha256": h.hexdigest()},
        "chunking": chunking or {},
    }


def cache_key(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def save_cache(directory, arrays: dict, manifest: dict) -> None:
    """Write the arrays and the manifest in a temporary folder, then rename it. A crash leaves no half-written cache under the real name."""
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(tmp / f"{name}.npy", array, allow_pickle=False)
    record = {**manifest, "arrays": {n: {"shape": list(a.shape), "dtype": str(a.dtype)} for n, a in arrays.items()}}
    (tmp / "manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def load_cache(directory, expected: dict, max_bytes: int = MAX_CACHE_BYTES):
    """The memory-mapped arrays of a cache folder, or None if there is no valid cache (missing, other inputs, damaged).
    Raises CacheTooLargeError, before it reads anything, if the arrays need more than `max_bytes`."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError):
        return None
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None                                                              # built from other inputs
    specs = manifest.get("arrays") or {}
    need = sum(int(np.prod(s["shape"])) * np.dtype(s["dtype"]).itemsize for s in specs.values())
    if need > max_bytes:
        raise CacheTooLargeError(f"the cache in {directory} needs {need / 1e9:.2f} GB, and the limit is {max_bytes / 1e9:.2f} GB. Not loaded.")
    arrays = {}
    try:
        for name, spec in specs.items():
            array = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)     # no copy: the OS pages the file in
            if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
                return None
            arrays[name] = array
    except (OSError, ValueError):
        return None                                                              # a missing or truncated file
    return arrays
