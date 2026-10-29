"""Retrievers: one interface, `search(query, k) -> list of Chunk, best first`, and two implementations.

  ChromaRetriever    the original: NoInstruct embedder, Chroma index, L2 distance (needs chromadb, langchain-chroma, torch)
  BGEM3Retriever     BGE-M3 hybrid (dense 0.4, sparse 0.2, ColBERT 0.4) scored exactly over all chunks in memory (m3.py);
                     the chunk encodings are a cache on disk, memory-mapped (first start about 80 s on the CPU, later starts about 3 s)

The text that the model reads is made by `format_results`, so it does not depend on the backend. A new backend (for example Vespa or
Qdrant for a larger corpus) is one more class with the same `search`. The tool returns ids and texts, never arrays.
"""
import time
from pathlib import Path
from typing import List, Protocol

from chunks import Chunk, read_chunks


class Retriever(Protocol):
    def search(self, query: str, k: int) -> List[Chunk]: ...


def format_results(chunks) -> str:
    """The AutoDidact format: 'Result i:', the chunk text, then '------'. Takes Chunk objects or plain strings."""
    return "".join(f"Result {i}:\n{c if isinstance(c, str) else c.text}\n------\n" for i, c in enumerate(chunks, start=1))


def load_chunks(chunks_path: Path):
    """(chunks, meta) from the JSONL file. The server never writes this file: scripts/export_chunks.py does."""
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk store not found at {chunks_path}. Create it once with: "
                                f"NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/export_chunks.py")
    return read_chunks(chunks_path)


class ChromaRetriever:
    def __init__(self, chroma_dir: Path, collection_name: str, embedding_model: str):
        from langchain_chroma import Chroma
        from embeddings import CustomHuggingFaceEmbeddings

        # The same class that built the index: documents use [CLS] pooling, queries use mean pooling.
        self.vectorstore = Chroma(collection_name=collection_name, embedding_function=CustomHuggingFaceEmbeddings(model_name=embedding_model),
                                  persist_directory=str(chroma_dir))

    def search(self, query: str, k: int) -> List[Chunk]:
        return [Chunk(id=f"chunk-{d.metadata['chunk_index']}", index=d.metadata["chunk_index"], text=d.page_content, start=d.metadata["chunk_start"],
                      end=d.metadata["chunk_end"], n_merged=d.metadata["n_merged"]) for d in self.vectorstore.similarity_search(query, k=k)]


class BGEM3Retriever:
    def __init__(self, chunks: List[Chunk], m3_dir: Path, cache_dir: Path = None, chunking: dict = None, log=print, max_cache_bytes: int = None, model=None):
        """`model`: an already loaded m3.BGEM3 to use (it is safe to share one between retrievers). If it is None, the model is loaded from m3_dir."""
        import m3

        m3_dir = Path(m3_dir)
        if not (m3_dir / "pytorch_model.bin").exists():
            raise FileNotFoundError(f"BGE-M3 not found in {m3_dir}. Expected pytorch_model.bin, colbert_linear.pt and sparse_linear.pt there.")
        max_cache_bytes = max_cache_bytes or m3.MAX_CACHE_BYTES
        self.chunks = list(chunks)
        manifest = m3.build_manifest(self.chunks, m3_dir, chunking)
        directory = Path(cache_dir) / f"m3-{m3.cache_key(manifest)}" if cache_dir else None
        t0 = time.time()
        self.model = model if model is not None else m3.BGEM3(m3_dir)
        arrays = m3.load_cache(directory, manifest, max_cache_bytes) if directory else None
        self.loaded_from_cache = arrays is not None
        if arrays is None:
            log(f"[retriever] encoding {len(self.chunks)} chunks with BGE-M3 (no valid cache yet: about 80 s on the CPU) ...")
            arrays = m3.pack(*self.model.encode([c.text for c in self.chunks], max_length=m3.DOC_MAX_LENGTH))
            if directory:
                m3.save_cache(directory, arrays, manifest)
                arrays = m3.load_cache(directory, manifest, max_cache_bytes) or arrays     # reopen memory-mapped: frees the copy in memory
        if cache_dir and Path(cache_dir).is_dir():
            stale = [p.name for p in Path(cache_dir).iterdir() if p.is_dir() and p.name.startswith("m3-") and p != directory]
            if stale:
                log(f"[retriever] other cache folders exist in {cache_dir} (from other chunks or settings): {stale}. Delete them if you do not need them.")
        self.index = m3.M3Index(arrays)
        self._m3 = m3
        log(f"[retriever] BGE-M3 hybrid ready in {time.time() - t0:.1f} s ({'cache hit' if self.loaded_from_cache else 'encoded and cached'}; {len(self.chunks)} chunks)")

    def search(self, query: str, k: int) -> List[Chunk]:
        q_dense, q_sparse, q_colbert = self.model.encode([query], max_length=self._m3.QUERY_MAX_LENGTH)
        scores = self.index.hybrid(q_dense[0], q_sparse[0], q_colbert[0])
        return [self.chunks[i] for i in scores.argsort()[::-1][:k]]
