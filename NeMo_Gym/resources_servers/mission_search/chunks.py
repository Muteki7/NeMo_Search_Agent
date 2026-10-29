"""The chunk store: the text and the metadata of every chunk, keyed by id, in one JSONL file. Embeddings are never stored here.

File format: the first line is {"_meta": {...}} (where the chunks come from and how they were made), then one JSON object for each chunk.
The search tool returns ids and texts. The arrays of an index (m3.py) are a cache that is built from these chunks.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class Chunk:
    id: str          # "chunk-<index>": the id that the Chroma index uses
    index: int       # position in the merged list
    text: str
    start: int       # the first old position (before the merge) that this chunk covers
    end: int         # the last old position
    n_merged: int


def write_chunks(path, chunks: List[Chunk], meta: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_chunks(path) -> Tuple[List[Chunk], dict]:
    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or "_meta" not in lines[0]:
        raise ValueError(f"{path} is not a chunk file: the first line must be the _meta line")
    chunks = [Chunk(**row) for row in lines[1:]]
    if [c.index for c in chunks] != list(range(len(chunks))):
        raise ValueError(f"{path}: the chunk indexes are not 0..{len(chunks) - 1} in order")
    if len({c.id for c in chunks}) != len(chunks) or any(not c.text.strip() or c.start > c.end for c in chunks):
        raise ValueError(f"{path}: a duplicate id, an empty text or a bad range")
    return chunks, lines[0]["_meta"]


def chunks_from_chroma(chroma_dir, collection_name: str) -> List[Chunk]:
    """Read the chunks of the index built by scripts/build_index.py, in the order of their chunk_index."""
    import chromadb
    got = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(collection_name).get(include=["documents", "metadatas"])
    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda r: r[2]["chunk_index"])
    return [Chunk(id=i, index=m["chunk_index"], text=t, start=m["chunk_start"], end=m["chunk_end"], n_merged=m["n_merged"]) for i, t, m in rows]
