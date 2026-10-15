"""Build the Chroma index of the document chunks (the search tool reads this index).

Mirrors AutoDidact's generate_data.py lines 46-52: `Chroma.from_documents` over the saved chunks, collection
`example_collection`, with `CustomHuggingFaceEmbeddings` (see ../embeddings.py). Differences from the old code:
  - Chunks shorter than --min-chars (default 100) are merged forward with the chunk(s) after them. The splitter cut
    each document element (a heading, a page ID, a caption) on its own, which left 206 of 753 chunks under 50
    characters; they filled 12.9% of the top-2 search results. Use --min-chars 0 to keep every chunk as it is.
  - questions.json numbers the chunks by their OLD list positions (`chunk_id` = list index of the middle chunk of each
    3-chunk window). So every chunk keeps the range of old positions that it covers (`chunk_start`, `chunk_end`), and
    the retrieval check uses that range. `chunk_index` is the new position in the merged list.
  - Ids are fixed (`chunk-<new index>`) instead of random.
  - A non-empty output folder is refused unless --rebuild is given (Chroma would otherwise add duplicate rows).
  - After the build, the index is reopened from disk and tested with real questions (retrieval check:
    window recall, centre-chunk recall and answer coverage).

Run it with the server venv (it needs chromadb, langchain-chroma, torch and transformers):
    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/build_index.py
"""
import argparse
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent                      # .../resources_servers/mission_search
REPO_ROOT = HERE.parents[3]                   # scripts -> mission_search -> resources_servers -> NeMo_Gym -> repo root
sys.path.insert(0, str(SERVER_DIR))           # so that `import embeddings` finds ../embeddings.py

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from embeddings import CustomHuggingFaceEmbeddings  # noqa: E402

COLLECTION = "example_collection"
DEFAULT_MIN_CHARS = 100    # a chunk shorter than this is merged with the chunk(s) after it
DEFAULT_MAX_CHARS = 1500   # merging never makes a chunk longer than this (the longest merged chunk was 1089)
# small words that carry no meaning, ignored when we compare an answer with a chunk
STOP_WORDS = frozenset("the a an of in on at to for and or is was were are be by with from as that this it its what which who how when where did does do".split())


def clean_metadata(meta: dict) -> dict:
    """Chroma accepts only str, int, float and bool values. Anything else is stored as a JSON string."""
    out = {}
    for k, v in meta.items():
        out[k] = v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str)
    return out

"""
Essentially these two functions ensure the metadata is first clean, they check languages, data types as mentioned and then dump to v
Now we load the chunks and go index by index, every index i covers a 100 chunks so we roughly run this 7 times since we have a total of 753 chunks
k checks the metadata items for each chunk and passes it to v, v holds whatever k listed, so v even holds data, k=last_modified, v=2026-09...

"""
def load_chunks(path: Path) -> list:
    with open(path, "rb") as f:
        chunks = pickle.load(f)  # list of langchain_core Document objects
    for i, doc in enumerate(chunks):
        doc.metadata = clean_metadata(doc.metadata)
    return chunks


def merge_small_chunks(chunks: list, min_chars: int = DEFAULT_MIN_CHARS, max_chars: int = DEFAULT_MAX_CHARS) -> list:
    """Glue each chunk that is shorter than `min_chars` onto the chunk(s) after it (joined with a newline).

    Go through the chunks in order. While the current piece is shorter than `min_chars` and adding the next chunk keeps
    it within `max_chars`, add the next chunk. Chunks that already have `min_chars` or more are not touched. A short
    chunk at the very end, or one that cannot grow without passing `max_chars`, stays short. `min_chars=0` merges nothing.

    Every returned chunk gets: `chunk_index` (new position), `chunk_start` and `chunk_end` (the old positions it covers,
    first and last) and `n_merged`. The other metadata comes from the first chunk of the group, so `category` is the
    category of that first chunk.
    """
    groups, current, current_len = [], [], 0
    for i, doc in enumerate(chunks):
        if current and current_len < min_chars and current_len + 1 + len(doc.page_content) <= max_chars:
            current.append(i)
            current_len += 1 + len(doc.page_content)
        else:
            if current:
                groups.append(current)
            current, current_len = [i], len(doc.page_content)
    if current:
        groups.append(current)

    merged = []
    for new_index, group in enumerate(groups):
        text = "\n".join(chunks[j].page_content for j in group)
        meta = {**chunks[group[0]].metadata, "chunk_index": new_index,
                "chunk_start": group[0], "chunk_end": group[-1], "n_merged": len(group)}
        merged.append(Document(page_content=text, metadata=meta))
    return merged


def key_words(text: str) -> set:
    """Lowercase words and numbers of `text`, without the small words in STOP_WORDS."""
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def covers(answer_words: set, chunk_text: str, share: float = 0.6) -> bool:
    """True if at least `share` of the answer's key words appear in the chunk text."""
    return len(answer_words & key_words(chunk_text)) / max(1, len(answer_words)) >= share


def retrieval_check(vectorstore, questions_path: Path, n: int = 400, ks=(2, 5)) -> dict:
    """Ask the reopened index real questions. A question with chunk_id i was written from OLD chunks i-1, i, i+1.

    Each result covers the old positions chunk_start..chunk_end (a merged chunk covers several). For the top k results:
      - window recall: a result covers at least one chunk of the question's window
      - centre recall: a result covers the middle chunk i
      - answer coverage: a result holds at least 60% of the key words of the answer. This reads the text, so it does
        not depend on chunk numbers. It is a rough proxy: answers are paraphrases.
    Random retrieval would reach window recall of only a few percent at k=5.
    """
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    step = max(1, len(questions) // n)
    sample = questions[::step][:n]
    kmax = max(ks)
    window_hits = {k: 0 for k in ks}
    centre_hits = {k: 0 for k in ks}
    answer_hits = {k: 0 for k in ks}
    for q in sample:
        results = vectorstore.similarity_search(q["question"], k=kmax)
        low, high = q["chunk_id"] - 1, q["chunk_id"] + 1
        answer_words = key_words(q["answer"])
        for k in ks:
            top = results[:k]
            window_hits[k] += any(d.metadata["chunk_start"] <= high and d.metadata["chunk_end"] >= low for d in top)
            centre_hits[k] += any(d.metadata["chunk_start"] <= q["chunk_id"] <= d.metadata["chunk_end"] for d in top)
            answer_hits[k] += any(covers(answer_words, d.page_content) for d in top)
    result = {"n": len(sample)}
    for k in ks:
        result[f"window_recall@{k}"] = window_hits[k] / len(sample)
        result[f"centre_recall@{k}"] = centre_hits[k] / len(sample)
        result[f"answer_coverage@{k}"] = answer_hits[k] / len(sample)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, default=REPO_ROOT / "Step2_Files" / "Saved_data" / "chunks.pkl")
    ap.add_argument("--questions", type=Path, default=REPO_ROOT / "Step2_Files" / "Saved_data" / "questions.json")
    ap.add_argument("--out-dir", type=Path, default=SERVER_DIR / "chroma_index")
    ap.add_argument("--rebuild", action="store_true", help="delete a non-empty --out-dir (it must be named chroma_index)")
    ap.add_argument("--no-check", action="store_true", help="skip the retrieval check")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help="merge a chunk shorter than this with the chunk(s) after it (0 keeps every chunk)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="never merge into a chunk longer than this")
    args = ap.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        if not args.rebuild:
            sys.exit(f"{args.out_dir} is not empty. Use --rebuild to delete it and build again.")
        if args.out_dir.name != "chroma_index":
            sys.exit(f"Refusing to delete {args.out_dir}: the folder must be named chroma_index.")
        shutil.rmtree(args.out_dir)

    raw = load_chunks(args.chunks)
    raw_lengths = [len(c.page_content) for c in raw]
    print(f"[chunks] {len(raw)} chunks from {args.chunks} (characters: min {min(raw_lengths)}, max {max(raw_lengths)}; "
          f"under 50: {sum(1 for x in raw_lengths if x < 50)})")
    chunks = merge_small_chunks(raw, args.min_chars, args.max_chars)
    assert sum(c.metadata["n_merged"] for c in chunks) == len(raw), "a chunk was lost or used twice by the merge"
    lengths = [len(c.page_content) for c in chunks]
    merged_groups = sum(1 for c in chunks if c.metadata["n_merged"] > 1)
    print(f"[merge] min-chars {args.min_chars}, max-chars {args.max_chars}: {len(raw)} -> {len(chunks)} chunks "
          f"({merged_groups} merged groups; characters: min {min(lengths)}, max {max(lengths)}; "
          f"under {args.min_chars}: {sum(1 for x in lengths if x < args.min_chars)})")
    print(f"[chunks] metadata keys on chunk 0: {sorted(chunks[0].metadata)}")

    t0 = time.time()
    embeddings = CustomHuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        collection_name=COLLECTION,
        persist_directory=str(args.out_dir),
    )
    print(f"[build] embedded and stored {len(chunks)} chunks in {time.time() - t0:.0f} s -> {args.out_dir}")
    del vectorstore  # prove below that the saved files work on their own

    reopened = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(args.out_dir))
    count = reopened._collection.count()
    size_mb = sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file()) / 1e6
    print(f"[reopen] collection '{COLLECTION}' holds {count} rows; index folder is {size_mb:.1f} MB")
    assert count == len(chunks), f"expected {len(chunks)} rows, found {count}"

    if not args.no_check:
        t1 = time.time()
        res = retrieval_check(reopened, args.questions)
        print(f"[check] {res['n']} questions in {time.time() - t1:.0f} s")
        for k in (2, 5):
            print(f"[check] top-{k}: window recall {res[f'window_recall@{k}']:.1%} | centre-chunk recall {res[f'centre_recall@{k}']:.1%} "
                  f"| answer coverage {res[f'answer_coverage@{k}']:.1%}")
        # Sanity floor only: a broken index (wrong model or pooling) scores near chance, under 2%.
        assert res["window_recall@5"] > 0.30, "retrieval is close to chance: the index or the embedding is broken"


if __name__ == "__main__":
    main()
