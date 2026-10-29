"""Write the chunk store (data/chunks.jsonl) from the Chroma index. The server does not write it: it stops with this command if the file is missing.

Run with the server venv, from the repository root:
    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/export_chunks.py [--force]
--min-chars and --max-chars only record how the index was built (the defaults of build_index.py). They do not change any chunk.
"""
import argparse
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))
from chunks import chunks_from_chroma, read_chunks, write_chunks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chroma-dir", type=Path, default=SERVER_DIR / "chroma_index")
    ap.add_argument("--collection", default="example_collection")
    ap.add_argument("--out", type=Path, default=SERVER_DIR / "data" / "chunks.jsonl")
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists. Use --force to overwrite it.")
    chunks = chunks_from_chroma(args.chroma_dir, args.collection)
    write_chunks(args.out, chunks, {"source": f"{args.chroma_dir.name} ({args.collection})", "chunking": {"min_chars": args.min_chars, "max_chars": args.max_chars}})
    back, _ = read_chunks(args.out)                      # read it back: the file must pass the same checks that the server applies
    sizes = [len(c.text) for c in back]
    print(f"[export] {len(back)} chunks -> {args.out} ({args.out.stat().st_size / 1e3:.0f} KB); characters: min {min(sizes)}, max {max(sizes)}")


if __name__ == "__main__":
    main()
