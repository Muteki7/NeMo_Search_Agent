"""Convert AutoDidact's questions.json into NeMo Gym JSONL rows (train / validation / example).

Each output row is one task in the OpenAI Responses-API format that NeMo Gym uses:

    {"responses_create_params": {"input": [system msg, user msg], "tools": [flat search_corpus tool],
                                 "parallel_tool_calls": false},
     "answer": ..., "difficulty": ..., "chunk_id": ...}        (validation rows also carry "block")

Two ways to split (--split):

  purged (default)  Validation = a few contiguous blocks of chunk ids. Each question was generated from a window
                    of 3 chunks (chunk_id - 1, chunk_id, chunk_id + 1), so two questions share source text when
                    their ids differ by 2 or less. Training rows within BUFFER ids of any validation row are
                    removed ("purged"), and so are training rows whose question text also appears in validation.
                    Blocks are chosen by hashing, not by a random generator, so the same blocks come out forever.
  legacy            AutoDidact's `search_module.get_qa_dataset()` exactly (random shuffle then 90/10 split,
                    seed 42). Kept for reference and regression checks; it leaks (see the printed report).

Needs the `datasets` package for the legacy split (use the training venv: trainer/temp/.venv-temp). Run from anywhere:
    python convert_questions.py [--split purged|legacy]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]  # scripts -> mission_search -> resources_servers -> NeMo_Gym -> repo root

# --- purged block split settings ---
SPLIT_SEED = 42       # used to hash-order the candidate blocks (and to order train rows); no random generator involved
VAL_FRACTION = 0.10   # target share of rows held out for validation
NUM_BLOCKS = 6        # validation = this many separate blocks of consecutive chunk ids
# generate_data.py built each question from chunks (chunk_id - 1, chunk_id, chunk_id + 1), so two questions share
# source text when their ids differ by 2 or less. Training rows this close to a validation row are purged.
BUFFER = 2

# ---------------------------------------------------------------------------------------------------
# Copied from AutoDidact/rl_helpers.py (get_system_prompt: line 22, SEARCH_TOOL_DEFINITION: line 34,
# build_user_prompt: line 52). Copied, not imported: rl_helpers pulls in torch, trl and the Chroma loader
# at import time. The only change is in get_system_prompt: AutoDidact wrote the "Cutting Knowledge Date /
# Today Date" header into the message by hand, but the Llama chat template already adds that header to
# every prompt, so keeping it made it appear twice (with a date that follows the calendar). The two
# instruction sentences are unchanged. The template's own date is pinned at render time (trainer/prompting.py).
# ---------------------------------------------------------------------------------------------------


def get_system_prompt():
    """System prompt: the two instruction sentences only (the chat template supplies the date header)."""
    return """When you receive a tool call response, use the output to format an answer to the original user question.

You are a helpful assistant with tool calling capabilities.
"""


# Tool definition for search corpus
SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": "Search over the knowledge corpus with a given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the knowledge corpus with"
                },
            },
            "required": ["query"]
        }
    }
}


def build_user_prompt(q):
    """Build a user prompt with the question and search tool definition."""
    user_prompt = f"""You are a research assistant, and you use the search_corpus tool to find answers to questions.
Given a question, answer it using by doing searches using the search_corpus tool.
To use the search_corpus tool, respond with a JSON for a function call with its proper arguments.

You may also reason in any message, thinking step by step about how to answer the question. Wrap your reasoning in <reasoning> and </reasoning> tags.

{json.dumps(SEARCH_TOOL_DEFINITION, indent=2)}

Question: {q}
"""
    return user_prompt


# ---------------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------------

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def normalize_difficulty(label: str) -> str:
    """Some generated rows have labels like '==Difficulty: easy=='; reduce them to easy/medium/hard."""
    m = re.fullmatch(r"\s*=*\s*(?:difficulty:\s*)?(easy|medium|hard)\s*=*\s*", label, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognised difficulty label: {label!r}")
    return m.group(1).lower()


def to_responses_tool(tool_def: dict) -> dict:
    """Chat-completions tool ({type, function: {...}}) -> flat Responses-API tool.

    `strict` is a required key in the Responses-API function tool. It is False here: the schema is not written
    for strict mode (no additionalProperties=false) and our rollout never uses strict decoding.
    """
    fn = tool_def["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
        "strict": False,
    }


def to_row(qa: dict, system_prompt: str, tools: list) -> dict:
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(qa["question"])},
            ],
            "tools": tools,
            "parallel_tool_calls": False,
        },
        "answer": qa["answer"],
        "difficulty": normalize_difficulty(qa["difficulty"]),
        "chunk_id": qa["chunk_id"],
    }
    if "block" in qa:  # validation rows of the purged split: which held-out block the question belongs to
        row["block"] = qa["block"]
    return row


def split_like_autodidact(questions: list):
    """Reproduce search_module.get_qa_dataset(): shuffle(seed=42), then train_test_split(test_size=0.1, seed=42)."""
    from datasets import Dataset  # imported here so the purged split does not need the package

    full = Dataset.from_list(questions).shuffle(seed=42)
    parts = full.train_test_split(test_size=0.1, seed=42)
    return list(parts["train"]), list(parts["test"])


def normalize_question(text: str) -> str:
    """Lowercase and collapse whitespace, so trivial variants of the same question compare equal."""
    return " ".join(text.lower().split())


def stable_key(seed: int, label: str) -> str:
    """Deterministic, library-independent ordering key (SHA-256), used instead of a random generator."""
    return hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()


def pick_blocks(max_id: int, num_blocks: int, val_fraction: float, seed: int):
    """Choose `num_blocks` separate blocks of consecutive chunk ids to hold out.

    The id axis 1..max_id is cut into equal cells of width round(max_id * val_fraction / num_blocks). Cells are
    ordered by a hash and taken greedily, skipping any cell next to one already chosen so the blocks stay separate
    regions of the document. Returns ([(first_id, last_id), ...] sorted by position, width).
    """
    width = max(1, round(max_id * val_fraction / num_blocks))
    n_cells = -(-max_id // width)  # ceil: the last cell may be shorter
    chosen = []
    for k in sorted(range(n_cells), key=lambda k: stable_key(seed, f"block-{k}")):
        if all(abs(k - c) > 1 for c in chosen):
            chosen.append(k)
        if len(chosen) == num_blocks:
            break
    if len(chosen) < num_blocks:
        raise ValueError(f"could not place {num_blocks} non-adjacent blocks of width {width} in ids 1..{max_id}")
    chosen.sort()
    return [(k * width + 1, min((k + 1) * width, max_id)) for k in chosen], width


def split_purged(questions: list):
    """Block split with a buffer and an exact-text purge. Returns (train, validation, info).

    1. Validation = every row whose chunk_id falls inside a chosen block (tagged with its block index).
    2. Train rows within BUFFER ids of any validation row are purged (their source windows overlap).
    3. Train rows whose normalised question text also appears in validation are purged (same question generated
       from distant parts of the document, which the id distance cannot catch).
    Purged rows are in neither set. Validation is never trimmed, so its size stays what the blocks gave.
    """
    max_id = max(r["chunk_id"] for r in questions)
    blocks, width = pick_blocks(max_id, NUM_BLOCKS, VAL_FRACTION, SPLIT_SEED)

    def block_of(chunk_id):
        return next((b for b, (lo, hi) in enumerate(blocks) if lo <= chunk_id <= hi), None)

    val, rest = [], []
    for r in questions:
        b = block_of(r["chunk_id"])
        if b is None:
            rest.append(r)
        else:
            val.append({**r, "block": b})

    val_ids = {r["chunk_id"] for r in val}
    val_texts = {normalize_question(r["question"]) for r in val}
    train, purged_buffer, purged_text = [], [], []
    for r in rest:
        if any(abs(r["chunk_id"] - v) <= BUFFER for v in val_ids):
            purged_buffer.append(r)
        elif normalize_question(r["question"]) in val_texts:
            purged_text.append(r)
        else:
            train.append(r)

    # deterministic shuffle of train (so example.jsonl is not just the first pages of the report); validation stays
    # ordered by block then id, which makes per-block reading easy
    train.sort(key=lambda r: stable_key(SPLIT_SEED, f"train-{r['chunk_id']}-{normalize_question(r['question'])}"))
    val.sort(key=lambda r: (r["block"], r["chunk_id"], r["question"]))
    info = {"blocks": blocks, "width": width, "purged_buffer": len(purged_buffer), "purged_text": len(purged_text)}
    return train, val, info


def purged_report(train_qa: list, val_qa: list, info: dict, total: int) -> None:
    """Print the split summary and fail (AssertionError) if a guarantee of the purged split is violated."""
    train_ids = {r["chunk_id"] for r in train_qa}
    val_ids = {r["chunk_id"] for r in val_qa}
    min_dist = min(abs(t - v) for t in train_ids for v in val_ids)
    train_texts = {normalize_question(r["question"]) for r in train_qa}
    text_overlap = sum(1 for r in val_qa if normalize_question(r["question"]) in train_texts)

    print(f"[split] purged: {len(info['blocks'])} blocks of up to {info['width']} ids (seed {SPLIT_SEED}, buffer {BUFFER})")
    for b, (lo, hi) in enumerate(info["blocks"]):
        n = sum(1 for r in val_qa if r["block"] == b)
        print(f"[split]   block {b}: ids {lo}-{hi} -> {n} validation rows")
    print(f"[split] validation={len(val_qa)} train={len(train_qa)} purged: buffer={info['purged_buffer']} "
          f"exact-text={info['purged_text']} (total {total})")
    print(f"[guarantee] min chunk-id distance train<->validation: {min_dist} (must be > {BUFFER}); "
          f"validation questions also in train: {text_overlap} (must be 0)")

    assert len(train_qa) + len(val_qa) + info["purged_buffer"] + info["purged_text"] == total, "rows unaccounted for"
    assert min_dist > BUFFER, "train and validation windows overlap"
    assert text_overlap == 0, "a validation question also appears in train"
    assert 0.05 * total <= len(val_qa) <= 0.15 * total, "validation size is far from the 10% target"


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def leakage_report(train_qa: list, val_qa: list) -> None:
    """Informational: how clean is the validation split? (The AutoDidact split does not guard against this.)"""
    train_questions = {r["question"] for r in train_qa}
    train_chunks = {r["chunk_id"] for r in train_qa}
    same_q = sum(1 for r in val_qa if r["question"] in train_questions)
    same_chunk = sum(1 for r in val_qa if r["chunk_id"] in train_chunks)
    print(f"[leakage] validation rows whose exact question text also appears in train: {same_q} of {len(val_qa)}")
    print(f"[leakage] validation rows whose chunk_id also appears in train: {same_chunk} of {len(val_qa)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO_ROOT / "Step2_Files" / "Saved_data" / "questions.json")
    ap.add_argument("--out-dir", type=Path, default=HERE.parent / "data")
    ap.add_argument("--split", choices=["purged", "legacy"], default="purged",
                    help="purged: block split with a buffer and an exact-text purge (default); "
                         "legacy: AutoDidact's random 90/10 split, for reference")
    ap.add_argument("--num-examples", type=int, default=5, help="rows copied from the start of train into example.jsonl")
    args = ap.parse_args()

    questions = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"[input] {len(questions)} rows from {args.input}")

    if args.split == "purged":
        train_qa, val_qa, info = split_purged(questions)
        purged_report(train_qa, val_qa, info, total=len(questions))
        n_purged = info["purged_buffer"] + info["purged_text"]
    else:
        train_qa, val_qa = split_like_autodidact(questions)
        leakage_report(train_qa, val_qa)
        n_purged = 0
    system_prompt = get_system_prompt()
    tools = [to_responses_tool(SEARCH_TOOL_DEFINITION)]

    train_rows = [to_row(r, system_prompt, tools) for r in train_qa]
    val_rows = [to_row(r, system_prompt, tools) for r in val_qa]
    example_rows = train_rows[: args.num_examples]

    # --- self-checks: fail loudly instead of writing a quietly broken dataset ---
    assert len(train_rows) + len(val_rows) + n_purged == len(questions), "rows lost or duplicated by the split"
    assert all(r["difficulty"] in VALID_DIFFICULTIES for r in train_rows + val_rows), "bad difficulty label"
    base_keys = {"responses_create_params", "answer", "difficulty", "chunk_id"}
    assert all(set(r) == base_keys for r in train_rows), "unexpected keys in a train row"
    val_keys = base_keys | {"block"} if args.split == "purged" else base_keys
    assert all(set(r) == val_keys for r in val_rows), "unexpected keys in a validation row"
    assert all(r["answer"].strip() for r in train_rows + val_rows), "empty answer"

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "validation.jsonl", val_rows)
    write_jsonl(args.out_dir / "example.jsonl", example_rows)
    print(f"[output] train={len(train_rows)} validation={len(val_rows)} example={len(example_rows)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
