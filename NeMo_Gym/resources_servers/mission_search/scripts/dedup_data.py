"""Remove the repeated questions from data/train.jsonl and data/validation.jsonl. The originals stay as they are.

Why: generate_data.py built each question from a window of 3 chunks, and neighbouring windows overlap, so the same question was written more than once
(train: 110 extra rows, validation: 18). In training the model sees the same question several times. In validation the rows are not independent, so the
standard error of a mean reward is too small.

Rule, for the rows with the same question text (`normalize_question` of convert_questions.py: lowercase, spaces collapsed):
  - the copies have the same reference (after `normalize` of app.py, the same text the verifier compares): keep the first copy, drop the others ("repeat")
  - the copies have different references: we cannot know which one is right, so drop ALL copies ("conflict")
The kept rows keep their order. Output: data/train_dedup.jsonl and data/validation_dedup.jsonl (the trainer loads a split by name: load_dataset("train_dedup")).

Run it with the server venv, from the repository root:
    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/dedup_data.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))                  # so that `import app` finds ../app.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import normalize  # noqa: E402
from convert_questions import normalize_question  # noqa: E402


def question_of(row: dict) -> str:
    """The question text: the end of the user message of the prompt, after "Question:"."""
    return row["responses_create_params"]["input"][-1]["content"].rsplit("Question:", 1)[-1].strip()


def dedup(rows: list) -> tuple:
    """(indices of the rows to keep, {index of a dropped row: "repeat" or "conflict"})."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[normalize_question(question_of(row))].append(i)
    kept, dropped = [], {}
    for indices in groups.values():
        if len({normalize(rows[i]["answer"]) for i in indices}) > 1:
            dropped.update({i: "conflict" for i in indices})
        else:
            kept.append(indices[0])
            dropped.update({i: "repeat" for i in indices[1:]})
    return sorted(kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=SERVER_DIR / "data")
    args = ap.parse_args()
    for split in ("train", "validation"):
        lines = [line for line in (args.data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        kept, dropped = dedup([json.loads(line) for line in lines])
        out = args.data_dir / f"{split}_dedup.jsonl"
        out.write_text("".join(lines[i] + "\n" for i in kept), encoding="utf-8")
        n_conflict = sum(1 for v in dropped.values() if v == "conflict")
        print(f"{split}: {len(lines)} rows -> {len(kept)} rows in {out.name} "
              f"(dropped {len(dropped)}: {len(dropped) - n_conflict} repeats, {n_conflict} rows of questions with different references)")


if __name__ == "__main__":
    main()
