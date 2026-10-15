"""Score the trajectories of a dump again with the CURRENT `score_answer` in app.py. No server, no GPU.

A dump is the JSONL file that `trainer/rollout_smoke.py` (--dump) or the training reward function (dump_path) writes.
Use it to try a change of the verifier on real trajectories: edit score_answer in app.py, then run this script.

Run it with the server venv, from the repository root:
    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/rescore_dump.py trainer/smoke_dump.jsonl --show 10
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so that `import app` finds ../app.py
from app import score_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--show", type=int, default=10, help="how many trajectories to print in each list")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.dump.read_text(encoding="utf-8").splitlines() if line]
    for r in rows:
        r["new_reward"] = score_answer(r["final_text"], r["answer"])

    print(f"{len(rows)} trajectories from {args.dump}")
    print(f"mean reward: recorded {sum(r['reward'] for r in rows) / len(rows):.3f} -> now {sum(r['new_reward'] for r in rows) / len(rows):.3f}")

    by_end = defaultdict(list)
    for r in rows:
        by_end[r["ended"]].append(r)
    print("\nby ending: count | mean recorded reward | mean reward now | how many now above 0")
    for ended, group in sorted(by_end.items()):
        print(f"  {ended:<10} {len(group):>4} | {sum(r['reward'] for r in group) / len(group):.3f} | "
              f"{sum(r['new_reward'] for r in group) / len(group):.3f} | {sum(r['new_reward'] > 0 for r in group)}")
    junk = [r for r in rows if r["ended"] != "answer" and r["new_reward"] > 0]
    print(f"\ntrajectories that did not end with an answer but still get reward now: {len(junk)}")

    print(f"\n{args.show} trajectories with the largest change (recorded -> now | ended | final text):")
    for r in sorted(rows, key=lambda r: abs(r["new_reward"] - r["reward"]), reverse=True)[:args.show]:
        print(f"  {r['reward']:.2f} -> {r['new_reward']:.2f} | {r['ended']:<9} | {(r['final_text'] or '(none)')[:80]!r}")

    print(f"\n{args.show} trajectories with the highest reward now:")
    for r in sorted(rows, key=lambda r: r["new_reward"], reverse=True)[:args.show]:
        print(f"  {r['new_reward']:.2f} | {r['ended']:<9} calls={r['n_tool_calls']} | {(r['final_text'] or '(none)')[:80]!r} "
              f"| expected {r['answer'][:50]!r}")


if __name__ == "__main__":
    main()
