"""Evaluate the base model and LoRA checkpoints on the validation questions, with the same loop as training.

Needs the Gym server (`ng_run` with resources_only.yaml, Learnings.md procedure 4.4) and a free GPU: do not run it
while training. It loads the model once, then evaluates the base model (no adapter) and each checkpoint in turn.
    cd trainer
    temp/.venv-temp/bin/python eval.py --run-dir runs/<run> --checkpoints runs/<run>/checkpoint-* runs/<run>/final
    temp/.venv-temp/bin/python eval.py --run-dir runs/<run> --checkpoints runs/<run>/checkpoint-2 --limit 16 --no-base   # quick check

- Decoding is greedy (temperature 0), so a run can be repeated and two models see the same conditions.
- The reward comes from the real /verify, so base and checkpoints must be evaluated with the same server (same app.py).
- Output in --run-dir: eval_base.jsonl, eval_<checkpoint name>.jsonl (one line for each question, same fields as
  train_dump.jsonl) and eval_summary.json. A table is printed at the end.
Metrics: mean reward (with its standard error), share of rewards at 1.0, "contains" (the expected answer is in the final
answer, whatever its length: it does not depend on the length cap), tool-call rate, answer rate, mean turns, mean words
in the final answer, and the paired change in reward against the base model.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from train_grpo import MAX_COMPLETION, MAX_NEW, MAX_TURNS, MODEL, load_dataset, load_model

from rollout import (GymClient, answer_words, append_jsonl, contains_answer, final_text, make_gym_verify_reward,
                     make_vllm_generate_fn, run_rollout, startup_checks, trajectory_record, vram_used_mib)


def evaluate(label, model, tok, client, rows, lora_request, dump_path, chunk=32):
    """Run every row to the end (greedy) and return one record for each. `lora_request=None` is the base model."""
    from vllm import SamplingParams
    generate_fn = make_vllm_generate_fn(model.vllm_engine, SamplingParams(temperature=0.0, max_tokens=MAX_NEW, logprobs=0),
                                        lora_request)
    reward_fn = make_gym_verify_reward(client)
    open(dump_path, "w").close()
    records, t0 = [], time.time()
    for start in range(0, len(rows), chunk):   # vLLM runs about 32 sequences at once: send the rows in chunks
        batch = rows[start:start + chunk]
        out = run_rollout([r["prompt"] for r in batch], tok, generate_fn, client.call_tool, max_turns=MAX_TURNS,
                          max_completion_tokens=MAX_COMPLETION, max_new_tokens=MAX_NEW)
        rewards = reward_fn(prompts=[r["prompt"] for r in batch], completions=[None] * len(batch),
                            trajectory=out["trajectory"], answer=[r["answer"] for r in batch],
                            responses_create_params=[r["responses_create_params"] for r in batch])
        new = [trajectory_record(r["prompt"], r["answer"], t, rew, model=label, index=start + i,
                                 contains=contains_answer(final_text(t), r["answer"]),
                                 final_words=len(answer_words(final_text(t))))
               for i, (r, t, rew) in enumerate(zip(batch, out["trajectory"], rewards))]
        append_jsonl(dump_path, new)
        records += new
        print(f"[eval] {label}: {len(records)}/{len(rows)} done, {time.time() - t0:.0f} s, VRAM {vram_used_mib()} MiB")
    return records


def adapter_size(ckpt: Path) -> str:
    """Sum of |lora_B| in the saved adapter. B starts at 0, so 0 means the adapter learned nothing."""
    from safetensors.torch import load_file
    files = list(ckpt.glob("adapter_model.safetensors"))
    if not files:
        return "no adapter_model.safetensors found"
    tensors = {k: v for k, v in load_file(str(files[0])).items() if "lora_B" in k}
    total = sum(v.float().abs().sum().item() for v in tensors.values())
    return f"{len(tensors)} lora_B tensors, sum of absolute values {total:.4f}" + (" (the adapter is still zero)" if total == 0 else "")


def summarize(label, records, base_records=None):
    n = len(records)
    rewards = [r["reward"] for r in records]
    se = statistics.stdev(rewards) / math.sqrt(n) if n > 1 else float("nan")
    out = {
        "model": label, "n": n, "reward": sum(rewards) / n, "reward_se": se,
        "exact": sum(x >= 0.999 for x in rewards) / n,
        "contains": sum(r["contains"] for r in records) / n,
        "tool_call_rate": sum(r["n_tool_calls"] > 0 for r in records) / n,
        "answer_rate": sum(r["ended"] == "answer" for r in records) / n,
        "mean_turns": sum(r["n_turns"] for r in records) / n,
        "mean_final_words": sum(r["final_words"] for r in records) / n,
        "endings": {k: sum(r["ended"] == k for r in records) / n
                    for k in ("answer", "truncated", "bad_call", "turn_cap", "budget")},
    }
    if base_records is not None:
        diffs = [a["reward"] - b["reward"] for a, b in zip(records, base_records)]
        out["delta_reward_vs_base"] = sum(diffs) / n
        out["delta_reward_se"] = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else float("nan")
    return out


def print_table(summaries):
    print("\n[eval] summary (greedy decoding; se = standard error; the change is paired on the same questions)")
    print(f"  {'model':<16} {'n':>4} {'reward':>14} {'exact':>6} {'contains':>8} {'tools':>6} {'answered':>8} {'turns':>6} {'words':>6} {'change vs base':>16}")
    for s in summaries:
        change = f"{s['delta_reward_vs_base']:+.3f} ± {s['delta_reward_se']:.3f}" if "delta_reward_vs_base" in s else "-"
        print(f"  {s['model']:<16} {s['n']:>4} {s['reward']:>7.3f} ± {s['reward_se']:.3f} {s['exact']:>6.2f} {s['contains']:>8.2f} "
              f"{s['tool_call_rate']:>6.2f} {s['answer_rate']:>8.2f} {s['mean_turns']:>6.2f} {s['mean_final_words']:>6.1f} {change:>16}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="where the dumps and the summary go")
    ap.add_argument("--checkpoints", nargs="*", default=[], type=Path, help="LoRA folders (checkpoint-N or final)")
    ap.add_argument("--no-base", action="store_true", help="skip the base model")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=None, help="use only this many questions (spread over the file)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--lora-rank", type=int, default=64, help="the rank used in training (vLLM must allow it)")
    args = ap.parse_args()

    for ckpt in args.checkpoints:   # fail before the slow model load
        if not (ckpt / "adapter_config.json").exists():
            raise SystemExit(f"{ckpt} has no adapter_config.json: not a LoRA folder")
    if args.no_base and not args.checkpoints:
        raise SystemExit("nothing to evaluate: give --checkpoints or drop --no-base")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    client = GymClient.discover()
    dataset = load_dataset(args.split)
    step = max(1, len(dataset) // args.limit) if args.limit else 1
    rows = [dataset[i] for i in range(0, len(dataset), step)][:args.limit]
    print(f"[eval] {len(rows)} questions from the {args.split} split; server {client.base_url}")

    model, tok = load_model(args.model, args.gpu_util, args.lora_rank)
    startup_checks(tok)

    summaries, base_records = [], None
    base_dump = args.run_dir / "eval_base.jsonl"
    if not args.no_base:
        base_records = evaluate("base", model, tok, client, rows, None, base_dump)
        summaries.append(summarize("base", base_records))
    elif base_dump.exists():   # an earlier invocation did the base model: use it for the paired change
        earlier = [json.loads(line) for line in base_dump.read_text(encoding="utf-8").splitlines() if line]
        if len(earlier) == len(rows):
            base_records = earlier
            summaries.append(summarize("base", base_records))
        else:
            print(f"[eval] eval_base.jsonl has {len(earlier)} rows, not {len(rows)}: no paired change this time")
    for ckpt in args.checkpoints:
        lora_request = model.load_lora(str(ckpt.resolve()), load_tensors=False)   # vLLM reads the saved adapter
        print(f"[eval] {ckpt.name}: {adapter_size(ckpt)}")
        records = evaluate(ckpt.name, model, tok, client, rows, lora_request, args.run_dir / f"eval_{ckpt.name}.jsonl")
        summaries.append(summarize(ckpt.name, records, base_records))

    summary_path = args.run_dir / "eval_summary.json"
    known = {s["model"]: s for s in (json.loads(summary_path.read_text()) if summary_path.exists() else [])}
    known.update({s["model"]: s for s in summaries})      # keep the results of earlier invocations
    summary_path.write_text(json.dumps(list(known.values()), indent=2))
    print_table(summaries)
    print(f"[eval] dumps and eval_summary.json are in {args.run_dir}")


if __name__ == "__main__":
    main()
