"""Test 2: run `rollout_func` alone on the real vLLM engine, the real LoRA handle and the live Gym server. No training.

Needs: `ng_run` running with resources_only.yaml (Learnings.md procedure 4.4) and a free GPU.
    cd trainer && temp/.venv-temp/bin/python rollout_smoke.py                    # 8 questions x 4 attempts, 1B model
    temp/.venv-temp/bin/python rollout_smoke.py --n-questions 32 --show 3       # bigger sample, print 3 trajectories

What it does:
  1. Finds the server through the head server and checks that the search tool answers.
  2. Loads the model and builds the real GRPOTrainer (it carries `trainer.model` and `trainer.processing_class`).
  3. Calls `rollout_func(prompts, trainer)` by hand on real dataset rows (each prompt repeated NUM_GEN times, as TRL does).
  4. Prints the checks (PASS / FAIL), one line for each trajectory, and full trajectories with the model tokens
     marked [MODEL] and the environment tokens marked [ENV].
  5. Sends the trajectories to the real /verify through the real reward function.
The statistics of the base model (tool-call rate, how the trajectories end, reward) are the main result.
"""
import argparse
import math
import subprocess
import sys

from train_grpo import (LORA_NAME, MAX_COMPLETION, MAX_NEW, MAX_TURNS, MODEL, NUM_GEN, build_trainer, load_dataset,
                        load_model)
from rollout import GymClient, append_jsonl, final_text, make_gym_verify_reward, make_rollout_func, trajectory_record

RESULTS = {}


def check(name, ok, detail=""):
    RESULTS[name] = bool(ok)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def vram_used_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return int(out.decode().strip().splitlines()[0])


def render_masked(tok, comp, mask):
    """The completion as text, with every run of model tokens and every run of environment tokens marked."""
    parts, start = [], 0
    for i in range(1, len(comp) + 1):
        if i == len(comp) or mask[i] != mask[start]:
            parts.append(f"[{'MODEL' if mask[start] else 'ENV'}]" + tok.decode(comp[start:i]) + f"[/{'MODEL' if mask[start] else 'ENV'}]")
            start = i
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-util", type=float, default=0.4)
    ap.add_argument("--n-questions", type=int, default=8, help="different questions; each gets NUM_GEN attempts")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--show", type=int, default=1, help="how many full trajectories to print")
    ap.add_argument("--dump", default="smoke_dump.jsonl", help="write every trajectory with its reward here (overwritten)")
    args = ap.parse_args()

    client = GymClient.discover()
    print(f"[smoke] resources server at {client.base_url}")
    text = client.call_tool("search_corpus", {"query": "gimbal angle deviations"})
    check("server_search_answers", text.startswith("Result 1:") and text.count("------") == 2)

    model, tok = load_model(args.model, args.gpu_util)
    vram_load = vram_used_mib()
    dataset = load_dataset(args.split)
    step = max(1, len(dataset) // args.n_questions)
    picks = [dataset[i] for i in range(0, len(dataset), step)][:args.n_questions]
    trainer = build_trainer(model, tok, client, dataset.select(range(8)), max_steps=1)
    vram_trainer = vram_used_mib()
    check("single_engine", vram_trainer - vram_load < 2000, f"VRAM after load {vram_load} MiB, after trainer {vram_trainer} MiB")

    prompts = [row["prompt"] for row in picks for _ in range(NUM_GEN)]    # TRL sends the repeats next to each other
    answers = [row["answer"] for row in picks for _ in range(NUM_GEN)]
    rcps = [row["responses_create_params"] for row in picks for _ in range(NUM_GEN)]
    n = len(prompts)

    rollout_func = make_rollout_func(client, LORA_NAME, max_turns=MAX_TURNS, max_new_tokens=MAX_NEW,
                                     max_completion_tokens=MAX_COMPLETION)
    out = rollout_func(prompts, trainer)
    stats = rollout_func.last_stats

    keys = ("prompt_ids", "completion_ids", "logprobs", "env_mask", "trajectory")
    check("one_trajectory_per_prompt", all(len(out[k]) == n for k in keys), f"{n} prompts")
    check("lengths_agree", all(len(c) == len(m) == len(l) for c, m, l in zip(out["completion_ids"], out["env_mask"], out["logprobs"])))
    has_env = [0 in m for m in out["env_mask"]]
    # A call that ended the trajectory by "budget" ran, but its result was not added. Earlier calls still have env tokens.
    expect_env = [t["n_tool_calls"] - (t["ended"] == "budget") > 0 for t in out["trajectory"]]
    check("env_tokens_only_after_a_tool_call", has_env == expect_env)
    sampled = [lp for m, ls in zip(out["env_mask"], out["logprobs"]) for k, lp in zip(m, ls) if k == 1]
    mean_lp = sum(sampled) / max(1, len(sampled))
    check("sampled_logprobs_valid", sampled and all(math.isfinite(x) and x <= 0 for x in sampled), f"mean {mean_lp:.3f}")
    groups = [tuple(map(tuple, out["completion_ids"][g * NUM_GEN:(g + 1) * NUM_GEN])) for g in range(len(picks))]
    check("attempts_differ_inside_a_group", any(len(set(g)) > 1 for g in groups))
    longest = max(len(c) for c in out["completion_ids"])
    check("completion_within_budget", longest <= MAX_COMPLETION, f"longest {longest} of {MAX_COMPLETION}")

    rewards = make_gym_verify_reward(client)(prompts=prompts, completions=[None] * n, trajectory=out["trajectory"],
                                             answer=answers, responses_create_params=rcps)
    check("rewards_from_real_verify", len(rewards) == n and all(0.0 <= r <= 1.0 for r in rewards))

    open(args.dump, "w").close()
    append_jsonl(args.dump, [trajectory_record(prompts[i], answers[i], out["trajectory"][i], rewards[i],
                                               group=i // NUM_GEN, attempt=i % NUM_GEN) for i in range(n)])
    print(f"[smoke] wrote {n} trajectories to {args.dump}")

    print("\n[smoke] one line for each trajectory: group.attempt ended n_calls turns tokens reward | final answer")
    for i, (t, r) in enumerate(zip(out["trajectory"], rewards)):
        ft = final_text(t)
        print(f"  {i // NUM_GEN}.{i % NUM_GEN}  {t['ended']:<9} calls={t['n_tool_calls']} turns={t['n_turns']} "
              f"tokens={len(out['completion_ids'][i]):<5} reward={r:.2f} | {(ft or '(no final message)')[:90]!r}")
    print(f"\n[smoke] statistics: {stats}")
    reward_by_group = [sum(rewards[g * NUM_GEN:(g + 1) * NUM_GEN]) / NUM_GEN for g in range(len(picks))]
    varying = sum(1 for g in range(len(picks)) if len(set(rewards[g * NUM_GEN:(g + 1) * NUM_GEN])) > 1)
    print(f"[smoke] mean reward {sum(rewards) / n:.3f}; groups whose attempts got different rewards: {varying} of {len(picks)} "
          f"(a group with equal rewards gives no learning signal)")
    print(f"[smoke] mean reward per group: {[round(x, 2) for x in reward_by_group]}")

    for i in range(min(args.show, n)):
        print(f"\n[smoke] ---- trajectory {i}: expected answer {answers[i]!r}, reward {rewards[i]:.2f} ----")
        print(render_masked(tok, out["completion_ids"][i], out["env_mask"][i]))

    if stats["tool_call_rate"] == 0:
        print("[smoke] WARN: no trajectory called the tool. The base model may not follow the tool format. Read a trajectory above.")
    if stats["ended_answer"] == 0:
        print("[smoke] WARN: no trajectory ended with a final answer.")
    failed = [k for k, v in RESULTS.items() if not v]
    print(f"\n[smoke] TEST 2 {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
