"""GRPO training of the search agent: builders (used by rollout_smoke.py and eval.py too) and the training run.

The Gym resources server must run first (`ng_run` with resources_only.yaml, see Learnings.md procedure 4.4).
    cd trainer
    temp/.venv-temp/bin/python train_grpo.py --max-steps 4 --save-steps 2 --name test       # test run: checks save and eval
    temp/.venv-temp/bin/python train_grpo.py --max-steps 200 --save-steps 50 2>&1 | tee run.log   # the first real run

Each run gets its own folder `trainer/runs/<start time>[_<name>]/` with:
    run_config.json     the arguments, the hyperparameters and the constants of this run
    train_dump.jsonl    every trajectory with its reward and the optimizer step (read it, or re-score it)
    checkpoint-<step>/  the LoRA adapter (and the optimizer state) at that step
    final/              the LoRA adapter after the last step
Evaluate a run with eval.py (it writes its dumps into the same folder).

Resume after a crash (it uses the settings saved in run_config.json and continues in the same run folder):
    temp/.venv-temp/bin/python train_grpo.py --resume runs/<run>/checkpoint-100
The dump lines of the steps after the checkpoint move to train_dump.jsonl.dropped_after_step<N>.jsonl (they run again).

`GRPOTrainer` below is the Unsloth-patched class (importing unsloth first swaps in unsloth_compiled_cache). It calls
our `rollout_func` for each batch, applies the env mask in the loss and does the group advantages itself.
"""
import os
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")  # WSL2: vLLM needs pinned memory

from unsloth import FastLanguageModel  # noqa: E402  (must come before trl and transformers)

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from rollout import (GymClient, checkpoint_step, make_gym_verify_reward, make_rollout_func,  # noqa: E402
                     prune_dump_after_step)

TRAINER_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINER_DIR.parent
DATA_DIR = REPO_ROOT / "NeMo_Gym" / "resources_servers" / "mission_search" / "data"
RUNS_DIR = TRAINER_DIR / "runs"

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"   # Unsloth loads its 4-bit version. The 1B plumbing model is unsloth/Llama-3.2-1B-Instruct.
LORA_NAME = "grpo_trainer_lora_model"
NUM_GEN = 8                    # attempts for each prompt (one group). With batch x accum = 8, one prompt for each step.
MAX_TURNS = 5
MAX_NEW = 512                  # tokens for one turn
MAX_SEQ = 4096                 # prompt (about 400) + completion must fit
MAX_COMPLETION = 3000          # all sampled tokens plus all tool results of one trajectory
TEMPERATURE = 0.9              # the rollout samples with it, and TRL divides its logits by it: keep the two equal
TOP_P = 0.95

# From AutoDidact (autodidact.ipynb). Dropped: use_agentic_generate (our rollout_func replaces it) and
# max_prompt_length (removed in TRL 0.29). beta = 0: no KL term, so no reference model.
# Run 2 differs from run 1 in two ways: scale_rewards="none" (below) and the retrieval reward in the verifier (app.py).
HYPERPARAMETERS = dict(
    learning_rate=5e-6, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
    warmup_steps=0.1,   # transformers 5 has no warmup_ratio: a warmup_steps below 1 is a ratio (AutoDidact: warmup_ratio=0.1)
    lr_scheduler_type="cosine", optim="paged_adamw_8bit", max_grad_norm=0.1, beta=0.0,
    scale_rewards="none",   # run 2: no division by the group std (Dr. GRPO). Run 1 used the TRL default, "group".
)


def load_model(model_name: str = MODEL, gpu_util: float = 0.55, lora_rank: int = 64):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=MAX_SEQ, load_in_4bit=True, fast_inference=True,
        gpu_memory_utilization=gpu_util, max_lora_rank=lora_rank,   # the default max_lora_rank is 16: vLLM refuses a larger LoRA
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_rank, lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    return model, tokenizer


def load_dataset(split: str = "train", limit: int = None) -> Dataset:
    """The `prompt` column is the message list. `answer` and `responses_create_params` pass through to the reward."""
    rows = [json.loads(line) for line in (DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if limit:
        rows = rows[:limit]
    return Dataset.from_list([{"prompt": r["responses_create_params"]["input"], "answer": r["answer"],
                               "responses_create_params": r["responses_create_params"]} for r in rows])


def make_run_dir(name: str = None) -> Path:
    run_dir = RUNS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{name}" if name else ""))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_trainer(model, tokenizer, client: GymClient, dataset: Dataset, max_steps: int = 5, output_dir: str = "grpo_out",
                  save_steps: int = None, batch: int = 4, accum: int = 2, dump_path: str = None,
                  step_offset: int = 0) -> GRPOTrainer:
    # One rollout call makes batch x accum trajectories (4 x 2 = 8 = one prompt, NUM_GEN attempts). The loss runs on
    # `batch` trajectories at a time, so a small `batch` saves GPU memory. The advantages are computed on all 8 first.
    config = GRPOConfig(
        output_dir=output_dir, max_steps=max_steps,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accum, num_generations=NUM_GEN,
        max_completion_length=MAX_COMPLETION, temperature=TEMPERATURE,
        logging_steps=1, report_to="none", bf16=True,
        save_strategy="steps" if save_steps else "no", save_steps=save_steps or 500,
        **HYPERPARAMETERS,
    )
    return GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[make_gym_verify_reward(client, dump_path=dump_path, step_offset=step_offset)],
        args=config, train_dataset=dataset,
        rollout_func=make_rollout_func(client, LORA_NAME, max_turns=MAX_TURNS, max_new_tokens=MAX_NEW,
                                       max_completion_tokens=MAX_COMPLETION, temperature=TEMPERATURE, top_p=TOP_P),
    )


def current_settings() -> dict:
    """Everything that must stay the same between a run and its resume."""
    return {"hyperparameters": HYPERPARAMETERS,
            "constants": {"NUM_GEN": NUM_GEN, "MAX_TURNS": MAX_TURNS, "MAX_NEW": MAX_NEW, "MAX_SEQ": MAX_SEQ,
                          "MAX_COMPLETION": MAX_COMPLETION, "TEMPERATURE": TEMPERATURE, "TOP_P": TOP_P}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=50, help="save the LoRA every this many steps (0: only at the end)")
    ap.add_argument("--name", default=None, help="added to the run folder name")
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4, help="trajectories for one loss step (batch x accum must be a multiple of NUM_GEN)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--train-split", default="train", help="the data file to train on: data/<split>.jsonl (run 3 uses train_dedup, see scripts/dedup_data.py)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="a checkpoint-N folder: continue that run from step N, in the same run folder, with its saved settings")
    ap.add_argument("--force", action="store_true",
                    help="with --resume: go on although the constants or hyperparameters in this file differ from the saved run")
    args = ap.parse_args()

    client = GymClient.discover()   # fails with a clear message if ng_run is not running
    resume_ckpt, resume_step = None, 0
    if args.resume:
        resume_ckpt = args.resume.resolve()
        missing = [f for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors")
                   if not (resume_ckpt / f).exists()]
        if missing:
            raise SystemExit(f"{resume_ckpt} is not a complete checkpoint (missing {missing}). `final/` cannot be resumed from.")
        run_dir = resume_ckpt.parent
        config = json.loads((run_dir / "run_config.json").read_text())
        for key, value in config["args"].items():          # the original settings win: the schedule must not change
            setattr(args, key, value)
        now = current_settings()
        differs = {f"{group}.{k}": (config[group].get(k), v) for group in now for k, v in now[group].items()
                   if config[group].get(k) != v}
        if differs and not args.force:
            raise SystemExit("These settings differ from the saved run (saved, now): "
                             + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in differs.items())
                             + ". Restore them in train_grpo.py, or add --force.")
        resume_step = checkpoint_step(resume_ckpt)
        if resume_step >= args.max_steps:
            raise SystemExit(f"{resume_ckpt.name} is at step {resume_step}, and max_steps is {args.max_steps}: nothing to do.")
        moved = prune_dump_after_step(run_dir / "train_dump.jsonl", resume_step)
        config.setdefault("resumes", []).append({
            "time": datetime.now().isoformat(timespec="seconds"), "checkpoint": resume_ckpt.name, "step": resume_step,
            "dump_lines_moved_aside": moved, "server": client.base_url, "settings_differ": sorted(differs)})
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        print(f"[train] resuming {run_dir.name} from {resume_ckpt.name} (step {resume_step} of {args.max_steps}); "
              f"{moved} dump lines of later steps moved to train_dump.jsonl.dropped_after_step{resume_step}.jsonl")
    else:
        run_dir = make_run_dir(args.name)
        (run_dir / "run_config.json").write_text(json.dumps({
            "args": {k: v for k, v in vars(args).items() if k not in ("resume", "force")},
            **current_settings(), "server": client.base_url}, indent=2))
        print(f"[train] run folder: {run_dir}")

    model, tokenizer = load_model(args.model, args.gpu_util, args.lora_rank)
    trainer = build_trainer(model, tokenizer, client, load_dataset(args.train_split), max_steps=args.max_steps,
                            output_dir=str(run_dir), save_steps=args.save_steps or None, batch=args.batch, accum=args.accum,
                            dump_path=str(run_dir / "train_dump.jsonl"), step_offset=resume_step)
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    trainer.save_model(str(run_dir / "final"))
    print(f"[train] done. LoRA saved in {run_dir}. Evaluate: python eval.py --run-dir {run_dir} "
          f"--checkpoints {run_dir}/checkpoint-* {run_dir}/final")


if __name__ == "__main__":
    main()
