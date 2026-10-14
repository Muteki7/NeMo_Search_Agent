"""GRPO training of the search agent: builders (used by rollout_smoke.py too) and the training run.

The Gym resources server must run first (`ng_run` with resources_only.yaml, see Learnings.md procedure 4.4).
    cd trainer && temp/.venv-temp/bin/python train_grpo.py --max-steps 5          # Test 3: smoke training

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
from pathlib import Path  # noqa: E402

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from rollout import GymClient, make_gym_verify_reward, make_rollout_func  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "NeMo_Gym" / "resources_servers" / "mission_search" / "data"

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"   # Unsloth loads its 4-bit version. The 1B plumbing model is unsloth/Llama-3.2-1B-Instruct.
LORA_NAME = "grpo_trainer_lora_model"
NUM_GEN = 4
MAX_TURNS = 5
MAX_NEW = 512                  # tokens for one turn
MAX_SEQ = 4096                 # prompt (about 400) + completion must fit
MAX_COMPLETION = 3000          # all sampled tokens plus all tool results of one trajectory


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


def build_trainer(model, tokenizer, client: GymClient, dataset: Dataset, max_steps: int = 5,
                  learning_rate: float = 5e-6, output_dir: str = "grpo_out", batch: int = NUM_GEN, accum: int = 2,
                  dump_path: str = None) -> GRPOTrainer:
    # TODO (before a real run): the hyperparameters. Plan.md says "from autodidact.ipynb cell 5", and we do not have
    # that notebook yet. These are first guesses for the smoke run only.
    config = GRPOConfig(
        output_dir=output_dir, max_steps=max_steps,
        # One rollout call makes batch x accum trajectories (4 x 2 = 8: 2 different prompts, 4 attempts each). The loss runs
        # on `batch` trajectories at a time, so a small `batch` saves GPU memory with the 8B model.
        per_device_train_batch_size=batch, gradient_accumulation_steps=accum, num_generations=NUM_GEN,
        max_completion_length=MAX_COMPLETION, learning_rate=learning_rate,
        logging_steps=1, report_to="none", save_strategy="no", bf16=True,
    )
    return GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[make_gym_verify_reward(client, dump_path=dump_path)],
        args=config, train_dataset=dataset,
        rollout_func=make_rollout_func(client, LORA_NAME, max_turns=MAX_TURNS, max_new_tokens=MAX_NEW,
                                       max_completion_tokens=MAX_COMPLETION),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--gpu-util", type=float, default=0.55)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--batch", type=int, default=NUM_GEN, help="trajectories for one loss step (a multiple of NUM_GEN)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--dump", default="train_dump.jsonl", help="every trajectory with its reward is appended here")
    args = ap.parse_args()

    client = GymClient.discover()   # fails with a clear message if ng_run is not running
    model, tokenizer = load_model(args.model, args.gpu_util, args.lora_rank)
    trainer = build_trainer(model, tokenizer, client, load_dataset("train"), max_steps=args.max_steps,
                            batch=args.batch, accum=args.accum, dump_path=args.dump)
    trainer.train()


if __name__ == "__main__":
    main()
