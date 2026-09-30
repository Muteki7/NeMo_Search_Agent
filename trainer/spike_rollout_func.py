"""Gate 1: can TRL `rollout_func` run on Unsloth's collocated vLLM engine?

Tiny model, dummy 2-turn echo tool, 2 GRPO steps. Prints a PASS/FAIL line per
criterion (a)-(d) from Plan.md. Run from trainer/ with .venv-train active.
"""
import os
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")  # WSL2: vLLM V2 runner needs pinned memory/UVA

from unsloth import FastLanguageModel  # noqa: E402  (must precede trl/transformers)

import gc, json, re, subprocess  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

MODEL = "unsloth/Llama-3.2-1B-Instruct"
NUM_GEN = 4
MAX_TURNS = 2
MAX_NEW = 64
LORA_NAME = "grpo_trainer_lora_model"

RESULTS = {}
CALL_LOG = {"generate_calls": 0, "lora_requests": 0, "batch_sizes": []}


def vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().strip().splitlines()[0])


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.4,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
vram_after_load = vram_used_mib()
print(f"[spike] VRAM after model load (one engine expected): {vram_after_load} MiB")


def echo_tool(text: str) -> str:
    return f"ECHO: {text}"


def make_rollout_func():
    def rollout_func(prompts, trainer):
        """TRL passes prompts ALREADY repeated num_generations times (one entry per completion),
        so return exactly one trajectory per incoming prompt, in the same order."""
        tok = trainer.processing_class
        eng = trainer.model.vllm_engine
        lora = trainer.model.load_lora(LORA_NAME, load_tensors=True)
        sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=MAX_NEW, logprobs=0)

        n = len(prompts)
        convs = [list(m) for m in prompts]
        p_ids = []
        for m in prompts:
            ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
            p_ids.append(list(ids["input_ids"] if isinstance(ids, dict) else ids))
        comp = [[] for _ in range(n)]
        lps = [[] for _ in range(n)]
        mask = [[] for _ in range(n)]
        prev_ids = [list(x) for x in p_ids]

        for turn in range(MAX_TURNS):
            texts = [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
            CALL_LOG["generate_calls"] += 1
            CALL_LOG["batch_sizes"].append(len(texts))
            outs = eng.generate(texts, sp, use_tqdm=False, lora_request=lora)
            if lora is not None:
                CALL_LOG["lora_requests"] += 1
            for i, out in enumerate(outs):
                o = out.outputs[0]
                gen_ids = list(o.token_ids)
                gen_lp = [
                    list(d.values())[0].logprob if d else 0.0
                    for d in (o.logprobs or [None] * len(gen_ids))
                ]
                # env tokens = template/tool tokens between previous_ids and this generation
                full_prompt_ids = tok(texts[i], add_special_tokens=False)["input_ids"]
                env_ids = full_prompt_ids[len(prev_ids[i]):] if turn > 0 else []
                comp[i] += env_ids + gen_ids
                lps[i] += [0.0] * len(env_ids) + gen_lp
                mask[i] += [0] * len(env_ids) + [1] * len(gen_ids)
                prev_ids[i] = full_prompt_ids + gen_ids
                convs[i].append({"role": "assistant", "content": o.text})
                if turn < MAX_TURNS - 1:
                    convs[i].append({"role": "user", "content": echo_tool(o.text[:40])})

        return {
            "prompt_ids": p_ids,
            "completion_ids": comp,
            "logprobs": lps,
            "env_mask": mask,
            "trajectory": [""] * n,  # extra field -> forwarded to reward fns
        }

    return rollout_func


def reward_len(prompts, completions, trajectory=None, **kw):
    # forces the extra-field plumbing to be exercised; also gives non-degenerate variance
    return [float(len(c[0]["content"]) % 7) for c in completions]


dataset = Dataset.from_list(
    [{"prompt": [{"role": "user", "content": f"Say something about number {i}."}]} for i in range(8)]
)

cfg = GRPOConfig(
    output_dir="spike_out",
    max_steps=2,
    per_device_train_batch_size=NUM_GEN,
    gradient_accumulation_steps=1,
    num_generations=NUM_GEN,
    max_completion_length=MAX_TURNS * MAX_NEW + 128,
    learning_rate=5e-6,
    logging_steps=1,
    report_to="none",
    save_strategy="no",
    bf16=True,
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_len],
    args=cfg,
    train_dataset=dataset,
    rollout_func=make_rollout_func(),
)

# (a) compiled cache contains the rollout_func branch
cache = os.path.join(os.getcwd(), "unsloth_compiled_cache", "UnslothGRPOTrainer.py")
n_rf = int(subprocess.run(["grep", "-c", "rollout_func", cache], capture_output=True, text=True).stdout.strip() or 0)
RESULTS["a_rollout_func_in_cache"] = n_rf > 0
print(f"[spike] (a) grep -c rollout_func {cache} -> {n_rf}")

# (b) only one vLLM engine: trainer must reuse model.vllm_engine, and VRAM must not jump
vram_after_trainer = vram_used_mib()
vg = getattr(trainer, "vllm_generation", None)
llm_attr = getattr(trainer, "llm", None)
reuses = (llm_attr is model.vllm_engine) or (
    vg is not None and any(getattr(vg, a, None) is model.vllm_engine for a in ("llm", "engine", "vllm_engine"))
)
print(f"[spike] (b) trainer.llm is model.vllm_engine: {llm_attr is model.vllm_engine}; "
      f"vllm_generation={type(vg).__name__ if vg is not None else None}; reuses={reuses}")
print(f"[spike] (b) VRAM load={vram_after_load} MiB, after trainer init={vram_after_trainer} MiB")
RESULTS["b_single_engine"] = reuses and (vram_after_trainer - vram_after_load) < 2000

trainer.train()

RESULTS["c_lora_used_in_rollout"] = CALL_LOG["generate_calls"] > 0 and CALL_LOG["lora_requests"] == CALL_LOG["generate_calls"]
print(f"[spike] (c) generate calls={CALL_LOG['generate_calls']}, with lora_request={CALL_LOG['lora_requests']}")

# (d) grouping + env_mask reflected in completions/mean_length
hist = [h for h in trainer.state.log_history if "completions/mean_length" in h]
ml = hist[0]["completions/mean_length"] if hist else None
# Every completion has MAX_TURNS generations of <= MAX_NEW tokens plus env tokens; with env_mask the
# logged length counts only generated tokens, so it must be <= MAX_TURNS*MAX_NEW.
RESULTS["d_env_mask_and_grouping"] = ml is not None and ml <= MAX_TURNS * MAX_NEW and all(
    b % NUM_GEN == 0 for b in CALL_LOG["batch_sizes"]
)
print(f"[spike] (d) completions/mean_length={ml} (cap {MAX_TURNS * MAX_NEW}); batch_sizes={CALL_LOG['batch_sizes'][:4]}")

print("\n===== GATE 1 RESULTS =====")
for k, v in RESULTS.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
print("GATE 1:", "PASS" if all(RESULTS.values()) else "FAIL")
