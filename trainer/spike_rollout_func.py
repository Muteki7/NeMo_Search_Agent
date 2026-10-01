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
#Add logs
ROLLOUT_LOG = [] #one dict per rollout_func call
VIOLATIONS = {"prefix": 0, "len_mismatch": 0, "env_lp_nonzero": 0, "gen_lp_positive": 0}
PREFIX_DETAILS = []  # what differed when re-rendering the conversation did not reproduce the previous token ids

#Module level logs

LORA_IDS = [] #lora_int_id seen by each rollout_func call

def lora_b_abs_sum(m):
    with torch.no_grad():
        """
        This process helps us monitor LoRA Magnitude: To track how much weight/magnitude the LoRA B matrices are
        accumulating
        
        for loop iterates through all trainable parameters in model m and filters specifically for 
        parameters who's name is lora_B

        p.detach(): Disables gradient tracking
        
        float(): converts parameters to 32-bit floating point precision (FP32) to prevent over/underflow
        if training was done in F16 or BF16
        
        abs().sum().item(): Takes abs value of every element and sums them up and converts the single-element
        
        PyTorch tensor to a standard Python number.
        return sum(..): sums abs values across all matched lora_B and returns a single floating-point number
        """
        return sum(p.detach().float().abs().sum().item() 
                   for n, p in m.named_parameters() if "lora_B" in n)

def fill_lora_b(m, seed, std=0.02):
    """
    We are essentially adding gaussian noise to LoRA B matrices.
    Now LoRA adaption is computed as $\nabal{W} = BxA$ generally B is initialized to 0 so the step 0 outputs are 100% identical
    to the base model. Since we are adding noise here that will break, so this is a test to see if LoRA adapter is even working.

    g=torch.Gen.. : Creates a deterministic CPU random num generator locked to seed

    with torch._no_grad(): Disables autograd gradient tracking so PyTorch allows directly modifying parameter values in-place
    
    for loop: finding every parameter tensor that belogns to LoRA B matrix

    p.copy_(...) : Overwrites existing values of parameters in-place with:
        torch.randn(..)*std: Draws values from the normal distribution with mean 0 and std (default to 0.22)
        .to(device=..): Ensures generated values match parameter's target GPU/CPU device and data type (F16, BF16, FP32)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=g) * std).to(device=p.device, dtype=p.dtype))

def greedy_ids(eng, text, lora_request):
    """
    Checking if our LoRA adapter is able to output tokens

    sp0 = SamplingParams(..): 
        temperature=0.0: Forces Greedy Decoding (pick highes probability token, making output 100% deterministic)
        max_tokens=32 : Limits generated output length to at most 32 tokens
    
        out = eng.generated(..): Passes hte prompt text into the inference engine (eng), applying specified dynamic
        LoRA adapter (lora_request) to process the request.

        return list(..): Extracts an array of generated token ID integers([1024, 882, 319..]) from first completion
        sequence of response objects and converts it to standard python list
    """
    sp0 = SamplingParams(temperature=0.0, max_tokens=32)
    out = eng.generate([text], sp0, use_tqdm=False, lora_request=lora_request)
    return list(out[0].outputs[0].token_ids)


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
        #Logging the adapter ID
        LORA_IDS.append(lora.lora_int_id)
        sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=MAX_NEW, logprobs=0)

        n = len(prompts)

        #Group checking
        """
        Since we use GRPO we check and calculate our advantage function with respect to the group of completion prompts
        This checks if the incoming prompts are structure in groups
        """
        groups_ok = all(prompts[j] == prompts[(j // NUM_GEN) * NUM_GEN] for j in range(n))
        n_groups = len({json.dumps(p) for p in prompts})

        convs = [list(m) for m in prompts]
        p_ids = []
        for m in prompts:
            ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
            p_ids.append(list(ids["input_ids"] if isinstance(ids, dict) else ids))
        comp = [[] for _ in range(n)]
        lps = [[] for _ in range(n)]
        mask = [[] for _ in range(n)]
        gen_count = [0] * n #Tokens the vLLM actually generated
        prev_ids = [list(x) for x in p_ids]
        TURN1_FINISH = [None] * n   # why turn 1 stopped ("stop" = EOS, "length" = hit max_tokens)

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
                if turn == 0:
                    #Storing Turn1's finish reason
                    TURN1_FINISH[i] = o.finish_reason
                gen_count[i]+=len(gen_ids)
                gen_lp = [
                    list(d.values())[0].logprob if d else 0.0
                    for d in (o.logprobs or [None] * len(gen_ids))
                ]
                # env tokens = template/tool tokens between previous_ids and this generation
                full_prompt_ids = tok(texts[i], add_special_tokens=False)["input_ids"]

                """
                For multi-turn generation it checks if the new tokens are being properly appened without
                changing the prefix tokens
                
                prev_ids[i]: The token IDs of conversation up to previous turn
                full_prompt_ids: The token IDs of current conversation with the appended generated tokens
                if the prefix of full_prompt_ids and prev_ids[i] doesn't match then our rollout_func is not properly appending the new tokens
                this can cause the policy to be slightly off-policy.
                """
                if turn>0 and full_prompt_ids[:len(prev_ids[i])] != prev_ids[i]:
                    VIOLATIONS["prefix"] +=1
                    #We find the first position where the two tokens differ and log the surrounding tokens for debugging
                    _j = next((k for k, (a, b) in enumerate(zip(full_prompt_ids, prev_ids[i])) if a != b),
                              min(len(full_prompt_ids), len(prev_ids[i])))
                    PREFIX_DETAILS.append({
                        "first_diff": _j, "len_prev": len(prev_ids[i]), "len_new": len(full_prompt_ids),
                        "prev_tokens": tok.convert_ids_to_tokens(prev_ids[i][max(0, _j - 4):_j + 4]),
                        "new_tokens": tok.convert_ids_to_tokens(full_prompt_ids[max(0, _j - 4):_j + 4]),
                        "turn1_finish": TURN1_FINISH[i],
                    })

                env_ids = full_prompt_ids[len(prev_ids[i]):] if turn > 0 else []
                comp[i] += env_ids + gen_ids
                lps[i] += [0.0] * len(env_ids) + gen_lp
                mask[i] += [0] * len(env_ids) + [1] * len(gen_ids)
                prev_ids[i] = full_prompt_ids + gen_ids
                convs[i].append({"role": "assistant", "content": o.text})
                if turn < MAX_TURNS - 1:
                    convs[i].append({"role": "user", "content": echo_tool(o.text[:40])})

        #Alignment, logprob and sample checks
        """
        comp: Completion tokens
        mask: loss mask
        lps: log-probabilities

        We ensure that these are all strictly equal
        """
        for i in range(n):
            if not (len(comp[i])==len(mask[i])==len(lps[i])):
                VIOLATIONS["len_mismatch"] +=1

            """Ensuring that the environment tokens don't contribute to the log_probs"""
            VIOLATIONS["env_lp_nonzero"] += sum(1 for m, lp in zip(mask[i], lps[i]) if m==0 and lp!=0.0)
            
            """Ensuring generated tokens have non-zero log probabilites"""
            VIOLATIONS["gen_lp_positive"] += sum(1 for m, lp in zip(mask[i], lps[i]) if m==1 and lp>0.0)
        ROLLOUT_LOG.append({
            "n": n, "groups_ok": groups_ok, "n_groups": n_groups,
            "mean_len": sum(len(c) for c in comp) / n,
            "mean_gen": sum(sum(m) for m in mask) / n,
            "min_env": min(len(m)-sum(m) for m in mask),
            "env_text": tok.decode([t for t, m in zip(comp[0], mask[0]) if m == 0]),
            "gen_text": tok.decode([t for t, m in zip(comp[0], mask[0]) if m == 1]),
            "gen_count_ok" : all(sum(mask[i])==gen_count[i] for i in range(n)),
        })
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
    per_device_train_batch_size= 2*NUM_GEN, #2 unique prompts repeats per rollout call
    gradient_accumulation_steps=1,
    num_generations=NUM_GEN,
    max_completion_length=MAX_TURNS * MAX_NEW + 128,
    learning_rate=1e-4, #Making training move the adapter more
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

#c1 before training
b_before = lora_b_abs_sum(trainer.model)

"""
d6 spy: record the surviving-token count before/after the env_mask (tool_mask) is multiplied into the
completion_mask inside the compiled trainer's loss path. If after < before, env tokens are zeroed in the loss.
Patch the module that defines the trainer class (grpo_accumulated_loss looks the function up in that module's globals).
"""
import sys
_trainer_mod = sys.modules[type(trainer).__module__]
if not hasattr(_trainer_mod, "align_completion_tool_mask"):
    _trainer_mod = next(m for k, m in sys.modules.items()
                        if k.endswith("UnslothGRPOTrainer") and hasattr(m, "align_completion_tool_mask"))
_orig_align = _trainer_mod.align_completion_tool_mask
LOSS_MASK = []   # (tokens before env_mask, tokens after env_mask) per call

def _spy_align(tool_mask, completion_mask):
    out = _orig_align(tool_mask, completion_mask)
    if tool_mask is not None:
        LOSS_MASK.append((int(completion_mask.sum()), int(out.sum())))
    return out

_trainer_mod.align_completion_tool_mask = _spy_align

trainer.train()

#c1,c2,c3 after training

"""
c0 & c1: Verifying LoRA Weight Mutation
 c0: Assets taht every single generation request during rollout passsed an active lora_request
 c1: Confirms that the initial LoRA B matrix was zero (<1e-9) and after initialzing and training
 it become non-zero (>0.0) indicating that our LoRA adapter is working 
"""

b_after=lora_b_abs_sum(trainer.model)
RESULTS["c0_lora_attached_each_call"] = (
    CALL_LOG["generate_calls"] > 0 and CALL_LOG["lora_requests"] == CALL_LOG["generate_calls"])

RESULTS["c1_adapter_updated"] = b_before <1e-9 and b_after > 0.0
print(f"[spike] (c1) sum|lora_b| before={b_before:.3e} after={b_after:.3e}")

eng = trainer.model.vllm_engine
prompt_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Say something about number 3."}],
    add_generation_prompt=True, tokenize=False)

"""
Testing Base model control
We are generate requests from base mode twice lora_request=None (greedy_ids)
verify deterministic greedy output => base_a==base_b
"""

base_a=greedy_ids(eng, prompt_text, None)
base_b=greedy_ids(eng, prompt_text, None)

"""
Testing Hot-swapping & live adapter influence (c2)
fill_lora_b(..): Manually adding Gaussian noise to Matrix B

trainer.model.load_lora: Reloading tensors with this added noise

c2_live_adapter..: Asserts that attaching lora1 changes the model's output tokens, proving that the
adapter is working 
"""
fill_lora_b(trainer.model, seed=1)
lora1 = trainer.model.load_lora(LORA_NAME, load_tensors=True)
out1 = greedy_ids(eng, prompt_text, lora1)
out1_again = greedy_ids(eng, prompt_text, lora1)

fill_lora_b(trainer.model, seed=2)
lora2 = trainer.model.load_lora(LORA_NAME, load_tensors=True)
out2= greedy_ids(eng, prompt_text, lora2)

RESULTS["c2_live_adapter_changes_output"] = base_a == base_b and out1 !=base_a

"""
c3: Guarding Against Caching & Stale-Adapter bugs
Engine implementations like vLLM uses LRU caches for dynamic LoRA adapters, if the engine caches adapater
by name/ID then even after updating weights under LORA_NAME in-place might cause inference to use old cached
adapter instead of fetching the updated weights

Generates lora1 output twice to confirm that it is stable

We then again generate lora2 and out2 and prove that lora2 produces a different output sequence to lora1 (out2!=out1)
This shows that cache is updated

lora2.lora_int_id > lora1...: Force vLLM to treat update as a fresh adapter instance.
"""
RESULTS["c3_no_stale_adapter"] = (
    out1==out1_again and out2 != out1 and lora2.lora_int_id > lora1.lora_int_id
    and LORA_IDS==sorted(set(LORA_IDS)) 
)

print(f"[spike] (c) lora ids in rollouts={LORA_IDS}; out1!=base={out1!=base_a}; out2!=out1={out2 != out1}")


#Before i test d
"""
TRL's in-built log checker to see completion token lenghts, idea is we want to compare that with our rollout token length
Since our custom rollout is properly masked (d2 test passed, env tokens are not added), properly appending without messing
up our prefix tokens (d3 pass), but we are failing up with the token matching for TRL vs Custom rollouts
"""
logged = [h["completions/mean_length"] for h in trainer.state.log_history if "completions/mean_length" in h]
RESULTS["d1_alignment"] = (VIOLATIONS["len_mismatch"]==0 and VIOLATIONS["env_lp_nonzero"]==0 and VIOLATIONS["gen_lp_positive"] == 0)

RESULTS["d2_env_tokens_present"] = bool(ROLLOUT_LOG) and all(r["min_env"] > 0 for r in ROLLOUT_LOG)

RESULTS["d3_prefix_preserving"] = VIOLATIONS["prefix"] == 0

"""
Dealing with d4 test failing we have a couple options:
len(logged) == len(ROLLOUT_LOG): To ensure TRL and Custom worker are logged 1-to-1 lockstep

abs(1-r["mean_gen"] < 1e-3): This is to check that our custom worker generated atleast ~1.0 tokens

l<r["mean_len"]: Important check to ensure that TRL is isolating environment tokens from policy updates
"""

"""
Checking where the failure point is by breaking down d4 check
"""
RESULTS["d4a_log_step_sync"] = (len(logged) == len(ROLLOUT_LOG))

if RESULTS["d4a_log_step_sync"]:
    #Rollout Drift check: Did our custom rollout generator drift from expected tokens generated
    RESULTS["d4b_mask_sum_matches_vllm_counts"] = all(
    r["gen_count_ok"] and 1 <= r["mean_gen"] <= MAX_TURNS * MAX_NEW for r in ROLLOUT_LOG)

    #if l>r["mean_gen"] then TRL is counting ENV/Prompt tokens as well
    #This reveals if TRL token is matching exactly to only model-generated tokens
    RESULTS["d4c_trl_mask_exact_match"] = all(
        abs(l-r["mean_gen"]) < 1e-3 for l,r in zip(logged, ROLLOUT_LOG) 
    )

    #Here true means TRL logged length is strictly below raw length
    RESULTS["d4d_trl_excludes_env_tokens"] = all(
    l < r["mean_len"] for l, r in zip(logged, ROLLOUT_LOG))

else:
    RESULTS["d4b_rollout_drift"]=False
    RESULTS["d4c_trl_mask_exact_match"]=False
    RESULTS["d4d_trl_includes_env_tokens"]=False

RESULTS["d4_metric_matches_mask"] = (
    RESULTS["d4a_log_step_sync"] and RESULTS["d4c_trl_mask_exact_match"]
    and RESULTS["d4d_trl_excludes_env_tokens"]
)

RESULTS["d5_grouping"] = bool(ROLLOUT_LOG) and all(
    r["groups_ok"] and r["n_groups"] >= 2 and r["n"] == r["n_groups"] * NUM_GEN for r in ROLLOUT_LOG)


#d6: env tokens are actually zeroed in the loss (surviving-token count drops after the env_mask is applied)
RESULTS["d6_env_zeroed_in_loss"] = bool(LOSS_MASK) and all(after < before for before, after in LOSS_MASK)
_tot_before = sum(b for b, _ in LOSS_MASK)
_tot_after = sum(a for _, a in LOSS_MASK)
print(f"[spike] (d6) calls={len(LOSS_MASK)} (before, after)={LOSS_MASK}; totals before={_tot_before} after={_tot_after}")
print(f"[spike] (d6) rollout totals: raw tokens={sum(r['mean_len'] * r['n'] for r in ROLLOUT_LOG):.0f} "
      f"generated tokens={sum(r['mean_gen'] * r['n'] for r in ROLLOUT_LOG):.0f}")

print(f"[spike] (d) violations={VIOLATIONS}; logged mean_length={logged}; mask-sum means={[round(r['mean_gen'],2) for r in ROLLOUT_LOG]}; raw means={[round(r['mean_len'], 2) for r in ROLLOUT_LOG]}")
print("[spike] (d) mask==0 text (env):", repr(ROLLOUT_LOG[0]["env_text"]))
print("[spike] (d) mask==1 text (model):", repr(ROLLOUT_LOG[0]["gen_text"]))
print("[spike] (d3) prefix violation details:", PREFIX_DETAILS)

# # (d) grouping + env_mask reflected in completions/mean_length
# hist = [h for h in trainer.state.log_history if "completions/mean_length" in h]
# ml = hist[0]["completions/mean_length"] if hist else None
# # Every completion has MAX_TURNS generations of <= MAX_NEW tokens plus env tokens; with env_mask the
# # logged length counts only generated tokens, so it must be <= MAX_TURNS*MAX_NEW.
# RESULTS["d_env_mask_and_grouping"] = ml is not None and ml <= MAX_TURNS * MAX_NEW and all(
#     b % NUM_GEN == 0 for b in CALL_LOG["batch_sizes"]
# )
# print(f"[spike] (d) completions/mean_length={ml} (cap {MAX_TURNS * MAX_NEW}); batch_sizes={CALL_LOG['batch_sizes'][:4]}")

print("\n===== GATE 1 RESULTS =====")
for k, v in RESULTS.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
print("GATE 1:", "PASS" if all(RESULTS.values()) else "FAIL")
