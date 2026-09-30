# Plan: Rebuild AutoDidact's search agent on NeMo Gym (fresh project, clean deps)

## Context

`use_agentic_generate` is gone from current Unsloth. TRL's `environment_factory` needs `transformers>=5.2.0`, and Unsloth's source patching crashes against that (`auto_docstring`, and `PreTrainedConfig` in unsloth#3786). You want a structure that scales to more complex environments, so overriding trainer internals is ruled out.

Research findings that shape the design:
- **Unsloth's NeMo Gym notebook (`Example_NeMo.ipynb`) is single-turn only.** It starts resources servers with `resources_only.yaml` and only POSTs the finished completion to `/verify` inside `reward_fn`. Tools are never called during training.
- **NeMo Gym's official multi-turn TRL route won't fit.** It needs a separate `trl vllm-serve` process on a second GPU, and Unsloth force-sets `vllm_mode='colocate'` (`unsloth/models/rl.py:869`). You have one RTX 5090 Laptop GPU (24 GB).
- **TRL `rollout_func` is the extension point that fits.** It exists from TRL 0.25, and `env_mask` (keeps tool-output tokens out of the loss) from 0.28. It returns `prompt_ids`/`completion_ids`/`logprobs`/`env_mask`, and any extra fields are forwarded to reward functions (`trl/trainer/grpo_trainer.py:1553-1605`).

**Decisions (confirmed with you):**
1. A generic multi-turn loop runs in `rollout_func` on Unsloth's collocated vLLM.
2. A NeMo Gym resources server owns the environment: the `search_corpus` tool plus `verify()`, both over HTTP.
3. `verify()` uses a deterministic answer match.
4. Fresh `search_gym/` directory with clean venvs.
5. Reuse the existing `questions.json`/`chunks.pkl`. No question regeneration.

**Why this scales:** the rollout loop is environment-agnostic. It only knows "generate → parse JSON tool call → POST `/{tool_name}` → append result → repeat" and "POST `/verify`". Adding a new environment later means adding a new resources server plus data rows carrying a server ref. The trainer does not change.

## Layout

```
reasoning_model/search_gym/
├── Gym/                                   # git clone NVIDIA-NeMo/Gym (own uv venv, Python 3.13.14 pinned)
│   └── resources_servers/mission_search/  # our environment (NeMo Gym scaffold layout)
│       ├── app.py            # tool endpoint + verify
│       ├── embeddings.py     # copied from AutoDidact/embeddings.py
│       ├── configs/mission_search.yaml, configs/resources_only.yaml
│       ├── data/{example,train,validation}.jsonl, chroma_index/
│       ├── scripts/build_index.py, scripts/convert_questions.py
│       ├── tests/{test_app.py, verifier_cases.jsonl}
│       └── requirements.txt  # -e nemo-gym[dev] @ ../../ , langchain-chroma, chromadb, transformers, torch (CPU)
└── trainer/                               # training venv (.venv-train, Python 3.12)
    ├── rollout.py      # generic multi-turn rollout_func + gym client
    ├── train_grpo.py   # Unsloth model + GRPOTrainer(rollout_func=..., reward_funcs=[gym_verify_reward])
    ├── eval.py         # held-out eval, with and without LoRA
    └── spike_rollout_func.py
```
AutoDidact stays untouched as a reference.

## Steps

### 0. Environments (clean, one-shot installs)
- **NeMo Gym:** `git clone https://github.com/NVIDIA-NeMo/Gym search_gym/Gym`, then `uv python install && uv sync` (honours `.python-version` 3.13.14, as the notebook's cell 5 explains). Scaffold with `gym env init --resources-server mission_search`; fall back to copying an existing server's layout if the CLI differs in the clone.
- **Training venv:** `uv venv search_gym/trainer/.venv-train --python 3.12`, then a single `uv pip install` with all pins:
  - `unsloth==2025.9.9`, `trl==0.29.1`, `transformers==4.56.2`, `vllm==0.22.0`, `torch==2.11.0`, `torchvision==0.26.0`, `datasets==3.6.0`, plus `requests pyyaml omegaconf ipykernel`.
  - This is the stack that is `pip check`-clean today. No langchain/chroma here, since those live in the env server.
  - Try `unsloth_zoo==2025.9.*` to match core, and accept whatever `uv` resolves if that's unsatisfiable. The current venv pairs `unsloth 2025.9.9` with a mismatched `unsloth_zoo 2026.9.7`.
  - Verify with `pip check`, then register a Jupyter kernel.

### 1. Gate: spike `rollout_func` on Unsloth (before building anything else)
`spike_rollout_func.py`, using a tiny model (`unsloth/Llama-3.2-1B-Instruct`, 4-bit, `fast_inference=True`), a dummy 2-turn echo tool, and 2 GRPO steps. Pass criteria:
- (a) The regenerated `unsloth_compiled_cache/UnslothGRPOTrainer.py` contains the `rollout_func` branch (`grep -c rollout_func`).
- (b) Only one vLLM engine is allocated. Watch VRAM. TRL 0.29 uses `self.vllm_generation`, but Unsloth 2025.9.9's regex targets the older `self.llm = LLM(...)`, so it may not reuse `model.vllm_engine`.
- (c) Generation inside `rollout_func` uses the live adapter: `trainer.model.vllm_engine.generate(..., lora_request=trainer.model.load_lora("grpo_trainer_lora_model", load_tensors=True))`, the same call Unsloth injects at `rl.py:1004`.
- (d) Output ordering matches `num_generations` grouping, and `completions/mean_length` reflects `env_mask`.

**If it fails:** switch the trainer to plain TRL + PEFT QLoRA with TRL's own `vllm_mode="colocate"`. Use a smaller model for 24 GB (Llama-3.2-3B-Instruct). The resources server, data, and `rollout.py` stay unchanged. That fallback is the payoff of the decoupled design.

### 2. Data: `scripts/convert_questions.py`
- Read `AutoDidact/saved_data/questions.json` (1434 rows of `{chunk_id, question, answer, difficulty}`).
- Emit NeMo Gym JSONL rows:
  - `responses_create_params`:
    - `input`: system prompt from `get_system_prompt()` and user prompt from `build_user_prompt(q)`, both in `rl_helpers.py:22,52`.
    - `tools`: `SEARCH_TOOL_DEFINITION` (`rl_helpers.py:34`) converted to the flat Responses-API form `{type, name, description, parameters, strict}`.
    - `parallel_tool_calls`: `false`.
  - Top-level fields: `answer`, `difficulty`, `chunk_id`.
- Split 90/10 with seed 42 (same as `search_module.get_qa_dataset`, `search_module.py:161`) into `train.jsonl`/`validation.jsonl`, plus 5 rows in `example.jsonl`.

### 3. Index: `scripts/build_index.py`
Rebuild Chroma inside the server venv from `AutoDidact/saved_data/chunks.pkl`, using `CustomHuggingFaceEmbeddings` on CPU and collection `example_collection`. This mirrors `generate_data.py:46-52` and avoids carrying the old sqlite across chromadb versions. It takes a couple of minutes for 753 chunks.

### 4. Resources server: `app.py` (pattern from the NeMo Gym single-step tutorial)
- **Config:** `MissionSearchConfig(BaseResourcesServerConfig)` with `chroma_dir`, `collection_name`, `embedding_model`, `top_k=2` (the old `run_tool_calls` used 2). Set `REVERIFY_MODE = STATELESS`.
- **Tool:** `POST /search_corpus` takes `SearchCorpusRequest{query: str}` and returns `SearchCorpusResponse{results: str}`. It reuses the `load_vectorstore` and `search` formatting logic (`search_module.py:23,49`).
- **Verify:** `MissionSearchVerifyRequest(BaseVerifyRequest)` adds `answer: str`.
  - Take the last `type=="message"` output text and strip reasoning with `remove_reasoning` (`rl_helpers.py:131`, copied).
  - Reward is 1.0 on a normalized exact or containment match, otherwise token-F1 against `answer`. It is 0.0 if there is no final message.
  - Return `BaseVerifyResponse(**body.model_dump(), reward=...)`.
- **Configs:** `mission_search.yaml` (resources server plus a `simple_agent` entry for later `gym eval run`) and `resources_only.yaml` (used for training, as in the notebook).
- **Tests:** a `VerifierFixture` with `verifier_cases.jsonl` (full, zero, and malformed cases), plus a unit test of `search_corpus`.

### 5. Trainer: `rollout.py` + `train_grpo.py`
- **Gym client:** discover server host/port from the head server's `GET :11000/global_config_dict_yaml`. Copy the pattern from `Example_NeMo.ipynb` cell 7.
- **`make_rollout_func(max_turns=5)`**, a batched loop modelled on `run_agent` (`rl_helpers.py:288`):
  - For each prompt × `num_generations`: render with the chat template and generate with the call from step 1(c).
  - Parse with `extract_json_objects` (`rl_helpers.py:89`, copied). Accept `{"name","parameters"|"arguments"}` or a nested `"function"` form.
  - If there's a tool call, POST `/{name}` and append `{"role":"ipython","content":...}`. Otherwise mark the trajectory done.
  - Token accounting is prefix-preserving: env tokens = `tokenize(conversation)[len(previous_ids):]` before each generation. `env_mask` is 1 for generated tokens and 0 for env/template tokens, and `logprobs` come from vLLM, with 0.0 for env tokens.
  - Return `prompt_ids`, `completion_ids`, `logprobs`, `env_mask`, and `trajectory`. The trajectory is Responses-API output items (`function_call`, `function_call_output`, final `message`).
- **`gym_verify_reward(prompts, completions, trajectory, answer, responses_create_params, **kw)`:** POST `/verify` per item. The request shape follows `Example_NeMo.ipynb` cell 11, but with the real multi-turn `output`.
- **`train_grpo.py`:**
  - Model and LoRA exactly as `autodidact.ipynb` cell 1: Llama-3.1-8B-Instruct, 4-bit, r=64, all 7 projection modules. Lower `gpu_memory_utilization` to ~0.55.
  - Hyperparameters from cell 5.
  - Build the dataset from `train.jsonl`: `prompt` = `responses_create_params.input`, with `answer` and `responses_create_params` as passthrough columns.
- **`eval.py`:** run `validation.jsonl` through the same loop, with and without the trained LoRA, and report mean reward. This replaces `run_eval` and notebook cells 9-10.

## Verification
1. Run `gym env test --resources-server mission_search` (verifier fixture plus tool test).
2. Start `ng_run "+config_paths=[resources_servers/mission_search/configs/resources_only.yaml]"`, then `curl` both `/search_corpus` (should return 2 chunks) and `/verify` with a hand-built correct and incorrect response (expect 1.0 and ~0.0).
3. Run the step-1 spike and check all pass criteria (a)-(d) hold.
4. Smoke-train `train_grpo.py` with `max_steps=5`: rewards are non-zero, no OOM, and logged completions show `search_corpus` calls with `ipython` results.
5. Full run (101 steps, as before), then `eval.py` comparing base vs. LoRA mean reward on the validation split.
