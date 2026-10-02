# Plan: Rebuild AutoDidact's search agent on NeMo Gym (fresh project, clean deps)

## Context

`use_agentic_generate` is gone from current Unsloth. TRL's `environment_factory` needs `transformers>=5.2.0`, and Unsloth's source patching crashes against that (`auto_docstring`, and `PreTrainedConfig` in unsloth#3786). You want a structure that scales to more complex environments, so overriding trainer internals is ruled out.

**Update (2026-10-01):** the `auto_docstring` crash was an older-Unsloth problem. Unsloth 2026.9.12 handles transformers 5.x and also patches TRL's `VLLMGeneration` to reuse its vLLM engine, with no monkeypatch from us (see `Dependency_hell.md`, rows 2, 21 and 23). `environment_factory` is still not used.

Research findings that shape the design:
- **Unsloth's NeMo Gym notebook (`Example_NeMo.ipynb`) is single-turn only.** It starts resources servers with `resources_only.yaml` and only POSTs the finished completion to `/verify` inside `reward_fn`. Tools are never called during training.
- **NeMo Gym's official multi-turn TRL route won't fit.** It needs a separate `trl vllm-serve` process on a second GPU, and Unsloth force-sets `vllm_mode='colocate'` (`unsloth/models/rl.py:869`). You have one RTX 5090 Laptop GPU (24 GB).
- **TRL `rollout_func` is the extension point that fits.** It exists from TRL 0.25, and `env_mask` (keeps tool-output tokens out of the loss) from 0.28. It returns `prompt_ids`/`completion_ids`/`logprobs`/`env_mask`, and any extra fields are forwarded to reward functions (`trl/trainer/grpo_trainer.py:1553-1605`).

**Decisions (confirmed with you):**
1. A generic multi-turn loop runs in `rollout_func` on Unsloth's collocated vLLM. It follows the Token-In, Token-Out (TITO) rule from the HF post: sampled token ids go into a running buffer and are never re-encoded (details in step 5).
2. A NeMo Gym resources server owns the environment: the `search_corpus` tool plus `verify()`, both over HTTP.
3. `verify()` uses a deterministic answer match.
4. Fresh `search_gym/` directory with clean venvs.
5. Reuse the existing `questions.json`/`chunks.pkl`. No question regeneration.

**Why this scales:** the rollout loop is environment-agnostic. It only knows "generate from token ids → decode only to parse the JSON tool call → POST `/{tool_name}` → append the tool message's token delta → repeat" and "POST `/verify`". Adding a new environment later means adding a new resources server plus data rows carrying a server ref. The trainer does not change.

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
└── trainer/                               # training venv: currently trainer/temp/.venv-temp (Python 3.12), see Dependency_hell.md
    ├── rollout.py      # generic multi-turn rollout_func + gym client (TITO token buffer, see step 5)
    ├── train_grpo.py   # Unsloth model + GRPOTrainer(rollout_func=..., reward_funcs=[gym_verify_reward])
    ├── eval.py         # held-out eval, with and without LoRA
    ├── prompting.py    # renders chat prompts with the template date pinned (26 Jul 2024); all render sites must use it
    ├── spike_rollout_func.py        # Gate 1 spike (passed); its rollout is the reference for rollout.py
    └── temp/working_stack.freeze.txt  # frozen package versions of the working venv
```
AutoDidact stays untouched as a reference.

## Steps

### 0. Environments (clean, one-shot installs)
- **NeMo Gym:** `git clone https://github.com/NVIDIA-NeMo/Gym search_gym/Gym`, then `uv python install && uv sync` (honours `.python-version` 3.13.14, as the notebook's cell 5 explains). Scaffold with `gym env init --resources-server mission_search`; fall back to copying an existing server's layout if the CLI differs in the clone.
- **Training venv:** done. The working environment is `trainer/temp/.venv-temp` (Python 3.12), built from Unsloth's Blackwell install path. The earlier `.venv-train` pin set (`unsloth==2025.9.9`, `vllm==0.22.0` and so on) failed at import and is obsolete.
  - Working stack: `unsloth 2026.9.12`, `unsloth_zoo 2026.9.8`, `trl 0.29.1`, `transformers 5.17.0`, `vllm 0.24.0+cu129` (GitHub wheel), `torch 2.11.0+cu128`, `torchvision 0.26.0`. The full list is `trainer/temp/working_stack.freeze.txt`, and the reasoning is in `Dependency_hell.md` section 1.
  - `trl` and `transformers` are above Unsloth's declared caps, so any install into this venv needs `uv pip install --override overrides.txt ...` (`trl==0.29.1`, `transformers==5.17.0`), always with the full package list plus the vLLM wheel URL. `pip check` reports exactly those two violations.
  - WSL2 needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (the spike sets it).
  - No langchain/chroma here, since those live in the env server. Register a Jupyter kernel when needed.

### 1. Gate: spike `rollout_func` on Unsloth (before building anything else): PASSED
`spike_rollout_func.py`, using a tiny model (`unsloth/Llama-3.2-1B-Instruct`, 4-bit, `fast_inference=True`), a dummy 2-turn echo tool, and 2 GRPO steps. Status (2026-10-01): all checks pass on branch `Gate1_d6`, run in `trainer/temp/.venv-temp`. How each criterion was verified:
- (a) The regenerated `unsloth_compiled_cache/UnslothGRPOTrainer.py` contains the `rollout_func` branch (`grep -c rollout_func` gives 15).
- (b) Only one vLLM engine is allocated. VRAM stays flat (10,454 MiB from model load to trainer init) and the trainer's `VLLMGeneration` points at `model.vllm_engine`. This needs Unsloth 2026.9.12, whose `patch_trl_vllm_generation()` replaces the old regex approach.
- (c) The live adapter is used inside `rollout_func`: `trainer.model.vllm_engine.generate(..., lora_request=trainer.model.load_lora("grpo_trainer_lora_model", load_tensors=True))`. Tightened into c0 (adapter attached on every call), c1 (LoRA B moved from 0 during training), c2 (a changed adapter changes greedy output) and c3 (no stale adapter, request ids rise).
- (d) Grouping and masking, tightened into d1 (alignment, logprobs), d2 (env tokens present for trajectories that continued), d3 (TITO invariants, see step 5), d4 (TRL's logged `completions/mean_length` equals the mask sum and is below the raw length), d5 (grouping with 2 unique prompts per batch) and d6 (a spy on `align_completion_tool_mask` showed the env tokens are zeroed in the real loss path: surviving-token counts matched the rollout's own counts exactly).
- **Limits:** d6 checks counts, not token positions. The sample is small, and `MAX_NEW=64` truncates many trajectories early (a real run needs a larger cap).
- **Lessons that carried into the design:** TRL passes prompts already repeated `num_generations` times, so `rollout_func` returns exactly one trajectory per incoming prompt. The fallback (plain TRL + PEFT, smaller model) was not needed.

### 2. Data: `scripts/convert_questions.py`: DONE
Script: `NeMo_Gym/resources_servers/mission_search/scripts/convert_questions.py` (needs only the standard library for the default split; `--split legacy` also needs `datasets`). Outputs: `NeMo_Gym/resources_servers/mission_search/data/{train,validation,example}.jsonl` (`train`/`validation` are git-ignored; regenerate with the script). Status (2026-10-02): 1,240 train / 148 validation / 5 example rows, all validated against Gym's `BaseRunRequest`.
- Reads `Step2_Files/Saved_data/questions.json` (1434 rows of `{chunk_id, question, answer, difficulty}`; `--input` overrides).
- Emits NeMo Gym JSONL rows:
  - `responses_create_params`:
    - `input`: a `system` message and a `user` message. Both functions are copied from `rl_helpers.py` (lines 22 and 52), not imported, because `rl_helpers` pulls in torch, trl and the Chroma loader. The system prompt keeps only the two instruction sentences: AutoDidact's hand-written "Cutting Knowledge Date / Today Date" header was dropped because the Llama template adds that header itself (it appeared twice). The tool definition stays inside the user prompt text, as in AutoDidact.
    - `tools`: `SEARCH_TOOL_DEFINITION` converted to the flat Responses-API form `{type, name, description, parameters, strict}` with `strict: false` (the key is required by Gym's schema; our rollout parses JSON from text and never uses strict decoding).
    - `parallel_tool_calls`: `false`.
  - Top-level fields: `answer`, `difficulty` (26 malformed labels like `==Difficulty: easy==` normalised to easy/medium/hard), `chunk_id`. Validation rows also carry `block`.
- **Split (deviation from the original plan, which copied AutoDidact's seed-42 random split).** Default `--split purged`:
  - `chunk_id` is the centre of a 3-chunk generation window, so questions share source text when their ids differ by 2 or less.
  - Validation = 6 separate blocks of 13 consecutive chunk ids (about 10%), chosen by SHA-256 hash order (seed 42), not by a random generator, so the blocks are the same forever (byte-identical output on Python 3.12 and 3.13).
  - Training rows within 2 ids of any validation row are purged (46 rows), and so are training rows whose normalised question text appears in validation (0 in the current blocks; the branch was unit-tested on synthetic data). Purged rows are in neither set.
  - The script asserts min id distance > 2 and zero question overlap between train and validation on every run.
  - `--split legacy` reproduces AutoDidact's `get_qa_dataset()` exactly (verified row for row), kept for reference. It leaks: 21 of 144 validation questions also appear in train, and 128 of 144 share a chunk with a train row.
  - 5 rows from the start of the (hash-shuffled) train set go to `example.jsonl`.
- Caveat: neighbouring chunks are not independent for a document-wide corpus, and the agent searches the whole corpus at evaluation time, so this reduces leakage rather than eliminating it. No separate test block yet (add one when tuning against validation begins).

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
  - TRL passes prompts already repeated `num_generations` times: return exactly one trajectory per incoming prompt, in the same order. Do not multiply by `num_generations` again.
  - **TITO (token-in, token-out).** Each trajectory keeps a running id buffer (`ctx_ids`), initialised from the tokenized prompt (`prompting.apply_chat_template(tok, messages, add_generation_prompt=True, tokenize=True, return_dict=False)`; every render goes through `prompting.py`, which pins the template's date so prompts are identical on any day). The buffer is the source of truth, and nothing decoded is ever re-encoded.
  - Generate from ids: `generate([{"prompt_token_ids": ctx_ids[i]}, ...], ..., lora_request=...)`, with the call from step 1(c). Sampled ids are appended to the buffer and to `completion_ids` exactly as vLLM produced them.
  - Decode the sampled text only to route the tool call: parse with `extract_json_objects` (`rl_helpers.py:89`, copied), accepting `{"name","parameters"|"arguments"}` or a nested `"function"` form. The decoded text is never fed back into the prompt.
  - If there's a tool call, POST `/{name}`, then append the tool message as a template-diff delta by id: render a dummy `[user, assistant]` conversation with and without the tool message (role `ipython`), subtract the token lists, and append the difference. Assert the prefix is unchanged on every call, and run the same prefix-preservation property check at startup. Otherwise mark the trajectory done.
  - Truncation (`finish_reason == "length"`) ends the trajectory: the buffer stays as sampled and nothing is added.
  - `env_mask` is 1 for sampled tokens and 0 for the appended tool delta, and `logprobs` come from vLLM, with 0.0 for env tokens. Check `out.prompt_token_ids` equals what was sent, and that the last prompt sent plus its generation equals `prompt_ids + completion_ids`.
  - **Open item (decide later): JSON quoting of tool output.** The Llama template JSON-quotes string content for the `ipython`/`tool` roles (`"…"`). For search chunks we probably don't want that. Decide how to append the retrieved chunks, and re-run the prefix-preservation check with the final format.
  - **Open item (later): history rewriting.** If a reasoning model or a harness strips or compacts earlier turns, TITO breaks. The mitigation from the HF post is to freeze everything before the last rewrite as prompt context (mask zero) and train only on the sampled tokens after it. Llama's template does not strip anything, so this is not needed for v1.
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
3. Step-1 spike: done, all checks pass (see step 1). Re-run it after any change to the rollout loop.
4. Smoke-train `train_grpo.py` with `max_steps=5`: rewards are non-zero, no OOM, and logged completions show `search_corpus` calls with `ipython` results.
5. Full run (101 steps, as before), then `eval.py` comparing base vs. LoRA mean reward on the validation split.
