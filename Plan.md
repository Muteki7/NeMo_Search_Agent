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
│       ├── embeddings.py     # copied from AutoDidact/embeddings.py (CPU, batched); used by build_index.py and the search tool
│       ├── configs/mission_search.yaml, configs/resources_only.yaml
│       ├── data/{example,train,validation}.jsonl
│       ├── chroma_index/     # built by scripts/build_index.py: 510 merged chunks (git-ignored)
│       ├── .venv/            # server venv, built from requirements.txt the way Gym builds it (git-ignored)
│       ├── scripts/build_index.py, scripts/convert_questions.py
│       ├── tests/{test_app.py, verifier_cases.jsonl}
│       └── requirements.txt  # -e nemo-gym[dev] @ ../../Gym; CPU-only torch pinned by URL; pinned transformers, chromadb, langchain-chroma, langchain-core
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

### 3. Index: `scripts/build_index.py`: DONE
Script: `NeMo_Gym/resources_servers/mission_search/scripts/build_index.py`. Run it with the server venv (`NeMo_Gym/resources_servers/mission_search/.venv/bin/python`). Output: `NeMo_Gym/resources_servers/mission_search/chroma_index/` (collection `example_collection`, 510 rows, 5.2 MB, git-ignored). Status (2026-10-05): built, reopened from disk, and tested with real questions.
- **Environment (deviation).** The server venv is built the way Gym builds it (`uv venv --seed --python 3.13.14`, then `uv pip install -r requirements.txt`). Torch is the CPU-only 2.11.0 wheel, pinned by URL: adding the PyTorch CPU index as an extra index breaks the resolve, because that index also hosts an old `packaging`. Pinned: `transformers 5.18.0`, `chromadb 1.5.9`, `langchain-chroma 1.1.0`, `langchain-core 1.6.6`. The venv adds about 1 GB of disk (`C:` is nearly full).
- **Embedder.** `embeddings.py` in the server folder is copied from AutoDidact with three changes: the `langchain_core` import (the old `langchain.embeddings.base` path is gone), CPU by default, and batches of 32. The pooling is unchanged: the `[CLS]` vector for documents, mean pooling for queries.
- **Build.** `Chroma.from_documents` over `Step2_Files/Saved_data/chunks.pkl`, with fixed ids `chunk-<index>`. It takes about 17 s, not "a couple of minutes". A non-empty output folder is refused unless `--rebuild` is given, and `--rebuild` only deletes a folder named `chroma_index`.
- **Merge of short chunks (deviation).** The splitter cut each document element (a heading, a page ID, a caption) on its own, so 206 of 753 chunks have fewer than 50 characters (124 titles, 46 page or document IDs, 33 captions). They filled 12.9% of the top-2 search results. `merge_small_chunks` glues each chunk shorter than `--min-chars` (default 100) onto the chunk(s) after it, up to `--max-chars` (default 1500). Result: 753 chunks became 510 (186 merged groups, shortest 101 characters, longest 1089). `--min-chars 0` keeps all 753 chunks.
- **How the threshold was chosen.** Five variants were compared on all 1,434 questions with an exact in-memory search, which matched the real index within 0.5 points. Answer coverage at top 2 / top 5: baseline 72.0 / 83.3, drop chunks under 50: 74.3 / 84.0, **merge to 100: 75.0 / 85.3**, merge to 200: 74.1 / 85.1, merge to 300: 73.2 / 84.7. Window recall at top 2 stayed flat for 100 (76.9 to 77.0) and fell for 200 (75.2) and 300 (72.2). Dropping loses content: 30 questions (2.1%) have an answer only in a tiny chunk. Not chosen: parent-child retrieval (it needs a second store and changes what the agent reads). Revisit it if the corpus grows.
- **Chunk numbers.** `questions.json` (`chunk_id`) and the purged split use the OLD chunk positions, so neither changes. Each merged chunk stores `chunk_start`, `chunk_end` (the old positions it covers) and `n_merged`. `chunk_index` is the new position.
- **Retrieval check** (400 questions, top 2 / top 5), before and after the merge. Window recall: 78.5 / 88.0 and 77.5 / 87.5. Centre-chunk recall: 34.0 / 46.5 and 44.2 / 57.2. Answer coverage: 73.0 / 82.0 and 75.0 / 83.8. The check asserts window recall at top 5 above 30%.
- **Limits.** Answer coverage is a token-overlap proxy (a result holds at least 60% of the answer's key words). The centre-chunk rise is partly mechanical, because a merged chunk covers more positions. The gain is small, about 2 points. The check samples every third question and stops at 400, so it never tests the last 16% of the questions.

### 4. Resources server: `app.py`: DONE
Files: `NeMo_Gym/resources_servers/mission_search/app.py`, `configs/resources_only.yaml`, `tests/test_app.py`, `tests/verifier_cases.jsonl`. Status (2026-10-07): 14 tests pass in the server venv, the server starts with `ng_run`, and the search call and the verify calls gave the expected results.
- **Config.** `MissionSearchResourcesServerConfig` has `chroma_dir` (default `chroma_index`, relative to the server folder), `collection_name` (`example_collection`), `embedding_model` and `top_k=2`. `REVERIFY_MODE` is STATELESS: `verify` never reads the index, so it is a pure function of the request body.
- **Tool.** `POST /search_corpus` takes `{query}` and returns `{results}`. `CorpusSearcher` opens the index once at server start with the same `CustomHuggingFaceEmbeddings` class that built it. The text format is the AutoDidact format (`Result i:`, the chunk text, `------`). The route name equals the tool name in the data rows, because the agent POSTs each tool call to `/{tool name}`. The embedding runs in a thread (`asyncio.to_thread`), so parallel rollouts do not block the server.
  - Deviation: `search_module.py` is not imported. It loads its own index at import time with old paths and `results=5`. Only its output format is copied.
- **Verify.** `MissionSearchVerifyRequest` adds `answer: str`. The scaffold had `expected_answer`, but the data rows use `answer`. The final answer is the last `message` item of the response output. `<reasoning>` blocks are removed, then both texts are normalized (lowercase, no punctuation, no articles, so `18:24` equals `1824`). Reward:
  - 1.0 for an exact match, or when the expected answer is a run of whole words inside a short answer.
  - Otherwise the token-F1 over the content words of the two texts.
  - 0.0 when there is no final message or no text is left after the reasoning is removed.
- **Containment cap (deviation).** "Short" means at most 5 x (expected words) + 15 words (`LENGTH_CAP_FACTOR` and `LENGTH_CAP_EXTRA` in `app.py`). Without a cap, a model could paste a whole chunk and get 1.0. It was 3 x + 10 at first. A dump of 40 trajectories from the 8B model showed two correct but wordy answers that got only 0.29 and 0.35, so it is now 5 x + 15 (the mean reward moved from 0.200 to 0.234, and no cap at all gives the same). Freeze it for a run: base and trained models must be scored with the same value.
- **F1 on content words (deviation, 2026-10-14).** The first smoke run (1B model) gave the highest rewards to non-answers (a bad tool call got 0.19), because the F1 counted small words such as "of" and "to". The F1 now drops stop words (`STOP_WORDS` in `app.py`; `not` and `no` stay in). A text of only small words gets 0.0.
- **No tool-call bonus (deviation).** Decide after the first rollouts show how often the base model calls the tool. Add a small bonus, capped so that the total stays at or below 1.0, only if almost no sample calls the tool.
- **Where the expected answers come from.** An 8B model wrote them from three-chunk windows (`generate_data.py`). Nobody checked them. A correct answer in other words scores below 1.0. See `Learnings.md` L13.
- **Configs.** `mission_search.yaml` keeps the resources server and the `simple_agent` entry. It cannot start alone, because `simple_agent` needs a `policy_model` that we do not have. Use it later for `gym eval run` with a model endpoint. `resources_only.yaml` has only the resources server, with no agent and no datasets. It is the file for training.
- **Where the agent and the trainer live (deviation from the Gym layout).** There is no agent server. `rollout_func` in the trainer (step 5) is the agent: it generates on the trainer's own vLLM engine, POSTs tool calls to `/search_corpus`, and the answer to `/verify`. `simple_agent` would need a second vLLM engine and works on text, so it breaks the single-engine gate and the TITO rule. The trainer lives in `trainer/` and is not a Gym server.
- **Two environments.** The server runs in `NeMo_Gym/resources_servers/mission_search/.venv`. The trainer runs in `trainer/temp/.venv-temp`. They share no packages (CPU against CUDA torch, transformers 5.18 against 5.17) and only talk over HTTP.
- **Start the server** (from `NeMo_Gym`): `resources_servers/mission_search/.venv/bin/ng_run "+config_paths=[resources_servers/mission_search/configs/resources_only.yaml]" +skip_venv_if_present=true`. Without the flag, `ng_run` runs `uv pip install` into the existing server venv and installs `nemo-gym` from PyPI. That could break the pins. With the flag, it only activates the venv.
- **Ports.** The head server is fixed at 11000. Our server gets a random port in 10001-20000 at each start. The trainer finds it with `GET :11000/global_config_dict_yaml`. The reply is one JSON string that holds YAML, so decode it with `json.loads` first. Ray (dashboard 8265, cluster 39107) also starts, but our server has `ray_enabled = False` and does not use it.
- **Tests.** `verifier_cases.jsonl` has six cases: two full reward, two zero reward and two malformed (no `response`, no `answer`). `test_app.py` also tests the scoring rules (partial F1, the length cap, reasoning-only answers, the last message) and the search tool (output format, `top_k`, parallel searches give the same text). The search tests skip if the index is missing.
- **Retrieval limit seen here.** For the example question ("What time was the flotation collar inflated?") the only chunk that mentions the collar is not in the top 5. A 60-question check gave window recall 77% at top 2 and 90% at top 5. Some questions need a reworded query. That is the behaviour that training should find.

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
1. From `NeMo_Gym`, run `resources_servers/mission_search/.venv/bin/python -m pytest resources_servers/mission_search/tests` (verifier fixture plus tool tests): done, 14 pass.
2. Start `ng_run "+config_paths=[resources_servers/mission_search/configs/resources_only.yaml]" +skip_venv_if_present=true`, then `curl` both `/search_corpus` (should return 2 chunks) and `/verify` with a hand-built correct and incorrect response (expect 1.0 and ~0.0): done.
3. Step-1 spike: done, all checks pass (see step 1). Re-run it after any change to the rollout loop.
4. Smoke-train `train_grpo.py` with `max_steps=5`: rewards are non-zero, no OOM, and logged completions show `search_corpus` calls with `ipython` results.
5. Full run (101 steps, as before), then `eval.py` comparing base vs. LoRA mean reward on the validation split.
