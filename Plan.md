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
3. `verify()` is a deterministic function of the response (STATELESS). Run 1 used the answer match alone. From run 2 it adds a retrieval part (step 4, step 6). A semantic match with BGE-M3 is planned (step 8).
4. Fresh `search_gym/` directory with clean venvs.
5. Reuse the existing `questions.json`/`chunks.pkl`. No question regeneration so far. A quality audit of the questions and references is planned (step 8), and it decides if we regenerate.

**Why this scales:** the rollout loop is environment-agnostic. It only knows "generate from token ids → decode only to parse the JSON tool call → POST `/{tool_name}` → append the tool message's token delta → repeat" and "POST `/verify`". Adding a new environment later means adding a new resources server plus data rows carrying a server ref. The trainer does not change.

## Layout

```
NeMo_search_agent/                         # the repository
├── Plan.md, Learnings.md, Dependency_hell.md, Debug_checks.md, rules.md, ASD-STE100.jpeg
├── Step2_Files/                           # AutoDidact files (questions.json, chunks.pkl, ...); git-ignored
├── Embedder/bge-m3/                       # BGE-M3 weights (2.2 GB), loaded from this path; git-ignored
├── NeMo_Gym/
│   ├── Gym/                               # clone of NVIDIA-NeMo/Gym (own .git, own uv venv); git-ignored
│   └── resources_servers/mission_search/  # our environment (NeMo Gym scaffold layout); tracked in git
│       ├── app.py            # tool endpoint + verify (answer reward + retrieval reward)
│       ├── embeddings.py     # NoInstruct embedder (CPU, batched); used by build_index.py and the search tool
│       ├── configs/mission_search.yaml, configs/resources_only.yaml
│       ├── data/{example,train,validation}.jsonl   # the scaffold's data/.gitignore hides train and validation
│       ├── chroma_index/     # 510 merged chunks, built by scripts/build_index.py (git-ignored)
│       ├── .venv/            # server venv (git-ignored)
│       ├── scripts/          # build_index.py, convert_questions.py, rescore_dump.py, report.py, retrieval_bench.py
│       ├── tests/            # test_app.py, test_report.py, test_retrieval_bench.py, verifier_cases.jsonl
│       └── requirements.txt  # CPU-only torch pinned by URL; pinned transformers, chromadb, langchain-chroma, langchain-core
└── trainer/                               # trainer venv: trainer/temp/.venv-temp (Python 3.12), see Dependency_hell.md
    ├── rollout.py        # Gym client, tool-call parser, TITO loop, vLLM wiring, verify reward, dump helpers
    ├── train_grpo.py     # Unsloth model + GRPOTrainer, per-run folder, resume
    ├── eval.py           # base model and checkpoints on the validation set (greedy)
    ├── rollout_smoke.py  # test 2: the rollout alone on vLLM and the live server
    ├── test_rollout.py   # test 1: 24 CPU tests with a fake model and a fake tool
    ├── prompting.py      # renders chat prompts with the template date pinned (26 Jul 2024); all render sites must use it
    ├── spike_rollout_func.py   # Gate 1 spike (passed)
    └── runs/<start time>_<name>/   # run_config.json, train_dump.jsonl, checkpoint-N/, final/, eval_*.jsonl, report.html (git-ignored)
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
Files: `NeMo_Gym/resources_servers/mission_search/app.py`, `configs/resources_only.yaml`, `tests/test_app.py`, `tests/verifier_cases.jsonl`. Status (2026-10-27): the tests pass in the server venv (`test_app.py` has 22), the server starts with `ng_run`, and the search call and the verify calls gave the expected results.
- **Config.** `MissionSearchResourcesServerConfig` has `chroma_dir` (default `chroma_index`, relative to the server folder), `collection_name` (`example_collection`), `embedding_model` and `top_k=2`. `REVERIFY_MODE` is STATELESS: `verify` never reads the index, so it is a pure function of the request body.
- **Tool.** `POST /search_corpus` takes `{query}` and returns `{results}`. `CorpusSearcher` opens the index once at server start with the same `CustomHuggingFaceEmbeddings` class that built it. The text format is the AutoDidact format (`Result i:`, the chunk text, `------`). The route name equals the tool name in the data rows, because the agent POSTs each tool call to `/{tool name}`. The embedding runs in a thread (`asyncio.to_thread`), so parallel rollouts do not block the server.
  - Deviation: `search_module.py` is not imported. It loads its own index at import time with old paths and `results=5`. Only its output format is copied.
- **Verify.** `MissionSearchVerifyRequest` adds `answer: str`. The scaffold had `expected_answer`, but the data rows use `answer`. The final answer is the last `message` item of the response output. `<reasoning>` blocks are removed, then both texts are normalized (lowercase, no punctuation, no articles, so `18:24` equals `1824`). Reward:
  - 1.0 for an exact match, or when the expected answer is a run of whole words inside a short answer.
  - Otherwise the token-F1 over the content words of the two texts.
  - 0.0 when there is no final message or no text is left after the reasoning is removed.
- **Containment cap (deviation).** "Short" means at most 5 x (expected words) + 15 words (`LENGTH_CAP_FACTOR` and `LENGTH_CAP_EXTRA` in `app.py`). Without a cap, a model could paste a whole chunk and get 1.0. It was 3 x + 10 at first. A dump of 40 trajectories from the 8B model showed two correct but wordy answers that got only 0.29 and 0.35, so it is now 5 x + 15 (the mean reward moved from 0.200 to 0.234, and no cap at all gives the same). Freeze it for a run: base and trained models must be scored with the same value.
- **F1 on content words (deviation, 2026-10-14).** The first smoke run (1B model) gave the highest rewards to non-answers (a bad tool call got 0.19), because the F1 counted small words such as "of" and "to". The F1 now drops stop words (`STOP_WORDS` in `app.py`; `not` and `no` stay in). A text of only small words gets 0.0.
- **Retrieval part of the reward (deviation, run 2, 2026-10-21).** No tool-call bonus was added. Instead `verify` pays for a search whose result holds the reference's facts: `reward = 0.75 x R_answer + 0.25 x gate x ret`. `cov_k` is the share of the reference's content words found in the result of search k. `s_k` is a ramp (0 below 0.3, 1 at 0.6 and above). `ret` is the maximum over searches of `0.5^(k-1) x s_k`, so extra searches cannot be farmed. `gate` is 1 only if the trajectory ends with a final message, so spam to the turn cap scores 0. A correct answer with a hit scores 1.0, a correct answer without one 0.75, a hit with no usable answer at most 0.25. The constants (`RETRIEVAL_WEIGHT`, `RETRIEVAL_DISCOUNT`, `RAMP_LOW`, `RAMP_HIGH`) are at the top of that block in `app.py`. Freeze them for a run. `/verify` returns the combined `reward` and the parts (`reward_answer`, `reward_retrieval`, `retrieval_gate`, `best_cov`, `n_searches`, `reward_components`), in Gym's `BaseMultiRewardVerifyResponse` pattern.
- **Where the expected answers come from.** An 8B model wrote them from three-chunk windows (`generate_data.py`). Nobody checked them. A correct answer in other words scores below 1.0. See `Learnings.md` L13.
- **Configs.** `mission_search.yaml` keeps the resources server and the `simple_agent` entry. It cannot start alone, because `simple_agent` needs a `policy_model` that we do not have. Use it later for `gym eval run` with a model endpoint. `resources_only.yaml` has only the resources server, with no agent and no datasets. It is the file for training.
- **Where the agent and the trainer live (deviation from the Gym layout).** There is no agent server. `rollout_func` in the trainer (step 5) is the agent: it generates on the trainer's own vLLM engine, POSTs tool calls to `/search_corpus`, and the answer to `/verify`. `simple_agent` would need a second vLLM engine and works on text, so it breaks the single-engine gate and the TITO rule. The trainer lives in `trainer/` and is not a Gym server.
- **Two environments.** The server runs in `NeMo_Gym/resources_servers/mission_search/.venv`. The trainer runs in `trainer/temp/.venv-temp`. They share no packages (CPU against CUDA torch, transformers 5.18 against 5.17) and only talk over HTTP.
- **Start the server** (from `NeMo_Gym`): `resources_servers/mission_search/.venv/bin/ng_run "+config_paths=[resources_servers/mission_search/configs/resources_only.yaml]" +skip_venv_if_present=true`. Without the flag, `ng_run` runs `uv pip install` into the existing server venv and installs `nemo-gym` from PyPI. That could break the pins. With the flag, it only activates the venv.
- **Ports.** The head server is fixed at 11000. Our server gets a random port in 10001-20000 at each start. The trainer finds it with `GET :11000/global_config_dict_yaml`. The reply is one JSON string that holds YAML, so decode it with `json.loads` first. Ray (dashboard 8265, cluster 39107) also starts, but our server has `ray_enabled = False` and does not use it.
- **Tests.** `verifier_cases.jsonl` has six cases: two full reward, two zero reward and two malformed (no `response`, no `answer`). `test_app.py` also tests the scoring rules (partial F1, the length cap, reasoning-only answers, the last message) and the search tool (output format, `top_k`, parallel searches give the same text). The search tests skip if the index is missing.
- **Retrieval limit seen here.** For the example question ("What time was the flotation collar inflated?") the only chunk that mentions the collar is not in the top 5. A 60-question check gave window recall 77% at top 2 and 90% at top 5. Some questions need a reworded query. That is the behaviour that training should find.

### 5. Trainer: `rollout.py`, `train_grpo.py`, `eval.py`: DONE
Files are in `trainer/`. Status (2026-10-27): two 200-step runs are trained, evaluated and compared (step 6).
- **Gym client.** `GymClient.discover()` asks the head server (port 11000) for the host and port of the resources server. It decodes the JSON string that holds YAML. Network and server faults raise `GymError`. They are never turned into a silent reward of 0.
- **The loop (`run_rollout`).** It takes a `generate_fn`, so it does not depend on vLLM, and CPU tests drive it with a fake model. TITO as planned: one token buffer per trajectory, sampled ids go in as they are, a tool result goes in as a template-diff delta (role `ipython`), and the prefix check runs at start. One trajectory per incoming prompt, in order. Tool calls of one turn run in parallel.
- **How a trajectory ends** (`ended`): `answer` (a turn without a tool call), `truncated` (a turn reached `MAX_NEW`), `bad_call` (broken JSON, an unknown tool, a missing query or two calls), `turn_cap` (a tool call on the last turn is not run), `budget` (a tool result plus one more turn would not fit `MAX_COMPLETION`). The last two have no final message, so their reward is 0.
- **Open item closed: JSON quoting.** The Llama template quotes the tool text. We left it as it is for v1 (about 0.9% more tokens).
- **Model and LoRA.** `unsloth/Meta-Llama-3.1-8B-Instruct` in 4-bit, LoRA rank 64 on all 7 projection modules. `max_lora_rank` must be passed to `from_pretrained`: its default (16) makes vLLM refuse the adapter. `gpu_memory_utilization=0.55` (Unsloth uses 51.8%). The KV cache holds 47,280 tokens.
- **Hyperparameters** (from AutoDidact, `autodidact.ipynb`): learning rate 5e-6, Adam betas 0.9 and 0.99, weight decay 0.1, cosine schedule with 10% warmup, `paged_adamw_8bit`, `max_grad_norm` 0.1, `beta` 0 (no KL term). Dropped: `use_agentic_generate` (our rollout replaces it) and `max_prompt_length` (removed in TRL 0.29). With transformers 5, `warmup_steps=0.1` replaces `warmup_ratio`.
- **Our settings.** `num_generations` 8, one prompt for each step: batch 4 and accumulation 2 give 8 trajectories for each rollout, and the loss runs on 4 at a time. `MAX_TURNS` 5, `MAX_NEW` 512, `MAX_SEQ` 4096, `MAX_COMPLETION` 3000. The rollout samples at temperature 0.9, and `GRPOConfig.temperature` has the same value, because TRL divides its logits by it.
- **Run folder.** `trainer/runs/<start time>_<name>/` holds `run_config.json`, `train_dump.jsonl` (every trajectory with its reward parts and its step), `checkpoint-N/` (about 1 GB each) and `final/`. `--save-steps 50` is the default.
- **Resume.** `train_grpo.py --resume runs/<run>/checkpoint-N` continues in the same folder with the saved settings. It stops if the constants or hyperparameters in the code differ from the saved run. Dump lines of the steps after the checkpoint move to a side file. Checked on a copy of a test run: the scheduler, the step numbers and the rewards came back. The final weights differed by 1.4%, which is normal GPU noise.
- **`eval.py`.** Greedy decoding on the validation set. It evaluates the base model (no adapter) once, then each checkpoint, on one loaded engine. The headline is the **answer-only** reward, so runs stay comparable. It also reports exact, contains, tool-call rate, answer rate, turns, final words, retrieval and the paired change against the base. It reuses `eval_base.jsonl` from the run folder with `--no-base`.
- **Speed and memory.** Steps take 12 to 60 s (the mean of the last 40 steps is 12 s). 200 steps take about 62 min. Peak VRAM is 19.9 GB of 24.4 GB.
- **Tests.** Test 1 (`test_rollout.py`): 24 CPU tests, each able to fail (checked with deliberate bugs). Test 2 (`rollout_smoke.py`): the rollout alone on the real engine and server.

### 6. Runs and results
Eval on the 148 validation questions, greedy, **answer-only reward** (± standard error). Run 1: `runs/20261015-141100_first200`, answer reward only, TRL scaling by the group std. Run 2: `runs/20261021-141800_run2_retrieval`, `scale_rewards="none"` and the retrieval part in the reward.

| Model | Reward | Exact | Contains | Retrieved (word overlap) | Searches |
|---|---|---|---|---|---|
| Base | 0.191 ± 0.029 | 0.15 | 0.19 | 0.72 | 1.95 |
| Run 1, step 50 / 100 / 150 / 200 | 0.361 / 0.460 / 0.462 / 0.461 ± 0.036 | 0.36 at 200 | 0.37 | - | 1.03 at 200 |
| Run 2, step 50 / 100 / 150 / 200 | 0.390 / 0.473 / 0.478 / 0.499 ± 0.037 | 0.41 at 200 | 0.43 | 0.68 at 200 | 1.03 at 200 |

- **Run 1.** The paired gain over the base is +0.27 ± 0.04 (72 questions better, 9 worse). It stops at step 100. The model learned the output contract (the answer outside `<reasoning>`: 32 of 148 base answers were inside it only) and to search once. It did not retrieve better.
- **Run 2.** Better by +0.039 ± 0.020 (paired, 1.9 standard errors; 19 questions better, 10 worse). That is suggestive, not proof: one training seed for each run, and two changes at once. The retrieval part carried only 7% of the learning signal inside groups, and retrieval did not improve.
- **What limits the score.** Retrieved (68%) times converted (62% of the retrieved answers end in the final text) is about 0.42. See `Learnings.md` L18.

### 7. Tools to read results
- `scripts/rescore_dump.py`: scores any dump again with the current `score_answer` (no GPU).
- `scripts/report.py`: one HTML report. A chart of 8 failure categories for each model, the definitions, and a question explorer. Categories come from rules over a lexical hit and a provenance hit (`Learnings.md` L18).
- `scripts/retrieval_bench.py`: compares retrievers on the chunks (step 8, `Learnings.md` L19).

### 8. Next steps (agreed order, not started)
1. **Retriever (stage 1). Done 2026-10-29** (code, 78 tests, real cache of 510 chunks, same rankings as the benchmark on 1,582 questions; the server run under `ng_run` is not yet tested). Add a BGE-M3 hybrid retriever to the server (dense 0.4, sparse 0.2, ColBERT 0.4, on all 510 chunks, in memory, encodings cached on disk), selected by a config value, behind a small `search(query, k)` interface. The 510 chunks, their metadata and the questions do not change. The Chroma index stays as the fallback and as the source of the chunk text.
2. **Eval only, no training.** Run the base model and run 2's final model with the new retriever at `top_k` 2 and 3. The verifier stays the same, so the numbers stay comparable. Re-run the base model whenever the retrieval changes.
3. **Data quality audit (in parallel, no training).** Sample rows through a validator and read about 30 by hand. Count defects by type (reference does not answer the question, question not self-contained, reference garbled). Remove the repeated question rows (validation: 18 extra rows, 7 repeats with different references).
4. **Semantic matching in the verifier (stage 2), after step 3.** BGE-M3 on the CPU in the server. The semantic score can only add credit and never replaces exact or containment. Calibrate the thresholds on labelled examples. Test it offline on stored dumps with `rescore_dump.py` first.
5. **Training run 3** on the new retrieval, with token limits that fit `top_k` 3, and with a decision about the retrieval part of the reward (it carried 7% of the signal in run 2).
6. **Version 2 of the corpus.** Two stages (dense and sparse candidates, then ColBERT rerank) and a vector engine such as Vespa or Qdrant. ColBERT vectors for all chunks stop fitting in memory at about 50,000 chunks.

## Verification
1. From `NeMo_Gym`, run `(ulimit -v 25165824; resources_servers/mission_search/.venv/bin/python -m pytest resources_servers/mission_search/tests)`: done 2026-10-29, 78 pass (verifier, report, retrieval benchmark, retriever), 2.7 GB resident, 16.3 GB virtual (cap 24 GB; see the header of `tests/test_retriever.py`). From `trainer`, run `temp/.venv-temp/bin/python test_rollout.py`: done, 24 pass.
2. Start `ng_run` (procedure 4.4), then `curl` `/search_corpus` and `/verify`: done.
3. Step-1 spike: done. Re-run it after any change to the rollout loop.
4. Smoke-train with `--max-steps 4 --save-steps 2`, then resume from `checkpoint-2` and evaluate the checkpoints: done.
5. Full runs of 200 steps, then `eval.py` on base and checkpoints: done twice (step 6).
6. After a change to the retriever or the verifier: re-run the base model first, because the yardstick changed.

## Surprises
Every "yes" from a RESULT / SURPRISE line goes here (rule in `AGENTS.md`). The date is at the start of the Prediction cell.

| Prediction | Check | Result | Surprise? | What we do |
|---|---|---|---|---|
| 2026-10-28: loading the cached BGE-M3 encodings (251 MB file, 510 chunks) takes a few seconds and a few hundred MB. | First start of the BGE-M3 retriever on the real cache. | The python process grew to 31 GB and was killed; WSL crashed 3 times. Cause: the `.npz` arrays were read again on every access, and the loop kept 510 full copies. The 5-chunk test never showed it, because the cost grows with the square of the chunk count. | yes | Loader replaced by a memory-mapped `.npy` cache with a manifest and a size guard. New test loads real-size arrays in a child process with a hard memory limit. Run every memory-heavy test under a cap. |
| 2026-10-29: the 15 no-model retriever tests peak under 300 MB (the pytest parent process). | `/usr/bin/time -v` around the run, 4 GB cap. | 15 pass in 3.4 s, peak 523 MB. The test that imports `app.py` takes 2.2 s (it loads the NeMo Gym and pydantic code). | yes | No plan change. Expect about 600 MB for this group; the 4 GB cap stays far above it. |
| 2026-10-29: the 3 model tests peak at 2.5 to 4.5 GB (RSS) and take 1 to 2 minutes. | `/usr/bin/time -v` around the run, 16 GB cap. | 2 pass, 1 skipped (needs `data/chunks.jsonl` and a real cache). 23 s. Peak 5.15 GB. Probable cause (not measured): each `BGEM3Retriever` in a test loads its own copy of the 2.3 GB model (the 3 model tests make 6 loads in all: 3 + 2 + 1; how many are alive at once is not measured), and loading `pytorch_model.bin` needs extra memory for a short time. | yes | No plan change. The server loads one model, so expect a lower peak there (to measure in the real cache build). Cap for these tests: set from the measured virtual peak (next row), not from RSS. |
| 2026-10-29: the virtual peak (`VmPeak`) of the 3 model tests is 8 to 13 GB, so a cap of about 2× the measured peak is workable. | Same tests, 16 GB cap, `VmPeak` and `VmHWM` read from `/proc` at the end. | Same result (2 pass, 1 skip). `VmHWM` 5.16 GB, `VmPeak` 14.48 GB (2.8× the resident peak). The 16 GB cap had only 10% room. A cap of 2× RSS (10 GB) would fail the tests; 2× `VmPeak` (29 GB) is about the whole machine (31 GB) and protects nothing. | yes | The "2× the measured peak" rule does not work for `ulimit -v`. User chose a cap of 20 GB virtual for the model tests (about 1.4× `VmPeak`) on 2026-10-29. Written in the header of `tests/test_retriever.py`. |
| 2026-10-29: each of 18 deliberate breaks of the retriever code is caught by a test (one sparse break might survive). | Scratch harness: one break in a copy, run the 15 no-model tests under a 4 GB cap. | 15 caught, 3 survived: largest sparse token dropped (real gap), negative sparse token not refused (not covered), chunk-store meta line not required (the test only checked the error type; another check raised the same type). | yes | Added `test_sparse_lookup_finds_the_largest_token_id_and_ignores_negative_ids` and made the chunk-store test check the reason of each error. All 3 breaks now caught; 16 no-model tests pass. |
| 2026-10-29: the 3 breaks of the model-test group give clear CAUGHT results. | Same harness, 20 GB cap. | First run: all skipped (the copied test could not find the model folder). Second run: the ranking test failed in every break (the copy had no `data/validation.jsonl`), so those failures meant nothing. Third run, after fixing the harness: 3 of 3 CAUGHT for the right reasons. | yes | Harness fault only, no code fault. A break counts only if the unbroken copy passes first and the failing test is the one named. |
| 2026-10-29: `data/chunks.jsonl` is 0.4 to 0.6 MB (510 chunks). | `export_chunks.py` under a 4 GB cap. | 510 chunks, 293 KB, 101 to 1,089 characters, 0.5 s, 112 MB. | yes | No plan change (chunks are shorter than the 1,500 limit, about 575 characters on average). User chose to track the file in git (not committed by me). |
| 2026-10-29: a search with the new retriever takes about 50 ms (30 to 150 ms) on the real 510-chunk cache. | Scratch script: 1,582 searches (1,434 + 148 questions), 6 CPU threads, 20 GB cap. Ranking compared with the stored benchmark numbers. | Ranking identical to the benchmark on all 8 metrics (for example validation prov@3 0.9054, cov@3 0.8716). Start from cache 2.7 s; first build 84 s. Search: median 144 ms, 95th percentile 158 ms. Peak: 2.5 GB resident, 6.4 GB virtual. | yes | No plan change. The latency split (query encoding or scoring) is not measured. About 0.15 s per search is small next to generation time; measure it only if the training step time shows it. |
| 2026-10-29: the whole `tests/` folder in one process, 20 GB cap, gives 58 passed. | `pytest tests` with `/usr/bin/time -v`. | 76 tests ran: 74 passed, 2 failed. (1) `real_scale`: the child's peak read 2,172 MB because `ru_maxrss` is inherited from the parent at fork (the parent had loaded the model in an earlier test file). Reproduced with 2 tests (2,097 MB). (2) a model test: `unable to mmap` because several loaded models in one process need more than 20 GB of virtual space. | yes | (1) Test fixed: the child reads its own `VmHWM`; the 510-copy break and the no-mmap break are still caught. (2) Not a code fault: the full suite needs one process for each file or group (decision asked from the user). |
| 2026-10-29: `test_app.py`, `test_report.py`, `test_retrieval_bench.py` together peak at about 7 GB virtual (one model copy). | Same 3 files in one process, 20 GB cap, `VmPeak` and `VmHWM` read at the end. | 57 pass. `VmPeak` 15.98 GB, `VmHWM` 2.22 GB (7× the resident peak). | yes | Virtual size does not follow real memory use, so a `ulimit -v` number is a rough guard only. Decision on how to cap the full suite asked from the user. |
| 2026-10-29: after the retriever accepts a loaded model and the tests share one (`tests/conftest.py`), the full suite gives 77 passed. | `pytest tests` in one process, 25 GB cap, `VmPeak` and `VmHWM` read at the end. | 76 passed, 1 failed: my new fake-model test. The fake hashed words into 8 slots and "battery" collided with "in", "the", "ocean". Fixed with a fixed word list. After the fix and one more test (the searcher passes its model on): 78 pass. Resident peak 2.68 GB (was 4.5 GB), virtual peak 16.31 GB. | yes | The 6 model loads in the retriever tests explained the resident peak, but not the virtual size: it is still about 16 GB with one model (probably the imported libraries). Cap for the full suite asked from the user. |
