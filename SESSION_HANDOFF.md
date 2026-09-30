# Session Handoff: AutoDidact → NeMo Gym migration

Read this first, then read `Plan.md` for the forward plan. This file covers what the project is, what was learned the hard way, and the current state. It does not repeat the plan.

## 1. Project in one paragraph

`AutoDidact/` teaches Llama-3.1-8B-Instruct (4-bit QLoRA, r=64) to answer questions about a document (Apollo 13 mission report) by calling a `search_corpus` tool, using GRPO.

Data flow:
1. `generate_data.py` splits `data/mission_report.md` into 753 chunks (1000 chars, no overlap) and embeds them into a Chroma index (`chroma_index/`, collection `example_collection`).
2. The same Llama model then writes 1434 synthetic `{question, answer, difficulty}` rows from sliding 3-chunk windows into `saved_data/questions.json`. These are the RL labels. The trainee sees only the question and must search to answer.
3. `rl_helpers.py` is the RL environment: OpenAI-style tool schema, JSON tool-call parsing (`extract_json_objects`), a multi-turn `run_agent` loop, and rewards (LLM-judge correctness + a flat 0.7 formatting reward).
4. `autodidact.ipynb` runs GRPO. `search_module.py` holds retrieval and the 90/10 train/test split (seed 42).

`simple_qa.py` is a manual human CLI, not a model eval. It is also broken (`verify` and `router` are undefined). `rl_helpers.check_correctness` is dead code for the same reason.

**Reusable artifacts:** `saved_data/questions.json` (1434 rows), `saved_data/chunks.pkl` (753 LangChain `Document`s), `chroma_index/`. Do NOT regenerate questions.

## 2. Roadblocks and fixes (chronological)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | LangChain import errors | `langchain.*` split into `langchain_community`, `langchain_text_splitters`, `langchain_core` | New paths. FAISS → Chroma: use `Chroma.from_documents(documents=..., embedding=..., collection_name=..., persist_directory=...)`. The constructor has no `chunks=` kwarg and no `save_local`; persistence is via `persist_directory`. `search_module.py` was updated to load Chroma. |
| 2 | `NameError: auto_docstring` in `unsloth/models/_utils.py` (`exec(config, globals())`) | Unsloth rewrites `transformers` source and `exec()`s it. It breaks on transformers ≥5.x internals. | Not fixable by pinning around it. See #3. |
| 3 | Dependency hell | pip only checks the packages in the current command, so incremental installs silently break others. Metadata is also disjoint: unsloth 2026.9.11 needs `torch<2.13`, `transformers<=5.5.0`, `trl<=0.24.0`, while vllm 0.30.0 needs `torch==2.13.0`, `transformers>=5.10.4`. | Resolve everything in ONE `uv pip install` with the full package list, then `pip check`. Narrow two-package resolves drift (`unsloth vllm` alone picked vllm 0.2.5). |
| 4 | "Latest" combos that pass metadata but crash at runtime | `unsloth 2025.9.5` has no `transformers` upper bound (stale, not tested) and crashes on transformers 5.x | A clean resolve only proves metadata. Always import-test. |
| 5 | vLLM version window | vLLM's `transformers` floor jumps from `>=4.56.0` (≤0.22.0) to `>=5.5.3` (0.24.0+) | `vllm==0.22.0` is the newest compatible with Unsloth's ceilings. |
| 6 | `wandb` circular-import `AttributeError` | Incomplete install (`platformdirs` missing) after `--no-deps` reinstalls | Reinstall with deps. Avoid `--no-deps` force-reinstalls. |
| 7 | `unstructured` 403 on NLTK download | Its hardcoded S3 mirror (`utic-public-cf.s3.amazonaws.com`) is dead | `python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng` (now in `~/nltk_data`). Also `unstructured` 0.15.14 required `numpy<2`, so it was upgraded to 0.27.8. |
| 8 | Re-running the model-load cell → OOM (0.0 GiB free) | The old vLLM engine still holds VRAM in the kernel process | Restart the Jupyter kernel. In-process teardown of vLLM is unreliable. |
| 9 | Kernel not selectable in VS Code | venv had no `ipykernel`, and lives outside the workspace | `ipykernel` installed and registered as kernel `vllm-workspace`. |
| 10 | `ImportError: AutoModelForCausalLM` from `trl.trainer.grpo_trainer` | Vendored `UnslothGRPOTrainerTemp.py` targets a year-old TRL | Dead end, superseded by #12. Unsloth now auto-generates `unsloth_compiled_cache/UnslothGRPOTrainer.py` (version-fingerprinted header; regenerated at import; safe to delete). Do not import from a frozen copy. |
| 11 | `torchvision::nms does not exist`, then `Could not import 'BertModel'` | Upgrading `trl` alone bumped torch to 2.14 but left torchvision 0.26 (built for torch 2.11). The BertModel error is collateral: transformers' lazy import chain reaches `torchvision.io`. | Restore the consistent stack (section 3). The `Enum ... register_constant()` warnings are torchao-internal noise, not actionable. |
| 12 | `use_agentic_generate` "not a valid GRPOConfig arg" | Feature was removed from current Unsloth (zero occurrences in the generated trainer). It was never an upstream TRL feature. | Replaced by the NeMo Gym design (see `Plan.md`). Notebook cells 5/6 still reference it and are obsolete. |
| 13 | TRL `environment_factory` unusable with Unsloth | It hard-requires `transformers>=5.2.0` (`ImportError` at trainer init). Every Unsloth version either caps below that or crashes at runtime. | Do not use `environment_factory` with Unsloth. Use TRL `rollout_func` (exists ≥0.25; `env_mask` ≥0.28). |

## 3. Known-good training stack (`pip check` clean, Python 3.12, RTX 5090 Laptop 24 GB)

`unsloth 2025.9.9` · `trl 0.29.1` · `transformers 4.56.2` · `vllm 0.22.0` · `torch 2.11.0` · `torchvision 0.26.0` · `datasets 3.6.0`

For a fresh venv, pin `unsloth_zoo==2025.9.10`. Do not leave it unpinned. `2025.9.14` caps `trl<=0.23.0`, which conflicts with `trl==0.29.1`. The existing `vllm-workspace/.venv` has a mismatched `unsloth_zoo 2026.9.7`.

Runtime behavior of this exact stack is not yet verified end to end.

## 4. Rules learned this session

- Never `pip install` one package into a resolved stack. Dry-run the full list with `uv pip install --dry-run`, apply it, then `pip check`.
- Check PyPI `requires_dist` (via `curl https://pypi.org/pypi/<pkg>/<version>/json`) before trusting any version combo.
- Verify claims from pasted material against the primary source. The claimed "precedent" that unsloth#3786 shows TRL `tools=` working was wrong. The issue is an open bug report.
- Ask before large or hard-to-reverse changes (uninstalling or downgrading packages, rewriting architecture).

## 5. Direction: see `Plan.md`

The decision is to rebuild in a fresh directory: NeMo Gym hosts the environment (`search_corpus` tool plus `verify`), and a separate Unsloth trainer venv drives multi-turn rollouts via TRL `rollout_func`. Two venvs are intentional: the Gym venv (Python 3.13.14, managed by `uv sync`) and the training venv (Python 3.12).

Two facts an agent must not lose:
- `Example_NeMo.ipynb` (Unsloth's notebook) is single-turn: it only POSTs finished completions to `/verify`. It never exercises tools during training.
- The first task is a gate: spike `rollout_func` on Unsloth before building anything else. The risk is that Unsloth 2025.9.9's patcher targets the older `self.llm = LLM(...)` layout while TRL 0.29 uses `self.vllm_generation`, so it may not reuse Unsloth's vLLM engine. Fallback: plain TRL + PEFT with TRL's own `vllm_mode="colocate"`.

## 6. Loose ends

- `requirements.txt` is stale (`transformers==4.49.0`, `unsloth==2025.3.6`, `faiss-gpu`).
- `generate_data.py` has stale FAISS text in its docstring, and its top-of-file `import unsloth` ordering warning is harmless.
- Unverified: whether `embeddings.py` was migrated to `from langchain_core.embeddings import Embeddings`.
- `Setup.md` holds the user's own learning notes. It is theirs, so don't overwrite it.

## 7. Working style the user likes

- One-line status updates while fixing things, so they stay in the loop.
- Conceptual "rough mental model" explanations alongside fixes, with their misconceptions corrected directly.
- Honest pushback with evidence over agreeing.
