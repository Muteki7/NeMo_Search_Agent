# NeMo_search_agent

Train Llama 3.1 8B to search a document and answer from it. The method is GRPO. The environment is a NeMo Gym resources server. The trainer is Unsloth with TRL. Everything runs on one 24 GB GPU.

![Answer reward of the base model, run 2 and run 3 on 126 held-out questions, and the training curves of run 2 and run 3](docs/results.svg)

The model gets a question about NASA's Apollo 13 Mission Report and one tool, `search_corpus`. It must search, read the two chunks that come back, and answer in its own words. A reward tells it how good the answer was. There are no example answers or reasoning traces to copy.

## Results

Held-out questions: 126, greedy decoding, BGE-M3 retriever with the top 2 chunks. All three rows use the same old exact-word answer reward, so they are comparable (± is one standard error).

| Model | Answer reward |
|---|---|
| Base Llama 3.1 8B Instruct | 0.161 ± 0.028 |
| Run 2 (retrieval reward) | 0.569 ± 0.038 |
| Run 3 (BGE-M3 retriever, semantic credit, de-duplicated data) | 0.558 ± 0.038 |

- **Training works.** The reward rises from 0.16 to 0.56 (+0.397 ± 0.041, paired on the same questions).
- **What the model learns is mostly behaviour.** In run 3 the share of answered questions rose from 92% to 99.5% during training, and searches per question fell from 1.65 to 1.02. In runs 1 and 2, retrieval itself did not get better.
- **Run 3 is not better than run 2.** Run 3 changed three things at once (retriever, semantic credit, cleaned data). The difference is -0.011 ± 0.019, which is no difference.
- **The semantic credit changes the score, not the model.** It adds about 0.10 to both models (run 2: 0.667, run 3: 0.653 with credit).
- **Limits.** One training run for each setting, so there is no seed variance. The questions and reference answers were written by an 8B model and nobody checked them all. In a hand check of 52 answers, 15 had a bad reference or a bad question. A perfect verifier would score about 0.68 on the labelled answers.

The numbers come from the eval and training dumps in [trainer/results/](trainer/results/) (about 10 MB, in this repository). `cd trainer && python3 plot_results.py` draws the chart from them.

## How it works

```
 TRAINER ENVIRONMENT (GPU)                   SERVER ENVIRONMENT (CPU)
 +------------------------------+            +------------------------------+
 | Unsloth + TRL GRPOTrainer    |            | NeMo Gym resources server    |
 | vLLM on the same GPU         |  tool call | search_corpus: BGE-M3 search |
 | rollout_func: the agent loop | ---------> |   returns the top 2 chunks   |
 | Llama 3.1 8B (4-bit) + LoRA  | <--------- |                              |
 |                              |  answer    | verify(): the reward         |
 |                              | ---------> |   returns reward + parts     |
 |                              | <--------- |                              |
 +------------------------------+            +------------------------------+
 (HTTP: POST /search_corpus for each tool call, POST /verify for each finished answer)
```

1. **The agent loop is in the trainer.** `trainer/rollout.py` generates text, parses a JSON tool call, sends it to `/search_corpus`, adds the result, and repeats for up to 5 turns. It keeps the sampled token ids and never encodes them again (Token-In, Token-Out). Tool-result tokens are masked out of the loss.
2. **The environment is a NeMo Gym resources server** (`NeMo_Gym/resources_servers/mission_search/`). It has two routes that matter: `POST /search_corpus` (the tool) and `POST /verify` (the reward).
3. **The reward** is `0.75 x answer + 0.25 x retrieval`.
   - Answer: 1.0 for an exact match, or when the reference is a run of whole words inside a short answer. Otherwise the token F1 over content words.
   - Semantic credit (run 3): an answer in other words gets at least 0.85 if every number of the reference is in the answer and the BGE-M3 ColBERT score (reference as the query) is at least 0.65. The credit never lowers a reward.
   - Retrieval: pays for a search whose result holds the reference's facts. A later search counts half. It counts only when the trajectory ends with a final answer.
4. **The retriever** of run 3 is BGE-M3 (dense 0.4, sparse 0.2, ColBERT 0.4). It scores all 510 chunks in memory, and the encodings are cached on disk. Runs 1 and 2 used a smaller embedder (NoInstruct) with a Chroma index.

| Run | Retriever | Reward | Training data | Server config |
|---|---|---|---|---|
| 1 | NoInstruct + Chroma | answer only | `train` (1,240 rows) | `resources_only.yaml` |
| 2 | NoInstruct + Chroma | answer + retrieval | `train` (1,240 rows) | `resources_only.yaml` |
| 3 | BGE-M3 hybrid | answer + retrieval + semantic credit | `train_dedup` (1,108 rows) | `resources_only_bge_m3_top2_semantic.yaml` |

This repository holds the code of run 3. Runs 1 and 2 need files that are not here (see [What you can reproduce](#what-you-can-reproduce)).

## Where the chunks come from

The agent searches 510 chunks of NASA's Apollo 13 Mission Report. The finished chunks are in `NeMo_Gym/resources_servers/mission_search/data/chunks.jsonl`. They were made in three steps:

1. **Split (not in this repository).** AutoDidact's `generate_data.py` loads the report as a markdown file (`UnstructuredLoader`) and splits it with `RecursiveCharacterTextSplitter` (1,000 characters, no overlap). Result: 753 chunks, saved as `chunks.pkl`. This project did not change that step. How the PDF was turned into markdown is not recorded here.
2. **Merge.** `scripts/build_index.py` glues every chunk of fewer than 100 characters onto the chunk after it, never above 1,500 characters. The splitter had cut each heading, page ID and caption on its own, so 206 of the 753 chunks had fewer than 50 characters. Result: 510 chunks of 101 to 1,089 characters. The same script builds the Chroma index that runs 1 and 2 used.
3. **Export.** `scripts/export_chunks.py` reads the merged chunks out of that index and writes `data/chunks.jsonl`. This is the only chunk file that run 3 reads, so repeating run 3 needs neither step 1 nor step 2.

The threshold of 100 characters was chosen by comparing five variants on all 1,434 questions: 100 gave the best answer coverage in the top 2 results (75.0%, against 72.0% with no merge). [docs/Plan.md](docs/Plan.md) has the comparison.

Each chunk in `chunks.jsonl` also keeps `start` and `end`, the positions in the old 753-chunk list that it covers. The questions in `data/` point at those old positions, and the retrieval check uses the ranges. Do not split the report again: different cuts would break the link between the questions and their chunks.

## Two environments

The server and the trainer have their own Python environments. They share no package. They talk over HTTP only. Never install one requirements file into the other environment.

| | Server environment | Trainer environment |
|---|---|---|
| Job | The search tool and the reward (NeMo Gym) | Training and evaluation |
| Python | 3.13 | 3.12 |
| torch | 2.11.0, CPU only | 2.11.0, CUDA 12.8 |
| transformers | 5.18.0 | 5.17.0 |
| GPU | not used | needed, 24 GB |
| Folder | `NeMo_Gym/resources_servers/mission_search/.venv` | `trainer/.venv` |
| Requirements | [`NeMo_Gym/resources_servers/mission_search/requirements.txt`](NeMo_Gym/resources_servers/mission_search/requirements.txt) | [`trainer/requirements.txt`](trainer/requirements.txt) and [`trainer/overrides.txt`](trainer/overrides.txt) |

Why two: the server needs CPU torch so the GPU stays free for training, and the two stacks pin different versions. `ng_run` also installs packages into the server environment on start, unless you pass `+skip_venv_if_present=true` (read in the NeMo Gym code, see `docs/Learnings.md`, L14). The docs name the trainer environment `trainer/temp/.venv-temp`. It is the same environment as `trainer/.venv` here.

The trainer file pins the versions that trained run 3. Three things in it need a word of explanation:
- `overrides.txt` is needed. Unsloth declares `trl<=0.24.0` and `transformers<=5.5.0`, but this project needs newer versions of both. It worked in all runs, but Unsloth has not tested it.
- torch, torchvision and torchaudio are pinned by wheel URL. A plain `torch==2.11.0` gives a different build, the PyPI torchaudio cannot load against the CUDA 12.8 torch, and `--torch-backend cu128` makes the resolver fail on `torchao`.
- `xgrammar` is pinned to 0.2.3. Version 0.2.5 declares `transformers<5` and adds a third problem to `pip check`.

## Requirements

- Linux or WSL2. It was built and tested on WSL2 Ubuntu.
- An NVIDIA GPU with 24 GB. It was trained on an RTX 5090 Laptop GPU (24 GB). The peak was 19.9 GB in earlier runs, with vLLM set to 55% of the GPU. A smaller GPU is untested.
- About 31 GB of RAM. The server with BGE-M3 uses about 2.5 GB of it.
- [uv](https://docs.astral.sh/uv/) and git. uv installs Python 3.12 and 3.13 for you.
- Disk: the trainer environment is about 13 GB (uv shares files between environments through its cache). Llama 3.1 8B in 4-bit is 5.6 GB, BGE-M3 is 2.3 GB, and each saved checkpoint is about 1 GB.

## Set up

The steps below run the final version (run 3). All commands start in the repository root.

**1. Get NeMo Gym.** The server installs it from a local clone, pinned to the commit used here.

```bash
git clone https://github.com/NVIDIA-NeMo/Gym NeMo_Gym/Gym
git -C NeMo_Gym/Gym checkout 12399ad815d410b47284547d73d933d12b0696c6
```

**2. Build the server environment.** Run the install from the server folder: the first line of its requirements file is a relative path.

```bash
cd NeMo_Gym/resources_servers/mission_search
uv venv .venv --seed --python 3.13.14
uv pip install --python .venv/bin/python -r requirements.txt
cd ../../..
```

**3. Download BGE-M3** (2.3 GB, MIT licence) into `Embedder/bge-m3`. The server reads it from that path.

```bash
NeMo_Gym/resources_servers/mission_search/.venv/bin/hf download BAAI/bge-m3 \
  pytorch_model.bin colbert_linear.pt sparse_linear.pt config.json tokenizer.json \
  tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 --local-dir Embedder/bge-m3
```

**4. Build the trainer environment.** Run the whole list in one command, never one package alone.

```bash
cd trainer
uv venv .venv --python 3.12 --seed
uv pip install --python .venv/bin/python --override overrides.txt -r requirements.txt
```

After the install, `pip check` must report exactly two problems: `unsloth` wants an older `trl` and an older `transformers`. Any other line is new.

### What was checked

The checks below ran on 2026-10-31 on the author's machine, unless a date says otherwise.

| Part | Checked | Not checked |
|---|---|---|
| Server environment | Built from its file on 2026-10-05. `uv pip install --dry-run` from the server folder reports no changes. | An install from the file into an empty environment since then. |
| BGE-M3 download | The file names and sizes of the command match the Hugging Face Hub (`hf download --dry-run`) and the author's local copy. | The download itself. |
| Trainer environment | The file was installed into an empty environment (about 4 minutes with downloads, 2.8 GB of new disk). `pip check` showed exactly the two problems above. `torch` is `2.11.0+cu128` and sees the GPU, all 17 `==` pins match, and the 26 offline trainer tests pass. `trainer/spike_rollout_func.py` (two GRPO steps on a 1B model, vLLM on the same GPU) passed all 19 checks in 3 minutes. | A full training run on this environment. About 70 packages that the file does not pin (for example pydantic, starlette, pillow) resolve to newer versions than in the environment that trained run 3. |
| Server start | `ng_run` with the run 3 config and no `chroma_index` folder reached `All 1 / 1 servers ready`, answered `/search_corpus` and `/verify`, and stopped without leaving a process. | Training against that server after the last change to `app.py`. |

Nobody has run the whole path from a clean clone, start to finish.

## Run it

Use two terminals. Start the server first and leave it running.

**Terminal 1: the server** (from `NeMo_Gym`).

```bash
cd NeMo_Gym
resources_servers/mission_search/.venv/bin/ng_run \
  "+config_paths=[resources_servers/mission_search/configs/resources_only_bge_m3_top2_semantic.yaml]" \
  +skip_venv_if_present=true
```

Wait for `All 1 / 1 servers ready`. The log must also show `[verify] semantic credit on`. The first start encodes the 510 chunks with BGE-M3 (about 80 s) and saves them to `m3_cache/`. Later starts take a few seconds. `+skip_venv_if_present=true` stops `ng_run` from installing packages into the server environment. NeMo Gym puts its head server on port 11000 and the resources server on a random port. The trainer finds that port by asking the head server.

**Terminal 2: train** (from `trainer`). Try a short run first (about 4 minutes), then the full run (about 74 minutes for 200 steps on the RTX 5090 Laptop GPU).

```bash
cd trainer
.venv/bin/python train_grpo.py --max-steps 4 --save-steps 2 --name smoke
.venv/bin/python train_grpo.py --max-steps 200 --save-steps 50 --name run3_semantic \
  --train-split train_dedup 2>&1 | tee run3_semantic.log
```

Each run gets a folder `trainer/runs/<start time>_<name>/` with `run_config.json`, `train_dump.jsonl` (every trajectory with its reward parts), the checkpoints and `final/`. After a crash, add `--resume runs/<run>/checkpoint-<N>`. `train_grpo.py` sets `VLLM_WSL2_ENABLE_PIN_MEMORY=1` itself, which WSL2 needs.

**Evaluate** (server still running, GPU free, about 25 minutes). The `reward` column of the table is the old-rule answer reward, the yardstick of the results above.

```bash
.venv/bin/python eval.py --run-dir runs/<run folder> --split validation_dedup \
  --checkpoints runs/<run folder>/checkpoint-50 runs/<run folder>/checkpoint-100 \
                runs/<run folder>/checkpoint-150 runs/<run folder>/checkpoint-200
```

`docs/Learnings.md`, section 4, has the same procedures with checks after each step, and the other tools: the failure report (`scripts/report.py`), the retriever benchmark (`scripts/retrieval_bench.py`) and the re-scoring of saved answers (`scripts/rescore_dump.py`).

## Tests

All tests run on the CPU. The trainer tests use a fake model and a fake tool.

| Tests | Command | Result on 2026-10-31 |
|---|---|---|
| Server (118) | `cd NeMo_Gym/resources_servers/mission_search && (ulimit -v 25165824; .venv/bin/python -m pytest tests)` | 118 passed in 32 s, 2.6 GB resident |
| Trainer rollout (26) | `cd trainer && .venv/bin/python test_rollout.py` | 26 of 26 passed |
| Chart (16) | `cd trainer && python3 test_plot_results.py` | 16 of 16 passed |

The `ulimit -v` cap protects the machine: the BGE-M3 tests are memory-heavy. Tests that need `Embedder/bge-m3`, the Chroma index, `data/train.jsonl` or the run folders skip themselves. With only the tracked files and the server environment, 105 pass and 13 skip (checked on a copy of the tracked files, 2026-10-31). Not covered by this table: `eval.py` on the GPU, the server under `ng_run` with 16 parallel verifies, and training quality. Two scripts need the GPU: `trainer/spike_rollout_func.py` (passed on 2026-10-31, see above) and `trainer/rollout_smoke.py` (the rollout on the real engine and a live server; not re-run on 2026-10-31).

## Repository layout

```
NeMo_search_agent/
├── README.md
├── LICENSE                     Apache-2.0 (code only, see Licence)
├── AGENTS.md                   how the author and the coding agent split the work (decision rights, test and reporting rules)
├── docs/
│   ├── Plan.md                 design, decisions, results of each run, and the table of wrong predictions
│   ├── Learnings.md            what went wrong, the rule taken from it (L1 to L25), and step-by-step procedures
│   ├── Dependency_hell.md      the trainer package stack: what broke, why, and how it was fixed
│   └── results.svg             the chart at the top of this file
├── NeMo_Gym/
│   └── resources_servers/mission_search/     the environment (the server environment)
│       ├── app.py              the /search_corpus tool and the /verify reward
│       ├── retriever.py, m3.py, chunks.py, embeddings.py     the retrievers (BGE-M3 hybrid; NoInstruct + Chroma)
│       ├── configs/            five ng_run configs (the one for run 3 is resources_only_bge_m3_top2_semantic.yaml)
│       ├── data/               chunks.jsonl (510 chunks), train_dedup.jsonl, validation_dedup.jsonl, verifier_labels.jsonl (72 hand labels)
│       ├── scripts/            data, index, report and benchmark tools
│       ├── tests/              the 118 server tests
│       └── requirements.txt    the server environment
└── trainer/                    the trainer environment
    ├── rollout.py              the agent loop, the Gym client, the vLLM wiring, the reward call
    ├── train_grpo.py           Unsloth model, GRPOTrainer, run folders, resume
    ├── eval.py                 base model and checkpoints on the validation questions
    ├── prompting.py            chat prompts with the template date pinned
    ├── plot_results.py         draws docs/results.svg
    ├── results/                the eval and training dumps behind the chart (9 files, about 10 MB; copies from the git-ignored runs/)
    ├── test_rollout.py, test_plot_results.py, rollout_smoke.py, spike_rollout_func.py
    ├── requirements.txt, overrides.txt     the trainer environment
    └── run2.log, run.log       the training logs of run 2 and run 3
```

## What you can reproduce

| You want to | What you need | Is it in the repository? |
|---|---|---|
| Train run 3 again, or evaluate a model | The Gym clone (Set up, step 1), the BGE-M3 weights (step 3) and the trainer environment | Yes, apart from those two downloads. `data/` holds the finished chunks (`chunks.jsonl`) and questions (`train_dedup.jsonl`, `validation_dedup.jsonl`). |
| Check the numbers and redraw the chart above | `trainer/results/`: the eval and training dumps (about 10 MB) | Yes. `cd trainer && python3 plot_results.py` |
| Use the trained models | The LoRA adapters (about 640 MB each) | No, they are not published. You get your own when you train. |
| Repeat run 1 or run 2 | `Step2_Files/` (git-ignored): the report, AutoDidact's `chunks.pkl` and `questions.json`. They build `data/train.jsonl` (`scripts/convert_questions.py`) and the Chroma index (`scripts/build_index.py`). | No. |

`chroma_index/` (built by `scripts/build_index.py`) and `m3_cache/` (builds itself on the first server start) sit in the server folder and are git-ignored.

## The docs

Read these when the README is not enough. They are long, and they are written as a record, in the order things happened.

- **[docs/Plan.md](docs/Plan.md)**: why it is built this way. The design and the decisions with their reasons, then each build step (data, index, server, trainer) with its status. It has the results of all three runs in tables, and a "Surprises" table: every prediction that turned out wrong, how it was checked, and what changed. Read it to see how the numbers above were reached and what they do not show.
- **[docs/Learnings.md](docs/Learnings.md)**: what went wrong and the rule taken from it. 25 entries (L1 to L25), each with the problem, the cause, the fix and a rule (for example: keep token ids and never encode them again, remove a validation leak with a purged split, compare runs with the same retriever, rule and rows). It also has a glossary, the step-by-step procedures for every task, and the open items.
- **[docs/Dependency_hell.md](docs/Dependency_hell.md)**: why the trainer stack is pinned the way it is. The working package versions, 28 install and run-time failures with cause and fix, and the rules for installing in this stack (one full install, import tests, never `-U`). It names a `trainer/temp/working_stack.freeze.txt` file: that folder is git-ignored, and `trainer/requirements.txt` holds the same pins.

## Acknowledgements

The idea comes from [AutoDidact](https://github.com/dCaples/AutoDidact), which trained Llama 3.1 8B to search the Apollo 13 report with GRPO. This project started from its question and chunk files, its prompts and its hyperparameters. The environment, the rollout inside TRL, the retrievers, the reward and the evaluation are built differently.

It builds on [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym), [Unsloth](https://github.com/unslothai/unsloth), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm), BGE-M3 by BAAI, Llama 3.1 by Meta, and NASA's Apollo 13 Mission Report (MSC-02680).

## Licence

The code in this repository is under the Apache License 2.0 (see [LICENSE](LICENSE)). The licence covers the code only. These parts keep their own terms:

- the text of the NASA report in `data/chunks.jsonl`;
- the questions and reference answers in `data/`, which an 8B Llama model wrote;
- the weights of Llama 3.1 (Meta's Llama 3.1 Community License) and BGE-M3 (MIT).

Some server files carry the NVIDIA header of the NeMo Gym scaffold they started from.
