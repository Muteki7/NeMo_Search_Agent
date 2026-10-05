# Debug checks for the index build (Step 3)

| Item | Data |
|---|---|
| Title | Debug checks for the index build |
| Specification | ASD-STE100 (Simplified Technical English) |
| Project | Train a small model to answer questions with a search tool |
| Scope | Two files: `scripts/build_index.py` and `embeddings.py` |
| Location | `NeMo_Gym/resources_servers/mission_search/` |
| Tool | The Python debugger in VS Code |
| Last update | 2026-10-03 |

## 1. How to read this file

Use this file with the debugger in VS Code. After you do all the stops, you must understand both files and their important operations.

Each stop has up to five parts: Breakpoint, Do, Look at, Expect and Learn. The stops follow the real flow of the program, from start to end.

Each number has a tag. The tag [real run] marks a number from an earlier real run. The tag [arithmetic] marks a number that we calculate. The tag [estimate] marks a guess from the code.

The values come from a read of the code and from earlier real runs. No new run was made for this file. The probe file `/tmp/claude-1000/-home-nk-projects-NeMo-search-agent/3959fcc0-83f3-4f41-931b-f1f2461ad4e7/scratchpad/debug_probe.py` is not used. You can delete it.

## 2. Technical names

Use the same name for the same thing every time.

| Name | What it is | Do not use |
|---|---|---|
| breakpoint | A mark on a line. The program stops at this line. | stop point |
| conditional breakpoint | A breakpoint with a test. The program stops only when the test is true. | - |
| stop | One place where the program waits at a breakpoint | pause |
| Step Over | The key F10. It runs the current line. It does not enter the functions that the line calls. | skip |
| Step Into | The key F11. It enters the function that the current line calls. | dive |
| Step Out | The keys Shift+F11. It runs to the end of the current function and goes back to the caller. | exit |
| Continue | The key F5. It runs to the next breakpoint. | resume |
| Call Stack | The panel with the list of active function calls. The newest call is at the top. | call chain |
| Debug Console | The panel where you run expressions in the stopped program | REPL |
| launch configuration | An entry in `.vscode/launch.json`. It tells the debugger what to run and how. | debug profile |
| interpreter | The Python program that runs the code. Here it is the `.venv` of the server. | python version |
| status bar | The bar at the bottom of the VS Code window. It shows the selected interpreter. | footer |
| Locals | The panel group with the variables of the current function | local scope |
| Globals | The panel group with the variables of the module | global scope |
| chunk | A piece of the source document, about 1000 characters long | part, segment |
| chunk id | The number of the middle chunk of a window | chunk number |
| window | Three chunks in a row. The question generator reads one window at a time. | span |
| token | A piece of text that the model reads | word |
| vector | A list of numbers that stands for one text | embedding |
| embedder | The object of the class `CustomHuggingFaceEmbeddings`. It turns text into vectors. | encoder |
| batch | A group of texts that the model reads together | bundle |
| CLS vector | The model output for the first token. We use it for documents. | first vector |
| mean pooling | The average of the token outputs, weighted by the attention mask. We use it for queries. | - |
| index | The Chroma files on disk. They hold the vectors and the chunk text. | database |
| collection | The named table in the index. Its name is `example_collection`. | table |
| distance | The number that Chroma gives for a match. A lower number means a closer match. | score |
| sample | The questions that the check tests | subset |
| window recall | The share of questions for which the top results hold a chunk of the question window | hit rate |
| centre-chunk recall | The share of questions for which the top results hold the middle chunk of the window | exact recall |

## 3. The files

- `scripts/build_index.py` is the flow. It goes from the chunks on disk to a tested index. Start here.
- `embeddings.py` turns text into vectors. It has the most important logic. You stop in it from inside `build_index.py`.
- `requirements.txt` has no code to step through.
- Both code files are in `NeMo_Gym/resources_servers/mission_search/`.

## 4. Setup

Do these steps one time:

1. Pick the interpreter `NeMo_Gym/resources_servers/mission_search/.venv/bin/python`. Use "Python: Select Interpreter". Enter the full path, not a relative path.
2. Look at the status bar. It must show `3.13.14 ('.venv': venv)`.
3. Add the launch configuration from section 5 to `.vscode/launch.json`.

Do this step after each session:

- Delete `/tmp/debug_index`. It holds about 6 MB.

All code runs in one CPU process. A breakpoint in our code hits as normal. This is not true for vLLM, which runs in other processes.

Open these panels: Variables (Locals and Globals), Watch, Call Stack and Debug Console.

## 5. The launch configuration

The file `.vscode/launch.json` is in the repository root. It tells the debugger what to run. Add this entry to the list `configurations`:

```json
{
    "name": "Debug build_index (server venv)",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/NeMo_Gym/resources_servers/mission_search/scripts/build_index.py",
    "python": "${workspaceFolder}/NeMo_Gym/resources_servers/mission_search/.venv/bin/python",
    "args": ["--out-dir", "/tmp/debug_index/chroma_index", "--rebuild"],
    "env": {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
    "justMyCode": false,
    "console": "integratedTerminal"
}
```

Each key has a job:

| Key | Value | Why |
|---|---|---|
| `name` | `Debug build_index (server venv)` | This is the name in the Run and Debug list. |
| `type` and `request` | `debugpy` and `launch` | They start the file under the debugger. |
| `program` | The path of `build_index.py` | The script finds its files from `__file__`. So the working directory does not matter. |
| `python` | The `.venv/bin/python` of the server | It pins the interpreter. A change of the global selection has no effect. |
| `args` | `--out-dir /tmp/debug_index/chroma_index --rebuild` | The index goes to a debug folder. Your real index stays safe. |
| `args` (`--rebuild`) | Part of the line above | It deletes the debug folder before the build. You can run again with no cleanup. |
| `env` | `HF_HUB_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false` | The first value keeps the model load off the network, because the model is in the cache. The second value stops a warning from the tokenizer. |
| `justMyCode` | `false` | Step Into enters library code. Breakpoints in library files hit. |
| `console` | `integratedTerminal` | The output shows in the terminal. |

> CAUTION: Do not use `--rebuild` without `--out-dir`. The script then deletes your real `chroma_index`. You can build it again in about 30 s.

Write `/tmp`, not `/temp`. The folder `/temp` does not exist, and the script cannot make it.

To start a debug check:

1. Open the view Run and Debug.
2. Choose "Debug build_index (server venv)" in the list at the top.
3. Set the breakpoints for the stop that you want (section 6).
4. Press F5.

To debug a different script:

1. Copy the entry.
2. Change `name`, `program`, `args` and `env` for the new script.
3. Keep `python` and `justMyCode`.

Keep the entry "Python Debugger: Current File" without these arguments. It runs the active file. Other scripts have their own `--out-dir` option. For example, `convert_questions.py` has one.

## 6. The stops

### Stop 1: The guard and the global names

**Breakpoint.** `build_index.py:83`

**Do.** Press F10 to run lines 83 to 88.

**Look at.**

- In Locals, look at `args`. `args.out_dir` is a `PosixPath`. `args.rebuild` is `True`, because the launch entry sets it. `args.no_check` is `False`.
- In Globals, look at `SERVER_DIR`, `REPO_ROOT` and `COLLECTION`.
- Look at `sys.path[0]`. It is the server folder.

**Expect.** In the Debug Console, run `args.out_dir.exists()`.

- On the first run, it is `False`. The program skips the guard.
- On a later run, it is `True`. The program enters the guard. `args.rebuild` passes the test and the folder name passes the test.
- Then line 88 runs `shutil.rmtree`. It deletes only your debug folder.

**Learn.** The guard protects an index that is on disk. `sys.path[0]` shows why `import embeddings` works from the `scripts/` folder.

### Stop 2: The metadata cleaning

**Breakpoint.** Make a conditional breakpoint at `build_index.py:37`. Use the condition `not isinstance(v, (str, int, float, bool))`. It stops only on a value that Chroma rejects.

**Do.**

1. Look at `k` and `v`.
2. Press F10 one time.

**Expect.** `languages` is a list [estimate]. The code changes it to a JSON string, for example `'["eng"]'`.

**Learn.** You see which keys the code changes. Then make a second breakpoint at `build_index.py:45` with the condition `i == 0`. Compare `doc.metadata` before and after the line. The final keys are these [real run]: `category`, `chunk_index`, `element_id`, `file_directory`, `filename`, `filetype`, `languages`, `last_modified` and `source`.

### Stop 3: The loaded chunks

**Breakpoint.** `build_index.py:92`

**Look at.**

- `chunks` has 753 items [real run]. The type of an item is `Document` [real run].
- `lengths` holds the character count of each chunk.
- In the Debug Console, run `sum(1 for l in lengths if l < 50)`.

**Expect.** The result is 206 [real run]. The minimum length is 3 and the maximum length is 1000.

**Learn.** This is the tiny-chunk problem. You can see it in the data.

### Stop 4: The embedder

**Breakpoint.** `build_index.py:96`

**Do.**

1. Press F11. The debugger enters `CustomHuggingFaceEmbeddings.__init__` at `embeddings.py:29`.
2. Press F10 over the two `from_pretrained` lines.
3. Stop at `embeddings.py:37`.

**Call Stack.** The Call Stack shows these frames, newest first:

1. `__init__` in `embeddings.py`
2. `main` at `build_index.py:96`
3. `<module>` at `build_index.py:124`

**Look at.**

- `self.device` is `"cpu"`.
- `self.batch_size` is 32.
- `self.default_mode` holds the default mode.
- `self.model.config` holds `hidden_size` and `num_hidden_layers`. Check both values.

**Expect.** `self.model.training` is `False`. This is an estimate: `from_pretrained` returns a model in eval mode, so the value was `False` before line 37.

**Learn.** The CPU default and the batch size are our choice. Do not step into the `from_pretrained` calls.

### Stop 5: Chroma calls the embedder

**Breakpoint.** Make breakpoints at `embeddings.py:71` and `embeddings.py:72`.

**Do.** Put the condition `start in (0, 736)` on line 72. The program then stops on the first batch and on the last batch only.

**Call Stack.** At line 71, the frames come from the source code. The list starts at the outermost call and ends at the innermost call:

1. `<module>` at `build_index.py:124`
2. `main` at `build_index.py:97`
3. `from_documents` at `langchain_chroma/vectorstores.py:1375`
4. `from_texts` at `vectorstores.py:1273`. The batch loop is at line 1359.
5. `add_texts` at `vectorstores.py:597`. It calls the embedder at line 627.
6. `embed_documents` at `embeddings.py:66`

Tell Claude if you see more frames.

**Look at.**

- `texts` and `len(texts)`. The length is 753 [estimate], because the batch limit of Chroma is in the thousands.
- `self.batch_size`, `start` and the length of `out`.

**Expect.** The loop runs 24 times. It has 23 full batches and one batch of 17. The sum is 753 = 23 × 32 + 17 [arithmetic].

**Learn.** The document path embeds in batches of 32. `out` grows by 32 in each pass.

**Optional route.** Do not use the breakpoints. Press F11 at `build_index.py:97` to go through the library frames above.

### Stop 6: One batch becomes vectors

**Breakpoint.** Make a conditional breakpoint at `embeddings.py:55`. Use the condition `mode == "sentence"`.

**Look at.**

- Run `{k: tuple(v.shape) for k, v in inp.items()}`. `input_ids` has the shape `(32, L)`. `L` is at most 512 [estimate]. `L` is the token count of the longest chunk in the batch.
- Run `inp['attention_mask'].sum(dim=1)`. The real lengths are different in each row.
- Run `self.tokenizer.convert_ids_to_tokens(inp['input_ids'][0][:10])`. The result must start with `[CLS]`.

**Do.**

1. Press F10 over line 55.
2. Check `output.last_hidden_state.shape`. It is `(32, L, hidden)`.
3. Press F10 until line 63.
4. Check `vectors.shape`. It is `(32, hidden)`.
5. Press Shift+F11 to go back to `embed_documents` at line 73.

Do not step into line 55. It leads into the call machinery of torch.

**Learn.** Documents use the CLS vector (`[:, 0, :]`).

### Stop 7: The index exists

**Breakpoint.** `build_index.py:104`

**Look at.**

- `vectorstore` holds the Chroma object.
- In the Debug Console, run `vectorstore._collection.count()`. The result is 753 [real run].
- Run `vectorstore._collection.metadata` and `vectorstore._collection.configuration`. Look for the distance setting. It is `l2` [estimate].
- Run `list(args.out_dir.iterdir())`. The list holds `chroma.sqlite3` and one folder with a UUID name [real run].

**Learn.** You see the ids and the metadata that we gave. The fixed ids are `chunk-0` to `chunk-752` (line 100).

### Stop 8: The index opens from disk

**Breakpoint.** `build_index.py:111`

**Look at.** `count` is 753. Look at `len(chunks)`. `size_mb` is about 6.3 [real run].

**Learn.** `del vectorstore` and a new `Chroma(...)` object show that the files work alone. We still give `embedding_function`, but only to embed queries.

### Stop 9: The sample

**Breakpoint.** `build_index.py:58`

**Do.** Press F10.

**Look at.**

- `len(questions)` is 1434 [real run].
- `step` is 3.
- `len(questions[::step])` is 478 [arithmetic].
- `len(sample)` is 400.
- In the Debug Console, `(len(sample)-1)*step` is 1197. This is the index of the last sample question.
- Compare `sample[-1]['chunk_id']` with `questions[-1]['chunk_id']`. The second value is the largest value in the file: 751 [real run].

**Notice.** `[:n]` cuts the list that has the step. So the check never tests the questions from index 1198 on. These are about 16% of the file, at the end of the document. We think this is a flaw in the check.

A fix is to use a larger step, so that the sample covers the whole file. The code is not changed.

### Stop 10: One real search

**Breakpoint.** Make a breakpoint at `embeddings.py:80` in `embed_query`. Make a second breakpoint at `build_index.py:62` with hit count 1.

**Call Stack.** At `embed_query`, the frames come from the source code. The list starts at the newest call:

1. `embed_query` at `embeddings.py:76`
2. `similarity_search_with_score` at `vectorstores.py:817`. The call is at line 848.
3. `similarity_search` at `vectorstores.py:730`
4. `retrieval_check` at `build_index.py:62`
5. `main` at `build_index.py:115`
6. `<module>`

Python 3.13 runs a comprehension in the same frame. So there is no `<listcomp>` frame [estimate].

**Look at.**

- `mode` is `"query"` here. It is `"sentence"` in Stop 6.
- At lines 57 to 60, look at `inp['attention_mask'].unsqueeze(2)` and the mean over the tokens.
- `vectors` has the shape `(1, hidden)`.

**Do.** In the Debug Console, run `vectorstore.similarity_search_with_score(questions[700]['question'], k=2)`. The result is chunk 364, then chunk 362. The distances are about 16.3 and about 22.0 [real run]. Disable the breakpoints in `embeddings.py` first. If you do not, the program stops again.

**Learn.** The query path uses mean pooling. This is the opposite of the document path. A lower distance means a closer match.

### Stop 11: The verdict

**Breakpoint.** `build_index.py:117`

**Look at.** `res` holds the result. `n` is 400. `window_recall@2` is 0.785 and `window_recall@5` is 0.88. The centre-chunk recall is 0.34 and 0.465 [real run].

**Learn.** The assertion at line 120 is only a floor.

## 7. When a breakpoint does not hit

- The storage of Chroma and the tokenizer run native code. You can step into their Python wrappers. You cannot step into the native part.
- Do not press F11 into torch at line 55. Press F10.
- The build takes about 32 s without the debugger. Use the conditions above. Then the program does not stop 24 times.

## 8. Problems and fixes

### The error `ModuleNotFoundError` for `langchain_chroma`

**Cause.** The debugger uses a different interpreter. The terminal where you activated `.venv` has no effect. The debugger starts its own process with the interpreter path from VS Code.

A relative path in "Enter interpreter path" can fail without a message. Then VS Code keeps the old interpreter. Here, the global selection was the trainer venv `.venv-temp`.

**Fix.**

1. Look at the status bar. Make sure that it shows `3.13.14 ('.venv': venv)`.
2. Pin `python` in the launch entry with the full path (section 5).
3. At the error, run `import sys; print(sys.executable, sys.prefix)` in the Debug Console.
4. If `sys.prefix` is not the `.venv` of the server, the interpreter is wrong.
5. If `sys.executable` shows a path with `cpython-3.13.14`, VS Code followed the link to the uv base interpreter. This is a rare case. The pinned `python` key is also the fix.

The file `.venv/bin/python` is a link to the uv base interpreter. This is normal for a uv venv.

### The program ends at once with the message "is not empty. Use --rebuild"

**Cause.** The run has no `--out-dir`. The script then uses your real index, which exists. Line 85 calls `sys.exit`, and the debugger ends.

**Fix.** Start the run with the entry "Debug build_index (server venv)". It has `--out-dir` and `--rebuild`.

## 9. Recap and questions

The flow of the program has nine parts:

1. The guard runs.
2. The code loads and cleans the chunks.
3. The code builds the embedder.
4. The code embeds 753 chunks in batches with the CLS vector.
5. The code stores the vectors.
6. The code opens the index again from disk.
7. The code makes a sample of the questions.
8. The code embeds each query with mean pooling.
9. The code counts the window hits.

Answer these questions from what you saw in the debugger:

1. The vectors are on disk. Why does the reopened index still need an embedder?
2. Documents use the CLS vector and queries use mean pooling. Where did you see each one? Why must they match the stored data?
3. Which distance setting did you find on the collection? Does a lower number mean closer?
4. Which questions does the check never sample? How can you change the sample?

Write down what you see in the debugger. Give your notes to Claude. Claude then corrects each difference from the expected values.
