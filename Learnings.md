# Learnings: changes in approach and the reasons for them

| Item | Data |
|---|---|
| Title | Learnings from the search agent project |
| Writing style | ASD-STE100 (Simplified Technical English) |
| Project | Train a small model to answer questions with a search tool |
| Scope | Changes in approach and changes in tests |
| Not in scope | Package problems. See `Dependency_hell.md`. |
| Last update | 2026-10-07 |

## 1. How to read this file

This file records the problems that we found and the decisions that we made. Each entry has five parts: Problem, Cause, Solution, Rule and Where.

Each paragraph has one topic. Each sentence is short. Each technical name in section 2 stands for one thing.

A CAUTION note shows a risk of damage to the results. No note in this file shows a risk of injury.

## 2. Technical names

Use the same name for the same thing every time.

| Name | What it is | Do not use |
|---|---|---|
| chunk | A piece of the source document, about 1000 characters long | part, segment |
| window | Three chunks in a row. The question generator reads one window at a time. | span |
| chunk id | The number of the middle chunk of a window. It is an old position (see below). | chunk number |
| prompt | The messages that start a trajectory: one system message and one user message | question text |
| turn | One answer from the model, and the tool message after it | step |
| context | All the tokens that the model reads before it chooses the next token | history |
| trajectory | One full run of the model for one question, with all turns | episode, conversation |
| rollout function | The function that makes the trajectories for the trainer | generation loop |
| token id | The whole number that stands for one piece of text | token number |
| sampled token | A token that the model chose | generated token |
| token buffer | The list of token ids that the model sees, in order | context list |
| tool message | The message that holds the result of a tool call | tool response, tool output |
| tool delta | The token ids that the chat template adds for a tool message | suffix, bridge |
| chat template | The set of rules that turns messages into text for the model | prompt template |
| environment mask | A list of 1 and 0 for each token. 1 means a sampled token. 0 means any other token. The code name is `env_mask`. | loss mask |
| smoke test | A small, fast test of the full path before the real build | quick test |
| check | One PASS or FAIL line in the smoke test | assertion line |
| run | One start of a script, from start to end | execution |
| train set | The rows that the model learns from | training split |
| validation set | The rows that we use to test the result | test set, val |
| validation block | A group of chunk ids that we hold out for the validation set | fold |
| purge distance | The number of chunk ids around a validation block. We remove the train rows in this distance. | buffer, margin |
| purge | To remove a row from the train set | drop, filter |
| pinned date | The fixed date that we give to the chat template | frozen date |
| vector | A list of numbers that stands for one text | embedding |
| embedder | The object that turns text into vectors. Its class is `CustomHuggingFaceEmbeddings`. | encoder |
| index | The Chroma files on disk. They hold the vectors and the chunk text. | database |
| top k | The k results that are nearest to the question | first k |
| old position | The number of a chunk before the merge. `chunk_id` in `questions.json` is an old position. | original id |
| merge | To glue a short chunk onto the chunk after it | join, combine |
| tiny chunk | A chunk with fewer than 50 characters | fragment |
| minimum chunk size | The value of `--min-chars`. A chunk with fewer characters is merged. | threshold |
| retrieval test | A test that asks the index real questions and counts the hits. The code name is `retrieval_check`. | index check |
| window recall | The share of questions for which the top k results hold a chunk of the question window | hit rate |
| centre-chunk recall | The share of questions for which the top k results hold the middle chunk of the window | exact recall |
| answer coverage | The share of questions for which a top k result holds at least 60% of the answer key words | answer hit rate |
| resources server | The Gym server that holds the tool and the reward. In this project it is `app.py`. | environment server |
| head server | The Gym server on port 11000. It knows the port of each other server. | main server, registry |
| server venv | The Python environment of the resources server: `NeMo_Gym/resources_servers/mission_search/.venv` | gym env |
| trainer venv | The Python environment of the trainer: `trainer/temp/.venv-temp` | training env |
| agent | The code that runs the loop: generate, call the tool, repeat. In this project it is the rollout function. | agent server |
| reward | The number from 0 to 1 that `verify` gives to one trajectory | score, label |
| expected answer | The text in the `answer` field of a row. An 8B model wrote it. | gold answer, label |
| final answer | The text of the last `message` item in a trajectory, without the reasoning | last reply |
| token F1 | A number from 0 to 1 for the words that two texts share | F-score |
| containment | The case where the expected answer is a run of whole words inside the final answer | substring match |
| length cap | The largest number of words in a final answer that gets 1.0 by containment | word limit |

## 3. Learnings

### L1. Run a smoke test before the full build

**Problem.** We did not know if Unsloth, TRL and vLLM can work together for a multi-turn rollout. A wrong guess was expensive.

**Cause.** The three programs have separate code and separate versions. Only a real run shows if they work together.

**Solution.** We wrote a small test before we built anything else. The test uses the model `unsloth/Llama-3.2-1B-Instruct` with 4-bit weights, a fake tool that repeats the text, and 2 train steps. The pass criteria were:

- (a) The trainer file that Unsloth makes contains the `rollout_func` branch.
- (b) The trainer starts only one vLLM engine.
- (c) The rollout uses the live LoRA adapter.
- (d) The output order matches the groups, and the environment mask works.

We also wrote a fallback: plain TRL with PEFT. We did not need it.

**Result.** The final run passes 19 of 19 checks. A package upgrade fixed (b). See `Dependency_hell.md`.

**Rule.**

- Test the riskiest part first.
- Use a small model and few steps.
- Write the pass criteria before you run the test.

**Where.** `trainer/spike_rollout_func.py`. The log of the final run is `trainer/spike_prompting.log`.

### L2. Make each test able to fail

**Problem.** Some checks passed when they proved nothing. The adapter check only showed that the code attached an adapter. It did not show that the adapter changed the output.

The length check compared the logged length to a cap of 128 tokens. The cap is 2 turns of 64 tokens, so the check always passed. Each batch also held only one unique prompt, so the group check had no way to fail.

**Cause.** The LoRA B matrices start at zero. At step 0 the adapter changes nothing, and the outputs equal the base model outputs. The checks did not use a control.

**Solution.** We replaced the weak checks with these checks:

- c1: The sum of the absolute values of the LoRA B matrices is zero before the run. It is greater than zero after the run.
- c2: We put random noise in the LoRA B matrices. The greedy output must change. Two base-model calls must give the same output.
- c3: We change the noise and load the adapter again. The output must change again, and the request id must rise.
- d1: For each trajectory, the completion, the mask and the log-probabilities have the same length. The log-probabilities of environment tokens are zero.
- d2: Each trajectory that continued has environment tokens.
- d4: The length that TRL logs equals the sum of the mask. This length is less than the raw length.
- d5: Each batch holds 2 unique prompts, with 4 completions for each prompt.

We also made two errors in the tests. One check expected 1.0 sampled tokens, but the real value was about 87. Another check had the wrong sign, so a correct result showed FAIL.

We split the check that failed into four small checks (d4a to d4d). Then we corrected each one.

**Rule.**

- Add a control to each test.
- Name each result so that True means good.
- Compare each constant with real data.
- Split a check that fails into small checks.
- Ask: "Can this test fail?"

**Where.** `trainer/spike_rollout_func.py`, in the code after `trainer.train()`.

### L3. Return one trajectory for each prompt that TRL sends

**Problem.** The run stopped with this error: "The expanded size of the tensor (4) must match the existing size (16)". The error came from the reward calculation.

**Cause.** TRL sends each prompt `num_generations` times. Here the value is 4. Our function made 4 trajectories for each prompt that TRL sent. So it made 16 trajectories for 4 prompts.

**Solution.** The function now returns exactly one trajectory for each prompt that TRL sends. The order is the same as the order of the prompts. TRL groups the rewards by position, so the order is important.

We first thought that the 4-bit weights needed a merge to 16-bit. The numbers in the error showed that this idea was wrong.

**Rule.** Read the numbers in an error message before you change the model.

**Where.** `rollout_func` in `trainer/spike_rollout_func.py`.

### L4. Keep the token ids and never re-encode them

**Problem.** The prefix check failed for 1 trajectory of 16. In that trajectory, the second turn used a context that was different from the context in the trainer.

**Cause.** The first turn stopped at the length limit. Its last sampled token was a space. The Llama chat template removes a space at the end of a message. So the new render did not hold that token.

Our code cut the new tokens from the render at the old length. The cut was one token too late. It dropped the end-of-turn token. The trainer then saw a context that vLLM did not see.

The general cause is that decode and then encode does not always give the same tokens. The same text can have more than one tokenization.

**Solution.** We changed the rollout to the TITO method (token-in, token-out):

1. Keep a token buffer for each trajectory.
2. Send the token buffer to vLLM as token ids.
3. Add each sampled token to the token buffer as the model made it.
4. Decode the text only to find the tool call. Do not use the decoded text again.
5. Add the tool message as a tool delta. Render a dummy conversation with the tool message and without it. Subtract the two token lists.
6. Make sure that the render with the tool message starts with the render without it. Do this at startup and in each call.

**Checks.**

- vLLM reports the same prompt token ids that we sent.
- The last prompt plus the last sampled tokens equals the prompt plus the completion that we return.
- A counter shows how often the old method gives different tokens. In the final run the count is 0 of 15 trajectories. The sample is small.

**Choices that we did not make.**

- We did not use the `return_token_ids` flag. It belongs to the vLLM server API. The offline engine already gives the token ids.
- We did not make one train sample for each model call. The TRL rollout function needs one completion for each prompt.
- We did not write a hand-made code module for each model that turns messages into tokens. The Hugging Face post says that the prefix check is enough.

An earlier run passed the old prefix check. We do not know why.

> CAUTION: Do not decode the sampled tokens and then encode the text again. The trainer can learn from tokens that the model did not sample.

**Rule.**

- Keep the sampled token ids as the source of truth.
- Use the decoded text only to route the tool call.
- Use a prefix check on the chat template for each new model.

**Where.** `tool_delta_ids` and `rollout_func` in `trainer/spike_rollout_func.py`.

**Source.** The Hugging Face post "TITO: Token-In, Token-Out in Agentic RL".

### L5. End the trajectory at the length limit

**Problem.** A turn can stop because it reached the maximum number of tokens. Then the turn has no end-of-turn token.

**Cause.** The model did not choose to stop. The structure of the turn is not complete.

**Solution.** We first planned to add the end-of-turn token by hand and then continue. We did not do this. The model did not sample that token. The Hugging Face post says that a broken structure is safe to leave.

Now the trajectory ends when the finish reason is `length`. The code adds no tokens and does not call the tool.

This rule changes the tests. Check d2 now counts only the trajectories that continued. With a limit of 64 tokens, many trajectories end early. In the first run with this rule, 10 of 16 trajectories ended at the limit. In the final run, 2 of 16 ended at the limit.

**Rule.** Use a larger token limit in the real run.

**Where.** `rollout_func` in `trainer/spike_rollout_func.py`.

### L6. Check that the environment mask reaches the loss

**Problem.** TRL logs a length that equals the sum of the mask. But the log is only a report. It does not show that the loss ignores the environment tokens.

**Cause.** The trainer calculates log-probabilities for all tokens. The model must read the tool message as context. The mask does not change the log-probabilities. It sets the loss weight of the masked tokens to zero.

**Solution.** We wrapped the function `align_completion_tool_mask` in the trainer file that Unsloth makes. The wrapper records the token count before the mask and after the mask in each real train step.

**Result.** In the final run the count before the mask is 3,495 tokens. The count after the mask is 3,134 tokens. These numbers equal the raw token count and the sampled token count from the rollout function.

The check counts tokens. It does not check the positions of the tokens.

**Rule.** Check the loss path. Do not trust only the log.

**Where.** The d6 check in `trainer/spike_rollout_func.py`.

### L7. Convert the questions to Gym rows

**Problem.** AutoDidact keeps the questions in one file in its own format. NeMo Gym needs one task in each row, in the Responses API format.

**Solution.** We wrote a script that makes the rows. These are the decisions:

- Each row has `responses_create_params` with a system message, a user message, one flat tool, and `parallel_tool_calls` set to false.
- Each row has the top-level fields `answer`, `difficulty` and `chunk_id`.
- We copied the prompt code from `rl_helpers.py`. We did not import it, because it loads torch, trl and the Chroma index.
- The system message has the role `system`. Llama has no `developer` role.
- The tool has `strict` set to false. Gym rejects a row that has no `strict` key. Our tool schema does not follow the rules of strict mode.
- The tool definition stays in the user prompt. The rollout reads it from there.
- We changed 26 malformed difficulty labels to easy, medium or hard.
- The script asserts its own guarantees.
- The Gym class `BaseRunRequest` accepts all 1,393 rows.

**Rule.** Check the data with the schema of the program that reads it.

**Where.** `NeMo_Gym/resources_servers/mission_search/scripts/convert_questions.py`. The output is in `NeMo_Gym/resources_servers/mission_search/data/`.

### L8. Pin the date in the chat template

**Problem.** The system prompt of each row held the header "Cutting Knowledge Date / Today Date" two times. The two dates can be different.

**Cause.** The Llama chat template adds this header to each system block. It uses the real date. The AutoDidact system prompt also had its own copy of the header.

The real date changes the prompt tokens. A change of the date changes 1 or 2 of about 295 tokens. A run on one day and a run on another day do not see the same input.

**Solution.**

- We removed our copy of the header. The system message keeps only the two instruction sentences.
- We made `trainer/prompting.py`. It gives the chat template the fixed date `26 Jul 2024`.
- The wrapper rejects any other date.

The average prompt length went from 291 tokens to 271 tokens. The data files do not hold the template date. A new date does not need new data files.

> CAUTION: Do not call `tokenizer.apply_chat_template` directly. The prompt then holds the real date, and runs on different days differ.

The pin covers only the prompts that we render. The Gym model server renders its own prompts with the real date.

**Rule.** Use `apply_chat_template` from `trainer/prompting.py` for each render call.

**Where.** `trainer/prompting.py` and `get_system_prompt` in `convert_questions.py`.

### L9. Remove the validation leak with a purged block split

**Problem.** The AutoDidact split puts rows in the validation set at random. In that split, 21 of 144 validation questions also appear in the train set. Also, 128 of 144 validation rows share a chunk with a train row.

**Cause.** The generator read one window of three chunks at a time. It made 2 questions for each window. The chunk id is the number of the middle chunk.

Two questions share source text when their chunk ids differ by 2 or less. The same question text appears more than once in 110 cases. In 9 of these cases the ids are more than 2 apart, up to 406.

A trained model scores too high on such a validation set. A comparison of the base model with the trained model is then not fair.

**Solution.** We made a purged block split. This is how it works:

1. Cut the chunk ids into cells of 13 ids.
2. Sort the cells by a SHA-256 hash with the seed 42. Do not use a random generator.
3. Take 6 cells that are not next to each other. These are the validation blocks.
4. Remove from the train set each row within 2 ids of a validation row. This is the purge distance.
5. Remove from the train set each row with the same question text as a validation row.

The result is 1,240 train rows and 148 validation rows. The purge distance removed 46 rows. The text purge removed 0 rows. Each validation row has a `block` field, so we can report the score for each block.

**Checks.**

- The script asserts that the minimum id distance is 3 or more. It also asserts that no question text is in both sets.
- The output files have the same bytes on Python 3.12 and Python 3.13.
- The option `--split legacy` gives the old split. A test showed the same rows in the same order as the old files, except for the removed header lines.
- A test with fake data showed that the text purge removes a copy of a validation question.

**What stays open.** Neighbor chunks are still related. The agent searches the whole document when we test it. So the split reduces the leak, but it does not remove the leak. We hold out no second set for a final test. For a reward of 0 or 1, 148 rows give an uncertainty of about 4 points.

**Choices that we did not make.** We did not add an embedding check or an LLM judge. We cannot prove the limit values for such checks, and our questions are short. We did not use combinatorial purged cross-validation. A train run costs too much for it.

**Rule.**

- Check the validation set for overlap with the train set.
- Choose the blocks with a hash, not with a random generator.
- Make the script fail if a guarantee is false.

**Where.** `split_purged` and `purged_report` in `convert_questions.py`.

### L10. Rules for our work

**Problem.** Some advice and some claims that we found were not correct for our case. An example is the `return_token_ids` flag in L4.

**Solution.** We use these rules:

- Read the code that raised an error before you choose a fix.
- Check a claim in the source code or in a test before you use it.
- Write the reason for a decision in this file when you make it.
- Say in the chat when the work is different from `Plan.md`.
- Say in the chat the full path of each new file, and the reason for it.
- Measure a problem before you fix it. Compare the options on the same questions.

**Where.** `rules.md` and `Plan.md`.

### L11. Test the index with real questions

**Problem.** After a build, the index can have the right row count and still search badly. A wrong model or a wrong pooling gives vectors that do not match. A row count cannot show this.

**Cause.** A test that counts rows has no way to fail on bad search. We also needed a way to tell if a change to the chunks helps.

**Solution.** The build script runs a retrieval test after each build:

1. Open the index again from disk. A new Chroma object must see all the rows. The index needs the embedder only for queries.
2. Ask the index 400 real questions. The test takes every third question.
3. Count three hit rates for the top 2 and the top 5: window recall, centre-chunk recall and answer coverage.
4. Assert that the window recall at top 5 is above 30%. Chance is only a few percent.

The three hit rates look at different things. Window recall counts chunk numbers: the search found the right area. Centre-chunk recall is strict: the search found the middle chunk. Answer coverage reads the text: a result holds the key words of the answer.

The script has two safety rules. It uses fixed ids `chunk-<index>`. It refuses a folder that is not empty, unless you add `--rebuild`. Without this rule, Chroma adds duplicate rows. The embedder runs on the CPU, so the GPU stays free for the train run.

**Result.** The values below are for the top 2 and the top 5, before the merge (L12):

- Window recall: 78.5% and 88.0%.
- Centre-chunk recall: 34.0% and 46.5%.
- Answer coverage: 73.0% and 82.0%.

**Limits.**

- The test takes every third question and stops at 400. So it never tests the last 16% of the questions.
- The floor of 30% finds only a broken index.
- Answer coverage is rough, because the answers are paraphrases.

**Rule.**

- Test the index after each build.
- Use more than one hit rate. Read the chunk numbers and the text.
- Keep the numbers from before a change. Then you can see if the change helps.

**Where.** `retrieval_check` in `NeMo_Gym/resources_servers/mission_search/scripts/build_index.py`.

### L12. Merge the short chunks

**Problem.** 206 of 753 chunks have fewer than 50 characters. They fill 12.9% of the top-2 results. The agent gets almost no text from such a result.

**Cause.** The loader made one element for each title, page ID and caption. The splitter cut each element alone. It never joined a short element to the element next to it. Of the 206 tiny chunks, 124 are titles, 46 are page or document IDs and 33 are captions.

**Solution.** We merge the short chunks:

1. Go through the chunks in order.
2. If the current chunk has fewer than 100 characters, add the next chunk to it. Put a newline between them.
3. Repeat until the chunk has 100 characters or more. Never go above 1500 characters.
4. Do not touch a chunk that already has 100 characters or more.

The result is 510 chunks, where there were 753. There are 186 merged groups. 324 chunks stay as they were. The shortest chunk has 101 characters and the longest has 1089, so the limit of 1500 never applies. A title joins the text of its section. The three pieces on the title page (9, 28 and 171 characters) become one chunk.

`questions.json` and the purged split use the old positions. We did not change them. Each merged chunk keeps `chunk_start`, `chunk_end` and `n_merged`, and the retrieval test uses these values. The option `--min-chars 0` gives the old 753 chunks.

**How we chose.** We compared five variants on all 1,434 questions. The search in this test was exact and ran in memory. It matched the real index within 0.5 points on the same 400 questions.

| Variant | Chunks | Answer coverage at top 2 and top 5 | Window recall at top 2 |
|---|---|---|---|
| Baseline | 753 | 72.0 and 83.3 | 76.9 |
| Drop the chunks under 50 | 547 | 74.3 and 84.0 | 76.8 |
| Merge to 100 | 510 | 75.0 and 85.3 | 77.0 |
| Merge to 200 | 459 | 74.1 and 85.1 | 75.2 |
| Merge to 300 | 389 | 73.2 and 84.7 | 72.2 |

Window recall stays the same for 100. It falls for 200 and 300. So we chose 100.

**Result in the real index.** On the same 400 questions, the values for the top 2 and the top 5 changed in this way:

- Answer coverage: 73.0% to 75.0%, and 82.0% to 83.8%.
- Window recall: 78.5% to 77.5%, and 88.0% to 87.5%.
- Centre-chunk recall: 34.0% to 44.2%, and 46.5% to 57.2%.

**Choices that we did not make.**

- We did not drop the tiny chunks. 30 questions (2.1%) have an answer only in a tiny chunk.
- We did not use 200 or 300. The window recall falls and the answer coverage does not improve.
- We did not use parent-child retrieval. It needs a second store and it changes what the agent reads. We can use it later if the corpus grows.

**What stays open.** The gain is small, about 2 points of answer coverage. The agent can search again, so the real gain can be smaller. 15.3% of the chunks are still shorter than 200 characters. The rise of the centre-chunk recall is partly mechanical, because a merged chunk covers more positions.

**Rule.**

- Measure the problem before you fix it.
- Compare the options on the same questions.
- Keep the old chunk numbers when you change the chunks.

**Where.** `merge_small_chunks` in `NeMo_Gym/resources_servers/mission_search/scripts/build_index.py`.

### L13. Score the final answer against the expected answer

**Problem.** An 8B model wrote the expected answers. The words differ from the words of a correct answer from our model. An exact match gives 0 to many correct answers. A plain containment test is easy to cheat.

**Cause.** The expected answers are paraphrases. Nobody checked them. A model that pastes a whole chunk into its answer always contains the expected text.

**Solution.** The function `score_answer` in `app.py` gives the reward in four steps:

1. Take the text of the last `message` item. If there is none, the reward is 0.0.
2. Remove the `<reasoning>` blocks. Then lowercase the text, delete the punctuation and delete the words a, an and the. Delete the punctuation without a space, so `18:24` and `1824` are the same.
3. Give 1.0 if the two texts are equal. Give 1.0 if the expected answer is a run of whole words in the final answer and the final answer is short.
4. Give the token F1 in all other cases. The token F1 is 0.0 if the two texts share no word.

A final answer is short if it has at most 3 x (expected words) + 10 words. For the expected answer `1824`, the limit is 13 words. For an expected answer of 6 words, the limit is 28 words. A final answer above the limit still gets a partial reward from the token F1.

`verify` does not read the index and does not call a model. It is a pure function of the request body, so the reverify mode is STATELESS. The same rollout always gets the same reward.

**Tests.** `tests/test_app.py` holds the checks:

- The fixture has six cases: two full reward, two zero reward and two malformed.
- A final answer with the answer only inside the reasoning gets 0.0.
- A final answer with the answer and 40 more words gets less than 1.0.
- The paraphrase "It was reinforced with Mylar tape" gets about 0.62 against "The Aclar will be reinforced with Mylar tape."

**Choices that we did not make.**

- We did not use a judge model. It breaks the STATELESS mode and costs GPU memory.
- We did not use an exact match alone. It gives 0 to correct paraphrases.
- We did not add a bonus for a tool call. A model calls the tool, ignores the result and still gets the bonus. Decide after the first rollouts show how often the model calls the tool. Cap such a bonus so that the total stays at or below 1.0.

**What stays open.**

- The numbers 3 and 10 in the length cap are a first guess. We did not measure them.
- The expected answers are not checked. Some are wrong, or the top 2 results do not hold the answer.
- The token F1 gives a small reward to a wrong answer that shares words with the expected answer.
- After the first rollouts, find the questions that no attempt scores on. They show bad expected answers.

**Rule.**

- Make the reward a pure function of the request.
- Test that the reward does not give 1.0 to a long answer that only contains the expected text.
- Keep the reward from 0 to 1.

**Where.** `score_answer`, `normalize` and `final_message_text` in `NeMo_Gym/resources_servers/mission_search/app.py`.

### L14. Keep the resources server and the trainer in separate environments

**Problem.** The Gym documents show an agent server and a model server next to the resources server. Our trainer has its own vLLM engine. We had to decide where the agent runs and which Python environment runs each part. `ng_run` also installs packages by default.

**Cause.** There are three reasons:

- The two stacks conflict. The server uses CPU torch and transformers 5.18. The trainer uses CUDA torch and transformers 5.17.
- The `simple_agent` needs a model server over HTTP. That means a second vLLM engine, and the GPU has no room for it. It also works with text, so it breaks the token buffer rule (L4).
- `ng_run` runs `uv pip install` into the server venv at each start, unless `skip_venv_if_present` is true. The code then installs `nemo-gym` from PyPI. We read this in `setup_command.py`. We did not run it.

**Solution.**

1. Run the resources server with `ng_run` in the server venv. Add `+skip_venv_if_present=true`. Then `ng_run` only activates the venv.
2. Run the trainer in the trainer venv, in a second terminal. The two environments share no package. They talk over HTTP.
3. The rollout function is the agent. It generates on the trainer engine. It sends each tool call to `POST /search_corpus` and the final answer to `POST /verify`.
4. The trainer finds the port of the server with `GET :11000/global_config_dict_yaml`. The reply is one JSON string that holds YAML. Decode it with `json.loads` first.

**What runs after the start.**

| Port | What | Notes |
|---|---|---|
| 11000 | Head server | Fixed port. It answers `status ok` at `/`. |
| 10001 to 20000 | Our resources server | A new random port at each start (18432 in the first run) |
| 8265 | Ray dashboard | Ray starts with `ng_run`. Our server has `ray_enabled = False` and does not use it. |
| 39107 | Ray cluster address | Not an HTTP page |

The first start did not install anything. The embedder loaded one time at the start (199 weights). A search call returned two results, and the verify calls gave the expected rewards.

**What stays open.** Ray runs but does nothing. If RAM or `/dev/shm` is short during training, run `ray stop`. We did not measure this.

**Rule.**

- Keep one Python environment for each stack. Never install the server packages into the trainer venv.
- Start `ng_run` with `+skip_venv_if_present=true` after the server venv works.
- Find the server port at run time. Never write it in the code.

**Where.** `NeMo_Gym/resources_servers/mission_search/configs/resources_only.yaml`. The agent code goes in `trainer/rollout.py` (step 5).

## 4. Procedures

### 4.1 Make the data files

1. Go to the repository root.
2. Run `python NeMo_Gym/resources_servers/mission_search/scripts/convert_questions.py`.
3. Find the line that starts with `[guarantee]`.
4. Make sure that the minimum distance is 3 or more.
5. Make sure that the number of questions in both sets is 0.

To use the old split, add `--split legacy`. Use the environment with the `datasets` package.

### 4.2 Run the smoke test

> CAUTION: Do not start the test while another program uses the GPU. The test needs about 10 GB of GPU memory.

1. Go to the `trainer` folder.
2. Activate the environment `temp/.venv-temp`.
3. Run `python spike_rollout_func.py`.
4. Find the line that starts with `GATE 1`.
5. Make sure that the line shows PASS.
6. If a check shows FAIL, read the `[spike]` lines for that check.

### 4.3 Build the index

> CAUTION: Do not use `--rebuild` with a folder that you want to keep. The script deletes the folder `chroma_index` and builds it again.

1. Go to the repository root.
2. Run `NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/build_index.py --rebuild`.
3. Find the line that starts with `[merge]`. Make sure that it shows 510 chunks and `under 100: 0`.
4. Find the lines that start with `[check]`. Make sure that the window recall at top 5 is above 30%.

The build takes about 20 s. To keep all 753 chunks, add `--min-chars 0`. To test without a change to the real index, add `--out-dir` with a new folder.

### 4.4 Start the server and test it

> CAUTION: Do not start `ng_run` without `+skip_venv_if_present=true`. It installs packages into the server venv.

1. Go to the folder `NeMo_Gym`.
2. Run `resources_servers/mission_search/.venv/bin/ng_run "+config_paths=[resources_servers/mission_search/configs/resources_only.yaml]" +skip_venv_if_present=true`.
3. Wait for the line `All 1 / 1 servers ready`.
4. Open a second terminal. The first terminal shows the log, and it does not take input.
5. Run `curl -s localhost:11000/global_config_dict_yaml | python3 -c 'import json,sys; print(json.load(sys.stdin))'`.
6. Find the `port` in the block `mission_search_resources_server`.
7. Send `{"query": "gimbal angle deviations"}` to `/search_corpus`. Make sure that the reply has `Result 1:` and `Result 2:`.
8. Send a row to `/verify` with a correct final answer. Make sure that the reward is 1.0. Use a wrong final answer. Make sure that the reward is 0.0.
9. Press Ctrl+C in the first terminal. Use `ps` to make sure that no `app.py` process stays.

To run the unit tests, go to `NeMo_Gym`. Run `resources_servers/mission_search/.venv/bin/python -m pytest resources_servers/mission_search/tests`. Make sure that all 14 tests pass.

## 5. Open items

- The Llama chat template puts quotation marks around the content of a tool message. We probably do not want this for search results. Decide how to add the search results, and run the prefix check again.
- A reasoning model can remove or shorten old turns. TITO does not work after such a change. Keep the text before the last change as the prompt, and train only on the tokens after it.
- Hold out a second set for a final test when we start to tune against the validation set.
- Delete the counter for the old render method after we measure it on a larger sample.
- The retrieval test never tests the last 16% of the questions. Use a larger step, so that the sample covers the whole file.
- `Debug_checks.md` and the pictures in `Schema_learning` show the index before the merge (753 chunks and the old line numbers). Update them before the next debug session.
- Check the expected answers. After the first rollouts, find the questions that no attempt scores on. Read them with their chunks.
- Measure the length cap (3 x expected words + 10). Look at the final answers that get 1.0 by containment.
- Decide on a bonus for a tool call after the first rollouts show how often the model calls the tool.
