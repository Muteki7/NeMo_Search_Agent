# Learnings: changes in approach and the reasons for them

| Item | Data |
|---|---|
| Title | Learnings from the search agent project |
| Writing style | ASD-STE100 (Simplified Technical English) |
| Project | Train a small model to answer questions with a search tool |
| Scope | Changes in approach and changes in tests |
| Not in scope | Package problems. See `Dependency_hell.md`. |
| Last update | 2026-10-27 |

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
| retriever | The code that finds the chunks for a query. Today it is `CorpusSearcher` with NoInstruct and Chroma. | search engine |
| NoInstruct | The small embedder of the server today (33M parameters) | the model |
| BGE-M3 | A larger embedder (568M parameters). One pass gives three scores: dense, sparse and ColBERT. | M3 model |
| dense score | The score from one vector for the whole text | semantic score |
| sparse score | The score from the weight of each token that the query and the chunk share | lexical score |
| ColBERT score | The score from the best match of each query token in the chunk, averaged | multi-vector score |
| lexical hit | One search result holds at least 60% of the content words of the expected answer | word hit |
| provenance hit | A retrieved chunk overlaps the window that the question came from | source hit |
| source support | The share of the content words of the expected answer that the source window holds | grounding |
| training run | One training from the first step to the last step, with its own folder in `trainer/runs` | run (alone) |
| checkpoint | The LoRA weights that the trainer saves at one step | snapshot |
| group | The 8 attempts at one prompt | batch |
| advantage | The number that shows how much better one attempt is than the mean of its group | bonus |
| answer-only reward | The reward from the final answer alone, without the retrieval part | base reward |
| combined reward | The reward that the trainer uses: the answer part plus the retrieval part | total reward |
| retrieved | The share of questions with a lexical hit | found |
| contains | The share of final answers that hold the expected answer, whatever their length | match rate |
| paired change | The mean over the same questions of (reward of the model minus reward of the base model) | difference |

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
4. Give the token F1 in all other cases. The token F1 counts only content words: it drops small words such as of, to and was. It is 0.0 if the two texts share no content word.

A final answer is short if it has at most 5 x (expected words) + 15 words. For the expected answer `1824`, the limit is 20 words. For an expected answer of 6 words, the limit is 45 words. The first limit was 3 x + 10. In a dump of 40 trajectories, two correct but wordy answers got only 0.29 and 0.35 with that limit. With 5 x + 15, the mean reward moved from 0.200 to 0.234, the same as with no limit at all. Do not change the limit during a run: the base model and the trained model need the same value. A final answer above the limit still gets a partial reward from the token F1.

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

- The numbers 5 and 15 in the length cap come from 40 trajectories of one model. Check them again on a larger dump.
- The expected answers are not checked. Some are wrong, or the top 2 results do not hold the answer.
- The token F1 gives a small reward to a wrong answer that shares content words with the expected answer. In the first smoke run (1B model), the F1 on all words gave the highest rewards to non-answers. We changed it to content words. A bad tool call whose query repeats words of the question still gets a little reward.
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

### L15. Read the reward the way the optimizer reads it

**Problem.** The reward went up in the first runs, but we did not know which signal the trainer used. The 1B smoke run gave its highest rewards to text that was not an answer.

**Cause.** GRPO does not use the reward itself. It uses the difference between each attempt and the mean of its group of 8. TRL then divides this difference by the standard deviation of the group.

**What we measured** in the 200 groups of run 1:

- 31 groups had 8 equal rewards. They gave no signal.
- 83 groups had an exact answer and a non-exact answer. They gave a clean signal.
- 86 groups had no exact answer. Their signal came only from small F1 differences.
- In 60 groups the standard deviation was below 0.1. The division gave an advantage of up to +2.3 to a difference of 0.02.

**Solution.** We took three steps:

1. The F1 counts content words only. A text of small words gets 0.0.
2. Run 2 sets `scale_rewards="none"`. TRL does not divide the difference.
3. We compare runs with the answer-only reward on fixed questions (L17), not with the training reward.

**Rule.**

- Write down the order of the outcomes before you choose numbers.
- Count the groups with no signal, with a clean signal and with a noise signal.
- Add a part to the reward only where the main signal is weak.

**Where.** `token_f1` in `NeMo_Gym/resources_servers/mission_search/app.py`. `HYPERPARAMETERS` in `trainer/train_grpo.py`.

### L16. Add a retrieval part to the reward and measure its share first

**Problem.** The reward gave no credit for a good search. A good search with a wrong answer scored the same as no search.

**Solution.** Run 2 uses `reward = 0.75 x answer + 0.25 x gate x ret`:

1. `cov_k` is the share of the content words of the expected answer that the result of search k holds.
2. `s_k` is a ramp: 0 below 0.3, 1 at 0.6 and above.
3. `ret` is the largest value of `0.5^(k-1) x s_k` over the searches. Extra searches add nothing after a hit.
4. `gate` is 1 if the trajectory ends with a final message. Spam to the turn cap scores 0.

A correct answer with a hit scores 1.0. A correct answer without a hit scores 0.75. A hit with no usable answer scores at most 0.25.

**Result.** The retrieval part carried 7% of the variance inside the groups. The answer part carried 78%. The covariance carried 16%. The retrieval part changed in 70 of 200 groups, and in 34 of the 112 groups with no exact answer. Retrieval did not improve: `retrieved` was 0.72 for the base model and 0.68 for run 2.

We think the 8 attempts of a group send almost the same query. We did not test this.

**Choices that we did not make.** We gave no negative reward for a bad call. Bad calls were 0% at the end of run 1, so a penalty has no target. A penalty also gives the model a reason to avoid the tool. A penalty for consecutive misses punishes a retry, and a retry is the only way to recover from a bad first search.

**Rule.**

- Measure how much of the variance inside the groups a new part carries. A part under 10% does not steer the model.
- Keep each part small, bounded and gated.
- Test each part against the ways to farm it. Spam, search and no answer, and search and ignore all score low.

**Where.** `retrieval_score` and `combine` in `app.py`. The tests are in `tests/test_app.py`.

### L17. Judge the runs with one fixed yardstick

**Problem.** The training reward comes from sampling at temperature 0.9 on a new question at each step. It does not compare two models.

**Solution.** `eval.py` makes the yardstick:

1. It decodes greedily (temperature 0). The same 148 questions run in the same order with the pinned date.
2. The headline is the answer-only reward. It stays the same after run 2 changed the training reward.
3. The base model runs through the same code, with no adapter.
4. The paired change against the base model comes with its standard error.
5. The metric `contains` does not depend on the length cap.

**Evidence of determinism.** We evaluated the base model in two run folders, six days apart. All 148 trajectories were identical: the queries, the retrieved text and the answers.

**Result.**

| Model | Answer-only reward | Exact |
|---|---|---|
| Base | 0.191 | 0.15 |
| Run 1 at step 200 | 0.461 | 0.36 |
| Run 2 at step 200 | 0.499 | 0.41 |

The paired change of run 2 against run 1 is +0.039 ± 0.020. That is 1.9 standard errors. Each run used one training seed, so this is not proof.

**Limits.**

- One greedy run for each model, on 148 questions. 130 of them are different texts (L18).
- GPU arithmetic changes with the batch size. Keep `--limit` and the chunk size of 32 the same between runs that you compare.
- A new index or a new verifier changes the yardstick. Run the base model again.
- Training is not deterministic. A resume gave weights that differed by 1.4%.

**Rule.** Evaluate with one fixed set of settings. Never judge a run with the number that it trained on.

**Where.** `trainer/eval.py`.

### L18. The score has two factors, and part of the data is weak

**Problem.** The exact rate stops near 0.4 on the validation set.

**Cause.** The score is the product of two rates. 68% of the questions had a lexical hit. For 62% of those, the answer is in the final text. 0.68 x 0.62 is about 0.42. Without a hit, the answer is in the final text in 2% of the cases.

**What the report shows** for the 148 questions:

| Category | Base | Run 1 | Run 2 |
|---|---|---|---|
| Success | 22 | 53 | 61 |
| Answer only inside `<reasoning>` | 32 | 0 | 0 |
| Retrieval miss | 31 | 44 | 40 |
| Right evidence, partial credit or wrong answer | 39 | 39 | 36 |
| False-hit suspect | 10 | 8 | 6 |
| Suspect bad reference | 4 | 4 | 4 |
| No usable final answer | 10 | 0 | 1 |

**Data findings.**

- 18 of the 148 validation rows repeat a question. The windows overlap, so two windows give the same question. 7 of the repeats have different references. In the train set, 110 of 1,240 rows repeat.
- Some references do not answer the question. One asks for the reason of an engine cutoff and gives the time of the cutoff. Both trained models gave a better answer than the reference and lost points.
- Some questions are not self-contained ("the first part of the maneuver").
- OCR swaps digits for letters. 1.3% of the references have such a token. 222 of the 510 chunks have one (`l3` for `13`).
- 25% of the validation questions have no chunk of their window in the top 5, even with the question as the query. Many of them are mission-level questions with answers in several places.

**Rule.**

- A semantic judge inherits the errors of the reference. Clean the references before you build the judge.
- Remove repeated questions before you make a split.

**Where.** `scripts/report.py` writes the report. `scripts/rescore_dump.py` scores a dump again.

### L19. Compare the retrievers on the chunks before you train

**Problem.** Retrieval is the first factor of the ceiling. We did not know whether BGE-M3 finds better chunks than NoInstruct.

**Solution.** `retrieval_bench.py` runs both on the same 510 chunks and the same queries, with exact search on the CPU. BGE-M3 has three scores. We built them from the model files and tested them against the example scores in the model card. The combined score uses the weights of the model card: dense 0.4, sparse 0.2, ColBERT 0.4. It divides the sum by the sum of the weights.

**Result.** The table shows provenance at top 2 and top 5.

| Query set | NoInstruct | BGE-M3 all three |
|---|---|---|
| All 1,434 questions | 0.771 and 0.873 | 0.895 and 0.955 |
| 148 validation questions | 0.649 and 0.750 | 0.824 and 0.932 |
| Queries of the trained model | 0.649 and 0.736 | 0.824 and 0.946 |

**Other results.**

- BGE-M3 at top 2 beats NoInstruct at top 5.
- ColBERT is the best single score. Sparse is better than dense. All three together are about equal to ColBERT alone.
- From top 2 to top 3 the gain is 8 points. From top 3 to top 5 it is 2.7 points.
- The OCR repair does not raise provenance. It raises the word-overlap coverage by about 2 points, because the spelling becomes equal.
- L2 distance, cosine and dot product give the same result with NoInstruct, because the vectors have almost the same length.
- The queries of the trained model find the chunks as well as the question itself.
- On the CPU, one chunk takes 162 ms and one query takes 15 ms. The NoInstruct values are 27 ms and 4 ms.

**Limits.** An LLM wrote each question from its window. This favours scores that match words. The validation set is harder than the mean of all questions.

**Rule.** Measure a new retriever on the chunks with exact search before you change the server. Test your own implementation against the published example scores.

**Where.** `scripts/retrieval_bench.py` and `tests/test_retrieval_bench.py`. The results are in `trainer/runs/retrieval_bench`.

### L20. Make a long run safe to stop

**Problem.** A 200-step run lasts about one hour. A crash at step 130 costs the whole run unless the run saves its state.

**Solution.**

1. `train_grpo.py` makes one folder for each training run. It saves a checkpoint every 50 steps and the `final` weights at the end.
2. `--resume` continues from a checkpoint in the same folder with the saved settings.
3. It stops if a constant or a hyperparameter in the code differs from the saved run.
4. It moves the dump lines of the steps after the checkpoint to a side file. It never deletes them.
5. We tested the resume on a copy of a test run. The learning rate and the step numbers came back.

**Rule.** Test a resume before the long run. Never edit `app.py` or the constants during a run.

**Where.** `main` in `trainer/train_grpo.py`. `prune_dump_after_step` in `trainer/rollout.py`.

### L21. Disk space under WSL2

**Problem.** WSL2 keeps the distribution in one disk file. The file grows and does not shrink. We deleted a copy of the model, and the space did not come back to drive C.

**Facts.**

- Space that we free inside WSL stays in the file. WSL reuses it for new files. A new copy of the same size does not grow the file.
- No setting named `autocompact` exists, as far as we found. The sparse disk option exists (`sparseVhd`). Recent WSL versions marked it as unsafe because of a risk of data corruption. The manual compaction tools do not work on a sparse disk.
- On 2026-10-23 the file was 177 GB and WSL used 159 GB. WSL was version 2.6.3.

**Solution.** We do not use the sparse option. When drive C is short on space, we free files inside WSL, run `wsl --shutdown`, and compact the file offline with `Optimize-VHD` or `diskpart`. We back up first. We did not run the compaction yet.

**Rule.** Delete the large files inside WSL before you compact. Keep model weights out of git. Git ignores `Embedder/`.

**Where.** `.gitignore`.

### L22. Test a loader on data of real size, under a memory limit

**Problem.** The first BGE-M3 retriever ran out of memory when it loaded its cache. One Python process grew to 31 GB, and the system stopped it. WSL stopped three times. The cache file was only 251 MB.

**Facts.**

- The cache was one `.npz` file. NumPy reads an array from an `.npz` file each time the code asks for it by name. The code asked for the whole ColBERT array once for each of the 510 chunks. Each request made a new copy of 249 MB. The copies added up to about 127 GB.
- The cost grows with the square of the number of chunks. A test with 5 chunks did not show the problem.
- A process limit (`ulimit -v`) limits the virtual address space. It does not limit the resident memory. In our tests, the virtual size was 2 to 7 times the resident size.
- A limit of 2 times the resident peak stopped the tests with no real memory problem. A limit of 2 times the virtual peak is almost all of the 31 GB of RAM, so it protects nothing.
- A child process reports `ru_maxrss` with the memory of its parent at the time of the fork. Read `VmHWM` from `/proc/self/status` in the child.
- Each retriever loaded its own copy of the model (2.3 GB). Six loads in the tests gave a resident peak of 4.5 GB. One shared model gave 2.7 GB.

**Solution.**

- Store each array of the cache in its own `.npy` file. Open the files with `mmap_mode="r"`. The six arrays are: dense, ColBERT tokens with their offsets, and the sparse weights in CSR form (pointers, indices, data).
- Write a manifest. It records the shapes, the types, the model revision, the chunk hash, and the chunking settings. The name of the cache folder is a hash of the manifest.
- Read the size from the manifest before the load. Refuse a load that is larger than 2 GiB.
- Compute the ColBERT scores in blocks. The temporary matrix stays under 256 MiB.
- Let the retriever accept a loaded model. The tests share one model.
- Run one test with arrays of real size in a child process that has a 3 GB limit. The old loader fails this test.

**Rule.** Test each loader with data of real size and under a hard memory limit. Break the code on purpose, and make sure that the test fails. Measure the limit. Do not guess it.

**Where.** `m3.py`, `retriever.py`, `tests/test_retriever.py`, `tests/conftest.py`.

### L23. Give credit for a correct answer in other words, but guard the numbers

**Problem.** The verifier gives 1.0 only for an exact match or a short answer that contains the reference. The final model of run 2 scored 0.578 on 148 validation questions. We read 52 of the 79 answers that did not get 1.0. Of these, 26 were correct answers in other words (for example "35,560" against "35 560", or a long answer around a short reference). They scored 0.39 on average. A perfect verifier would give about 0.68.

**Facts.**

- A BGE-M3 score looks at the topic of two texts. It does not check a fact. We made 44 answers from correct answers by changing one number. In each mode, 73 to 95% of them stayed above the best threshold.
- The ColBERT score is not symmetric. With the reference as the query, a long correct answer is not punished for its extra words. An answer that leaves out part of the reference is punished: only 5 of 39 half answers kept the credit. This direction had the smallest overlap between correct and wrong answers.
- A guard that needs every number of the reference in the answer caught all 44 changed answers. It blocked 1 of 46 correct answers. A blocked answer keeps its old reward, so the cost is small. On real data, two wrong answers scored 0.60 and 0.61. The guard blocked both.
- The labelled set had no hard negatives at first. We made them with a script, and they changed the design.
- The guard sees digits only. A wrong answer without a number can still get credit (row 97 of the run-2 eval: score 0.66). Two answers that agree with a bad reference also got credit.

**Solution.**

- Credit rule: if the number guard passes and the ColBERT score (reference as the query) is at least 0.65, the answer reward is at least 0.85. The rule never lowers a reward. The model runs only when the cheap checks pass.
- The points are below 1.0, so an exact answer still earns more than a long answer in other words.
- `reward_answer` stays the answer reward of the old rules. The new field `reward_answer_semantic` holds the credited value, and `reward` uses it. The yardstick between runs (L17) does not change.
- The server loads one BGE-M3 for the retriever and the verifier. The credit is off unless the config sets `semantic_credit: true`.
- We replayed the 148 saved answers. The mean went from 0.578 to 0.677, and 30 answers got credit. None of the 9 wrong answers or 7 bad questions got credit. The threshold was chosen on the same 72 labels.

**Rule.** A semantic score checks the topic, not the facts. Guard the facts that a rule can check. Test a new reward on hard negatives, not only on easy ones. Keep the old reward as a yardstick, because a rise of the new reward alone does not show that the model is better.

**Where.** `app.py` (`number_guard`, `semantic_credit`), `m3.py` (`reference_coverage`), `scripts/rescore_dump.py --semantic`, `tests/test_semantic_credit.py`, `data/verifier_labels.jsonl`.

### L24. Remove repeated questions, and drop the ones with conflicting references

**Problem.** The same question text appears more than once in the data. Train has 110 extra rows (88 questions), validation has 18 (15 questions). The model sees the same question many times in training. In validation, 148 rows are not 148 independent samples, so the standard error is too small.

**Facts.**

- The generator built each question from a window of three chunks, and neighbouring windows overlap. Every repeat comes from neighbouring chunks (for example 99 and 100). None comes from the same chunk.
- Most repeats have the same reference. Some have different references (22 questions in train, 4 in validation, after the verifier's `normalize`). We cannot know which reference is right. A correct answer can score low against the wrong one.
- No question is in both train and validation.

**Solution.**

- Keep the first copy of a repeated question. Drop all copies when the references differ. Two references are the same when the verifier's `normalize` gives the same text.
- Write new files (`train_dedup.jsonl`: 1,108 rows, `validation_dedup.jsonl`: 126 rows). Do not change the originals. The trainer loads a split by name.
- Recompute old results on the 126 rows from the saved dumps. The means fell by about 0.01 and the standard errors grew by about 0.003. No conclusion changed.

**Rule.** Compare the old and the new numbers on the same rows. Keep the original files, so an old result can still be reproduced.

**Where.** `scripts/dedup_data.py`, `tests/test_dedup_data.py`, `data/*_dedup.jsonl`.

### L25. A better verifier changed the score, not the model: compare like with like

**Problem.** Run 3 trained with the BGE-M3 retriever, the semantic credit (L23) and the de-duplicated data (L24). We wanted to know if the model was better than run 2.

**Facts.**

- Eval on the 126 validation questions, greedy, both final models with BGE-M3 at top 2. Old rule: run 3 0.558, run 2 0.569 (run 3 minus run 2: -0.011 ± 0.019). With the credit: run 3 0.653, run 2 0.667 (-0.014 ± 0.022). Each run has 25 answers with credit. There is no difference.
- The credit adds about 0.10 to both models. It changes the scale of the score, not the model.
- The training reward rose from 0.43 to 0.71. This does not show a better model. The reward is not the same in the two runs (L15, L17).
- One set of answers has three scores. `reward` in `eval_summary.json` is the old rule (the yardstick). `reward_answer_semantic` is the answer part with the credit. `reward_total` is the training formula: 0.75 times the credited answer plus 0.25 times the retrieval part.
- Run 2 checkpoints were evaluated with the old retriever. Only its final model was evaluated again with BGE-M3. A step by step comparison mixes the retriever and the training. Only the comparison of the final models is fair.
- The room was small. A perfect verifier gave about 0.68 on the labelled answers (L23), and the model already answers in words close to the reference. Bad references and bad questions limit the score too (L18).
- The answers did not drift. Mean final words went from 46 to 35 in training, and the eval answers have 22 words (run 2: 22).
- In the training dump, 2 of the 424 rows with credit had no search. The model wrote a bad tool call as text, and the text repeated the question words (ColBERT 0.675 and 0.709, reward 0.6375). We left this open on purpose (decision of the user, 2026-10-30).

**Not known.** Run 3 changed three things at once, so we cannot say which one did nothing. There is one run for each setting. Nobody read the 25 answers with credit in the eval.

**Rule.** Compare two runs with the same retriever, the same rule and the same rows. Write which of the three scores a number is. Before you train on a new reward, measure how much room it has (the labelled ceiling), because a verifier that fixes the scale may not change the model.

**Where.** `trainer/runs/20261030-080054_run3_semantic`, `trainer/runs/eval_run3_semantic`, Plan step 6.

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

To run the unit tests, go to `NeMo_Gym`. Run `resources_servers/mission_search/.venv/bin/python -m pytest resources_servers/mission_search/tests`. Make sure that all tests pass.

### 4.5 Train

> CAUTION: Do not start while another program uses the GPU. A training run uses up to 20 GB of GPU memory.

1. Start the server (procedure 4.4). Do not close its terminal.
2. Go to the `trainer` folder.
3. Run `temp/.venv-temp/bin/python train_grpo.py --max-steps 200 --save-steps 50 --name <name> 2>&1 | tee <name>.log`.
4. Find the line `[train] run folder:`.
5. After the first step, look in `train_dump.jsonl` for the field `reward_answer`. If it is missing, the server runs an old verifier. Stop and restart the server.
6. Do not change `app.py` or the constants in `train_grpo.py` during the run.
7. After a crash, run `train_grpo.py --resume runs/<run>/checkpoint-<N>` with the highest N.

A test run uses `--max-steps 4 --save-steps 2`. It takes about 4 minutes.

### 4.6 Evaluate

1. Start the server (procedure 4.4). Make sure that the GPU is free.
2. Go to the `trainer` folder.
3. Run `temp/.venv-temp/bin/python eval.py --run-dir runs/<run> --checkpoints runs/<run>/checkpoint-50 runs/<run>/checkpoint-100 runs/<run>/checkpoint-150 runs/<run>/checkpoint-200`.
4. Read the table at the end. The column `reward` is the answer-only reward.
5. Wait about 25 minutes for the base model and four checkpoints.

For a fast check, add `--limit 16`.

### 4.7 Make the report

1. Go to the repository root.
2. Run `NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/report.py --run-dir <run folder> --model "<label>=<path to an eval dump>"`. Repeat `--model` for each model, up to four.
3. Open `<run folder>/report.html` in a browser. In WSL, run `explorer.exe "$(wslpath -w <file>)"`.

### 4.8 Run the retrieval benchmark

1. Go to the repository root. Make sure that BGE-M3 is in `Embedder/bge-m3`.
2. Run `NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/retrieval_bench.py --model-queries <eval dump>`.
3. Wait about 10 minutes. The script uses the CPU only.
4. Read the tables. The results also go to `trainer/runs/retrieval_bench`.

For a fast check, add `--limit 60 --no-repair`.

### 4.9 Run the tests with a memory limit

1. Go to the folder `NeMo_Gym/resources_servers/mission_search`.
2. Run `(ulimit -v 25165824; .venv/bin/python -m pytest tests)`.
3. Make sure that all tests pass (78 tests on 2026-10-29, 110 on 2026-10-30).
4. To find a new limit, run the tests with a script that prints `VmPeak` and `VmHWM` from `/proc/self/status` at the end.
5. Set the limit to 1.5 times `VmPeak`. It must stay below the RAM of WSL (31 GB).

## 5. Open items

- The Llama chat template puts quotation marks around the tool text. We kept this for version 1. Compare quoted and plain text after the retriever change.
- Some reasoning models remove or shorten old turns. TITO does not work after such a change. Keep the text before the last change as the prompt, and train only on the tokens after it.
- Hold out a second set for a final test when we tune against the validation set.
- The retrieval test in `build_index.py` never tests the last 16% of the questions. Use a larger step, so that the sample covers the whole file.
- The pictures in `Schema_learning` and the local debug notes (not in the repository) show the index before the merge (753 chunks and the old line numbers). Update them before the next debug session.
- Audit the questions and the references (Plan, step 8). Count the defects by type before we decide to write new questions.
- Measure the length cap (5 x expected words + 15) on a larger dump. Look at the final answers that get 1.0 by containment.
- The ramp of the retrieval part (0.3 to 0.6) fits NoInstruct at top 2. Fit it again for the new retriever, or drop the part (L16).
- Run 2 changed two things. Run an ablation with the scaling change alone if the cause matters. One run costs about 1 hour, and the evaluation about 25 minutes.
- Decide whether git tracks `data/train.jsonl`, `data/validation.jsonl` and the new `*_dedup.jsonl` files. The scaffold ignores them, and git also ignores `Step2_Files`. (The repeated rows are removed in the `_dedup` files: L24.)
- Put the path of BGE-M3 in a config value before the server loads it.
- A `top_k` of 3 gives about 420 tokens for each search. Raise `MAX_COMPLETION` and `MAX_SEQ`, and check the VRAM. The peak was 19.9 GB of 24.4 GB.
- Compact the WSL disk file when drive C is short on space (L21).
- Read the 25 answers with credit in `eval_run3_semantic/eval_final.jsonl` and mark the wrong ones (L25). The credit has no check for a search: 2 of 424 credited training rows had none.
- Run 2 checkpoints 50, 100 and 150 with BGE-M3 at top 2 (about 15 minutes of GPU) give a fair step by step table against run 3 (L25).
