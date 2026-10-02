# Learnings: changes in approach and the reasons for them

| Item | Data |
|---|---|
| Title | Learnings from the search agent project |
| Writing style | ASD-STE100 (Simplified Technical English) |
| Project | Train a small model to answer questions with a search tool |
| Scope | Changes in approach and changes in tests |
| Not in scope | Package problems. See `Dependency_hell.md`. |
| Last update | 2026-10-02 |

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
| chunk id | The number of the middle chunk of a window | chunk number |
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

**Where.** `rules.md` and `Plan.md`.

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

## 5. Open items

- The Llama chat template puts quotation marks around the content of a tool message. We probably do not want this for search results. Decide how to add the search results, and run the prefix check again.
- A reasoning model can remove or shorten old turns. TITO does not work after such a change. Keep the text before the last change as the prompt, and train only on the tokens after it.
- Hold out a second set for a final test when we start to tune against the validation set.
- Delete the counter for the old render method after we measure it on a larger sample.
