# How we work

## Decision rights
The human owns design. The agent owns execution. When a decision is not on this list, treat it as the human's and ask.

**Agent decides alone. Report after.**
- Implementing steps of a plan the human has approved.
- Writing and running tests that live in the repo.
- Fixing a bug found by a failing test (failing test first, then the fix).
- Names, file layout and small structure inside an approved design. Note them in the report.

**Agent proposes, human decides. Stop and ask first.**
- Any design, architecture, data shape, default, or scope change.
- UX, wording, and prompt text.
- Changes to original or shared code, or to a public contract. Flag as a deviation.
- Adding a dependency.
- Changing production code only to make a test tool work.
- Spending real money or quota (live model calls, paid APIs) beyond [budget].
- Anything that touches credentials, auth, or security boundaries.

**Never without explicit approval, every time.**
- Commit, push, merge, tag, publish, open a PR.
- Delete or overwrite files the agent did not create in this session.
- Anything outward-facing (messages, posts, uploads) or legal (terms, licences).

**Stop and report. Do not push on.**
- A result contradicts the plan or the agent's own claim.
- A test fails and the cause is not understood.
- The agent finds it "a little unsure" about a decision. Unsure means ask.

## How to ask
- One question at a time. Be specific. Give options with a recommendation.
- Make each question stand alone: say what you found, then ask.

## Tests
- Tests that support a claim of "done" live in `[tests/]` in the repo, never only in a scratch folder.
- Scratch folders are for throwaway experiments. The agent says which is which.
- Each test file starts with a comment saying why it exists. Each test name is a sentence about behaviour.
- A bug found by hand gets a failing test first, then the fix.
- A new test must be seen failing once (break the code on purpose, watch it fail, restore). Report this.
- Running the tests must leave the working tree clean. If it does not, report what it left.
- Say what is NOT tested, and why.

## Reporting (predict → check, once per unit of work)
Applies only to the verification run: the first full test run after a unit of
work is code-complete, or a live check that supports a "done" claim. It does
NOT apply to iterating, debugging, lint, typecheck, or re-runs after a fix.

- Once the code is complete, before writing or running the tests, write:
PREDICT: the behaviours I expect (max 5 bullets), and for each, what
result would prove me wrong.
- Write the tests from the prediction: one test per predicted behaviour,
plus at least one that would fail if the prediction were wrong. Then run them.
- After the run, write:
RESULT: what happened. SURPRISE: yes/no, and why.
- A surprise is a mismatch with PREDICT (the code behaved differently than I
expected), not an ordinary failure from a typo or slip I fix straight away.
- Never edit PREDICT after the run.
- For every surprise:
   - Add a test that covers it, if it captures behaviour worth guarding. If
     not, say why no test was added.
   - Log it on the Surprises list in Plan.md (date, expected, got, test
     added or reason none, next step).
   - If it changes the plan, stop and tell me.
- Failures and skips come first in the report. Say what was NOT tested.