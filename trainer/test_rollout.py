"""Test 1 for rollout.py: logic tests on the CPU. No vLLM, no GPU, no Gym server.

A fake generator plays the model: it returns scripted text, turned into token ids with the real Llama tokenizer.
A fake tool plays the server. The real tokenizer and the real chat template are used, so the token ids are real.

Run (from the trainer folder, in the trainer venv):
    temp/.venv-temp/bin/python test_rollout.py            # the offline tests
    temp/.venv-temp/bin/python test_rollout.py --live     # also test against a running `ng_run` server
Each test prints one PASS or FAIL line. The exit code is 1 if one test fails.
"""
import json
import sys
import traceback
from functools import lru_cache

import rollout
from rollout import (GenOut, GymClient, GymError, build_response, make_gym_verify_reward, parse_turn, run_rollout,
                     startup_checks, tool_delta_ids)
from prompting import apply_chat_template

MODEL = "unsloth/Llama-3.2-1B-Instruct"   # only its tokenizer and chat template are used
CALL = '{"type": "function", "function": {"name": "search_corpus", "parameters": {"query": "%s"}}}'


@lru_cache(maxsize=1)
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def messages(question):
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": question}]


class ScriptedModel:
    """generate_fn. `scripts` maps a question to a list of (text, finish_reason), one entry for each turn."""

    def __init__(self, scripts, prompts):
        self.scripts = scripts
        self.turn = {q: 0 for q in scripts}
        self.prompt_ids = {q: list(apply_chat_template(tok(), messages(q), add_generation_prompt=True,
                                                       tokenize=True, return_dict=False)) for q in scripts}
        self.calls = 0

    def __call__(self, id_lists, wrong_prompt_ids=False):
        self.calls += 1
        eot = tok().convert_tokens_to_ids("<|eot_id|>")
        outs = []
        for ids in id_lists:
            q = next(q for q, p in self.prompt_ids.items() if ids[:len(p)] == p)
            text, reason = self.scripts[q][self.turn[q]]
            self.turn[q] += 1
            token_ids = tok().encode(text, add_special_tokens=False) + ([eot] if reason == "stop" else [])
            outs.append(GenOut(token_ids, [-0.5] * len(token_ids), text, reason,
                               ids[:-1] if wrong_prompt_ids else list(ids)))
        return outs


class FakeTool:
    def __init__(self):
        self.calls = []

    def __call__(self, name, args):
        self.calls.append((name, args))
        return f"Result 1:\nfake text for {args['query']}\n------\n"


def rollout_for(scripts, **kw):
    questions = list(scripts)
    model, tool = ScriptedModel(scripts, questions), FakeTool()
    out = run_rollout([messages(q) for q in questions], tok(), model, tool, **kw)
    return out, model, tool


# ---- 1. the parser --------------------------------------------------------------------------------------------
def test_parse_nested_form():
    p = parse_turn("<reasoning>I search.</reasoning>\n" + CALL % "flotation collar")
    assert (p.kind, p.name, p.args) == ("call", "search_corpus", {"query": "flotation collar"}), p


def test_parse_flat_forms():
    flat = '{"name": "search_corpus", "parameters": {"query": "a"}}'
    as_string = '{"name": "search_corpus", "arguments": "{\\"query\\": \\"b\\"}"}'
    assert parse_turn(flat).args == {"query": "a"}
    assert parse_turn(as_string).args == {"query": "b"}


def test_parse_answers():
    assert parse_turn("The answer is 1824.").kind == "answer"
    assert parse_turn('Here is data {"x": 1} and more.').kind == "answer"       # JSON that is not a call
    assert parse_turn(CALL[:-3] % "cut off").kind == "answer"                  # unbalanced braces: no object


def test_parse_errors():
    assert parse_turn(CALL % "a" + " " + CALL % "b").kind == "error"            # two calls
    assert parse_turn('{"name": "delete_all", "parameters": {"query": "a"}}').kind == "error"   # unknown tool
    assert parse_turn('{"name": "search_corpus", "parameters": {}}').kind == "error"            # no query
    assert parse_turn('{"name": "search_corpus", "parameters": {"query": 5}}').kind == "error"  # not a string
    assert parse_turn('{"name": "search_corpus", "parameters": "oops"}').kind == "error"        # not an object


# ---- 2. the template, the delta and the pinned date ----------------------------------------------------------
def test_startup_checks_and_delta_shape():
    delta = startup_checks(tok())
    text = tok().decode(delta)
    assert text.startswith("<|start_header_id|>ipython<|end_header_id|>"), text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"), text


def test_delta_does_not_depend_on_the_dummy_turns():
    # the delta must hold only the tool message: its text is in it and the dummy words are not
    text = tok().decode(tool_delta_ids(tok(), "ipython", "Result 1:\nabc"))
    assert "abc" in text and "dummy" not in text, text


# ---- 3. the loop ----------------------------------------------------------------------------------------------
def test_call_then_answer_and_direct_answer():
    scripts = {"Q-call": [(CALL % "gimbal", "stop"), ("It is 1824.", "stop")], "Q-direct": [("Just 7.", "stop")]}
    out, model, tool = rollout_for(scripts)
    assert [t["ended"] for t in out["trajectory"]] == ["answer", "answer"]
    assert [i["type"] for i in out["trajectory"][0]["output"]] == ["function_call", "function_call_output", "message"]
    assert [i["type"] for i in out["trajectory"][1]["output"]] == ["message"]
    assert tool.calls == [("search_corpus", {"query": "gimbal"})]
    assert model.calls == 2                                      # turn 1 for both, turn 2 for the first only
    comp, mask, lps = out["completion_ids"][0], out["env_mask"][0], out["logprobs"][0]
    delta = tool_delta_ids(tok(), "ipython", "Result 1:\nfake text for gimbal\n------\n")
    assert sum(1 for m in mask if m == 0) == len(delta), "the zeros in the mask are not exactly the tool delta"
    start = mask.index(0)
    assert comp[start:start + len(delta)] == delta, "the delta is not in the completion where the mask says"
    assert all(lp == 0.0 for m, lp in zip(mask, lps) if m == 0) and all(lp < 0.0 for m, lp in zip(mask, lps) if m == 1)
    assert all(m == 1 for m in out["env_mask"][1]), "a direct answer has no environment tokens"
    assert out["stats"]["tool_call_rate"] == 0.5 and out["stats"]["ended_answer"] == 1.0


def test_one_trajectory_per_prompt_in_order():
    # TRL sends each prompt num_generations times, next to each other. We return one trajectory for each, in order.
    scripts = {"QA": [("a", "stop")] * 4, "QB": [("b", "stop")] * 4}     # one scripted turn for each of the 4 copies
    questions = ["QA"] * 4 + ["QB"] * 4
    out = run_rollout([messages(q) for q in questions], tok(), ScriptedModel(scripts, questions), FakeTool())
    assert len(out["prompt_ids"]) == len(out["completion_ids"]) == len(out["trajectory"]) == 8
    assert out["prompt_ids"][0] == out["prompt_ids"][3] != out["prompt_ids"][4] == out["prompt_ids"][7]
    assert tok().decode(out["completion_ids"][0]).startswith("a") and tok().decode(out["completion_ids"][7]).startswith("b")


def test_truncated_turn_ends_with_nothing_added():
    out, _, tool = rollout_for({"Q": [(CALL % "x", "length")]})
    assert out["trajectory"][0]["ended"] == "truncated" and tool.calls == []
    assert all(m == 1 for m in out["env_mask"][0])


def test_call_on_the_last_turn_is_not_run():
    out, _, tool = rollout_for({"Q": [(CALL % "a", "stop"), (CALL % "b", "stop")]}, max_turns=2)
    t = out["trajectory"][0]
    assert t["ended"] == "turn_cap" and len(tool.calls) == 1
    assert [i["type"] for i in t["output"]] == ["function_call", "function_call_output", "function_call"]   # no message


def test_bad_call_ends_the_trajectory_without_a_tool_run():
    out, _, tool = rollout_for({"Q": [(CALL % "a" + " " + CALL % "b", "stop")]})
    assert out["trajectory"][0]["ended"] == "bad_call" and tool.calls == []
    assert out["trajectory"][0]["output"][-1]["type"] == "message"


def test_budget_stops_before_the_delta():
    out, _, tool = rollout_for({"Q": [(CALL % "a", "stop"), ("done", "stop")]}, max_completion_tokens=50, max_new_tokens=40)
    t = out["trajectory"][0]
    assert t["ended"] == "budget" and len(tool.calls) == 1
    assert all(m == 1 for m in out["env_mask"][0]), "the delta was added although it did not fit"


def test_server_failure_is_raised_not_hidden():
    def broken_tool(name, args):
        raise GymError("server down")
    scripts = {"Q": [(CALL % "a", "stop"), ("x", "stop")]}
    try:
        run_rollout([messages("Q")], tok(), ScriptedModel(scripts, ["Q"]), broken_tool)
    except GymError:
        return
    raise AssertionError("a tool server failure was swallowed")


def test_engine_prompt_mismatch_is_raised():
    model = ScriptedModel({"Q": [("x", "stop")]}, ["Q"])
    try:
        run_rollout([messages("Q")], tok(), lambda ids: model(ids, wrong_prompt_ids=True), FakeTool())
    except AssertionError:
        return
    raise AssertionError("a prompt mismatch was not detected")


# ---- 4. the Gym client (the network is faked) ----------------------------------------------------------------
class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_discover_decodes_the_head_server_reply():
    yaml_text = ("mission_search_resources_server:\n  resources_servers:\n    mission_search:\n"
                 "      entrypoint: app.py\n      host: 127.0.0.1\n      port: 18432\nport_range_low: 10001\n")
    real = rollout.requests.get
    # The head server sends a JSON string that holds the YAML. `requests` decodes the JSON, so .json() returns the YAML text.
    rollout.requests.get = lambda url, timeout: _Reply(yaml_text)
    try:
        assert GymClient.discover().base_url == "http://127.0.0.1:18432"
        try:
            GymClient.discover(server_name="other_server")
        except GymError as error:
            assert "not in the head server config" in str(error)
        else:
            raise AssertionError("an unknown server name was accepted")
    finally:
        rollout.requests.get = real


def test_discover_says_what_to_do_when_the_head_server_is_down():
    try:
        GymClient.discover(head_url="http://127.0.0.1:9", timeout=2)     # nothing listens on port 9
    except GymError as error:
        assert "Start `ng_run` first" in str(error)
        return
    raise AssertionError("no error although the head server is down")


# ---- 5. the reward function (the network is faked) -----------------------------------------------------------
def test_reward_function_builds_the_verify_request():
    seen = []

    class FakeClient:
        def post(self, path, body):
            seen.append((path, body))
            return {"reward": 1}

    scripts = {"Q": [(CALL % "a", "stop"), ("It is 1824.", "stop")]}
    out, _, _ = rollout_for(scripts)
    reward = make_gym_verify_reward(FakeClient())(
        prompts=[messages("Q")], completions=[None], trajectory=out["trajectory"], answer=["1824"],
        responses_create_params=[{"input": messages("Q"), "tools": []}])
    path, body = seen[0]
    assert reward == [1.0] and path == "/verify" and body["answer"] == "1824"
    assert body["response"]["output"][-1]["type"] == "message" and body["responses_create_params"]["tools"] == []
    assert set(body["response"]) >= {"id", "created_at", "model", "object", "output", "tools", "tool_choice"}


# ---- 6. live tests: need `ng_run` with resources_only.yaml -----------------------------------------------------
def live_search_and_verify_end_to_end():
    client = GymClient.discover()
    text = client.call_tool("search_corpus", {"query": "gimbal angle deviations"})
    assert text.startswith("Result 1:") and text.count("------") == 2, text[:200]
    scripts = {"Q-right": [(CALL % "flotation collar", "stop"), ("<reasoning>found it</reasoning>1824", "stop")],
               "Q-wrong": [("Nobody knows.", "stop")],
               "Q-cap": [(CALL % "a", "stop"), (CALL % "b", "stop")]}
    model = ScriptedModel(scripts, list(scripts))
    out = run_rollout([messages(q) for q in scripts], tok(), model, client.call_tool, max_turns=2)
    rewards = make_gym_verify_reward(client)(
        prompts=[messages(q) for q in scripts], completions=[None] * 3, trajectory=out["trajectory"],
        answer=["1824"] * 3)
    assert rewards == [1.0, 0.0, 0.0], rewards     # right answer, wrong answer, a tool call and no final answer


# ---- runner ---------------------------------------------------------------------------------------------------
def main():
    live = "--live" in sys.argv
    tests = [(n, f) for n, f in globals().items()
             if callable(f) and (n.startswith("test_") or (live and n.startswith("live_")))]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}\n" + "".join("      " + line for line in traceback.format_exc().splitlines(True)))
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed" + ("" if live else " (offline only; add --live for the server tests)"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
