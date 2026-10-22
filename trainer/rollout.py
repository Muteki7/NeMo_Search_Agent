"""Multi-turn rollout for GRPO: the agent loop, the Gym client and the verify reward.

How the pieces fit (see Plan.md step 5 and Learnings.md L4, L5, L13, L14):
  - The NeMo Gym resources server (started by `ng_run`) owns the tool (`/search_corpus`) and the reward (`/verify`).
    `GymClient.discover` asks the head server (port 11000) where it listens. After that we talk to it directly.
  - `run_rollout` is the agent. It does not know about vLLM: it takes a `generate_fn` that turns token ids into
    sampled token ids. The Test 1 file (`test_rollout.py`) drives it with a fake generator.
  - TITO (token in, token out): each trajectory keeps one list of token ids (`ctx_ids`). The ids that vLLM samples go
    in as they are. A tool result goes in as a "delta": the ids that the chat template adds for a tool message
    (found by rendering a dummy conversation with and without it). Nothing is ever decoded and encoded again.
  - Decoded text is used for one thing only: to find a tool call.

This file must not import unsloth or vllm at module level, so that the tests run on the CPU.
"""
import json
import re
import string
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
import yaml

from prompting import apply_chat_template, assert_date_pinned

HEAD_URL = "http://127.0.0.1:11000"
SERVER_NAME = "mission_search_resources_server"   # the instance name in resources_only.yaml
TOOL_ROLE = "ipython"                             # Llama 3.x role for a tool message (the template quotes the text)
# tool name -> names of the arguments that must be strings. The route on the server is /{tool name}.
ALLOWED_TOOLS = {"search_corpus": ("query",)}

_DUMMY = [{"role": "user", "content": "dummy"}, {"role": "assistant", "content": "dummy"}]


# ---------------------------------------------------------------------------------------------------------------
# Gym client
# ---------------------------------------------------------------------------------------------------------------
class GymError(RuntimeError):
    """The environment server did not answer as it should. This is an infrastructure fault, not a model fault."""


class GymClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def discover(cls, server_name: str = SERVER_NAME, head_url: str = HEAD_URL, timeout: float = 10.0) -> "GymClient":
        """Ask the head server for the host and port of the resources server (they change at each `ng_run` start)."""
        try:
            reply = requests.get(f"{head_url}/global_config_dict_yaml", timeout=timeout)
            reply.raise_for_status()
        except requests.RequestException as error:
            raise GymError(f"Head server not reachable at {head_url}. Start `ng_run` first. ({error})") from error
        # The reply is one JSON string that holds YAML. Decode the JSON first, then the YAML.
        config = yaml.safe_load(reply.json())
        for type_block in (config.get(server_name) or {}).values():
            for impl_block in type_block.values():
                if isinstance(impl_block, dict) and "port" in impl_block:
                    return cls(f"http://{impl_block.get('host', '127.0.0.1')}:{impl_block['port']}")
        raise GymError(f"Server {server_name!r} is not in the head server config. Is `ng_run` using resources_only.yaml?")

    def post(self, path: str, payload: dict) -> dict:
        try:
            reply = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            reply.raise_for_status()
            return reply.json()
        except (requests.RequestException, ValueError) as error:
            raise GymError(f"POST {self.base_url}{path} failed: {error}") from error

    def call_tool(self, name: str, args: dict) -> str:
        """Run one tool and return its text. The search tool answers {"results": "..."}."""
        reply = self.post(f"/{name}", args)
        return reply["results"] if isinstance(reply.get("results"), str) else json.dumps(reply)


# ---------------------------------------------------------------------------------------------------------------
# Tool-call parsing (decoded text is used for this only)
# ---------------------------------------------------------------------------------------------------------------
def extract_json_objects(text: str) -> list:
    """Every top-level JSON object (a dict) in `text`. Copied from AutoDidact rl_helpers.py (no unsloth import here)."""
    results, i, length = [], 0, len(text)
    while i < length:
        if text[i] == "{":
            start, stack = i, 1
            i += 1
            while i < length and stack > 0:
                if text[i] == "{":
                    stack += 1
                elif text[i] == "}":
                    stack -= 1
                i += 1
            if stack == 0:
                try:
                    obj = json.loads(text[start:i])
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
        else:
            i += 1
    return results


@dataclass
class ParsedTurn:
    kind: str                      # "answer" (no tool call), "call" (one valid tool call) or "error" (a bad tool call)
    name: Optional[str] = None
    args: Optional[dict] = None
    reason: str = ""


def _call_shape(obj: dict):
    """(name, arguments) if `obj` looks like a tool call, else None. Two forms:
    {"name": ..., "parameters"|"arguments": {...}} and {"type": "function", "function": {"name": ..., "parameters": {...}}}"""
    inner = obj["function"] if isinstance(obj.get("function"), dict) else obj
    name = inner.get("name")
    if not isinstance(name, str):
        return None
    args = inner.get("parameters", inner.get("arguments"))
    if isinstance(args, str):  # OpenAI style: the arguments are a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return name, args


def parse_turn(text: str, allowed_tools: dict = ALLOWED_TOOLS) -> ParsedTurn:
    """Decide what a sampled turn is. A JSON object that is not call-shaped is part of an answer, not a call."""
    calls = [shape for shape in map(_call_shape, extract_json_objects(text)) if shape is not None]
    if not calls:
        return ParsedTurn("answer")
    if len(calls) > 1:
        return ParsedTurn("error", reason="more than one tool call in one turn")
    name, args = calls[0]
    if name not in allowed_tools:
        return ParsedTurn("error", name=name, reason=f"unknown tool {name!r}")
    if not isinstance(args, dict):
        return ParsedTurn("error", name=name, reason="arguments are not an object")
    missing = [k for k in allowed_tools[name] if not isinstance(args.get(k), str)]
    if missing:
        return ParsedTurn("error", name=name, reason=f"missing or non-string argument {missing}")
    return ParsedTurn("call", name=name, args={k: args[k] for k in allowed_tools[name]})


# ---------------------------------------------------------------------------------------------------------------
# TITO helpers
# ---------------------------------------------------------------------------------------------------------------
def tool_delta_ids(tok, role: str, text: str) -> list:
    """The ids to append after a sampled assistant turn: the tool message and the opening of the next assistant turn.
    Render a dummy conversation with and without the tool message. The ids that the second render adds are the delta."""
    before = apply_chat_template(tok, _DUMMY, tokenize=True, return_dict=False)
    after = apply_chat_template(tok, _DUMMY + [{"role": role, "content": text}],
                                tokenize=True, return_dict=False, add_generation_prompt=True)
    assert list(after[:len(before)]) == list(before), f"chat template is not prefix-preserving for role={role!r}"
    return list(after[len(before):])


def startup_checks(tok, role: str = TOOL_ROLE) -> list:
    """Run before the first rollout. Raises if the template is not prefix-preserving or the date is not pinned."""
    probe = tool_delta_ids(tok, role, "dummy")
    assert_date_pinned(apply_chat_template(
        tok, [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], tokenize=False))
    return probe


# ---------------------------------------------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class GenOut:
    """One sampled turn, as returned by a generate function."""
    token_ids: list
    logprobs: list
    text: str
    finish_reason: str             # "stop" or "length"
    prompt_token_ids: list         # the ids that the engine says it received


@dataclass
class _Traj:
    prompt_ids: list
    ctx_ids: list                  # the TITO buffer: exactly what the engine sees
    comp: list = field(default_factory=list)
    lps: list = field(default_factory=list)
    mask: list = field(default_factory=list)
    items: list = field(default_factory=list)      # Responses-API output items
    n_calls: int = 0
    n_gen_turns: int = 0
    active: bool = True
    ended: str = ""                # answer | truncated | bad_call | turn_cap | budget
    last_sent: list = None
    last_gen: list = None


def _message_item(i: int, turn: int, text: str) -> dict:
    return {"id": f"msg_{i}_{turn}", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def run_rollout(prompts, tok, generate_fn: Callable, call_tool: Callable, max_turns: int = 5,
                max_completion_tokens: Optional[int] = None, max_new_tokens: int = 256,
                allowed_tools: dict = ALLOWED_TOOLS, tool_role: str = TOOL_ROLE, max_workers: int = 16) -> dict:
    """Run every prompt to the end. Returns one trajectory per incoming prompt, in the same order.

    prompts      : list of message lists. TRL already repeats each prompt `num_generations` times, so do not repeat it again.
    generate_fn  : list of id lists -> list of GenOut (same order)
    call_tool    : (name, args dict) -> tool text. It raises GymError if the server fails (we do not hide that).
    max_turns    : the most generations for one trajectory. A tool call on the last turn is not run (no answer is left).
    max_completion_tokens : if set, a tool call is not run when its result plus one more turn would not fit.
    """
    n = len(prompts)
    trajs = []
    for messages in prompts:
        ids = list(apply_chat_template(tok, messages, add_generation_prompt=True, tokenize=True, return_dict=False))
        trajs.append(_Traj(prompt_ids=ids, ctx_ids=list(ids)))

    for turn in range(max_turns):
        idx = [i for i in range(n) if trajs[i].active]
        if not idx:
            break
        sent = {i: list(trajs[i].ctx_ids) for i in idx}
        outs = generate_fn([sent[i] for i in idx])
        assert len(outs) == len(idx), "generate_fn returned the wrong number of outputs"

        pending = []   # (trajectory index, ParsedTurn, text) for turns that ask for a tool
        for i, out in zip(idx, outs):
            t = trajs[i]
            assert len(out.token_ids) == len(out.logprobs), "token ids and logprobs differ in length"
            if list(out.prompt_token_ids) != sent[i]:
                raise AssertionError("the engine received other ids than we sent (hidden re-tokenization)")
            # token-out: the sampled ids go into the buffer exactly as produced
            t.comp += out.token_ids
            t.lps += out.logprobs
            t.mask += [1] * len(out.token_ids)
            t.ctx_ids += out.token_ids
            t.last_sent, t.last_gen = sent[i], list(out.token_ids)
            t.n_gen_turns += 1

            if out.finish_reason == "length":           # truncated: the buffer ends as sampled
                t.items.append(_message_item(i, turn, out.text))
                t.active, t.ended = False, "truncated"
                continue
            parsed = parse_turn(out.text, allowed_tools)
            if parsed.kind == "answer":
                t.items.append(_message_item(i, turn, out.text))
                t.active, t.ended = False, "answer"
            elif parsed.kind == "error":
                t.items.append(_message_item(i, turn, out.text))
                t.active, t.ended = False, "bad_call"
            elif turn == max_turns - 1:                  # a call on the last turn: no turn is left to use the result
                t.items.append({"id": f"fc_{i}_{turn}", "type": "function_call", "call_id": f"call_{i}_{turn}",
                                "name": parsed.name, "arguments": json.dumps(parsed.args), "status": "completed"})
                t.active, t.ended = False, "turn_cap"
            else:
                pending.append((i, parsed))

        if pending:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(pending)))) as pool:
                texts = list(pool.map(lambda p: call_tool(p[1].name, p[1].args), pending))
            for (i, parsed), tool_text in zip(pending, texts):
                t = trajs[i]
                call = {"id": f"fc_{i}_{turn}", "type": "function_call", "call_id": f"call_{i}_{turn}",
                        "name": parsed.name, "arguments": json.dumps(parsed.args), "status": "completed"}
                t.items.append(call)
                t.n_calls += 1
                delta = tool_delta_ids(tok, tool_role, tool_text)
                if max_completion_tokens is not None and len(t.comp) + len(delta) + max_new_tokens > max_completion_tokens:
                    t.active, t.ended = False, "budget"  # the result and a next turn would not fit: stop before the delta
                    continue
                t.items.append({"type": "function_call_output", "call_id": call["call_id"], "output": tool_text})
                t.comp += delta
                t.lps += [0.0] * len(delta)
                t.mask += [0] * len(delta)
                t.ctx_ids += delta

    for t in trajs:
        if t.active:  # cannot happen: the last turn always ends a trajectory. Keep the check to catch a loop bug.
            raise AssertionError("a trajectory is still active after the last turn")
        assert len(t.comp) == len(t.lps) == len(t.mask), "completion, logprobs and mask differ in length"
        assert all(lp == 0.0 for m, lp in zip(t.mask, t.lps) if m == 0), "an environment token has a logprob"
        assert all(lp <= 0.0 for m, lp in zip(t.mask, t.lps) if m == 1), "a sampled token has a positive logprob"
        # TITO invariant: the last prompt sent plus its generation equals the prompt plus the completion. It holds
        # because every trajectory ends right after a generation (a tool delta is added only when a next turn follows).
        assert t.last_sent + t.last_gen == t.prompt_ids + t.comp, "token buffer does not match prompt + completion"

    stats = {
        "n": n,
        "tool_call_rate": sum(t.n_calls > 0 for t in trajs) / n if n else 0.0,
        "mean_turns": sum(t.n_gen_turns for t in trajs) / n if n else 0.0,
        **{f"ended_{k}": sum(t.ended == k for t in trajs) / n if n else 0.0
           for k in ("answer", "truncated", "bad_call", "turn_cap", "budget")},
    }
    return {
        "prompt_ids": [t.prompt_ids for t in trajs],
        "completion_ids": [t.comp for t in trajs],
        "logprobs": [t.lps for t in trajs],
        "env_mask": [t.mask for t in trajs],
        # one dict for each trajectory: the reward function turns "output" into the /verify request
        "trajectory": [{"output": t.items, "ended": t.ended, "n_tool_calls": t.n_calls, "n_turns": t.n_gen_turns}
                       for t in trajs],
        "stats": stats,
    }


# ---------------------------------------------------------------------------------------------------------------
# Wiring to Unsloth / TRL / vLLM (tested in Test 2, not in Test 1)
# ---------------------------------------------------------------------------------------------------------------
def vram_used_mib() -> Optional[int]:
    """GPU memory in use (all programs), from nvidia-smi. None if it cannot be read."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def make_vllm_generate_fn(eng, sampling_params, lora_request=None) -> Callable:
    """A generate_fn for run_rollout on a vLLM engine: token ids in, sampled token ids and their logprobs out.
    `lora_request=None` runs the base model (no adapter)."""
    def generate_fn(id_lists):
        outs = eng.generate([{"prompt_token_ids": ids} for ids in id_lists], sampling_params, use_tqdm=False,
                            lora_request=lora_request)
        results = []
        for out in outs:
            o = out.outputs[0]
            ids = list(o.token_ids)
            lps = [(d[t].logprob if t in d else list(d.values())[0].logprob) if d else 0.0
                   for t, d in zip(ids, o.logprobs or [None] * len(ids))]
            results.append(GenOut(ids, lps, o.text, o.finish_reason, list(out.prompt_token_ids)))
        return results
    return generate_fn


def make_rollout_func(client: GymClient, lora_name: str, max_turns: int = 5, max_new_tokens: int = 256,
                      max_completion_tokens: Optional[int] = None, temperature: float = 0.9, top_p: float = 0.95):
    """Build the `rollout_func(prompts, trainer)` that GRPOTrainer calls. It returns one trajectory per prompt.
    Set the GRPOConfig `temperature` to the same value: TRL divides its own logits by it."""
    from vllm import SamplingParams  # imported here so that the CPU tests do not need vllm

    def rollout_func(prompts, trainer):
        tok = trainer.processing_class
        eng = trainer.model.vllm_engine
        startup_checks(tok)
        lora = trainer.model.load_lora(lora_name, load_tensors=True)   # a fresh handle at each step: the weights changed
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, logprobs=0)
        result = run_rollout(prompts, tok, make_vllm_generate_fn(eng, sp, lora), client.call_tool, max_turns=max_turns,
                             max_completion_tokens=max_completion_tokens, max_new_tokens=max_new_tokens)
        stats = result.pop("stats")
        stats["vram_mib"] = vram_used_mib()
        rollout_func.last_stats = stats
        print("[rollout] " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
        return result

    rollout_func.last_stats = None
    return rollout_func


# ---------------------------------------------------------------------------------------------------------------
# Answer text helpers (for eval.py). The same normalization as app.py in the server: the two venvs cannot import each
# other, so keep them in sync. They do not decide the reward (the server does), only the extra eval metrics.
# ---------------------------------------------------------------------------------------------------------------
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def remove_reasoning(text: str) -> str:
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)


def normalize_text(text: str) -> str:
    """Lowercase, delete punctuation (without a space: `18:24` equals `1824`), delete a/an/the, collapse spaces."""
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower().translate(_PUNCTUATION))
    return " ".join(text.split())


def answer_words(final_text: Optional[str]) -> list:
    """The words of a final answer without the reasoning."""
    return normalize_text(remove_reasoning(final_text or "")).split()


def contains_answer(final_text: Optional[str], expected: str) -> bool:
    """True if the expected answer is a run of whole words in the final answer. Ignores the length cap of the reward."""
    haystack, needle = answer_words(final_text), normalize_text(expected).split()
    return bool(needle) and any(haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


# ---------------------------------------------------------------------------------------------------------------
# Dump: one JSON line for each trajectory, to read the outputs and to re-score them offline
# ---------------------------------------------------------------------------------------------------------------
def final_text(trajectory: dict) -> Optional[str]:
    """Text of the last `message` item of a trajectory, or None if it has no final message."""
    messages = [i for i in trajectory["output"] if i["type"] == "message"]
    return messages[-1]["content"][0]["text"] if messages else None


def trajectory_record(prompt_messages: list, answer: str, trajectory: dict, reward: float, **extra) -> dict:
    """One dump line. `question` is the text after the last "Question:" in the last user message."""
    user = next((m["content"] for m in reversed(prompt_messages) if m["role"] == "user"), "")
    return {**extra, "question": user.rsplit("Question:", 1)[-1].strip(), "answer": answer, "ended": trajectory["ended"],
            "n_tool_calls": trajectory["n_tool_calls"], "n_turns": trajectory["n_turns"], "reward": reward,
            "final_text": final_text(trajectory), "output": trajectory["output"]}


def append_jsonl(path, records: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def checkpoint_step(checkpoint_dir) -> int:
    """The optimizer step of a TRL checkpoint folder (`global_step` in trainer_state.json)."""
    with open(Path(checkpoint_dir) / "trainer_state.json", encoding="utf-8") as f:
        return int(json.load(f)["global_step"])


def prune_dump_after_step(dump_path, step: int) -> int:
    """Before a resume: the steps after the checkpoint will be run again, so move their dump lines into
    `<dump>.dropped_after_step<step>.jsonl` (appended, never deleted). Returns the number of lines moved."""
    path = Path(dump_path)
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [line for line in lines if json.loads(line)["step"] <= step]
    drop = [line for line in lines if json.loads(line)["step"] > step]
    if drop:
        with open(path.with_name(path.name + f".dropped_after_step{step}.jsonl"), "a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in drop))
        path.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    return len(drop)


# ---------------------------------------------------------------------------------------------------------------
# Reward: ask /verify
# ---------------------------------------------------------------------------------------------------------------
def build_response(output_items: list) -> dict:
    """The `response` object that /verify expects (the Responses API shape), around our output items."""
    return {"id": "rollout", "created_at": 0, "model": "policy", "object": "response", "output": output_items,
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}


# The parts of the reward that /verify returns next to `reward` (the combined value the trainer uses).
REWARD_PARTS = ("reward_answer", "reward_retrieval", "retrieval_gate", "best_cov", "n_searches")


def verify_batch(client: GymClient, prompts, trajectory, answer, responses_create_params=None, max_workers: int = 16) -> list:
    """POST every trajectory to /verify and return the full replies (a dict for each: reward, reward_answer, ...)."""
    def one(i):
        params = responses_create_params[i] if responses_create_params else {"input": prompts[i]}
        body = {"responses_create_params": params, "response": build_response(trajectory[i]["output"]), "answer": answer[i]}
        return client.post("/verify", body)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, range(len(prompts))))


def make_gym_verify_reward(client: GymClient, max_workers: int = 16, dump_path=None, step_offset: int = 0):
    """`dump_path`: if set, every trajectory is appended there with its reward and the parts of it (see `trajectory_record`).
    TRL calls the reward once for each rollout, so `step` counts the optimizer steps of this run."""
    calls = {"n": step_offset}   # after a resume, the steps continue from the checkpoint step

    def gym_verify_reward(prompts, completions, trajectory, answer, responses_create_params=None, **kwargs):
        """One reward for each trajectory. TRL passes the dataset columns `answer` and `responses_create_params`
        and the `trajectory` field that rollout_func returned."""
        replies = verify_batch(client, prompts, trajectory, answer, responses_create_params, max_workers)
        rewards = [float(r["reward"]) for r in replies]
        if dump_path:
            calls["n"] += 1
            append_jsonl(dump_path, [trajectory_record(prompts[i], answer[i], trajectory[i], rewards[i], step=calls["n"], index=i,
                                                       **{k: replies[i][k] for k in REWARD_PARTS if k in replies[i]})
                                     for i in range(len(prompts))])
        return rewards

    return gym_verify_reward
