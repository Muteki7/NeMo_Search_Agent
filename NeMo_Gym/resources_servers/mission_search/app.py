# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the Apollo 13 mission-report search task.

Two jobs:
  - `POST /search_corpus` : the tool. Embeds the query and returns the top-k chunks from the Chroma index (step 3).
  - `POST /verify`        : the reward. Compares the final answer of the model with the expected answer.
`verify` does not read the index. It is a pure function of the request body, so REVERIFY_MODE is STATELESS.
"""
import asyncio
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import ClassVar, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.verifier_fixture import VerifierFixture


SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))  # so that `import embeddings` finds ./embeddings.py, however app.py is started


class MissionSearchResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # Set these as flat keys under the server block of the YAML file. A relative chroma_dir is read from this folder.
    chroma_dir: str = "chroma_index"
    collection_name: str = "example_collection"
    embedding_model: str = "avsolatorio/NoInstruct-small-Embedding-v0"
    top_k: int = 2  # the old AutoDidact loop used 2 results per search


# ---------------------------------------------------------------------------------------------------------------
# Tool: search_corpus
# ---------------------------------------------------------------------------------------------------------------
class SearchCorpusRequest(BaseModel):
    query: str


class SearchCorpusResponse(BaseModel):
    results: str


class CorpusSearcher:
    """The Chroma index plus the embedder. Loaded once. Needs chromadb, langchain-chroma, torch and transformers."""

    def __init__(self, chroma_dir: Path, collection_name: str, embedding_model: str, top_k: int):
        from langchain_chroma import Chroma
        from embeddings import CustomHuggingFaceEmbeddings

        if not chroma_dir.is_dir():
            raise FileNotFoundError(f"Chroma index not found at {chroma_dir}. Build it with scripts/build_index.py.")
        # The same class that built the index: documents use [CLS] pooling, queries use mean pooling.
        embeddings = CustomHuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )
        self.top_k = top_k

    def search(self, query: str) -> str:
        """Same output format as AutoDidact search_module.search: 'Result i:', the chunk text, then '------'."""
        results = self.vectorstore.similarity_search(query, k=self.top_k)
        return "".join(f"Result {i}:\n{doc.page_content}\n------\n" for i, doc in enumerate(results, start=1))


# ---------------------------------------------------------------------------------------------------------------
# Verify: reward in [0, 1]
# ---------------------------------------------------------------------------------------------------------------
def remove_reasoning(text: str) -> str:
    """Remove every <reasoning>...</reasoning> block, tags included (copied from AutoDidact rl_helpers.py)."""
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)


# The containment cap (see score_answer). 5 x + 15 was chosen on a dump of 40 trajectories: 3 x + 10 gave 0.29 and 0.35
# to two correct but wordy answers. A pasted chunk has 100 words or more, far above the cap for short answers.
LENGTH_CAP_FACTOR = 5
LENGTH_CAP_EXTRA = 15

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, delete punctuation, delete the articles a/an/the, collapse spaces.
    Punctuation is deleted (not replaced by a space), so '18:24' and '1824' are the same."""
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# Small words that carry no meaning. The F1 ignores them, so a long text of common words cannot earn reward.
# Negations (not, no) are NOT in the list: they change the meaning of an answer.
STOP_WORDS = frozenset(
    "the a an of in on at to for and or is was were are be been being by with from as that this these those it its "
    "he she they we you i his her their them our your there what which who whom how when where why did does do done "
    "had has have will would can could should may might shall also but if then than so into over after before "
    "during about up out".split()
)


def content_tokens(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS]


def token_f1(prediction_tokens: list, expected_tokens: list) -> float:
    """F1 over the content words (the stop words are dropped first). 0.0 if no content word is shared.
    If the expected answer has no content word at all, all its words count."""
    expected_content = content_tokens(expected_tokens) or expected_tokens
    prediction_content = content_tokens(prediction_tokens)
    common = sum((Counter(prediction_content) & Counter(expected_content)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_content)
    recall = common / len(expected_content)
    return 2 * precision * recall / (precision + recall)


def contains_run(haystack: list, needle: list) -> bool:
    """True if `needle` appears in `haystack` as one run of whole tokens."""
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def score_answer(final_text: Optional[str], expected: str) -> float:
    """Reward for one final answer.
      1.0   normalized exact match, or the expected answer appears in the answer as a run of whole words and the
            answer is short (at most LENGTH_CAP_FACTOR x expected length + LENGTH_CAP_EXTRA words, now 5 x + 15). The cap
            stops a model from pasting a whole chunk. Freeze it for a run: base and trained model must use the same value.
      F1    otherwise, the token-F1 over the content words (stop words dropped; 0.0 if no content word is shared)
      0.0   no final message, or nothing left after the reasoning is removed
    """
    if final_text is None:
        return 0.0
    prediction = normalize(remove_reasoning(final_text)).split()
    expected_tokens = normalize(expected).split()
    if not prediction or not expected_tokens:
        return 0.0
    if prediction == expected_tokens:
        return 1.0
    if len(prediction) <= LENGTH_CAP_FACTOR * len(expected_tokens) + LENGTH_CAP_EXTRA and contains_run(prediction, expected_tokens):
        return 1.0
    return token_f1(prediction, expected_tokens)


def final_message_text(output: list) -> Optional[str]:
    """Text of the LAST `message` item of the response output (the final answer), or None if there is none.
    In a multi-turn rollout the earlier items are function_call and function_call_output (and earlier messages)."""
    for item in reversed(output):
        if getattr(item, "type", None) == "message":
            return "".join(part.text for part in item.content if getattr(part, "type", None) == "output_text")
    return None


# ---------------------------------------------------------------------------------------------------------------
# Retrieval reward (run 2): credit for a search whose results contain the reference answer's facts
# ---------------------------------------------------------------------------------------------------------------
# reward = (1 - RETRIEVAL_WEIGHT) * R_answer + RETRIEVAL_WEIGHT * gate * ret
#   cov_k = share of the reference answer's content words found in the k-th search result (bag of words)
#   s_k   = ramp(cov_k): 0 below RAMP_LOW, rising linearly to 1 at RAMP_HIGH
#   ret   = max over searches k = 1, 2, ... of RETRIEVAL_DISCOUNT^(k-1) * s_k
#           (a later search is worth half as much, so extra searches cannot be farmed: after a hit they add nothing)
#   gate  = 1 if the trajectory ends with a final message, else 0 (spam to the turn cap, or searching and never answering, scores 0)
# A correct answer with a retrieval hit scores 1.0, a correct answer without one 0.75, a hit with no usable answer at most 0.25.
# Freeze these numbers for a run, like the length cap above.
RETRIEVAL_WEIGHT = 0.25
RETRIEVAL_DISCOUNT = 0.5
RAMP_LOW = 0.3
RAMP_HIGH = 0.6


def word_set(text: str) -> set:
    """Lowercase letters and digits, split on everything else (so `18:24` gives {18, 24})."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def coverage(result_text: str, expected: str) -> float:
    """cov_k: the share of the reference answer's content words (stop words removed) that appear in `result_text`."""
    needed = word_set(expected) - STOP_WORDS
    return len(needed & word_set(result_text)) / len(needed) if needed else 0.0


def ramp(cov: float) -> float:
    return min(1.0, max(0.0, (cov - RAMP_LOW) / (RAMP_HIGH - RAMP_LOW)))


def search_results(output: list) -> list:
    """The text of each tool result, in the order of the searches."""
    return [item.output for item in output if getattr(item, "type", None) == "function_call_output" and isinstance(item.output, str)]


def retrieval_score(output: list, expected: str) -> tuple:
    """(ret, best_cov, number of searches). ret is 0.0 when there was no search."""
    covs = [coverage(text, expected) for text in search_results(output)]
    ret = max((RETRIEVAL_DISCOUNT**k * ramp(c) for k, c in enumerate(covs)), default=0.0)
    return ret, max(covs, default=0.0), len(covs)


def combine(answer_reward: float, ret: float, answered: bool) -> float:
    return (1.0 - RETRIEVAL_WEIGHT) * answer_reward + RETRIEVAL_WEIGHT * (ret if answered else 0.0)


class MissionSearchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")  # data rows also carry difficulty, chunk_id, block
    answer: str


class MissionSearchVerifyResponse(BaseMultiRewardVerifyResponse):
    # `reward` is the combined value that the trainer uses. The parts are also top-level fields, so a caller can read the
    # answer-only reward (the yardstick for comparing runs) and the retrieval numbers.
    reward_answer: float = 0.0       # R_answer: the answer reward alone (what run 1 trained on)
    reward_retrieval: float = 0.0    # ret, before the gate
    retrieval_gate: float = 0.0      # 1.0 if the trajectory ended with a final message
    best_cov: float = 0.0            # the best cov_k over the searches, without the discount
    n_searches: float = 0.0


class MissionSearchVerifier:
    async def verify(self, body: MissionSearchVerifyRequest) -> MissionSearchVerifyResponse:
        final_text = final_message_text(body.response.output)
        answer_reward = score_answer(final_text, body.answer)
        ret, best_cov, n_searches = retrieval_score(body.response.output, body.answer)
        answered = final_text is not None
        return MissionSearchVerifyResponse(
            **body.model_dump(),
            reward=combine(answer_reward, ret, answered),
            reward_components={"answer": answer_reward, "retrieval": ret if answered else 0.0},
            reward_answer=answer_reward, reward_retrieval=ret, retrieval_gate=float(answered),
            best_cov=best_cov, n_searches=float(n_searches),
        )


# ---------------------------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------------------------
class MissionSearchResourcesServer(MissionSearchVerifier, SimpleResourcesServer):
    ray_enabled = False

    config: MissionSearchResourcesServerConfig
    _searcher: Optional[CorpusSearcher] = PrivateAttr(default=None)  # set in setup_webserver

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        chroma_dir = Path(self.config.chroma_dir)
        self._searcher = CorpusSearcher(
            chroma_dir if chroma_dir.is_absolute() else SERVER_DIR / chroma_dir,
            self.config.collection_name,
            self.config.embedding_model,
            self.config.top_k,
        )
        # The route name must equal the tool name in the data rows ("search_corpus"): simple_agent and our
        # rollout_func both POST a tool call to /{tool name}.
        app.post("/search_corpus")(self.search_corpus)

        return app

    async def search_corpus(self, body: SearchCorpusRequest) -> SearchCorpusResponse:
        # The CPU embedding blocks. Run it in a thread so that parallel rollouts do not stall the event loop.
        results = await asyncio.to_thread(self._searcher.search, body.query)
        return SearchCorpusResponse(results=results)


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=MissionSearchVerifier,
    request_model=MissionSearchVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    MissionSearchResourcesServer.run_webserver()
