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
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
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


class MissionSearchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")  # data rows also carry difficulty, chunk_id, block
    answer: str


class MissionSearchVerifier:
    async def verify(self, body: MissionSearchVerifyRequest) -> BaseVerifyResponse:
        reward = score_answer(final_message_text(body.response.output), body.answer)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


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
