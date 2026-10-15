"""Embedding model for the search tool and for the index build.

Copied from AutoDidact/embeddings.py (Step2_Files/embeddings.py). Changes from the original:
  1. `Embeddings` is imported from `langchain_core.embeddings` (the old `langchain.embeddings.base` path is gone).
  2. The device defaults to "cpu" (was: CUDA if available). The GPU belongs to training; this model is tiny.
  3. `embed_documents` works in batches (default 32) instead of padding the whole list into one batch.
The pooling is unchanged: [CLS] vectors for documents ("sentence" mode), mean pooling for queries ("query" mode).
Documents and queries must always go through this same class, or the vectors do not match.
"""
from typing import List, Union

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

# Set a default model here
DEFAULT_MODEL_NAME = "avsolatorio/NoInstruct-small-Embedding-v0"


class CustomHuggingFaceEmbeddings(Embeddings):
    """
    A custom embeddings class that wraps a Hugging Face model for generating embeddings.

    Supports two modes:
    - "sentence": uses the [CLS] token representation for sentence/document embeddings.
    - "query": uses mean pooling over tokens (weighted by the attention mask) for query embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, default_mode: str = "sentence",
                 device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.default_mode = default_mode  # "sentence" or "query"
        self.model.eval()  # Set model to evaluation mode

    def get_embedding(self, text: Union[str, List[str]], mode: str = None):
        if mode is None:
            mode = self.default_mode
        assert mode in ("query", "sentence"), f"Unsupported mode: {mode}. Only 'query' and 'sentence' are supported."

        # Ensure we are working with a list of texts
        if isinstance(text, str):
            text = [text]

        # Tokenize the input texts
        inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Move the input tensors to the same device as the model
        inp = {key: value.to(self.device) for key, value in inp.items()}

        # Forward pass (no gradients needed)
        with torch.no_grad():
            output = self.model(**inp)

        if mode == "query":
            # Mean pooling: weight by attention mask and average across tokens
            vectors = output.last_hidden_state * inp["attention_mask"].unsqueeze(2)
            vectors = vectors.sum(dim=1) / inp["attention_mask"].sum(dim=-1).view(-1, 1)
        else:
            # Sentence/document embedding: use the [CLS] token (first token) representation
            vectors = output.last_hidden_state[:, 0, :]
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of documents (using sentence mode), `batch_size` texts at a time.
        """
        out: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors = self.get_embedding(texts[start:start + self.batch_size], mode="sentence")
            out.extend(vectors.cpu().numpy().tolist())
        return out

    def embed_query(self, text: str) -> List[float]:
        """
        Compute an embedding for a single query.
        """
        vector = self.get_embedding(text, mode="query")
        return vector.cpu().numpy()[0].tolist()


# For quick testing
if __name__ == "__main__":
    embeddings = CustomHuggingFaceEmbeddings()
    doc_embeddings = embeddings.embed_documents(["The Apollo 13 service module oxygen tank ruptured.", "A short test."])
    print("Document embeddings:", len(doc_embeddings), "vectors of size", len(doc_embeddings[0]))
    query_embedding = embeddings.embed_query("What ruptured on Apollo 13?")
    print("Query embedding: size", len(query_embedding))
