"""FAISS vector store with Sentence Transformers embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from api.schemas import SearchResult, SeedDocumentsResponse
from config import get_settings
from llm import call_llm_json

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.json"
SEED_COUNT = 50


class VectorStore:
    """FAISS-backed vector store for document retrieval."""

    def __init__(self) -> None:
        settings = get_settings()
        self.index_path = Path(settings.faiss_index_path)
        self.embedding_model_name = settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[dict[str, Any]] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model %s", self.embedding_model_name)
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def _index_exists(self) -> bool:
        return (self.index_path / INDEX_FILE).exists() and (
            self.index_path / METADATA_FILE
        ).exists()

    def _save(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path / INDEX_FILE))
        with (self.index_path / METADATA_FILE).open("w", encoding="utf-8") as f:
            json.dump(self._documents, f)

    def _load(self) -> None:
        index_file = self.index_path / INDEX_FILE
        meta_file = self.index_path / METADATA_FILE
        self._index = faiss.read_index(str(index_file))
        with meta_file.open("r", encoding="utf-8") as f:
            self._documents = json.load(f)
        logger.info("Loaded FAISS index with %d documents", len(self._documents))

    async def _generate_seed_documents(self) -> list[dict[str, str]]:
        """Use Gemini to generate seed knowledge documents."""
        prompt = f"""Generate exactly {SEED_COUNT} short factual documents for a knowledge base.
Cover health, legal, and financial general knowledge (roughly equal split).
Each document should be 2-4 sentences of general, safe, educational information.

Return JSON with a "documents" array where each item has:
- "content": the document text
- "category": one of "health", "legal", "financial"
- "title": a short title"""

        result = await call_llm_json(prompt, SeedDocumentsResponse)
        docs = []
        for i, doc in enumerate(result.data.documents[:SEED_COUNT]):
            docs.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": {
                        "category": doc.get("category", "general"),
                        "title": doc.get("title", f"doc_{i}"),
                        "url": f"internal://seed/{i}",
                    },
                }
            )
        return docs

    def _build_fallback_seed(self) -> list[dict[str, str]]:
        """Fallback seed documents when Gemini is unavailable."""
        categories = ["health", "legal", "financial"]
        docs = []
        for i in range(SEED_COUNT):
            cat = categories[i % 3]
            docs.append(
                {
                    "content": (
                        f"This is a general knowledge document about {cat} topic {i + 1}. "
                        f"It provides educational information and recommends consulting "
                        f"a qualified professional for specific advice."
                    ),
                    "metadata": {
                        "category": cat,
                        "title": f"{cat}_knowledge_{i + 1}",
                        "url": f"internal://seed/{i}",
                    },
                }
            )
        return docs

    async def ensure_index(self) -> None:
        """Create the FAISS index if it does not exist."""
        if self._index_exists():
            self._load()
            return

        logger.info("Building new FAISS index...")
        try:
            docs = await self._generate_seed_documents()
        except Exception as exc:
            logger.warning("Gemini seed generation failed, using fallback: %s", exc)
            docs = self._build_fallback_seed()

        self.add_documents(docs)
        self._save()
        logger.info("FAISS index built with %d documents", len(self._documents))

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Add documents to the FAISS index."""
        if not docs:
            return

        texts = [d["content"] for d in docs]
        embeddings = self._embed(texts)

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(embeddings)
        self._documents.extend(docs)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Search the FAISS index for the top-k most relevant chunks."""
        if self._index is None or not self._documents:
            if self._index_exists():
                self._load()
            else:
                return []

        query_vec = self._embed([query])
        scores, indices = self._index.search(query_vec, min(k, len(self._documents)))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            results.append(
                SearchResult(
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    score=float(score),
                )
            )
        return results
