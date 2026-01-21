"""Embeddings module - Sentence embeddings for semantic search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from news_sentiment.domain.models import Article

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        """Encode sentences to embeddings."""
        ...


class SentenceEmbedder:
    """Generate sentence embeddings for semantic search.

    Uses sentence-transformers for high-quality embeddings.
    Falls back to a simple average if transformers unavailable.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Dimension for default model

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize embedder with specified model.

        Args:
            model_name: HuggingFace model name. Defaults to all-MiniLM-L6-v2.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: EmbeddingModel | None = None
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Check if sentence-transformers is available."""
        if self._available is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401

                self._available = True
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install news-sentiment[transformers]"
                )
                self._available = False
        return self._available

    @property
    def model(self) -> EmbeddingModel | None:
        """Lazy-load the embedding model."""
        if self._model is None and self.is_available:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> NDArray[np.float32] | None:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector or None if model unavailable.
        """
        if not self.is_available or self.model is None:
            return None

        embedding = self.model.encode([text], show_progress_bar=False)
        return embedding[0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> NDArray[np.float32] | None:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            Array of embeddings or None if model unavailable.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        if not self.is_available or self.model is None:
            return None

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return embeddings

    def embed_article(self, article: Article) -> NDArray[np.float32] | None:
        """Generate embedding for an article.

        Combines title and content for richer representation.

        Args:
            article: Article to embed.

        Returns:
            Embedding vector or None if model unavailable.
        """
        text = f"{article.title}. {article.content[:2000]}"
        return self.embed_text(text)

    def embed_articles(
        self,
        articles: list[Article],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[tuple[str, NDArray[np.float32]]]:
        """Generate embeddings for multiple articles.

        Args:
            articles: List of articles to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            List of (article_id, embedding) tuples.
        """
        if not articles:
            return []

        texts = [f"{a.title}. {a.content[:2000]}" for a in articles]
        embeddings = self.embed_texts(texts, batch_size, show_progress)

        if embeddings is None:
            return []

        return [(a.id, emb) for a, emb in zip(articles, embeddings, strict=True)]

    def compute_similarity(
        self,
        embedding1: NDArray[np.float32],
        embedding2: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def find_similar(
        self,
        query_embedding: NDArray[np.float32],
        candidate_embeddings: list[tuple[str, NDArray[np.float32]]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find most similar items to query.

        Args:
            query_embedding: Query embedding vector.
            candidate_embeddings: List of (id, embedding) tuples.
            top_k: Number of results to return.

        Returns:
            List of (id, similarity_score) tuples sorted by similarity.
        """
        if not candidate_embeddings:
            return []

        similarities = []
        for item_id, embedding in candidate_embeddings:
            sim = self.compute_similarity(query_embedding, embedding)
            similarities.append((item_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Singleton instance for convenience
_embedder: SentenceEmbedder | None = None


def get_embedder(model_name: str | None = None) -> SentenceEmbedder:
    """Get or create singleton embedder instance.

    Args:
        model_name: Model name (only used on first call).

    Returns:
        SentenceEmbedder instance.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceEmbedder(model_name)
    return _embedder
