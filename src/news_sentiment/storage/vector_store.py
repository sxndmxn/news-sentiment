"""Vector store module - ChromaDB for semantic search."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from news_sentiment.domain.models import Article

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for article embeddings.

    Provides semantic search capabilities for finding similar articles.
    """

    COLLECTION_NAME = "articles"
    DEFAULT_PERSIST_DIR = "data/chroma"

    def __init__(self, persist_directory: str | None = None) -> None:
        """Initialize vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence.
                Defaults to data/chroma.
        """
        if persist_directory is None:
            data_dir = os.getenv("NEWS_SENTIMENT_DATA_DIR", "data")
            persist_directory = str(Path(data_dir) / "chroma")

        self.persist_directory = persist_directory
        self._client: Any | None = None
        self._collection: Any | None = None
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        if self._available is None:
            try:
                import chromadb  # noqa: F401

                self._available = True
            except ImportError:
                logger.warning(
                    "chromadb not installed. "
                    "Install with: pip install news-sentiment[vector]"
                )
                self._available = False
        return self._available

    @property
    def client(self) -> Any | None:
        """Lazy-load ChromaDB client."""
        if self._client is None and self.is_available:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        return self._client

    @property
    def collection(self) -> Any | None:
        """Get or create the articles collection."""
        if self._collection is None and self.client is not None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection '{self.COLLECTION_NAME}' ready")
        return self._collection

    def add_article(
        self,
        article_id: str,
        embedding: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add a single article embedding to the store.

        Args:
            article_id: Unique article identifier.
            embedding: Article embedding vector.
            metadata: Optional metadata (title, source, etc.).

        Returns:
            True if successful, False otherwise.
        """
        if self.collection is None:
            return False

        try:
            self.collection.upsert(
                ids=[article_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata] if metadata else None,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add article {article_id}: {e}")
            return False

    def add_articles(
        self,
        articles: list[tuple[str, NDArray[np.float32], dict[str, Any] | None]],
    ) -> int:
        """Add multiple article embeddings in batch.

        Args:
            articles: List of (article_id, embedding, metadata) tuples.

        Returns:
            Number of articles successfully added.
        """
        if self.collection is None or not articles:
            return 0

        ids = []
        embeddings = []
        metadatas = []

        for article_id, embedding, metadata in articles:
            ids.append(article_id)
            embeddings.append(embedding.tolist())
            metadatas.append(metadata or {})

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to add {len(articles)} articles: {e}")
            return 0

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar articles.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional filter conditions.

        Returns:
            List of results with id, distance, and metadata.
        """
        if self.collection is None:
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
                include=["distances", "metadatas"],
            )

            output = []
            if results["ids"] and results["ids"][0]:
                for i, article_id in enumerate(results["ids"][0]):
                    output.append({
                        "id": article_id,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    })
            return output
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_text(
        self,
        text: str,
        embedder: Any,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar articles using text query.

        Args:
            text: Text query to search for.
            embedder: SentenceEmbedder instance for encoding.
            top_k: Number of results to return.
            where: Optional filter conditions.

        Returns:
            List of results with id, distance, and metadata.
        """
        embedding = embedder.embed_text(text)
        if embedding is None:
            return []
        return self.search(embedding, top_k, where)

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        """Get a specific article by ID.

        Args:
            article_id: Article identifier.

        Returns:
            Article data or None if not found.
        """
        if self.collection is None:
            return None

        try:
            result = self.collection.get(
                ids=[article_id],
                include=["embeddings", "metadatas"],
            )
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "embedding": np.array(result["embeddings"][0]) if result["embeddings"] else None,
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception as e:
            logger.error(f"Failed to get article {article_id}: {e}")
        return None

    def delete_article(self, article_id: str) -> bool:
        """Delete an article from the store.

        Args:
            article_id: Article identifier.

        Returns:
            True if successful, False otherwise.
        """
        if self.collection is None:
            return False

        try:
            self.collection.delete(ids=[article_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete article {article_id}: {e}")
            return False

    def count(self) -> int:
        """Get total number of articles in store."""
        if self.collection is None:
            return 0
        return self.collection.count()

    def clear(self) -> bool:
        """Clear all articles from the store."""
        if self.client is None:
            return False

        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store(persist_directory: str | None = None) -> VectorStore:
    """Get or create singleton vector store instance.

    Args:
        persist_directory: Persistence directory (only used on first call).

    Returns:
        VectorStore instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(persist_directory)
    return _vector_store


class ArticleIndexer:
    """High-level indexer for articles combining embeddings and vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Any | None = None,
    ) -> None:
        """Initialize indexer.

        Args:
            vector_store: VectorStore instance.
            embedder: SentenceEmbedder instance.
        """
        from news_sentiment.analysis.embeddings import get_embedder

        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()

    def index_article(self, article: Article) -> bool:
        """Index a single article.

        Args:
            article: Article to index.

        Returns:
            True if successful.
        """
        embedding = self.embedder.embed_article(article)
        if embedding is None:
            return False

        metadata = {
            "title": article.title[:500],
            "source_id": article.source_id,
            "source_name": article.source_name or "",
            "source_bias": article.source_bias.value if article.source_bias else "",
            "published_at": article.published_at.isoformat() if article.published_at else "",
            "word_count": article.word_count,
        }

        return self.vector_store.add_article(article.id, embedding, metadata)

    def index_articles(
        self,
        articles: list[Article],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> int:
        """Index multiple articles.

        Args:
            articles: Articles to index.
            batch_size: Embedding batch size.
            show_progress: Show progress bar.

        Returns:
            Number of articles indexed.
        """
        if not articles:
            return 0

        # Get embeddings
        article_embeddings = self.embedder.embed_articles(
            articles, batch_size, show_progress
        )

        if not article_embeddings:
            return 0

        # Build batch for vector store
        batch = []
        article_map = {a.id: a for a in articles}

        for article_id, embedding in article_embeddings:
            article = article_map.get(article_id)
            if article:
                metadata = {
                    "title": article.title[:500],
                    "source_id": article.source_id,
                    "source_name": article.source_name or "",
                    "source_bias": article.source_bias.value if article.source_bias else "",
                    "published_at": article.published_at.isoformat() if article.published_at else "",
                    "word_count": article.word_count,
                }
                batch.append((article_id, embedding, metadata))

        return self.vector_store.add_articles(batch)

    def search_similar(
        self,
        query: str,
        top_k: int = 10,
        bias_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for articles similar to query text.

        Args:
            query: Text query.
            top_k: Number of results.
            bias_filter: Optional bias label filter.

        Returns:
            Search results.
        """
        where = {"source_bias": bias_filter} if bias_filter else None
        return self.vector_store.search_by_text(query, self.embedder, top_k, where)

    def find_similar_to_article(
        self,
        article: Article,
        top_k: int = 10,
        exclude_same_source: bool = True,
    ) -> list[dict[str, Any]]:
        """Find articles similar to a given article.

        Args:
            article: Reference article.
            top_k: Number of results.
            exclude_same_source: Exclude articles from same source.

        Returns:
            Search results.
        """
        embedding = self.embedder.embed_article(article)
        if embedding is None:
            return []

        # Get more results to filter
        results = self.vector_store.search(embedding, top_k + 5)

        # Filter out the article itself and optionally same source
        filtered = []
        for r in results:
            if r["id"] == article.id:
                continue
            if exclude_same_source and r.get("metadata", {}).get("source_id") == article.source_id:
                continue
            filtered.append(r)
            if len(filtered) >= top_k:
                break

        return filtered
