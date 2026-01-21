"""Tests for vector store module."""

import numpy as np
import pytest

from news_sentiment.storage.vector_store import VectorStore, get_vector_store


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_init_default_path(self) -> None:
        """Test initialization with default path."""
        store = VectorStore()
        assert "chroma" in store.persist_directory

    def test_init_custom_path(self) -> None:
        """Test initialization with custom path."""
        store = VectorStore("/custom/path")
        assert store.persist_directory == "/custom/path"

    def test_is_available_detection(self) -> None:
        """Test availability detection."""
        store = VectorStore()
        assert isinstance(store.is_available, bool)

    def test_count_no_client(self) -> None:
        """Test count returns 0 when unavailable."""
        store = VectorStore()
        store._available = False
        assert store.count() == 0

    def test_search_no_collection(self) -> None:
        """Test search returns empty when unavailable."""
        store = VectorStore()
        store._available = False
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query)
        assert results == []


class TestGetVectorStore:
    """Tests for get_vector_store function."""

    def test_returns_vector_store(self) -> None:
        """Test that get_vector_store returns a VectorStore."""
        store = get_vector_store()
        assert isinstance(store, VectorStore)


@pytest.mark.skipif(
    not VectorStore().is_available,
    reason="chromadb not installed"
)
class TestVectorStoreWithChroma:
    """Tests that require ChromaDB to be available."""

    @pytest.fixture
    def temp_store(self, tmp_path) -> VectorStore:
        """Create a temporary vector store."""
        return VectorStore(str(tmp_path / "chroma"))

    def test_add_and_retrieve_article(self, temp_store: VectorStore) -> None:
        """Test adding and retrieving an article."""
        embedding = np.random.randn(384).astype(np.float32)
        metadata = {"title": "Test Article", "source": "test"}

        success = temp_store.add_article("article-1", embedding, metadata)
        assert success

        result = temp_store.get_article("article-1")
        assert result is not None
        assert result["id"] == "article-1"
        assert result["metadata"]["title"] == "Test Article"

    def test_add_articles_batch(self, temp_store: VectorStore) -> None:
        """Test batch adding articles."""
        articles = [
            ("a1", np.random.randn(384).astype(np.float32), {"title": "Article 1"}),
            ("a2", np.random.randn(384).astype(np.float32), {"title": "Article 2"}),
            ("a3", np.random.randn(384).astype(np.float32), {"title": "Article 3"}),
        ]

        count = temp_store.add_articles(articles)
        assert count == 3
        assert temp_store.count() == 3

    def test_search_finds_similar(self, temp_store: VectorStore) -> None:
        """Test that search finds similar vectors."""
        # Add some articles with known embeddings
        base = np.zeros(384, dtype=np.float32)
        base[0] = 1.0

        similar = base.copy()
        similar[1] = 0.1  # Slightly different

        different = np.zeros(384, dtype=np.float32)
        different[100] = 1.0  # Very different direction

        temp_store.add_article("similar", similar, {"title": "Similar"})
        temp_store.add_article("different", different, {"title": "Different"})

        # Search with base vector
        results = temp_store.search(base, top_k=2)

        assert len(results) == 2
        # Similar should be first (higher similarity / lower distance)
        assert results[0]["id"] == "similar"

    def test_delete_article(self, temp_store: VectorStore) -> None:
        """Test deleting an article."""
        embedding = np.random.randn(384).astype(np.float32)
        temp_store.add_article("to-delete", embedding, {})

        assert temp_store.count() == 1

        success = temp_store.delete_article("to-delete")
        assert success
        assert temp_store.count() == 0

    def test_clear_collection(self, temp_store: VectorStore) -> None:
        """Test clearing all articles."""
        for i in range(5):
            embedding = np.random.randn(384).astype(np.float32)
            temp_store.add_article(f"article-{i}", embedding, {})

        assert temp_store.count() == 5

        success = temp_store.clear()
        assert success
        # After clear, collection is deleted - need to re-access
        assert temp_store.count() == 0
