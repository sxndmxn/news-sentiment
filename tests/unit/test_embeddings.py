"""Tests for embeddings module."""

import numpy as np
import pytest

from news_sentiment.analysis.embeddings import SentenceEmbedder, get_embedder
from news_sentiment.domain.models import Article
from news_sentiment.domain.value_objects import BiasLabel


class TestSentenceEmbedder:
    """Tests for SentenceEmbedder class."""

    def test_init_default_model(self) -> None:
        """Test initialization with default model."""
        embedder = SentenceEmbedder()
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder._model is None  # Lazy loading

    def test_init_custom_model(self) -> None:
        """Test initialization with custom model."""
        embedder = SentenceEmbedder("custom-model")
        assert embedder.model_name == "custom-model"

    def test_is_available_detection(self) -> None:
        """Test availability detection."""
        embedder = SentenceEmbedder()
        # Should be either True or False, not None
        assert isinstance(embedder.is_available, bool)

    def test_compute_similarity_identical(self) -> None:
        """Test similarity of identical vectors."""
        embedder = SentenceEmbedder()
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = embedder.compute_similarity(vec, vec)
        assert sim == pytest.approx(1.0)

    def test_compute_similarity_orthogonal(self) -> None:
        """Test similarity of orthogonal vectors."""
        embedder = SentenceEmbedder()
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = embedder.compute_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0)

    def test_compute_similarity_opposite(self) -> None:
        """Test similarity of opposite vectors."""
        embedder = SentenceEmbedder()
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        sim = embedder.compute_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0)

    def test_compute_similarity_zero_vector(self) -> None:
        """Test similarity with zero vector."""
        embedder = SentenceEmbedder()
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sim = embedder.compute_similarity(vec1, vec2)
        assert sim == 0.0

    def test_find_similar_empty(self) -> None:
        """Test find_similar with empty candidates."""
        embedder = SentenceEmbedder()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = embedder.find_similar(query, [])
        assert results == []

    def test_find_similar_basic(self) -> None:
        """Test find_similar with basic candidates."""
        embedder = SentenceEmbedder()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            ("a", np.array([1.0, 0.0, 0.0], dtype=np.float32)),  # Identical
            ("b", np.array([0.5, 0.5, 0.0], dtype=np.float32)),  # Similar
            ("c", np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # Orthogonal
        ]
        results = embedder.find_similar(query, candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "a"  # Most similar
        assert results[0][1] == pytest.approx(1.0)

    def test_embed_texts_empty(self) -> None:
        """Test embed_texts with empty list."""
        embedder = SentenceEmbedder()
        result = embedder.embed_texts([])
        assert result is not None
        assert len(result) == 0


class TestGetEmbedder:
    """Tests for get_embedder function."""

    def test_returns_embedder(self) -> None:
        """Test that get_embedder returns a SentenceEmbedder."""
        embedder = get_embedder()
        assert isinstance(embedder, SentenceEmbedder)

    def test_singleton_behavior(self) -> None:
        """Test that get_embedder returns same instance."""
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2


@pytest.mark.skipif(
    not SentenceEmbedder().is_available, reason="sentence-transformers not installed"
)
class TestSentenceEmbedderWithModel:
    """Tests that require the actual model to be loaded."""

    def test_embed_text(self) -> None:
        """Test embedding a single text."""
        embedder = SentenceEmbedder()
        embedding = embedder.embed_text("Hello world")

        assert embedding is not None
        assert embedding.shape[0] == 384  # Default model dimension
        assert embedding.dtype == np.float32

    def test_embed_texts_batch(self) -> None:
        """Test embedding multiple texts."""
        embedder = SentenceEmbedder()
        texts = ["Hello world", "Goodbye world", "Test sentence"]
        embeddings = embedder.embed_texts(texts)

        assert embeddings is not None
        assert embeddings.shape == (3, 384)

    def test_embed_article(self) -> None:
        """Test embedding an article."""
        embedder = SentenceEmbedder()
        article = Article(
            source_id="test",
            url="https://example.com/article",
            title="Test Article Title",
            content="This is a test article with some content for embedding.",
            source_bias=BiasLabel.CENTER,
        )

        embedding = embedder.embed_article(article)
        assert embedding is not None
        assert embedding.shape[0] == 384

    def test_similar_texts_have_high_similarity(self) -> None:
        """Test that similar texts have high cosine similarity."""
        embedder = SentenceEmbedder()

        text1 = "The economy is growing rapidly"
        text2 = "Economic growth is accelerating"
        text3 = "The cat sat on the mat"

        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)

        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13
        assert sim_12 > 0.5  # Related texts
