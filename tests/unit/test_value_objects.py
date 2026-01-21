"""Tests for domain value objects."""

import pytest

from news_sentiment.domain.value_objects import (
    BiasLabel,
    ContentHash,
    LoadedLanguageScore,
    SentimentScore,
)


class TestBiasLabel:
    """Tests for BiasLabel enum."""

    def test_all_labels_exist(self) -> None:
        """Test all expected bias labels are defined."""
        expected = {
            "far_left",
            "left",
            "center_left",
            "center",
            "center_right",
            "right",
            "far_right",
        }
        actual = {label.value for label in BiasLabel}
        assert actual == expected

    def test_from_string_exact_match(self) -> None:
        """Test conversion from exact string value."""
        assert BiasLabel.from_string("center_left") == BiasLabel.CENTER_LEFT
        assert BiasLabel.from_string("far_right") == BiasLabel.FAR_RIGHT

    def test_from_string_with_hyphen(self) -> None:
        """Test conversion handles hyphens."""
        assert BiasLabel.from_string("center-left") == BiasLabel.CENTER_LEFT
        assert BiasLabel.from_string("far-right") == BiasLabel.FAR_RIGHT

    def test_from_string_with_spaces(self) -> None:
        """Test conversion handles spaces."""
        assert BiasLabel.from_string("center left") == BiasLabel.CENTER_LEFT

    def test_from_string_invalid(self) -> None:
        """Test conversion raises for invalid input."""
        with pytest.raises(ValueError, match="Unknown bias label"):
            BiasLabel.from_string("invalid_label")


class TestSentimentScore:
    """Tests for SentimentScore value object."""

    def test_positive_sentiment_label(self) -> None:
        """Test positive sentiment label detection."""
        score = SentimentScore(compound=0.5, positive=0.6, negative=0.1, neutral=0.3)
        assert score.sentiment_label == "positive"

    def test_negative_sentiment_label(self) -> None:
        """Test negative sentiment label detection."""
        score = SentimentScore(compound=-0.5, positive=0.1, negative=0.6, neutral=0.3)
        assert score.sentiment_label == "negative"

    def test_neutral_sentiment_label(self) -> None:
        """Test neutral sentiment label detection."""
        score = SentimentScore(compound=0.01, positive=0.1, negative=0.1, neutral=0.8)
        assert score.sentiment_label == "neutral"

    def test_is_strong_positive(self) -> None:
        """Test strong positive sentiment detection."""
        score = SentimentScore(compound=0.6, positive=0.7, negative=0.0, neutral=0.3)
        assert score.is_strong is True

    def test_is_strong_negative(self) -> None:
        """Test strong negative sentiment detection."""
        score = SentimentScore(compound=-0.6, positive=0.0, negative=0.7, neutral=0.3)
        assert score.is_strong is True

    def test_is_not_strong(self) -> None:
        """Test weak sentiment detection."""
        score = SentimentScore(compound=0.2, positive=0.3, negative=0.1, neutral=0.6)
        assert score.is_strong is False

    def test_neutral_score_factory(self) -> None:
        """Test neutral score factory method."""
        score = SentimentScore.neutral_score()
        assert score.compound == 0.0
        assert score.neutral == 1.0
        assert score.sentiment_label == "neutral"

    def test_immutability(self) -> None:
        """Test score is immutable."""
        score = SentimentScore(compound=0.5, positive=0.6, negative=0.1, neutral=0.3)
        with pytest.raises(Exception):  # ValidationError for frozen model
            score.compound = 0.0  # type: ignore

    def test_validation_bounds(self) -> None:
        """Test value bounds validation."""
        with pytest.raises(ValueError):
            SentimentScore(compound=1.5, positive=0.5, negative=0.5, neutral=0.0)


class TestContentHash:
    """Tests for ContentHash value object."""

    def test_from_content(self) -> None:
        """Test hash creation from content."""
        content = "This is test content"
        hash_obj = ContentHash.from_content(content)
        assert hash_obj.algorithm == "sha256"
        assert len(hash_obj.value) == 64  # SHA256 hex length

    def test_hash_consistency(self) -> None:
        """Test same content produces same hash."""
        content = "Test content for hashing"
        hash1 = ContentHash.from_content(content)
        hash2 = ContentHash.from_content(content)
        assert hash1.value == hash2.value

    def test_hash_whitespace_normalization(self) -> None:
        """Test whitespace is normalized before hashing."""
        content1 = "Hello   world"
        content2 = "Hello world"
        hash1 = ContentHash.from_content(content1)
        hash2 = ContentHash.from_content(content2)
        assert hash1.value == hash2.value

    def test_matches(self) -> None:
        """Test hash matching."""
        hash1 = ContentHash.from_content("test")
        hash2 = ContentHash.from_content("test")
        hash3 = ContentHash.from_content("different")
        assert hash1.matches(hash2) is True
        assert hash1.matches(hash3) is False

    def test_invalid_hash_value(self) -> None:
        """Test validation rejects invalid hex strings."""
        with pytest.raises(ValueError, match="hexadecimal"):
            ContentHash(algorithm="sha256", value="not-a-valid-hex")


class TestLoadedLanguageScore:
    """Tests for LoadedLanguageScore value object."""

    def test_total_loaded(self) -> None:
        """Test total loaded word count."""
        score = LoadedLanguageScore(
            hedges_count=5, intensifiers_count=3, weasel_words_count=2, total_words=100
        )
        assert score.total_loaded == 10

    def test_loaded_ratio(self) -> None:
        """Test loaded language ratio calculation."""
        score = LoadedLanguageScore(
            hedges_count=5, intensifiers_count=3, weasel_words_count=2, total_words=100
        )
        assert score.loaded_ratio == 0.1

    def test_loaded_ratio_zero_words(self) -> None:
        """Test ratio is zero when no words."""
        score = LoadedLanguageScore(hedges_count=0, intensifiers_count=0, weasel_words_count=0)
        assert score.loaded_ratio == 0.0

    def test_is_high_loaded(self) -> None:
        """Test high loaded language detection."""
        high = LoadedLanguageScore(
            hedges_count=6, intensifiers_count=0, weasel_words_count=0, total_words=100
        )
        low = LoadedLanguageScore(
            hedges_count=1, intensifiers_count=0, weasel_words_count=0, total_words=100
        )
        assert high.is_high_loaded is True
        assert low.is_high_loaded is False
