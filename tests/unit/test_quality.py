"""Tests for quality validation."""

import pytest

from news_sentiment.domain.exceptions import ContentTooShortError
from news_sentiment.ingestion.quality import (
    ContentLengthValidator,
    DuplicateDetector,
    HTMLCleaner,
    LanguageValidator,
    QualityPipeline,
)


class TestContentLengthValidator:
    """Tests for ContentLengthValidator."""

    def test_valid_length(self) -> None:
        """Test content within bounds passes."""
        validator = ContentLengthValidator(min_length=10, max_length=100)
        assert validator.validate("x" * 50) is True

    def test_too_short(self) -> None:
        """Test short content raises error."""
        validator = ContentLengthValidator(min_length=100)
        with pytest.raises(ContentTooShortError) as exc_info:
            validator.validate("short")
        assert exc_info.value.actual_length == 5
        assert exc_info.value.min_length == 100

    def test_whitespace_trimmed(self) -> None:
        """Test whitespace is trimmed for length check."""
        validator = ContentLengthValidator(min_length=10)
        with pytest.raises(ContentTooShortError):
            validator.validate("   short   ")


class TestLanguageValidator:
    """Tests for LanguageValidator."""

    def test_english_content_passes(self) -> None:
        """Test English content is detected."""
        validator = LanguageValidator()
        content = "The quick brown fox jumps over the lazy dog. This is a test."
        assert validator.validate(content) is True

    def test_threshold_boundary(self) -> None:
        """Test content at threshold boundary."""
        validator = LanguageValidator(threshold=0.1)
        # Mix of English and gibberish
        content = "the a is xyz abc qrs lmn opq"
        # Should pass if enough English markers
        assert validator.validate(content) is True


class TestHTMLCleaner:
    """Tests for HTMLCleaner."""

    def test_removes_tags(self) -> None:
        """Test HTML tags are removed."""
        cleaner = HTMLCleaner()
        result = cleaner.clean("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "world" in result

    def test_converts_entities(self) -> None:
        """Test common HTML entities are converted."""
        cleaner = HTMLCleaner()
        result = cleaner.clean("Hello&nbsp;&amp;&quot;world&quot;")
        assert "&nbsp;" not in result
        assert "&amp;" not in result
        assert "&" in result
        assert '"' in result

    def test_normalizes_whitespace(self) -> None:
        """Test multiple whitespace is collapsed."""
        cleaner = HTMLCleaner()
        result = cleaner.clean("Hello    world\n\ntest")
        assert "    " not in result
        assert "\n\n" not in result


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    def test_fingerprint_generation(self) -> None:
        """Test fingerprint is generated consistently."""
        detector = DuplicateDetector()
        fp1 = detector.get_fingerprint("Hello World!")
        fp2 = detector.get_fingerprint("Hello World!")
        assert fp1 == fp2

    def test_fingerprint_normalized(self) -> None:
        """Test fingerprint normalizes content."""
        detector = DuplicateDetector()
        fp1 = detector.get_fingerprint("Hello  World!")
        fp2 = detector.get_fingerprint("hello world")
        assert fp1 == fp2

    def test_similar_content_detected(self) -> None:
        """Test similar content is detected."""
        detector = DuplicateDetector(similarity_threshold=0.8)
        content1 = "The quick brown fox jumps over the lazy dog today"
        content2 = "The quick brown fox jumps over the lazy dog yesterday"
        assert detector.is_similar(content1, content2) is True

    def test_different_content_not_similar(self) -> None:
        """Test different content is not flagged."""
        detector = DuplicateDetector(similarity_threshold=0.8)
        content1 = "Breaking news about technology today"
        content2 = "Sports update from the weekend game"
        assert detector.is_similar(content1, content2) is False


class TestQualityPipeline:
    """Tests for QualityPipeline."""

    def test_valid_content_passes(self) -> None:
        """Test valid content passes all checks."""
        pipeline = QualityPipeline(min_content_length=10)
        result = pipeline.validate("This is valid content that should pass all checks.")
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_html_cleaned(self) -> None:
        """Test HTML is cleaned from content."""
        pipeline = QualityPipeline(min_content_length=10, clean_html=True)
        result = pipeline.validate("<p>This is <b>HTML</b> content.</p>")
        assert result.is_valid is True
        assert "<" not in (result.cleaned_content or "")

    def test_short_content_fails(self) -> None:
        """Test short content fails validation."""
        pipeline = QualityPipeline(min_content_length=100)
        result = pipeline.validate("Short")
        assert result.is_valid is False
        assert len(result.issues) > 0

    def test_content_truncated(self) -> None:
        """Test long content is truncated."""
        pipeline = QualityPipeline(max_content_length=50)
        result = pipeline.validate("x" * 100)
        assert result.is_valid is True
        assert len(result.cleaned_content or "") <= 50
        assert any("truncated" in issue.lower() for issue in result.issues)
