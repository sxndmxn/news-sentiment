"""Data quality validators for ingestion pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from news_sentiment.domain.exceptions import (
    ContentTooShortError,
    InvalidLanguageError,
)


class QualityValidator(Protocol):
    """Protocol for quality validators."""

    def validate(self, content: str, **kwargs: object) -> bool:
        """Validate content, raise QualityError if invalid."""
        ...


@dataclass
class QualityResult:
    """Result of quality validation."""

    is_valid: bool
    issues: list[str]
    cleaned_content: str | None = None

    @property
    def has_issues(self) -> bool:
        """Check if there are any quality issues."""
        return len(self.issues) > 0


class ContentLengthValidator:
    """Validate content meets minimum length requirements."""

    def __init__(self, min_length: int = 100, max_length: int = 100000):
        """Initialize validator.

        Args:
            min_length: Minimum content length in characters
            max_length: Maximum content length in characters
        """
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, content: str, **kwargs: object) -> bool:
        """Validate content length."""
        length = len(content.strip())

        if length < self.min_length:
            raise ContentTooShortError(length, self.min_length)

        if length > self.max_length:
            # Truncate rather than fail
            return True

        return True


class LanguageValidator:
    """Validate content is in expected language."""

    # Common English words for basic detection
    ENGLISH_MARKERS = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
    }

    def __init__(self, expected_language: str = "en", threshold: float = 0.1):
        """Initialize validator.

        Args:
            expected_language: Expected language code (only 'en' supported)
            threshold: Minimum ratio of English markers to detect
        """
        self.expected_language = expected_language
        self.threshold = threshold

    def validate(self, content: str, **kwargs: object) -> bool:
        """Validate content language using basic word detection."""
        if self.expected_language != "en":
            # Only English detection implemented
            return True

        words = set(re.findall(r"\b[a-z]+\b", content.lower()))
        if not words:
            return True  # Can't determine, allow it

        english_count = len(words & self.ENGLISH_MARKERS)
        ratio = english_count / len(words)

        if ratio < self.threshold:
            raise InvalidLanguageError("unknown", self.expected_language)

        return True


class HTMLCleaner:
    """Clean HTML tags and entities from content."""

    # Patterns to remove
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z]+;|&#\d+;")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def clean(self, content: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        # Remove HTML tags
        text = self.HTML_TAG_PATTERN.sub(" ", content)

        # Convert common entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # Remove remaining entities
        text = self.HTML_ENTITY_PATTERN.sub(" ", text)

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text)

        return text.strip()


class DuplicateDetector:
    """Detect potential duplicate content."""

    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize detector.

        Args:
            similarity_threshold: Threshold for considering content duplicate
        """
        self.similarity_threshold = similarity_threshold

    def get_fingerprint(self, content: str, length: int = 100) -> str:
        """Generate a fingerprint for quick duplicate detection.

        Uses first N characters of normalized content.
        """
        # Normalize: lowercase, remove non-alpha, collapse whitespace
        normalized = re.sub(r"[^a-z\s]", "", content.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized[:length]

    def is_similar(self, content1: str, content2: str) -> bool:
        """Check if two pieces of content are similar.

        Uses Jaccard similarity on word sets.
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / union if union > 0 else 0
        return similarity >= self.similarity_threshold


class QualityPipeline:
    """Pipeline for running multiple quality checks."""

    def __init__(
        self,
        min_content_length: int = 100,
        max_content_length: int = 100000,
        check_language: bool = True,
        clean_html: bool = True,
    ):
        """Initialize quality pipeline.

        Args:
            min_content_length: Minimum content length
            max_content_length: Maximum content length
            check_language: Whether to check language
            clean_html: Whether to clean HTML
        """
        self.length_validator = ContentLengthValidator(min_content_length, max_content_length)
        self.language_validator = LanguageValidator() if check_language else None
        self.html_cleaner = HTMLCleaner() if clean_html else None
        self.max_content_length = max_content_length

    def validate(self, content: str) -> QualityResult:
        """Run all quality checks on content.

        Returns:
            QualityResult with validation status and cleaned content
        """
        issues: list[str] = []
        cleaned_content = content

        # Clean HTML first
        if self.html_cleaner:
            cleaned_content = self.html_cleaner.clean(cleaned_content)

        # Truncate if too long
        if len(cleaned_content) > self.max_content_length:
            cleaned_content = cleaned_content[: self.max_content_length]
            issues.append(f"Content truncated to {self.max_content_length} characters")

        # Check length
        try:
            self.length_validator.validate(cleaned_content)
        except ContentTooShortError as e:
            issues.append(str(e))
            return QualityResult(is_valid=False, issues=issues, cleaned_content=cleaned_content)

        # Check language
        if self.language_validator:
            try:
                self.language_validator.validate(cleaned_content)
            except InvalidLanguageError as e:
                issues.append(str(e))
                # Language issues are warnings, not failures
                pass

        return QualityResult(is_valid=True, issues=issues, cleaned_content=cleaned_content)
