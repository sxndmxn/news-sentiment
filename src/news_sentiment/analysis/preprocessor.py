"""Text preprocessing for analysis pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""

    original: str
    cleaned: str
    word_count: int
    sentence_count: int
    chunks: list[str] | None = None


class TextPreprocessor:
    """Text preprocessing utilities for sentiment analysis."""

    # Patterns for cleaning
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#\w+")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_html: bool = True,
        lowercase: bool = False,
        max_chunk_words: int = 500,
    ):
        """Initialize preprocessor.

        Args:
            remove_urls: Remove URL patterns
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_html: Remove HTML tags
            lowercase: Convert to lowercase
            max_chunk_words: Maximum words per chunk for long text
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_html = remove_html
        self.lowercase = lowercase
        self.max_chunk_words = max_chunk_words

    def clean(self, text: str) -> str:
        """Clean text by removing unwanted patterns."""
        if not text:
            return ""

        result = text

        if self.remove_html:
            result = self.HTML_TAG_PATTERN.sub(" ", result)

        if self.remove_urls:
            result = self.URL_PATTERN.sub(" ", result)

        if self.remove_emails:
            result = self.EMAIL_PATTERN.sub(" ", result)

        if self.remove_mentions:
            result = self.MENTION_PATTERN.sub(" ", result)

        if self.remove_hashtags:
            result = self.HASHTAG_PATTERN.sub(" ", result)

        # Normalize whitespace
        result = self.WHITESPACE_PATTERN.sub(" ", result)
        result = result.strip()

        if self.lowercase:
            result = result.lower()

        return result

    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        if not text:
            return 0
        sentences = self.SENTENCE_PATTERN.split(text)
        return len([s for s in sentences if s.strip()])

    def count_words(self, text: str) -> int:
        """Count words in text."""
        if not text:
            return 0
        return len(text.split())

    def split_into_chunks(self, text: str) -> list[str]:
        """Split long text into chunks for processing.

        Splits at sentence boundaries where possible.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text:
            return []

        word_count = self.count_words(text)
        if word_count <= self.max_chunk_words:
            return [text]

        # Split into sentences
        sentences = self.SENTENCE_PATTERN.split(text)

        chunks = []
        current_chunk: list[str] = []
        current_words = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = self.count_words(sentence)

            if current_words + sentence_words > self.max_chunk_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0

            current_chunk.append(sentence)
            current_words += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def preprocess(self, text: str, chunk: bool = False) -> PreprocessedText:
        """Preprocess text for analysis.

        Args:
            text: Text to preprocess
            chunk: Whether to split into chunks

        Returns:
            PreprocessedText with cleaned text and metadata
        """
        cleaned = self.clean(text)

        result = PreprocessedText(
            original=text,
            cleaned=cleaned,
            word_count=self.count_words(cleaned),
            sentence_count=self.count_sentences(cleaned),
        )

        if chunk:
            result.chunks = self.split_into_chunks(cleaned)

        return result


# Default preprocessor instance
_preprocessor: TextPreprocessor | None = None


def get_preprocessor() -> TextPreprocessor:
    """Get the default preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = TextPreprocessor()
    return _preprocessor


def preprocess_text(text: str, chunk: bool = False) -> PreprocessedText:
    """Convenience function to preprocess text."""
    return get_preprocessor().preprocess(text, chunk=chunk)
