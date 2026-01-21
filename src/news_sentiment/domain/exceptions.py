"""Domain exceptions - Application-specific error types."""

from __future__ import annotations


class NewsSentimentError(Exception):
    """Base exception for all news sentiment errors."""

    pass


class DomainError(NewsSentimentError):
    """Base exception for domain layer errors."""

    pass


class ValidationError(DomainError):
    """Raised when domain validation fails."""

    pass


class ArticleValidationError(ValidationError):
    """Raised when article validation fails."""

    pass


class DuplicateArticleError(DomainError):
    """Raised when attempting to create a duplicate article."""

    def __init__(self, url: str | None = None, content_hash: str | None = None):
        self.url = url
        self.content_hash = content_hash
        message = "Duplicate article detected"
        if url:
            message += f" (url: {url})"
        if content_hash:
            message += f" (hash: {content_hash[:16]}...)"
        super().__init__(message)


class IngestionError(NewsSentimentError):
    """Base exception for ingestion layer errors."""

    pass


class FeedParseError(IngestionError):
    """Raised when RSS feed parsing fails."""

    def __init__(self, source: str, reason: str):
        self.source = source
        self.reason = reason
        super().__init__(f"Failed to parse feed '{source}': {reason}")


class FeedFetchError(IngestionError):
    """Raised when RSS feed fetching fails."""

    def __init__(self, url: str, status_code: int | None = None):
        self.url = url
        self.status_code = status_code
        message = f"Failed to fetch feed from '{url}'"
        if status_code:
            message += f" (status: {status_code})"
        super().__init__(message)


class CircuitBreakerOpenError(IngestionError):
    """Raised when circuit breaker is open for a source."""

    def __init__(self, source: str, retry_after: float | None = None):
        self.source = source
        self.retry_after = retry_after
        message = f"Circuit breaker open for source '{source}'"
        if retry_after:
            message += f" (retry after {retry_after:.1f}s)"
        super().__init__(message)


class QualityError(IngestionError):
    """Raised when content fails quality checks."""

    pass


class ContentTooShortError(QualityError):
    """Raised when content is below minimum length."""

    def __init__(self, actual_length: int, min_length: int):
        self.actual_length = actual_length
        self.min_length = min_length
        super().__init__(f"Content too short: {actual_length} chars (minimum: {min_length})")


class InvalidLanguageError(QualityError):
    """Raised when content is not in expected language."""

    def __init__(self, detected: str, expected: str = "en"):
        self.detected = detected
        self.expected = expected
        super().__init__(f"Invalid language: detected '{detected}', expected '{expected}'")


class AnalysisError(NewsSentimentError):
    """Base exception for analysis layer errors."""

    pass


class AnalyzerNotFoundError(AnalysisError):
    """Raised when requested analyzer is not available."""

    def __init__(self, analyzer_name: str):
        self.analyzer_name = analyzer_name
        super().__init__(f"Analyzer not found: '{analyzer_name}'")


class AnalysisFailedError(AnalysisError):
    """Raised when analysis fails for an article."""

    def __init__(self, article_id: str, reason: str):
        self.article_id = article_id
        self.reason = reason
        super().__init__(f"Analysis failed for article '{article_id}': {reason}")


class StorageError(NewsSentimentError):
    """Base exception for storage layer errors."""

    pass


class RepositoryError(StorageError):
    """Raised when repository operation fails."""

    pass


class ArticleNotFoundError(RepositoryError):
    """Raised when article is not found in repository."""

    def __init__(self, article_id: str):
        self.article_id = article_id
        super().__init__(f"Article not found: '{article_id}'")


class SourceNotFoundError(RepositoryError):
    """Raised when source is not found."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        super().__init__(f"Source not found: '{source_id}'")


class ConfigurationError(NewsSentimentError):
    """Raised when configuration is invalid or missing."""

    pass


class AlertError(NewsSentimentError):
    """Base exception for alerting errors."""

    pass


class NotificationError(AlertError):
    """Raised when notification delivery fails."""

    def __init__(self, channel: str, reason: str):
        self.channel = channel
        self.reason = reason
        super().__init__(f"Failed to send notification via '{channel}': {reason}")
