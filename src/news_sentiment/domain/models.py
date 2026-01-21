"""Domain models - Core aggregates and entities."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, Field

from news_sentiment.domain.value_objects import (
    AnalysisVersion,
    BiasLabel,
    ContentHash,
    LoadedLanguageScore,
    SentimentScore,
)


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


class Source(BaseModel):
    """News source aggregate - represents an RSS feed source."""

    id: str = Field(default_factory=generate_uuid)
    name: str = Field(min_length=1, max_length=100)
    slug: str = Field(min_length=1, max_length=50, pattern=r"^[a-z0-9_-]+$")
    feed_url: str = Field(min_length=1)
    website_url: str | None = None
    bias_label: BiasLabel
    description: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Circuit breaker state (transient, not persisted)
    failure_count: int = Field(default=0, exclude=True)
    last_failure_at: datetime | None = Field(default=None, exclude=True)

    def record_failure(self) -> None:
        """Record a fetch failure for circuit breaker tracking."""
        self.failure_count += 1
        self.last_failure_at = datetime.utcnow()

    def reset_failures(self) -> None:
        """Reset failure count after successful fetch."""
        self.failure_count = 0
        self.last_failure_at = None


class Article(BaseModel):
    """Article aggregate - represents a news article."""

    id: str = Field(default_factory=generate_uuid)
    source_id: str
    url: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    content: str
    summary: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: ContentHash | None = None
    word_count: int = Field(default=0, ge=0)

    # Denormalized source info for efficient queries
    source_name: str | None = None
    source_bias: BiasLabel | None = None

    def model_post_init(self, __context: object) -> None:
        """Post-init processing to compute derived fields."""
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())
        if self.content_hash is None and self.content:
            self.content_hash = ContentHash.from_content(self.content)

    @property
    def is_valid_for_analysis(self) -> bool:
        """Check if article has sufficient content for analysis."""
        return self.word_count >= 50


class SentimentResult(BaseModel):
    """Sentiment analysis result entity."""

    id: str = Field(default_factory=generate_uuid)
    article_id: str
    sentiment_score: SentimentScore
    loaded_language: LoadedLanguageScore | None = None
    analysis_version: AnalysisVersion
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    # Lineage tracking
    processing_duration_ms: Annotated[int, Field(ge=0)] | None = None
    raw_output: dict | None = None

    # Denormalized article info for efficient queries
    article_title: str | None = None
    source_id: str | None = None
    source_bias: BiasLabel | None = None
    published_at: datetime | None = None


class Entity(BaseModel):
    """Named entity extracted from article content."""

    id: str = Field(default_factory=generate_uuid)
    article_id: str
    text: str = Field(min_length=1)
    label: str = Field(min_length=1)  # e.g., PERSON, ORG, GPE
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    sentiment_context: SentimentScore | None = None

    @property
    def span_length(self) -> int:
        """Length of entity span in characters."""
        return self.end_char - self.start_char


class DailySummary(BaseModel):
    """Aggregated daily summary for analytics."""

    id: str = Field(default_factory=generate_uuid)
    date: datetime
    source_id: str | None = None  # None for aggregate across all sources
    bias_label: BiasLabel | None = None  # None for aggregate across all biases

    article_count: int = Field(ge=0)
    avg_sentiment: float = Field(ge=-1.0, le=1.0)
    min_sentiment: float = Field(ge=-1.0, le=1.0)
    max_sentiment: float = Field(ge=-1.0, le=1.0)
    sentiment_std_dev: float = Field(ge=0.0)

    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)

    avg_loaded_ratio: float | None = None
    top_entities: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
