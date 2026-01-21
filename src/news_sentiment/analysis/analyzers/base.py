"""Base analyzer interface using Protocol (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from news_sentiment.domain.models import Article, SentimentResult
from news_sentiment.domain.value_objects import AnalysisVersion, SentimentScore

if TYPE_CHECKING:
    pass


@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the analyzer interface (Strategy pattern).

    All analyzers must implement this protocol to be usable in the pipeline.
    """

    @property
    def name(self) -> str:
        """Unique name for this analyzer."""
        ...

    @property
    def version(self) -> str:
        """Version string for this analyzer."""
        ...

    def analyze(self, text: str) -> SentimentScore:
        """Analyze text and return sentiment score.

        Args:
            text: Text content to analyze

        Returns:
            SentimentScore with compound and component values
        """
        ...

    def analyze_article(self, article: Article) -> SentimentResult:
        """Analyze an article and return full result with metadata.

        Args:
            article: Article to analyze

        Returns:
            SentimentResult with score, lineage, and metadata
        """
        ...


class BaseAnalyzer(ABC):
    """Abstract base class for analyzers.

    Provides common functionality and enforces the Analyzer protocol.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this analyzer."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string for this analyzer."""
        pass

    @property
    def model_name(self) -> str | None:
        """Optional model name (for ML-based analyzers)."""
        return None

    @property
    def model_version(self) -> str | None:
        """Optional model version."""
        return None

    def get_analysis_version(self) -> AnalysisVersion:
        """Get the analysis version information."""
        return AnalysisVersion(
            analyzer_name=self.name,
            analyzer_version=self.version,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    @abstractmethod
    def analyze(self, text: str) -> SentimentScore:
        """Analyze text and return sentiment score.

        Args:
            text: Text content to analyze

        Returns:
            SentimentScore with compound and component values
        """
        pass

    def analyze_article(self, article: Article) -> SentimentResult:
        """Analyze an article and return full result with metadata.

        Default implementation uses analyze() on article content.

        Args:
            article: Article to analyze

        Returns:
            SentimentResult with score, lineage, and metadata
        """
        import time

        start_time = time.perf_counter()

        # Analyze the content
        sentiment_score = self.analyze(article.content)

        end_time = time.perf_counter()
        duration_ms = int((end_time - start_time) * 1000)

        return SentimentResult(
            article_id=article.id,
            sentiment_score=sentiment_score,
            analysis_version=self.get_analysis_version(),
            processing_duration_ms=duration_ms,
            article_title=article.title,
            source_id=article.source_id,
            source_bias=article.source_bias,
            published_at=article.published_at,
        )
