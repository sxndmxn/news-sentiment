"""Analysis pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from news_sentiment.analysis.analyzers.factory import AnalyzerFactory
from news_sentiment.analysis.lexicon import LoadedLanguageDetector
from news_sentiment.domain.exceptions import AnalysisFailedError
from news_sentiment.domain.models import Article, SentimentResult

if TYPE_CHECKING:
    from news_sentiment.analysis.analyzers.base import BaseAnalyzer

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the full analysis pipeline for articles.

    Combines:
    - Sentiment analysis (VADER or transformer)
    - Loaded language detection
    - Result aggregation
    """

    def __init__(
        self,
        analyzer_name: str = "vader",
        detect_loaded_language: bool = True,
    ):
        """Initialize analysis pipeline.

        Args:
            analyzer_name: Name of sentiment analyzer to use
            detect_loaded_language: Whether to detect loaded language
        """
        self.analyzer: BaseAnalyzer = AnalyzerFactory.create(analyzer_name)
        self.loaded_language_detector = (
            LoadedLanguageDetector() if detect_loaded_language else None
        )
        self._detect_loaded_language = detect_loaded_language

    def analyze_article(self, article: Article) -> SentimentResult:
        """Analyze a single article.

        Args:
            article: Article to analyze

        Returns:
            SentimentResult with all analysis data

        Raises:
            AnalysisFailedError: If analysis fails
        """
        try:
            # Get base sentiment result
            result = self.analyzer.analyze_article(article)

            # Add loaded language detection if enabled
            if self.loaded_language_detector and self._detect_loaded_language:
                loaded_score = self.loaded_language_detector.analyze(article.content)
                # Create new result with loaded language
                result = SentimentResult(
                    id=result.id,
                    article_id=result.article_id,
                    sentiment_score=result.sentiment_score,
                    loaded_language=loaded_score,
                    analysis_version=result.analysis_version,
                    analyzed_at=result.analyzed_at,
                    processing_duration_ms=result.processing_duration_ms,
                    article_title=result.article_title,
                    source_id=result.source_id,
                    source_bias=result.source_bias,
                    published_at=result.published_at,
                )

            return result

        except Exception as e:
            logger.error(f"Analysis failed for article {article.id}: {e}")
            raise AnalysisFailedError(article.id, str(e))

    def analyze_batch(
        self,
        articles: list[Article],
        on_progress: callable | None = None,
    ) -> list[SentimentResult]:
        """Analyze a batch of articles.

        Args:
            articles: List of articles to analyze
            on_progress: Optional callback called after each article (article_index, total)

        Returns:
            List of SentimentResult for successfully analyzed articles
        """
        results = []
        total = len(articles)

        for i, article in enumerate(articles):
            try:
                result = self.analyze_article(article)
                results.append(result)
            except AnalysisFailedError as e:
                logger.warning(f"Skipping article {article.id}: {e}")
                continue

            if on_progress:
                on_progress(i + 1, total)

        return results

    @property
    def analyzer_info(self) -> dict:
        """Get information about the current analyzer."""
        return {
            "name": self.analyzer.name,
            "version": self.analyzer.version,
            "model_name": self.analyzer.model_name,
            "model_version": self.analyzer.model_version,
            "loaded_language_detection": self._detect_loaded_language,
        }


# Convenience function
def create_pipeline(
    analyzer_name: str = "vader",
    detect_loaded_language: bool = True,
) -> AnalysisPipeline:
    """Create an analysis pipeline with the specified configuration."""
    return AnalysisPipeline(
        analyzer_name=analyzer_name,
        detect_loaded_language=detect_loaded_language,
    )
