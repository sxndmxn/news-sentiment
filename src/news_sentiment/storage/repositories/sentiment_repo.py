"""Sentiment result repository - data access for analysis results."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from news_sentiment.domain.models import SentimentResult
from news_sentiment.domain.value_objects import (
    AnalysisVersion,
    BiasLabel,
    LoadedLanguageScore,
    SentimentScore,
)
from news_sentiment.storage.database import SentimentResultTable, get_database

if TYPE_CHECKING:
    from news_sentiment.storage.database import Database


class SentimentRepository:
    """Repository for sentiment result persistence operations."""

    def __init__(self, db: Database | None = None):
        """Initialize repository.

        Args:
            db: Database instance. Uses global instance if not provided.
        """
        self.db = db or get_database()

    def _to_domain(self, row: SentimentResultTable) -> SentimentResult:
        """Convert database row to domain model."""
        sentiment_score = SentimentScore(
            compound=row.compound,
            positive=row.positive,
            negative=row.negative,
            neutral=row.neutral,
        )

        loaded_language = None
        if row.hedges_count is not None:
            loaded_language = LoadedLanguageScore(
                hedges_count=row.hedges_count,
                intensifiers_count=row.intensifiers_count or 0,
                weasel_words_count=row.weasel_words_count or 0,
                total_words=row.total_words or 0,
            )

        analysis_version = AnalysisVersion(
            analyzer_name=row.analyzer_name,
            analyzer_version=row.analyzer_version,
            model_name=row.model_name,
            model_version=row.model_version,
        )

        return SentimentResult(
            id=row.id,
            article_id=row.article_id,
            sentiment_score=sentiment_score,
            loaded_language=loaded_language,
            analysis_version=analysis_version,
            analyzed_at=row.analyzed_at,
            processing_duration_ms=row.processing_duration_ms,
            article_title=row.article_title,
            source_id=row.source_id,
            source_bias=BiasLabel(row.source_bias) if row.source_bias else None,
            published_at=row.published_at,
        )

    def _to_row(self, result: SentimentResult) -> SentimentResultTable:
        """Convert domain model to database row."""
        return SentimentResultTable(
            id=result.id,
            article_id=result.article_id,
            compound=result.sentiment_score.compound,
            positive=result.sentiment_score.positive,
            negative=result.sentiment_score.negative,
            neutral=result.sentiment_score.neutral,
            hedges_count=result.loaded_language.hedges_count if result.loaded_language else None,
            intensifiers_count=(
                result.loaded_language.intensifiers_count if result.loaded_language else None
            ),
            weasel_words_count=(
                result.loaded_language.weasel_words_count if result.loaded_language else None
            ),
            total_words=result.loaded_language.total_words if result.loaded_language else None,
            analyzer_name=result.analysis_version.analyzer_name,
            analyzer_version=result.analysis_version.analyzer_version,
            model_name=result.analysis_version.model_name,
            model_version=result.analysis_version.model_version,
            analyzed_at=result.analyzed_at,
            processing_duration_ms=result.processing_duration_ms,
            article_title=result.article_title,
            source_id=result.source_id,
            source_bias=result.source_bias.value if result.source_bias else None,
            published_at=result.published_at,
        )

    def save(self, result: SentimentResult) -> SentimentResult:
        """Save a sentiment result to the database."""
        with self.db.session() as session:
            # Check for existing result for this article
            existing = session.execute(
                select(SentimentResultTable).where(
                    SentimentResultTable.article_id == result.article_id
                )
            ).scalar_one_or_none()

            if existing:
                # Update existing result
                existing.compound = result.sentiment_score.compound
                existing.positive = result.sentiment_score.positive
                existing.negative = result.sentiment_score.negative
                existing.neutral = result.sentiment_score.neutral
                if result.loaded_language:
                    existing.hedges_count = result.loaded_language.hedges_count
                    existing.intensifiers_count = result.loaded_language.intensifiers_count
                    existing.weasel_words_count = result.loaded_language.weasel_words_count
                    existing.total_words = result.loaded_language.total_words
                existing.analyzer_name = result.analysis_version.analyzer_name
                existing.analyzer_version = result.analysis_version.analyzer_version
                existing.model_name = result.analysis_version.model_name
                existing.model_version = result.analysis_version.model_version
                existing.analyzed_at = result.analyzed_at
                existing.processing_duration_ms = result.processing_duration_ms
                session.flush()
                return self._to_domain(existing)
            else:
                row = self._to_row(result)
                session.add(row)
                session.flush()
                return self._to_domain(row)

    def save_batch(self, results: list[SentimentResult]) -> list[SentimentResult]:
        """Save multiple sentiment results."""
        saved = []
        with self.db.session() as session:
            for result in results:
                row = self._to_row(result)
                session.add(row)
                saved.append(result)
            session.flush()
        return saved

    def get_by_article_id(self, article_id: str) -> SentimentResult | None:
        """Get sentiment result for an article."""
        with self.db.session() as session:
            row = session.execute(
                select(SentimentResultTable).where(SentimentResultTable.article_id == article_id)
            ).scalar_one_or_none()
            return self._to_domain(row) if row else None

    def list_recent(
        self,
        days: int = 7,
        source_id: str | None = None,
        bias: BiasLabel | None = None,
        limit: int = 100,
    ) -> list[SentimentResult]:
        """List recent sentiment results."""
        with self.db.session() as session:
            query = select(SentimentResultTable).order_by(SentimentResultTable.analyzed_at.desc())

            if days > 0:
                cutoff = datetime.utcnow() - timedelta(days=days)
                query = query.where(SentimentResultTable.analyzed_at >= cutoff)

            if source_id:
                query = query.where(SentimentResultTable.source_id == source_id)

            if bias:
                query = query.where(SentimentResultTable.source_bias == bias.value)

            query = query.limit(limit)
            rows = session.execute(query).scalars().all()
            return [self._to_domain(row) for row in rows]

    def get_average_sentiment(
        self,
        days: int = 7,
        source_id: str | None = None,
        bias: BiasLabel | None = None,
    ) -> float | None:
        """Get average compound sentiment score."""
        with self.db.session() as session:
            query = select(func.avg(SentimentResultTable.compound))

            if days > 0:
                cutoff = datetime.utcnow() - timedelta(days=days)
                query = query.where(SentimentResultTable.analyzed_at >= cutoff)

            if source_id:
                query = query.where(SentimentResultTable.source_id == source_id)

            if bias:
                query = query.where(SentimentResultTable.source_bias == bias.value)

            result = session.execute(query).scalar()
            return result

    def get_sentiment_distribution(
        self,
        days: int = 7,
        source_id: str | None = None,
    ) -> dict[str, int]:
        """Get count of positive, negative, neutral results."""
        with self.db.session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days) if days > 0 else None

            counts = {"positive": 0, "negative": 0, "neutral": 0}

            for label, (low, high) in [
                ("positive", (0.05, 2.0)),
                ("negative", (-2.0, -0.05)),
                ("neutral", (-0.05, 0.05)),
            ]:
                query = select(func.count()).select_from(SentimentResultTable)
                query = query.where(SentimentResultTable.compound >= low)
                query = query.where(SentimentResultTable.compound < high)

                if cutoff:
                    query = query.where(SentimentResultTable.analyzed_at >= cutoff)
                if source_id:
                    query = query.where(SentimentResultTable.source_id == source_id)

                counts[label] = session.execute(query).scalar() or 0

            return counts

    def count_total(self) -> int:
        """Count total sentiment results."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count()).select_from(SentimentResultTable)
            ).scalar()
            return result or 0

    def count_by_source(self, source_id: str) -> int:
        """Count sentiment results for a source."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count())
                .select_from(SentimentResultTable)
                .where(SentimentResultTable.source_id == source_id)
            ).scalar()
            return result or 0
