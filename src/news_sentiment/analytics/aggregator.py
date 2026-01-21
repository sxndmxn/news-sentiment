"""Analytics aggregator for computing daily summaries."""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from news_sentiment.domain.models import DailySummary
from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import (
    DailySummaryTable,
    SentimentResultTable,
    get_database,
)

if TYPE_CHECKING:
    from sqlalchemy import select
    from news_sentiment.storage.database import Database

logger = logging.getLogger(__name__)


class Aggregator:
    """Aggregator for computing daily statistics."""

    def __init__(self, db: Database | None = None):
        """Initialize aggregator.

        Args:
            db: Database instance. Uses global instance if not provided.
        """
        self.db = db or get_database()

    def compute_daily_summary(
        self,
        date: datetime,
        source_id: str | None = None,
        bias_label: BiasLabel | None = None,
    ) -> DailySummary | None:
        """Compute summary statistics for a specific date.

        Args:
            date: Date to compute summary for
            source_id: Optional filter by source
            bias_label: Optional filter by bias

        Returns:
            DailySummary or None if no data
        """
        from sqlalchemy import select, func

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        with self.db.session() as session:
            # Build base query
            query = select(SentimentResultTable).where(
                SentimentResultTable.analyzed_at >= start_of_day,
                SentimentResultTable.analyzed_at < end_of_day,
            )

            if source_id:
                query = query.where(SentimentResultTable.source_id == source_id)

            if bias_label:
                query = query.where(SentimentResultTable.source_bias == bias_label.value)

            results = session.execute(query).scalars().all()

            if not results:
                return None

            # Calculate statistics
            compounds = [r.compound for r in results]
            article_count = len(compounds)

            positive_count = sum(1 for c in compounds if c >= 0.05)
            negative_count = sum(1 for c in compounds if c <= -0.05)
            neutral_count = article_count - positive_count - negative_count

            avg_sentiment = statistics.mean(compounds)
            min_sentiment = min(compounds)
            max_sentiment = max(compounds)
            sentiment_std_dev = statistics.stdev(compounds) if len(compounds) > 1 else 0.0

            # Calculate loaded language average
            loaded_ratios = []
            for r in results:
                if r.total_words and r.total_words > 0:
                    total_loaded = (r.hedges_count or 0) + (r.intensifiers_count or 0) + (r.weasel_words_count or 0)
                    loaded_ratios.append(total_loaded / r.total_words)

            avg_loaded_ratio = statistics.mean(loaded_ratios) if loaded_ratios else None

            return DailySummary(
                date=start_of_day,
                source_id=source_id,
                bias_label=bias_label,
                article_count=article_count,
                avg_sentiment=avg_sentiment,
                min_sentiment=min_sentiment,
                max_sentiment=max_sentiment,
                sentiment_std_dev=sentiment_std_dev,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                avg_loaded_ratio=avg_loaded_ratio,
            )

    def save_summary(self, summary: DailySummary) -> DailySummary:
        """Save a daily summary to the database."""
        with self.db.session() as session:
            # Check for existing summary
            from sqlalchemy import select

            query = select(DailySummaryTable).where(
                DailySummaryTable.date == summary.date,
            )
            if summary.source_id:
                query = query.where(DailySummaryTable.source_id == summary.source_id)
            else:
                query = query.where(DailySummaryTable.source_id.is_(None))

            if summary.bias_label:
                query = query.where(DailySummaryTable.bias_label == summary.bias_label.value)
            else:
                query = query.where(DailySummaryTable.bias_label.is_(None))

            existing = session.execute(query).scalar_one_or_none()

            if existing:
                # Update existing
                existing.article_count = summary.article_count
                existing.avg_sentiment = summary.avg_sentiment
                existing.min_sentiment = summary.min_sentiment
                existing.max_sentiment = summary.max_sentiment
                existing.sentiment_std_dev = summary.sentiment_std_dev
                existing.positive_count = summary.positive_count
                existing.negative_count = summary.negative_count
                existing.neutral_count = summary.neutral_count
                existing.avg_loaded_ratio = summary.avg_loaded_ratio
                existing.top_entities = json.dumps(summary.top_entities) if summary.top_entities else None
                session.flush()
            else:
                row = DailySummaryTable(
                    id=summary.id,
                    date=summary.date,
                    source_id=summary.source_id,
                    bias_label=summary.bias_label.value if summary.bias_label else None,
                    article_count=summary.article_count,
                    avg_sentiment=summary.avg_sentiment,
                    min_sentiment=summary.min_sentiment,
                    max_sentiment=summary.max_sentiment,
                    sentiment_std_dev=summary.sentiment_std_dev,
                    positive_count=summary.positive_count,
                    negative_count=summary.negative_count,
                    neutral_count=summary.neutral_count,
                    avg_loaded_ratio=summary.avg_loaded_ratio,
                    top_entities=json.dumps(summary.top_entities) if summary.top_entities else None,
                )
                session.add(row)
                session.flush()

        return summary

    def aggregate_day(self, date: datetime) -> list[DailySummary]:
        """Aggregate all summaries for a day.

        Creates summaries for:
        - Overall (no filters)
        - Each bias category

        Args:
            date: Date to aggregate

        Returns:
            List of created summaries
        """
        summaries = []

        # Overall summary
        overall = self.compute_daily_summary(date)
        if overall:
            self.save_summary(overall)
            summaries.append(overall)
            logger.info(f"Created overall summary for {date.date()}: {overall.article_count} articles")

        # Per-bias summaries
        for bias in BiasLabel:
            bias_summary = self.compute_daily_summary(date, bias_label=bias)
            if bias_summary and bias_summary.article_count > 0:
                self.save_summary(bias_summary)
                summaries.append(bias_summary)
                logger.info(f"Created {bias.value} summary for {date.date()}: {bias_summary.article_count} articles")

        return summaries

    def get_trend_data(
        self,
        days: int = 7,
        bias_label: BiasLabel | None = None,
    ) -> list[dict]:
        """Get sentiment trend data for visualization.

        Args:
            days: Number of days to fetch
            bias_label: Optional filter by bias

        Returns:
            List of dicts with date and avg_sentiment
        """
        from sqlalchemy import select

        cutoff = datetime.utcnow() - timedelta(days=days)

        with self.db.session() as session:
            query = select(DailySummaryTable).where(
                DailySummaryTable.date >= cutoff,
            )

            if bias_label:
                query = query.where(DailySummaryTable.bias_label == bias_label.value)
            else:
                query = query.where(DailySummaryTable.bias_label.is_(None))

            query = query.where(DailySummaryTable.source_id.is_(None))
            query = query.order_by(DailySummaryTable.date)

            results = session.execute(query).scalars().all()

            return [
                {
                    "date": r.date.isoformat(),
                    "avg_sentiment": r.avg_sentiment,
                    "article_count": r.article_count,
                    "positive_count": r.positive_count,
                    "negative_count": r.negative_count,
                    "neutral_count": r.neutral_count,
                }
                for r in results
            ]


# Convenience function
def get_aggregator() -> Aggregator:
    """Get an aggregator instance."""
    return Aggregator()
