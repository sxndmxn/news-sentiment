"""Data export functionality - CSV and JSON exports."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import get_database
from news_sentiment.storage.repositories.article_repo import ArticleRepository
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Exporter:
    """Export data to various formats."""

    def __init__(self, output_dir: Path | str | None = None):
        """Initialize exporter.

        Args:
            output_dir: Directory for export files. Defaults to data/exports/
        """
        if output_dir is None:
            output_dir = Path("data/exports")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db = get_database()
        self.article_repo = ArticleRepository(self.db)
        self.sentiment_repo = SentimentRepository(self.db)

    def export_articles_json(
        self,
        days: int = 7,
        filename: str | None = None,
        include_sentiment: bool = True,
    ) -> Path:
        """Export articles to JSON.

        Args:
            days: Number of days to export
            filename: Output filename. Auto-generated if not provided.
            include_sentiment: Whether to include sentiment data

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.json"

        output_path = self.output_dir / filename

        articles = self.article_repo.list_recent(days=days, limit=10000)

        data = []
        for article in articles:
            article_data = {
                "id": article.id,
                "title": article.title,
                "url": article.url,
                "source_name": article.source_name,
                "source_bias": article.source_bias.value if article.source_bias else None,
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "fetched_at": article.fetched_at.isoformat() if article.fetched_at else None,
                "word_count": article.word_count,
            }

            if include_sentiment:
                sentiment = self.sentiment_repo.get_by_article_id(article.id)
                if sentiment:
                    article_data["sentiment"] = {
                        "compound": sentiment.sentiment_score.compound,
                        "positive": sentiment.sentiment_score.positive,
                        "negative": sentiment.sentiment_score.negative,
                        "neutral": sentiment.sentiment_score.neutral,
                        "label": sentiment.sentiment_score.sentiment_label,
                    }
                    if sentiment.loaded_language:
                        article_data["loaded_language"] = {
                            "hedges": sentiment.loaded_language.hedges_count,
                            "intensifiers": sentiment.loaded_language.intensifiers_count,
                            "weasel_words": sentiment.loaded_language.weasel_words_count,
                            "ratio": sentiment.loaded_language.loaded_ratio,
                        }

            data.append(article_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(data)} articles to {output_path}")
        return output_path

    def export_articles_csv(
        self,
        days: int = 7,
        filename: str | None = None,
        include_sentiment: bool = True,
    ) -> Path:
        """Export articles to CSV.

        Args:
            days: Number of days to export
            filename: Output filename. Auto-generated if not provided.
            include_sentiment: Whether to include sentiment data

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.csv"

        output_path = self.output_dir / filename

        articles = self.article_repo.list_recent(days=days, limit=10000)

        # Define columns
        fieldnames = [
            "id",
            "title",
            "url",
            "source_name",
            "source_bias",
            "published_at",
            "fetched_at",
            "word_count",
        ]

        if include_sentiment:
            fieldnames.extend([
                "sentiment_compound",
                "sentiment_positive",
                "sentiment_negative",
                "sentiment_neutral",
                "sentiment_label",
                "loaded_hedges",
                "loaded_intensifiers",
                "loaded_weasel_words",
                "loaded_ratio",
            ])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for article in articles:
                row = {
                    "id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "source_name": article.source_name,
                    "source_bias": article.source_bias.value if article.source_bias else "",
                    "published_at": article.published_at.isoformat() if article.published_at else "",
                    "fetched_at": article.fetched_at.isoformat() if article.fetched_at else "",
                    "word_count": article.word_count,
                }

                if include_sentiment:
                    sentiment = self.sentiment_repo.get_by_article_id(article.id)
                    if sentiment:
                        row.update({
                            "sentiment_compound": sentiment.sentiment_score.compound,
                            "sentiment_positive": sentiment.sentiment_score.positive,
                            "sentiment_negative": sentiment.sentiment_score.negative,
                            "sentiment_neutral": sentiment.sentiment_score.neutral,
                            "sentiment_label": sentiment.sentiment_score.sentiment_label,
                        })
                        if sentiment.loaded_language:
                            row.update({
                                "loaded_hedges": sentiment.loaded_language.hedges_count,
                                "loaded_intensifiers": sentiment.loaded_language.intensifiers_count,
                                "loaded_weasel_words": sentiment.loaded_language.weasel_words_count,
                                "loaded_ratio": sentiment.loaded_language.loaded_ratio,
                            })

                writer.writerow(row)

        logger.info(f"Exported {len(articles)} articles to {output_path}")
        return output_path

    def export_sentiment_summary_json(
        self,
        days: int = 7,
        filename: str | None = None,
    ) -> Path:
        """Export sentiment summary statistics.

        Args:
            days: Number of days to summarize
            filename: Output filename

        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_summary_{timestamp}.json"

        output_path = self.output_dir / filename

        # Collect summary data
        summary = {
            "period": {
                "days": days,
                "start": (datetime.utcnow() - timedelta(days=days)).isoformat(),
                "end": datetime.utcnow().isoformat(),
            },
            "overall": {
                "total_articles": self.article_repo.count_total(),
                "total_analyzed": self.sentiment_repo.count_total(),
                "avg_sentiment": self.sentiment_repo.get_average_sentiment(days=days),
                "distribution": self.sentiment_repo.get_sentiment_distribution(days=days),
            },
            "by_bias": {},
        }

        # Per-bias statistics
        for bias in BiasLabel:
            avg = self.sentiment_repo.get_average_sentiment(days=days, bias=bias)
            if avg is not None:
                summary["by_bias"][bias.value] = {
                    "avg_sentiment": avg,
                    "article_count": self.article_repo.count_by_bias(bias),
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported sentiment summary to {output_path}")
        return output_path


def export_to_json(days: int = 7, output_dir: str | None = None) -> Path:
    """Convenience function to export articles to JSON."""
    exporter = Exporter(output_dir)
    return exporter.export_articles_json(days=days)


def export_to_csv(days: int = 7, output_dir: str | None = None) -> Path:
    """Convenience function to export articles to CSV."""
    exporter = Exporter(output_dir)
    return exporter.export_articles_csv(days=days)
