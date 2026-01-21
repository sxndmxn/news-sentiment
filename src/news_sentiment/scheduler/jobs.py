"""Scheduled job definitions."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def ingest_all_sources() -> dict:
    """Job: Ingest articles from all active sources.

    Returns:
        Dict with ingestion results
    """
    from news_sentiment.ingestion.deduplication import DeduplicationService
    from news_sentiment.ingestion.rss_parser import RSSParser
    from news_sentiment.ingestion.source_registry import get_registry
    from news_sentiment.storage.database import init_database
    from news_sentiment.storage.repositories.article_repo import (
        ArticleRepository,
        SourceRepository,
    )

    logger.info("Starting scheduled ingestion job")

    db = init_database()
    registry = get_registry()
    parser = RSSParser()
    article_repo = ArticleRepository(db)
    source_repo = SourceRepository(db)
    dedup_service = DeduplicationService(article_repo)

    sources = registry.get_active_sources()
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "sources_processed": 0,
        "articles_fetched": 0,
        "articles_saved": 0,
        "errors": [],
    }

    for source in sources:
        try:
            source_repo.save(source)
            articles = parser.fetch_feed(source)
            unique_articles = dedup_service.filter_duplicates(articles)
            saved = article_repo.save_batch(unique_articles)

            results["sources_processed"] += 1
            results["articles_fetched"] += len(articles)
            results["articles_saved"] += len(saved)

            logger.info(f"Ingested {len(saved)} articles from {source.name}")

        except Exception as e:
            logger.error(f"Error ingesting from {source.slug}: {e}")
            results["errors"].append({"source": source.slug, "error": str(e)})

    logger.info(
        f"Ingestion complete: {results['articles_saved']} articles from {results['sources_processed']} sources"
    )
    return results


def analyze_pending_articles(limit: int = 100) -> dict:
    """Job: Analyze articles that haven't been analyzed yet.

    Args:
        limit: Maximum articles to analyze

    Returns:
        Dict with analysis results
    """
    from news_sentiment.analysis.pipeline import AnalysisPipeline
    from news_sentiment.storage.database import init_database
    from news_sentiment.storage.repositories.article_repo import ArticleRepository
    from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository

    logger.info("Starting scheduled analysis job")

    db = init_database()
    article_repo = ArticleRepository(db)
    sentiment_repo = SentimentRepository(db)
    pipeline = AnalysisPipeline()

    articles = article_repo.list_unanalyzed(limit=limit)
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "articles_analyzed": 0,
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "errors": [],
    }

    for article in articles:
        try:
            result = pipeline.analyze_article(article)
            sentiment_repo.save(result)
            results["articles_analyzed"] += 1

            label = result.sentiment_score.sentiment_label
            results[label] += 1

        except Exception as e:
            logger.error(f"Error analyzing article {article.id}: {e}")
            results["errors"].append({"article_id": article.id, "error": str(e)})

    logger.info(f"Analysis complete: {results['articles_analyzed']} articles")
    return results


def compute_daily_summaries() -> dict:
    """Job: Compute daily summary statistics.

    Returns:
        Dict with aggregation results
    """
    from news_sentiment.analytics.aggregator import Aggregator
    from news_sentiment.storage.database import init_database

    logger.info("Starting scheduled aggregation job")

    init_database()
    aggregator = Aggregator()

    # Compute for yesterday (complete data)
    yesterday = datetime.utcnow() - timedelta(days=1)
    yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    summaries = aggregator.aggregate_day(yesterday)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": yesterday.isoformat(),
        "summaries_created": len(summaries),
    }

    logger.info(f"Aggregation complete: {len(summaries)} summaries for {yesterday.date()}")
    return results


def check_alerts() -> dict:
    """Job: Check for alert conditions.

    Returns:
        Dict with alert check results
    """
    # Placeholder for Phase 5
    logger.info("Alert check job - not yet implemented")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alerts_triggered": 0,
    }


def cleanup_old_data(days_to_keep: int = 90) -> dict:
    """Job: Clean up old data from the database.

    Args:
        days_to_keep: Number of days of data to retain

    Returns:
        Dict with cleanup results
    """
    # Placeholder for maintenance
    logger.info(f"Cleanup job - keeping {days_to_keep} days of data")
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "records_deleted": 0,
    }
