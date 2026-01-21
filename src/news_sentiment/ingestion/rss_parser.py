"""RSS feed parser with resilient fetching."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import feedparser
from dateutil import parser as date_parser

from news_sentiment.domain.exceptions import FeedFetchError, FeedParseError
from news_sentiment.domain.models import Article, Source
from news_sentiment.ingestion.circuit_breaker import (
    CircuitBreaker,
    get_circuit_breaker_registry,
)
from news_sentiment.ingestion.quality import QualityPipeline

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RSSParser:
    """Parser for RSS feeds with circuit breaker protection."""

    def __init__(
        self,
        quality_pipeline: QualityPipeline | None = None,
        user_agent: str = "NewsSentimentAnalyzer/1.0",
    ):
        """Initialize RSS parser.

        Args:
            quality_pipeline: Pipeline for content quality validation
            user_agent: User agent string for requests
        """
        self.quality_pipeline = quality_pipeline or QualityPipeline()
        self.user_agent = user_agent
        self._cb_registry = get_circuit_breaker_registry()

    def _get_circuit_breaker(self, source: Source) -> CircuitBreaker:
        """Get circuit breaker for a source."""
        return self._cb_registry.get_breaker(source.slug)

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse date string to datetime."""
        if not date_str:
            return None
        try:
            return date_parser.parse(date_str)
        except (ValueError, TypeError):
            return None

    def _extract_content(self, entry: dict) -> str:
        """Extract content from feed entry, trying multiple fields."""
        # Try content field first (usually has full content)
        if "content" in entry and entry["content"]:
            content_list = entry["content"]
            if isinstance(content_list, list) and content_list:
                return content_list[0].get("value", "")

        # Try summary
        if "summary" in entry and entry["summary"]:
            return entry["summary"]

        # Try description
        if "description" in entry and entry["description"]:
            return entry["description"]

        # Fallback to title
        return entry.get("title", "")

    def _entry_to_article(self, entry: dict, source: Source) -> Article | None:
        """Convert feed entry to Article domain model."""
        url = entry.get("link", "")
        if not url:
            return None

        title = entry.get("title", "").strip()
        if not title:
            return None

        content = self._extract_content(entry)

        # Validate and clean content
        quality_result = self.quality_pipeline.validate(content)
        if not quality_result.is_valid:
            logger.debug(f"Article failed quality check: {url} - {quality_result.issues}")
            return None

        cleaned_content = quality_result.cleaned_content or content

        # Get author
        author = None
        if "author" in entry:
            author = entry["author"]
        elif "author_detail" in entry:
            author = entry["author_detail"].get("name")

        # Parse published date
        published_at = self._parse_date(entry.get("published"))
        if not published_at:
            published_at = self._parse_date(entry.get("updated"))

        return Article(
            source_id=source.id,
            url=url,
            title=title[:500],  # Truncate to max length
            content=cleaned_content,
            summary=entry.get("summary", "")[:1000] if entry.get("summary") else None,
            author=author[:200] if author else None,
            published_at=published_at,
            source_name=source.name,
            source_bias=source.bias_label,
        )

    def fetch_feed(self, source: Source) -> list[Article]:
        """Fetch and parse RSS feed for a source.

        Uses circuit breaker for resilience.

        Args:
            source: News source to fetch

        Returns:
            List of Article domain models

        Raises:
            CircuitBreakerOpenError: If circuit is open for this source
            FeedFetchError: If feed cannot be fetched
            FeedParseError: If feed cannot be parsed
        """
        cb = self._get_circuit_breaker(source)

        def _do_fetch() -> list[Article]:
            return self._fetch_and_parse(source)

        return cb.call(_do_fetch)

    def _fetch_and_parse(self, source: Source) -> list[Article]:
        """Internal method to fetch and parse feed (called by circuit breaker)."""
        logger.info(f"Fetching feed: {source.name} ({source.feed_url})")

        # Parse the feed
        feed = feedparser.parse(
            source.feed_url,
            agent=self.user_agent,
        )

        # Check for errors
        if feed.bozo and feed.bozo_exception:
            error_msg = str(feed.bozo_exception)
            logger.warning(f"Feed parse warning for {source.slug}: {error_msg}")
            # Some bozo errors are recoverable, continue if we have entries
            if not feed.entries:
                raise FeedParseError(source.slug, error_msg)

        # Check HTTP status if available
        if hasattr(feed, "status"):
            if feed.status >= 400:
                raise FeedFetchError(source.feed_url, feed.status)

        if not feed.entries:
            logger.warning(f"No entries in feed: {source.slug}")
            return []

        # Convert entries to articles
        articles = []
        for entry in feed.entries:
            try:
                article = self._entry_to_article(entry, source)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Error processing entry from {source.slug}: {e}")
                continue

        logger.info(f"Fetched {len(articles)} articles from {source.name}")
        return articles

    def fetch_multiple(
        self,
        sources: list[Source],
        max_per_source: int = 50,
    ) -> dict[str, list[Article]]:
        """Fetch from multiple sources.

        Args:
            sources: List of sources to fetch
            max_per_source: Maximum articles per source

        Returns:
            Dict mapping source slug to list of articles
        """
        results: dict[str, list[Article]] = {}

        for source in sources:
            try:
                articles = self.fetch_feed(source)
                results[source.slug] = articles[:max_per_source]
            except Exception as e:
                logger.error(f"Failed to fetch {source.slug}: {e}")
                results[source.slug] = []

        return results

    def get_feed_info(self, url: str) -> dict | None:
        """Get basic feed information without full parsing.

        Useful for validating feed URLs.
        """
        try:
            feed = feedparser.parse(url, agent=self.user_agent)

            if feed.bozo and not feed.entries:
                return None

            return {
                "title": feed.feed.get("title", "Unknown"),
                "description": feed.feed.get("description", ""),
                "link": feed.feed.get("link", url),
                "entry_count": len(feed.entries),
                "last_updated": feed.feed.get("updated", ""),
            }
        except Exception:
            return None
