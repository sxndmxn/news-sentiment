"""Deduplication utilities for article ingestion."""

from __future__ import annotations

from news_sentiment.domain.models import Article
from news_sentiment.domain.value_objects import ContentHash
from news_sentiment.storage.repositories.article_repo import ArticleRepository


class DeduplicationService:
    """Service for detecting and preventing duplicate articles."""

    def __init__(self, article_repo: ArticleRepository | None = None):
        """Initialize deduplication service.

        Args:
            article_repo: Repository for checking existing articles
        """
        self.article_repo = article_repo or ArticleRepository()

    def is_duplicate(self, article: Article) -> bool:
        """Check if an article is a duplicate.

        Checks both URL and content hash.
        """
        # Check URL first (faster)
        if self.article_repo.exists_by_url(article.url):
            return True

        # Check content hash
        if article.content_hash and self.article_repo.exists_by_hash(article.content_hash):
            return True

        return False

    def filter_duplicates(self, articles: list[Article]) -> list[Article]:
        """Filter out duplicate articles from a list.

        Also handles duplicates within the batch (same URL/hash).
        """
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        unique_articles: list[Article] = []

        for article in articles:
            # Skip if URL already seen in this batch
            if article.url in seen_urls:
                continue

            # Skip if hash already seen in this batch
            if article.content_hash:
                hash_key = article.content_hash.value
                if hash_key in seen_hashes:
                    continue

            # Skip if exists in database
            if self.is_duplicate(article):
                continue

            # Mark as seen
            seen_urls.add(article.url)
            if article.content_hash:
                seen_hashes.add(article.content_hash.value)

            unique_articles.append(article)

        return unique_articles

    def get_duplicate_urls(self, urls: list[str]) -> list[str]:
        """Get list of URLs that already exist in the database."""
        return [url for url in urls if self.article_repo.exists_by_url(url)]
