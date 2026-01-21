"""Article repository - Data access layer for articles."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from news_sentiment.domain.exceptions import ArticleNotFoundError, DuplicateArticleError
from news_sentiment.domain.models import Article, Source
from news_sentiment.domain.value_objects import BiasLabel, ContentHash
from news_sentiment.storage.database import ArticleTable, SourceTable, get_database

if TYPE_CHECKING:
    from news_sentiment.storage.database import Database


class ArticleRepository:
    """Repository for article persistence operations."""

    def __init__(self, db: Database | None = None):
        """Initialize repository.

        Args:
            db: Database instance. Uses global instance if not provided.
        """
        self.db = db or get_database()

    def _to_domain(self, row: ArticleTable) -> Article:
        """Convert database row to domain model."""
        content_hash = None
        if row.content_hash_value:
            content_hash = ContentHash(
                algorithm=row.content_hash_algorithm or "sha256",
                value=row.content_hash_value,
            )
        return Article(
            id=row.id,
            source_id=row.source_id,
            url=row.url,
            title=row.title,
            content=row.content,
            summary=row.summary,
            author=row.author,
            published_at=row.published_at,
            fetched_at=row.fetched_at,
            content_hash=content_hash,
            word_count=row.word_count,
            source_name=row.source_name,
            source_bias=BiasLabel(row.source_bias) if row.source_bias else None,
        )

    def _to_row(self, article: Article) -> ArticleTable:
        """Convert domain model to database row."""
        return ArticleTable(
            id=article.id,
            source_id=article.source_id,
            url=article.url,
            title=article.title,
            content=article.content,
            summary=article.summary,
            author=article.author,
            published_at=article.published_at,
            fetched_at=article.fetched_at,
            content_hash_algorithm=article.content_hash.algorithm if article.content_hash else None,
            content_hash_value=article.content_hash.value if article.content_hash else None,
            word_count=article.word_count,
            source_name=article.source_name,
            source_bias=article.source_bias.value if article.source_bias else None,
        )

    def save(self, article: Article) -> Article:
        """Save an article to the database."""
        with self.db.session() as session:
            # Check for URL duplicate
            existing = session.execute(
                select(ArticleTable).where(ArticleTable.url == article.url)
            ).scalar_one_or_none()

            if existing:
                raise DuplicateArticleError(url=article.url)

            # Check for content hash duplicate
            if article.content_hash:
                hash_match = session.execute(
                    select(ArticleTable).where(
                        ArticleTable.content_hash_value == article.content_hash.value
                    )
                ).scalar_one_or_none()

                if hash_match:
                    raise DuplicateArticleError(content_hash=article.content_hash.value)

            row = self._to_row(article)
            session.add(row)
            session.flush()
            return self._to_domain(row)

    def save_batch(self, articles: list[Article]) -> list[Article]:
        """Save multiple articles, skipping duplicates."""
        saved = []
        with self.db.session() as session:
            for article in articles:
                # Check for URL duplicate
                existing = session.execute(
                    select(ArticleTable).where(ArticleTable.url == article.url)
                ).scalar_one_or_none()

                if existing:
                    continue

                # Check for content hash duplicate
                if article.content_hash:
                    hash_match = session.execute(
                        select(ArticleTable).where(
                            ArticleTable.content_hash_value == article.content_hash.value
                        )
                    ).scalar_one_or_none()
                    if hash_match:
                        continue

                row = self._to_row(article)
                session.add(row)
                saved.append(article)

            session.flush()
        return saved

    def get_by_id(self, article_id: str) -> Article:
        """Get an article by ID."""
        with self.db.session() as session:
            row = session.execute(
                select(ArticleTable).where(ArticleTable.id == article_id)
            ).scalar_one_or_none()

            if not row:
                raise ArticleNotFoundError(article_id)

            return self._to_domain(row)

    def get_by_url(self, url: str) -> Article | None:
        """Get an article by URL."""
        with self.db.session() as session:
            row = session.execute(
                select(ArticleTable).where(ArticleTable.url == url)
            ).scalar_one_or_none()

            return self._to_domain(row) if row else None

    def exists_by_url(self, url: str) -> bool:
        """Check if an article exists by URL."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count()).select_from(ArticleTable).where(ArticleTable.url == url)
            ).scalar()
            return (result or 0) > 0

    def exists_by_hash(self, content_hash: ContentHash) -> bool:
        """Check if an article exists by content hash."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count())
                .select_from(ArticleTable)
                .where(ArticleTable.content_hash_value == content_hash.value)
            ).scalar()
            return (result or 0) > 0

    def list_recent(
        self,
        days: int = 7,
        source_id: str | None = None,
        bias: BiasLabel | None = None,
        limit: int = 100,
    ) -> list[Article]:
        """List recent articles with optional filters."""
        with self.db.session() as session:
            query = select(ArticleTable).order_by(ArticleTable.fetched_at.desc())

            if days > 0:
                cutoff = datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                from datetime import timedelta

                cutoff = cutoff - timedelta(days=days)
                query = query.where(ArticleTable.fetched_at >= cutoff)

            if source_id:
                query = query.where(ArticleTable.source_id == source_id)

            if bias:
                query = query.where(ArticleTable.source_bias == bias.value)

            query = query.limit(limit)
            rows = session.execute(query).scalars().all()
            return [self._to_domain(row) for row in rows]

    def list_unanalyzed(self, limit: int = 100) -> list[Article]:
        """List articles that haven't been analyzed yet."""
        from news_sentiment.storage.database import SentimentResultTable

        with self.db.session() as session:
            subquery = select(SentimentResultTable.article_id)
            query = (
                select(ArticleTable)
                .where(ArticleTable.id.notin_(subquery))
                .where(ArticleTable.word_count >= 50)  # Valid for analysis
                .order_by(ArticleTable.fetched_at.desc())
                .limit(limit)
            )
            rows = session.execute(query).scalars().all()
            return [self._to_domain(row) for row in rows]

    def count_by_source(self, source_id: str) -> int:
        """Count articles for a specific source."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count())
                .select_from(ArticleTable)
                .where(ArticleTable.source_id == source_id)
            ).scalar()
            return result or 0

    def count_by_bias(self, bias: BiasLabel) -> int:
        """Count articles by bias label."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count())
                .select_from(ArticleTable)
                .where(ArticleTable.source_bias == bias.value)
            ).scalar()
            return result or 0

    def count_total(self) -> int:
        """Count total articles in database."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count()).select_from(ArticleTable)
            ).scalar()
            return result or 0


class SourceRepository:
    """Repository for source persistence operations."""

    def __init__(self, db: Database | None = None):
        """Initialize repository."""
        self.db = db or get_database()

    def _to_domain(self, row: SourceTable) -> Source:
        """Convert database row to domain model."""
        return Source(
            id=row.id,
            name=row.name,
            slug=row.slug,
            feed_url=row.feed_url,
            website_url=row.website_url,
            bias_label=BiasLabel(row.bias_label),
            description=row.description,
            is_active=row.is_active,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _to_row(self, source: Source) -> SourceTable:
        """Convert domain model to database row."""
        return SourceTable(
            id=source.id,
            name=source.name,
            slug=source.slug,
            feed_url=source.feed_url,
            website_url=source.website_url,
            bias_label=source.bias_label.value,
            description=source.description,
            is_active=source.is_active,
            created_at=source.created_at,
            updated_at=source.updated_at,
        )

    def save(self, source: Source) -> Source:
        """Save a source to the database."""
        with self.db.session() as session:
            existing = session.execute(
                select(SourceTable).where(SourceTable.slug == source.slug)
            ).scalar_one_or_none()

            if existing:
                # Update existing source
                existing.name = source.name
                existing.feed_url = source.feed_url
                existing.website_url = source.website_url
                existing.bias_label = source.bias_label.value
                existing.description = source.description
                existing.is_active = source.is_active
                existing.updated_at = datetime.utcnow()
                session.flush()
                return self._to_domain(existing)
            else:
                row = self._to_row(source)
                session.add(row)
                session.flush()
                return self._to_domain(row)

    def get_by_slug(self, slug: str) -> Source | None:
        """Get a source by slug."""
        with self.db.session() as session:
            row = session.execute(
                select(SourceTable).where(SourceTable.slug == slug)
            ).scalar_one_or_none()
            return self._to_domain(row) if row else None

    def get_by_id(self, source_id: str) -> Source | None:
        """Get a source by ID."""
        with self.db.session() as session:
            row = session.execute(
                select(SourceTable).where(SourceTable.id == source_id)
            ).scalar_one_or_none()
            return self._to_domain(row) if row else None

    def list_active(self) -> list[Source]:
        """List all active sources."""
        with self.db.session() as session:
            rows = session.execute(
                select(SourceTable)
                .where(SourceTable.is_active == True)  # noqa: E712
                .order_by(SourceTable.name)
            ).scalars().all()
            return [self._to_domain(row) for row in rows]

    def list_all(self) -> list[Source]:
        """List all sources."""
        with self.db.session() as session:
            rows = session.execute(
                select(SourceTable).order_by(SourceTable.name)
            ).scalars().all()
            return [self._to_domain(row) for row in rows]

    def list_by_bias(self, bias: BiasLabel) -> list[Source]:
        """List sources by bias label."""
        with self.db.session() as session:
            rows = session.execute(
                select(SourceTable)
                .where(SourceTable.bias_label == bias.value)
                .where(SourceTable.is_active == True)  # noqa: E712
                .order_by(SourceTable.name)
            ).scalars().all()
            return [self._to_domain(row) for row in rows]

    def count_total(self) -> int:
        """Count total sources."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count()).select_from(SourceTable)
            ).scalar()
            return result or 0

    def count_active(self) -> int:
        """Count active sources."""
        with self.db.session() as session:
            result = session.execute(
                select(func.count())
                .select_from(SourceTable)
                .where(SourceTable.is_active == True)  # noqa: E712
            ).scalar()
            return result or 0
