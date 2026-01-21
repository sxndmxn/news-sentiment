"""Database connection management using SQLAlchemy 2.0."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    relationship,
    sessionmaker,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class SourceTable(Base):
    """SQLAlchemy model for news sources."""

    __tablename__ = "sources"

    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    slug = Column(String(50), nullable=False, unique=True, index=True)
    feed_url = Column(Text, nullable=False)
    website_url = Column(Text, nullable=True)
    bias_label = Column(String(20), nullable=False, index=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    articles = relationship("ArticleTable", back_populates="source")


class ArticleTable(Base):
    """SQLAlchemy model for articles."""

    __tablename__ = "articles"

    id = Column(String(36), primary_key=True)
    source_id = Column(String(36), ForeignKey("sources.id"), nullable=False, index=True)
    url = Column(Text, nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    author = Column(String(200), nullable=True)
    published_at = Column(DateTime, nullable=True, index=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, index=True)
    content_hash_algorithm = Column(String(20), default="sha256")
    content_hash_value = Column(String(64), nullable=True, index=True)
    word_count = Column(Integer, default=0)
    source_name = Column(String(100), nullable=True)
    source_bias = Column(String(20), nullable=True, index=True)

    source = relationship("SourceTable", back_populates="articles")
    sentiment_results = relationship("SentimentResultTable", back_populates="article")
    entities = relationship("EntityTable", back_populates="article")

    __table_args__ = (
        Index("ix_articles_published_source", "published_at", "source_id"),
        Index("ix_articles_bias_published", "source_bias", "published_at"),
    )


class SentimentResultTable(Base):
    """SQLAlchemy model for sentiment analysis results."""

    __tablename__ = "sentiment_results"

    id = Column(String(36), primary_key=True)
    article_id = Column(String(36), ForeignKey("articles.id"), nullable=False, index=True)
    compound = Column(Float, nullable=False)
    positive = Column(Float, nullable=False)
    negative = Column(Float, nullable=False)
    neutral = Column(Float, nullable=False)
    hedges_count = Column(Integer, default=0)
    intensifiers_count = Column(Integer, default=0)
    weasel_words_count = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    analyzer_name = Column(String(50), nullable=False)
    analyzer_version = Column(String(20), nullable=False)
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(50), nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    processing_duration_ms = Column(Integer, nullable=True)

    # Denormalized fields for efficient queries
    article_title = Column(String(500), nullable=True)
    source_id = Column(String(36), nullable=True, index=True)
    source_bias = Column(String(20), nullable=True, index=True)
    published_at = Column(DateTime, nullable=True, index=True)

    article = relationship("ArticleTable", back_populates="sentiment_results")

    __table_args__ = (
        Index("ix_sentiment_source_analyzed", "source_id", "analyzed_at"),
        Index("ix_sentiment_bias_published", "source_bias", "published_at"),
    )


class EntityTable(Base):
    """SQLAlchemy model for named entities."""

    __tablename__ = "entities"

    id = Column(String(36), primary_key=True)
    article_id = Column(String(36), ForeignKey("articles.id"), nullable=False, index=True)
    text = Column(String(200), nullable=False, index=True)
    label = Column(String(20), nullable=False, index=True)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)

    article = relationship("ArticleTable", back_populates="entities")

    __table_args__ = (Index("ix_entities_text_label", "text", "label"),)


class DailySummaryTable(Base):
    """SQLAlchemy model for daily summaries."""

    __tablename__ = "daily_summaries"

    id = Column(String(36), primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    source_id = Column(String(36), nullable=True, index=True)
    bias_label = Column(String(20), nullable=True, index=True)
    article_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    min_sentiment = Column(Float, default=0.0)
    max_sentiment = Column(Float, default=0.0)
    sentiment_std_dev = Column(Float, default=0.0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    avg_loaded_ratio = Column(Float, nullable=True)
    top_entities = Column(Text, nullable=True)  # JSON serialized
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_summaries_date_source", "date", "source_id"),
        Index("ix_summaries_date_bias", "date", "bias_label"),
    )


class Database:
    """Database connection manager."""

    def __init__(self, db_url: str | None = None):
        """Initialize database connection.

        Args:
            db_url: SQLAlchemy database URL. Defaults to SQLite in data/ directory.
        """
        if db_url is None:
            data_dir = Path(os.getenv("NEWS_SENTIMENT_DATA_DIR", "data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "news_sentiment.db"
            db_url = f"sqlite:///{db_path}"

        self.db_url = db_url
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.db_url,
                echo=os.getenv("NEWS_SENTIMENT_DB_ECHO", "").lower() == "true",
                pool_pre_ping=True,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        return self._session_factory

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic commit/rollback."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Dispose of the engine and connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


# Global database instance (lazy initialization)
_db: Database | None = None


def get_database(db_url: str | None = None) -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database(db_url)
    return _db


def init_database(db_url: str | None = None) -> Database:
    """Initialize database and create tables."""
    db = get_database(db_url)
    db.create_tables()
    return db
