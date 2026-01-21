"""Source registry - Load and manage news source configurations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from news_sentiment.domain.exceptions import ConfigurationError, SourceNotFoundError
from news_sentiment.domain.models import Source
from news_sentiment.domain.value_objects import BiasLabel

if TYPE_CHECKING:
    pass


class SourceRegistry:
    """Registry for managing news source configurations."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize source registry.

        Args:
            config_path: Path to feeds.yaml. Defaults to config/feeds.yaml.
        """
        if config_path is None:
            config_path = Path(os.getenv("NEWS_SENTIMENT_CONFIG_DIR", "config")) / "feeds.yaml"
        self.config_path = Path(config_path)
        self._sources: dict[str, Source] | None = None
        self._config: dict | None = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            raise ConfigurationError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {self.config_path}: {e}")

        if not self._config or "sources" not in self._config:
            raise ConfigurationError(f"No 'sources' key in {self.config_path}")

        return self._config

    def _load_sources(self) -> dict[str, Source]:
        """Load and parse all sources from config."""
        if self._sources is not None:
            return self._sources

        config = self._load_config()
        self._sources = {}

        for source_data in config.get("sources", []):
            try:
                source = Source(
                    name=source_data["name"],
                    slug=source_data["slug"],
                    feed_url=source_data["feed_url"],
                    website_url=source_data.get("website_url"),
                    bias_label=BiasLabel.from_string(source_data["bias_label"]),
                    description=source_data.get("description"),
                    is_active=source_data.get("is_active", True),
                )
                self._sources[source.slug] = source
            except (KeyError, ValueError) as e:
                raise ConfigurationError(
                    f"Invalid source configuration for '{source_data.get('slug', 'unknown')}': {e}"
                )

        return self._sources

    def get_source(self, slug: str) -> Source:
        """Get a source by slug."""
        sources = self._load_sources()
        if slug not in sources:
            raise SourceNotFoundError(slug)
        return sources[slug]

    def get_all_sources(self) -> list[Source]:
        """Get all configured sources."""
        return list(self._load_sources().values())

    def get_active_sources(self) -> list[Source]:
        """Get all active sources."""
        return [s for s in self._load_sources().values() if s.is_active]

    def get_sources_by_bias(self, bias: BiasLabel) -> list[Source]:
        """Get sources filtered by bias label."""
        return [s for s in self._load_sources().values() if s.bias_label == bias and s.is_active]

    def get_ingestion_settings(self) -> dict:
        """Get ingestion settings from config."""
        config = self._load_config()
        return config.get("ingestion", {
            "max_articles_per_source": 50,
            "min_content_length": 100,
            "max_content_length": 100000,
            "fetch_interval_minutes": 60,
        })

    def get_health_check_settings(self) -> dict:
        """Get health check settings from config."""
        config = self._load_config()
        return config.get("health_check", {
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "retry_delay_seconds": 5,
        })

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._sources = None
        self._config = None

    @property
    def source_count(self) -> int:
        """Total number of configured sources."""
        return len(self._load_sources())

    @property
    def active_count(self) -> int:
        """Number of active sources."""
        return len(self.get_active_sources())


# Global registry instance
_registry: SourceRegistry | None = None


def get_registry(config_path: str | Path | None = None) -> SourceRegistry:
    """Get or create the global source registry."""
    global _registry
    if _registry is None:
        _registry = SourceRegistry(config_path)
    return _registry
