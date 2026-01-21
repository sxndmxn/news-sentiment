"""Analyzer factory for creating analyzers based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from news_sentiment.analysis.analyzers.base import BaseAnalyzer
from news_sentiment.analysis.analyzers.vader import VaderAnalyzer
from news_sentiment.domain.exceptions import AnalyzerNotFoundError

if TYPE_CHECKING:
    pass


class AnalyzerFactory:
    """Factory for creating sentiment analyzers.

    Implements the Factory pattern to instantiate the appropriate analyzer
    based on configuration or content requirements.
    """

    # Registry of available analyzers
    _analyzers: dict[str, type[BaseAnalyzer]] = {
        "vader": VaderAnalyzer,
    }

    @classmethod
    def register(cls, name: str, analyzer_class: type[BaseAnalyzer]) -> None:
        """Register a new analyzer type.

        Args:
            name: Unique name for the analyzer
            analyzer_class: Class implementing BaseAnalyzer
        """
        cls._analyzers[name] = analyzer_class

    @classmethod
    def create(cls, name: str = "vader") -> BaseAnalyzer:
        """Create an analyzer by name.

        Args:
            name: Name of the analyzer to create

        Returns:
            Instance of the requested analyzer

        Raises:
            AnalyzerNotFoundError: If analyzer name is not registered
        """
        if name not in cls._analyzers:
            raise AnalyzerNotFoundError(name)

        return cls._analyzers[name]()

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available analyzer names."""
        return list(cls._analyzers.keys())

    @classmethod
    def get_default(cls) -> BaseAnalyzer:
        """Get the default analyzer (VADER)."""
        return cls.create("vader")


def get_analyzer(name: str = "vader") -> BaseAnalyzer:
    """Convenience function to get an analyzer by name."""
    return AnalyzerFactory.create(name)
