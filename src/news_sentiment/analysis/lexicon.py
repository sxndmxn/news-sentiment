"""Loaded language detection using lexicon-based approach."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from news_sentiment.domain.value_objects import LoadedLanguageScore

if TYPE_CHECKING:
    pass


class LexiconLoader:
    """Load and manage lexicon word lists from files."""

    def __init__(self, lexicons_dir: Path | str | None = None):
        """Initialize lexicon loader.

        Args:
            lexicons_dir: Directory containing lexicon text files.
                         Defaults to config/lexicons/
        """
        if lexicons_dir is None:
            lexicons_dir = self._find_lexicons_dir()
        self.lexicons_dir = Path(lexicons_dir)

    def _find_lexicons_dir(self) -> Path:
        """Find the lexicons directory."""
        # Check relative to cwd
        cwd_path = Path.cwd() / "config" / "lexicons"
        if cwd_path.exists():
            return cwd_path

        # Check relative to package
        package_dir = Path(__file__).parent.parent.parent.parent
        package_path = package_dir / "config" / "lexicons"
        if package_path.exists():
            return package_path

        # Check environment variable
        config_dir = os.getenv("NEWS_SENTIMENT_CONFIG_DIR", "config")
        env_path = Path(config_dir) / "lexicons"
        if env_path.exists():
            return env_path

        return Path("config/lexicons")

    def load_lexicon(self, name: str) -> set[str]:
        """Load a lexicon from a text file.

        Args:
            name: Name of lexicon file (without .txt extension)

        Returns:
            Set of words/phrases from the lexicon
        """
        file_path = self.lexicons_dir / f"{name}.txt"
        if not file_path.exists():
            return set()

        words = set()
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#"):
                    words.add(line.lower())

        return words


class LoadedLanguageDetector:
    """Detect loaded language patterns in text.

    Loaded language includes:
    - Hedges: Words that indicate uncertainty (maybe, possibly, seemingly)
    - Intensifiers: Words that amplify claims (absolutely, extremely, definitely)
    - Weasel words: Vague attributions (sources say, experts believe, some people)
    """

    def __init__(self, lexicons_dir: Path | str | None = None):
        """Initialize detector with lexicons.

        Args:
            lexicons_dir: Directory containing lexicon files
        """
        loader = LexiconLoader(lexicons_dir)

        self.hedges = loader.load_lexicon("hedges")
        self.intensifiers = loader.load_lexicon("intensifiers")
        self.weasel_words = loader.load_lexicon("weasel_words")

        # Compile patterns for multi-word phrases
        self._hedge_patterns = self._compile_patterns(self.hedges)
        self._intensifier_patterns = self._compile_patterns(self.intensifiers)
        self._weasel_patterns = self._compile_patterns(self.weasel_words)

    def _compile_patterns(self, words: set[str]) -> list[re.Pattern[str]]:
        """Compile regex patterns for word/phrase matching."""
        patterns = []
        for word in words:
            # Escape special regex characters and create word boundary pattern
            escaped = re.escape(word)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _count_matches(self, text: str, patterns: list[re.Pattern[str]]) -> int:
        """Count total matches of patterns in text."""
        count = 0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count

    def analyze(self, text: str) -> LoadedLanguageScore:
        """Analyze text for loaded language.

        Args:
            text: Text content to analyze

        Returns:
            LoadedLanguageScore with counts for each category
        """
        if not text or not text.strip():
            return LoadedLanguageScore()

        # Count words
        words = text.split()
        total_words = len(words)

        # Count loaded language occurrences
        hedges_count = self._count_matches(text, self._hedge_patterns)
        intensifiers_count = self._count_matches(text, self._intensifier_patterns)
        weasel_count = self._count_matches(text, self._weasel_patterns)

        return LoadedLanguageScore(
            hedges_count=hedges_count,
            intensifiers_count=intensifiers_count,
            weasel_words_count=weasel_count,
            total_words=total_words,
        )

    def get_matches(self, text: str) -> dict[str, list[str]]:
        """Get all loaded language matches with their categories.

        Args:
            text: Text to analyze

        Returns:
            Dict mapping category to list of matched words/phrases
        """
        matches: dict[str, list[str]] = {
            "hedges": [],
            "intensifiers": [],
            "weasel_words": [],
        }

        for pattern in self._hedge_patterns:
            found = pattern.findall(text)
            matches["hedges"].extend(found)

        for pattern in self._intensifier_patterns:
            found = pattern.findall(text)
            matches["intensifiers"].extend(found)

        for pattern in self._weasel_patterns:
            found = pattern.findall(text)
            matches["weasel_words"].extend(found)

        return matches


# Global instance for convenience
_detector: LoadedLanguageDetector | None = None


def get_detector() -> LoadedLanguageDetector:
    """Get the global loaded language detector."""
    global _detector
    if _detector is None:
        _detector = LoadedLanguageDetector()
    return _detector


def analyze_loaded_language(text: str) -> LoadedLanguageScore:
    """Convenience function to analyze loaded language."""
    return get_detector().analyze(text)
