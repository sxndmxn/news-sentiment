"""VADER sentiment analyzer implementation."""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from news_sentiment.analysis.analyzers.base import BaseAnalyzer
from news_sentiment.domain.value_objects import SentimentScore


class VaderAnalyzer(BaseAnalyzer):
    """VADER (Valence Aware Dictionary and sEntiment Reasoner) analyzer.

    A lexicon and rule-based sentiment analysis tool specifically attuned to
    sentiments expressed in social media. Works well for news content too.

    VADER outputs:
    - pos: Proportion of text that is positive
    - neg: Proportion of text that is negative
    - neu: Proportion of text that is neutral
    - compound: Normalized, weighted composite score (-1 to +1)
    """

    _instance: SentimentIntensityAnalyzer | None = None

    def __init__(self) -> None:
        """Initialize VADER analyzer with singleton pattern for efficiency."""
        if VaderAnalyzer._instance is None:
            VaderAnalyzer._instance = SentimentIntensityAnalyzer()
        self._analyzer = VaderAnalyzer._instance

    @property
    def name(self) -> str:
        return "vader"

    @property
    def version(self) -> str:
        return "3.3.2"

    def analyze(self, text: str) -> SentimentScore:
        """Analyze text using VADER.

        Args:
            text: Text content to analyze

        Returns:
            SentimentScore with VADER scores
        """
        if not text or not text.strip():
            return SentimentScore.neutral_score()

        # Get VADER scores
        scores = self._analyzer.polarity_scores(text)

        return SentimentScore(
            compound=scores["compound"],
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
        )

    def analyze_sentences(self, text: str) -> list[SentimentScore]:
        """Analyze text sentence by sentence.

        Useful for understanding sentiment distribution within an article.

        Args:
            text: Text content to analyze

        Returns:
            List of SentimentScore, one per sentence
        """
        import re

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [self.analyze(sentence) for sentence in sentences if sentence.strip()]
