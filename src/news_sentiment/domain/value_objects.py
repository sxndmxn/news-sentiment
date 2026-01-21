"""Domain value objects - Immutable objects defined by their attributes."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class BiasLabel(str, Enum):
    """Political bias classification for news sources."""

    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"

    @classmethod
    def from_string(cls, value: str) -> BiasLabel:
        """Convert string to BiasLabel, handling various formats."""
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        try:
            return cls(normalized)
        except ValueError:
            # Try matching by name
            for member in cls:
                if member.name.lower() == normalized:
                    return member
            raise ValueError(f"Unknown bias label: {value}") from None


class SentimentScore(BaseModel):
    """Sentiment analysis score with compound and component values."""

    model_config = {"frozen": True}

    compound: Annotated[float, Field(ge=-1.0, le=1.0)]
    positive: Annotated[float, Field(ge=0.0, le=1.0)]
    negative: Annotated[float, Field(ge=0.0, le=1.0)]
    neutral: Annotated[float, Field(ge=0.0, le=1.0)]

    @property
    def sentiment_label(self) -> str:
        """Get human-readable sentiment label based on compound score."""
        if self.compound >= 0.05:
            return "positive"
        elif self.compound <= -0.05:
            return "negative"
        return "neutral"

    @property
    def is_strong(self) -> bool:
        """Check if sentiment is strongly positive or negative."""
        return abs(self.compound) >= 0.5

    @classmethod
    def neutral_score(cls) -> SentimentScore:
        """Create a neutral sentiment score."""
        return cls(compound=0.0, positive=0.0, negative=0.0, neutral=1.0)


class ContentHash(BaseModel):
    """Content hash for deduplication purposes."""

    model_config = {"frozen": True}

    algorithm: str = "sha256"
    value: str

    @field_validator("value")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate hash is a valid hex string."""
        if not v:
            raise ValueError("Hash value cannot be empty")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("Hash must be a valid hexadecimal string") from None
        return v.lower()

    @classmethod
    def from_content(cls, content: str, algorithm: str = "sha256") -> ContentHash:
        """Create a content hash from text content."""
        # Normalize whitespace for consistent hashing
        normalized = " ".join(content.split())
        hash_value = hashlib.new(algorithm, normalized.encode("utf-8")).hexdigest()
        return cls(algorithm=algorithm, value=hash_value)

    def matches(self, other: ContentHash) -> bool:
        """Check if two hashes match."""
        return self.algorithm == other.algorithm and self.value == other.value


class LoadedLanguageScore(BaseModel):
    """Score for loaded language detection."""

    model_config = {"frozen": True}

    hedges_count: int = Field(ge=0, default=0)
    intensifiers_count: int = Field(ge=0, default=0)
    weasel_words_count: int = Field(ge=0, default=0)
    total_words: int = Field(ge=0, default=0)

    @property
    def total_loaded(self) -> int:
        """Total count of loaded language markers."""
        return self.hedges_count + self.intensifiers_count + self.weasel_words_count

    @property
    def loaded_ratio(self) -> float:
        """Ratio of loaded language to total words."""
        if self.total_words == 0:
            return 0.0
        return self.total_loaded / self.total_words

    @property
    def is_high_loaded(self) -> bool:
        """Check if content has unusually high loaded language."""
        return self.loaded_ratio > 0.05  # More than 5% loaded words


class AnalysisVersion(BaseModel):
    """Version information for analysis results (model versioning)."""

    model_config = {"frozen": True}

    analyzer_name: str
    analyzer_version: str
    model_name: str | None = None
    model_version: str | None = None

    def __str__(self) -> str:
        """String representation of version."""
        base = f"{self.analyzer_name}:{self.analyzer_version}"
        if self.model_name:
            base += f"/{self.model_name}"
            if self.model_version:
                base += f":{self.model_version}"
        return base
