"""Alert detection - Monitor for threshold violations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import get_database
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts that can be triggered."""

    SENTIMENT_SPIKE = "sentiment_spike"
    SENTIMENT_DROP = "sentiment_drop"
    HIGH_NEGATIVITY = "high_negativity"
    BIAS_DIVERGENCE = "bias_divergence"
    NO_DATA = "no_data"
    HIGH_LOADED_LANGUAGE = "high_loaded_language"


@dataclass
class Alert:
    """Represents a triggered alert."""

    alert_type: AlertType
    level: AlertLevel
    message: str
    value: float | None = None
    threshold: float | None = None
    source_id: str | None = None
    bias_label: BiasLabel | None = None
    triggered_at: datetime = None

    def __post_init__(self) -> None:
        if self.triggered_at is None:
            self.triggered_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "level": self.level.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "source_id": self.source_id,
            "bias_label": self.bias_label.value if self.bias_label else None,
            "triggered_at": self.triggered_at.isoformat(),
        }


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    alert_type: AlertType
    threshold: float
    level: AlertLevel = AlertLevel.WARNING
    enabled: bool = True
    lookback_hours: int = 24


class AlertDetector:
    """Detects alert conditions based on sentiment data."""

    DEFAULT_RULES = [
        AlertRule(AlertType.SENTIMENT_SPIKE, threshold=0.3, level=AlertLevel.INFO),
        AlertRule(AlertType.SENTIMENT_DROP, threshold=-0.3, level=AlertLevel.WARNING),
        AlertRule(AlertType.HIGH_NEGATIVITY, threshold=0.6, level=AlertLevel.WARNING),
        AlertRule(AlertType.BIAS_DIVERGENCE, threshold=0.4, level=AlertLevel.INFO),
        AlertRule(AlertType.NO_DATA, threshold=0, level=AlertLevel.WARNING),
        AlertRule(AlertType.HIGH_LOADED_LANGUAGE, threshold=0.1, level=AlertLevel.INFO),
    ]

    def __init__(self, rules: list[AlertRule] | None = None):
        """Initialize detector.

        Args:
            rules: Alert rules to use. Defaults to DEFAULT_RULES.
        """
        self.rules = {r.alert_type: r for r in (rules or self.DEFAULT_RULES)}
        self.db = get_database()
        self.sentiment_repo = SentimentRepository(self.db)

    def check_all(self) -> list[Alert]:
        """Check all alert conditions.

        Returns:
            List of triggered alerts
        """
        alerts = []

        for alert_type, rule in self.rules.items():
            if not rule.enabled:
                continue

            try:
                triggered = self._check_rule(alert_type, rule)
                alerts.extend(triggered)
            except Exception as e:
                logger.error(f"Error checking rule {alert_type}: {e}")

        return alerts

    def _check_rule(self, alert_type: AlertType, rule: AlertRule) -> list[Alert]:
        """Check a specific alert rule.

        Args:
            alert_type: Type of alert to check
            rule: Rule configuration

        Returns:
            List of triggered alerts
        """
        if alert_type == AlertType.SENTIMENT_SPIKE:
            return self._check_sentiment_change(rule, positive=True)
        elif alert_type == AlertType.SENTIMENT_DROP:
            return self._check_sentiment_change(rule, positive=False)
        elif alert_type == AlertType.HIGH_NEGATIVITY:
            return self._check_high_negativity(rule)
        elif alert_type == AlertType.BIAS_DIVERGENCE:
            return self._check_bias_divergence(rule)
        elif alert_type == AlertType.NO_DATA:
            return self._check_no_data(rule)
        elif alert_type == AlertType.HIGH_LOADED_LANGUAGE:
            return self._check_loaded_language(rule)
        return []

    def _check_sentiment_change(self, rule: AlertRule, positive: bool) -> list[Alert]:
        """Check for significant sentiment changes."""
        alerts = []

        # Compare recent sentiment to baseline
        _ = rule.lookback_hours  # Reserved for future hourly granularity
        recent = self.sentiment_repo.get_average_sentiment(days=1)
        baseline = self.sentiment_repo.get_average_sentiment(days=7)

        if recent is None or baseline is None:
            return alerts

        change = recent - baseline

        if positive and change >= rule.threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.SENTIMENT_SPIKE,
                    level=rule.level,
                    message=f"Sentiment increased by {change:.3f} (threshold: {rule.threshold})",
                    value=change,
                    threshold=rule.threshold,
                )
            )
        elif not positive and change <= -abs(rule.threshold):
            alerts.append(
                Alert(
                    alert_type=AlertType.SENTIMENT_DROP,
                    level=rule.level,
                    message=f"Sentiment decreased by {abs(change):.3f} (threshold: {rule.threshold})",
                    value=change,
                    threshold=-abs(rule.threshold),
                )
            )

        return alerts

    def _check_high_negativity(self, rule: AlertRule) -> list[Alert]:
        """Check for high proportion of negative articles."""
        alerts = []

        distribution = self.sentiment_repo.get_sentiment_distribution(days=1)
        total = sum(distribution.values())

        if total == 0:
            return alerts

        negative_ratio = distribution["negative"] / total

        if negative_ratio >= rule.threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.HIGH_NEGATIVITY,
                    level=rule.level,
                    message=f"High negativity: {negative_ratio:.1%} of articles are negative",
                    value=negative_ratio,
                    threshold=rule.threshold,
                )
            )

        return alerts

    def _check_bias_divergence(self, rule: AlertRule) -> list[Alert]:
        """Check for divergence between bias categories."""
        alerts = []

        left_avg = self.sentiment_repo.get_average_sentiment(days=1, bias=BiasLabel.LEFT)
        right_avg = self.sentiment_repo.get_average_sentiment(days=1, bias=BiasLabel.RIGHT)

        if left_avg is None or right_avg is None:
            return alerts

        divergence = abs(left_avg - right_avg)

        if divergence >= rule.threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.BIAS_DIVERGENCE,
                    level=rule.level,
                    message=f"Sentiment divergence between left ({left_avg:.3f}) and right ({right_avg:.3f}): {divergence:.3f}",
                    value=divergence,
                    threshold=rule.threshold,
                )
            )

        return alerts

    def _check_no_data(self, rule: AlertRule) -> list[Alert]:
        """Check if no new data has been ingested recently."""
        alerts = []

        recent = self.sentiment_repo.list_recent(days=1, limit=1)

        if not recent:
            alerts.append(
                Alert(
                    alert_type=AlertType.NO_DATA,
                    level=rule.level,
                    message="No articles analyzed in the last 24 hours",
                    value=0,
                    threshold=rule.threshold,
                )
            )

        return alerts

    def _check_loaded_language(self, rule: AlertRule) -> list[Alert]:
        """Check for high loaded language in recent articles."""
        alerts = []

        recent = self.sentiment_repo.list_recent(days=1, limit=100)

        if not recent:
            return alerts

        loaded_ratios = []
        for r in recent:
            if r.loaded_language:
                loaded_ratios.append(r.loaded_language.loaded_ratio)

        if not loaded_ratios:
            return alerts

        avg_loaded = sum(loaded_ratios) / len(loaded_ratios)

        if avg_loaded >= rule.threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.HIGH_LOADED_LANGUAGE,
                    level=rule.level,
                    message=f"High loaded language average: {avg_loaded:.1%}",
                    value=avg_loaded,
                    threshold=rule.threshold,
                )
            )

        return alerts


def check_alerts() -> list[Alert]:
    """Convenience function to check all alerts."""
    detector = AlertDetector()
    return detector.check_all()
