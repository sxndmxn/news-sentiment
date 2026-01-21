"""Alert notification - Multi-channel dispatch using Apprise."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from news_sentiment.alerts.detector import Alert

logger = logging.getLogger(__name__)


class Notifier:
    """Multi-channel notification dispatcher using Apprise."""

    def __init__(self, config_path: Path | str | None = None):
        """Initialize notifier.

        Args:
            config_path: Path to alerts.yaml config file
        """
        if config_path is None:
            config_path = self._find_config_path()
        self.config_path = Path(config_path) if config_path else None
        self._apprise = None
        self._config = None

    def _find_config_path(self) -> Path | None:
        """Find the alerts config file."""
        # Check relative to cwd
        cwd_path = Path.cwd() / "config" / "alerts.yaml"
        if cwd_path.exists():
            return cwd_path

        # Check environment variable
        config_dir = os.getenv("NEWS_SENTIMENT_CONFIG_DIR", "config")
        env_path = Path(config_dir) / "alerts.yaml"
        if env_path.exists():
            return env_path

        return None

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self._config is not None:
            return self._config

        if self.config_path is None or not self.config_path.exists():
            self._config = {"channels": [], "enabled": False}
            return self._config

        try:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading alerts config: {e}")
            self._config = {"channels": [], "enabled": False}

        return self._config

    def _get_apprise(self):
        """Get or create Apprise instance."""
        if self._apprise is not None:
            return self._apprise

        try:
            import apprise

            self._apprise = apprise.Apprise()

            config = self._load_config()
            channels = config.get("channels", [])

            for channel in channels:
                if channel.get("enabled", True):
                    url = channel.get("url")
                    if url:
                        self._apprise.add(url)
                        logger.debug(
                            f"Added notification channel: {channel.get('name', 'unnamed')}"
                        )

        except ImportError:
            logger.warning("Apprise not installed. Notifications disabled.")
            self._apprise = None

        return self._apprise

    def send(self, alert: Alert) -> bool:
        """Send a notification for an alert.

        Args:
            alert: Alert to notify about

        Returns:
            True if notification was sent successfully
        """
        config = self._load_config()
        if not config.get("enabled", True):
            logger.debug("Notifications disabled in config")
            return False

        apprise = self._get_apprise()
        if apprise is None:
            logger.warning("Apprise not available, cannot send notification")
            return False

        # Format the message
        title = f"[{alert.level.value.upper()}] {alert.alert_type.value}"
        body = self._format_alert(alert)

        try:
            # Determine notification type based on level
            notify_type = "info"
            if alert.level.value == "warning":
                notify_type = "warning"
            elif alert.level.value == "critical":
                notify_type = "failure"

            result = apprise.notify(
                title=title,
                body=body,
                notify_type=notify_type,
            )

            if result:
                logger.info(f"Sent notification for alert: {alert.alert_type.value}")
            else:
                logger.warning(f"Failed to send notification for alert: {alert.alert_type.value}")

            return result

        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    def send_batch(self, alerts: list[Alert]) -> dict:
        """Send notifications for multiple alerts.

        Args:
            alerts: List of alerts to notify about

        Returns:
            Dict with success/failure counts
        """
        results = {"sent": 0, "failed": 0}

        for alert in alerts:
            if self.send(alert):
                results["sent"] += 1
            else:
                results["failed"] += 1

        return results

    def _format_alert(self, alert: Alert) -> str:
        """Format an alert into a notification message."""
        lines = [
            alert.message,
            "",
            f"Type: {alert.alert_type.value}",
            f"Level: {alert.level.value}",
        ]

        if alert.value is not None:
            lines.append(f"Value: {alert.value:.4f}")

        if alert.threshold is not None:
            lines.append(f"Threshold: {alert.threshold:.4f}")

        if alert.bias_label:
            lines.append(f"Bias: {alert.bias_label.value}")

        if alert.source_id:
            lines.append(f"Source: {alert.source_id}")

        lines.append(f"Time: {alert.triggered_at.isoformat()}")

        return "\n".join(lines)

    def test_connection(self) -> bool:
        """Test the notification channels.

        Returns:
            True if test notification was sent successfully
        """
        apprise = self._get_apprise()
        if apprise is None:
            return False

        try:
            return apprise.notify(
                title="News Sentiment Analyzer - Test",
                body="This is a test notification from the News Sentiment Analyzer.",
                notify_type="info",
            )
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
            return False


def send_alert(alert: Alert) -> bool:
    """Convenience function to send a single alert."""
    notifier = Notifier()
    return notifier.send(alert)


def send_alerts(alerts: list[Alert]) -> dict:
    """Convenience function to send multiple alerts."""
    notifier = Notifier()
    return notifier.send_batch(alerts)
