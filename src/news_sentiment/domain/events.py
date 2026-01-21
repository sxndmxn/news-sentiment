"""Domain events for event-driven architecture."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base class for domain events."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.event_type = self.__class__.__name__


@dataclass
class ArticleIngested(DomainEvent):
    """Event raised when an article is ingested."""

    article_id: str = ""
    source_id: str = ""
    url: str = ""
    title: str = ""


@dataclass
class ArticleAnalyzed(DomainEvent):
    """Event raised when an article is analyzed."""

    article_id: str = ""
    sentiment_result_id: str = ""
    compound_score: float = 0.0
    sentiment_label: str = ""


@dataclass
class DailySummaryComputed(DomainEvent):
    """Event raised when a daily summary is computed."""

    summary_id: str = ""
    date: datetime | None = None
    article_count: int = 0
    avg_sentiment: float = 0.0


@dataclass
class CircuitBreakerStateChanged(DomainEvent):
    """Event raised when a circuit breaker changes state."""

    source_slug: str = ""
    previous_state: str = ""
    new_state: str = ""
    failure_count: int = 0


@dataclass
class AlertTriggered(DomainEvent):
    """Event raised when an alert condition is triggered."""

    alert_id: str = ""
    alert_type: str = ""
    threshold: float = 0.0
    actual_value: float = 0.0
    message: str = ""


class EventBus:
    """Simple event bus for publishing and subscribing to domain events."""

    _handlers: dict[str, list[Callable[[DomainEvent], None]]]

    def __init__(self) -> None:
        self._handlers = {}

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        """Subscribe a handler to an event type."""
        type_name = event_type.__name__
        if type_name not in self._handlers:
            self._handlers[type_name] = []
        self._handlers[type_name].append(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribed handlers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't fail on handler errors
                import logging

                logging.getLogger(__name__).error(
                    f"Error in event handler for {event.event_type}: {e}"
                )

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def publish_event(event: DomainEvent) -> None:
    """Convenience function to publish an event."""
    get_event_bus().publish(event)
