"""Circuit breaker pattern for resilient RSS feed fetching."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from news_sentiment.domain.exceptions import (
    CircuitBreakerOpenError,
    FeedFetchError,
    IngestionError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests are rejected
    - HALF_OPEN: Recovery testing, one request allowed to test

    Uses tenacity for retry logic within the CLOSED state.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_base_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker (e.g., source slug)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            retry_attempts: Number of retries within closed state
            retry_base_wait: Base wait time for exponential backoff
            retry_max_wait: Maximum wait time between retries
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.retry_attempts = retry_attempts
        self.retry_base_wait = retry_base_wait
        self.retry_max_wait = retry_max_wait

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._last_success_time: datetime | None = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
        return self._state

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._last_success_time = datetime.utcnow()
        if self._state != CircuitState.CLOSED:
            logger.info(f"Circuit '{self.name}' transitioning to CLOSED")
            self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()

        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit '{self.name}' OPEN after {self._failure_count} failures"
                )
            self._state = CircuitState.OPEN

    def call(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Zero-argument callable to execute

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenError: If circuit is open
            IngestionError: If call fails after retries
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            retry_after = None
            if self._last_failure_time:
                elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
                retry_after = max(0, self.recovery_timeout - elapsed)
            raise CircuitBreakerOpenError(self.name, retry_after)

        try:
            # Use tenacity for retries in CLOSED/HALF_OPEN state
            retryer = Retrying(
                stop=stop_after_attempt(self.retry_attempts),
                wait=wait_exponential(
                    multiplier=self.retry_base_wait,
                    max=self.retry_max_wait,
                ),
                retry=retry_if_exception_type((FeedFetchError, IngestionError, IOError)),
                reraise=True,
            )

            result = retryer(func)
            self._record_success()
            return result

        except RetryError as e:
            self._record_failure()
            # Re-raise the underlying exception
            if e.last_attempt.exception():
                raise e.last_attempt.exception()
            raise

        except Exception as e:
            self._record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit '{self.name}' manually reset")

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
            "last_success": self._last_success_time.isoformat() if self._last_success_time else None,
            "recovery_timeout_seconds": self.recovery_timeout,
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        retry_attempts: int = 3,
    ):
        """Initialize circuit breaker registry.

        Args:
            failure_threshold: Default failure threshold for new breakers
            recovery_timeout: Default recovery timeout for new breakers
            retry_attempts: Default retry attempts for new breakers
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.retry_attempts = retry_attempts
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                retry_attempts=self.retry_attempts,
            )
        return self._breakers[name]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_all_status(self) -> list[dict[str, Any]]:
        """Get status of all circuit breakers."""
        return [breaker.get_status() for breaker in self._breakers.values()]

    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]


# Global registry instance
_cb_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _cb_registry
    if _cb_registry is None:
        _cb_registry = CircuitBreakerRegistry()
    return _cb_registry
