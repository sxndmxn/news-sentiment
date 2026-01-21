"""Data lineage tracking for analysis pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class ProcessingStep:
    """Record of a single processing step in the pipeline."""

    step_name: str
    component: str
    version: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    input_size: int | None = None
    output_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def complete(self, output_size: int | None = None) -> None:
        """Mark step as completed."""
        self.completed_at = datetime.utcnow()
        if self.completed_at and self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        if output_size is not None:
            self.output_size = output_size

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.completed_at = datetime.utcnow()
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "component": self.component,
            "version": self.version,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class LineageRecord:
    """Complete lineage record for an analysis run."""

    id: str = field(default_factory=lambda: str(uuid4()))
    article_id: str | None = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    steps: list[ProcessingStep] = field(default_factory=list)
    source_version: str | None = None
    pipeline_version: str = "1.0.0"

    def add_step(
        self,
        step_name: str,
        component: str,
        version: str,
        input_size: int | None = None,
        **metadata: Any,
    ) -> ProcessingStep:
        """Add a new processing step."""
        step = ProcessingStep(
            step_name=step_name,
            component=component,
            version=version,
            started_at=datetime.utcnow(),
            input_size=input_size,
            metadata=metadata,
        )
        self.steps.append(step)
        return step

    def complete(self) -> None:
        """Mark entire processing as complete."""
        self.completed_at = datetime.utcnow()

    @property
    def total_duration_ms(self) -> int | None:
        """Total processing duration in milliseconds."""
        if not self.completed_at:
            return None
        return int((self.completed_at - self.started_at).total_seconds() * 1000)

    @property
    def has_errors(self) -> bool:
        """Check if any step had an error."""
        return any(step.error is not None for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "article_id": self.article_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "steps": [step.to_dict() for step in self.steps],
            "source_version": self.source_version,
            "pipeline_version": self.pipeline_version,
            "has_errors": self.has_errors,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class LineageTracker:
    """Track lineage across multiple articles."""

    def __init__(self, pipeline_version: str = "1.0.0"):
        """Initialize lineage tracker.

        Args:
            pipeline_version: Version of the analysis pipeline
        """
        self.pipeline_version = pipeline_version
        self._records: dict[str, LineageRecord] = {}

    def start_record(self, article_id: str) -> LineageRecord:
        """Start a new lineage record for an article."""
        record = LineageRecord(
            article_id=article_id,
            pipeline_version=self.pipeline_version,
        )
        self._records[article_id] = record
        return record

    def get_record(self, article_id: str) -> LineageRecord | None:
        """Get lineage record for an article."""
        return self._records.get(article_id)

    def complete_record(self, article_id: str) -> LineageRecord | None:
        """Complete a lineage record."""
        record = self._records.get(article_id)
        if record:
            record.complete()
        return record

    def get_all_records(self) -> list[LineageRecord]:
        """Get all lineage records."""
        return list(self._records.values())

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
