"""Scheduler runner using APScheduler."""

from __future__ import annotations

import logging
import signal
import sys
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from news_sentiment.scheduler.jobs import (
    analyze_pending_articles,
    check_alerts,
    compute_daily_summaries,
    ingest_all_sources,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SchedulerRunner:
    """Manages scheduled jobs using APScheduler."""

    def __init__(self) -> None:
        """Initialize scheduler runner."""
        self.scheduler = BackgroundScheduler()
        self._running = False

    def configure_jobs(self) -> None:
        """Configure all scheduled jobs."""
        # Ingest every hour
        self.scheduler.add_job(
            ingest_all_sources,
            trigger=IntervalTrigger(hours=1),
            id="ingest_hourly",
            name="Hourly RSS Ingestion",
            replace_existing=True,
        )

        # Analyze every 30 minutes
        self.scheduler.add_job(
            analyze_pending_articles,
            trigger=IntervalTrigger(minutes=30),
            id="analyze_30min",
            name="Analyze Pending Articles",
            replace_existing=True,
        )

        # Daily summary at 1 AM
        self.scheduler.add_job(
            compute_daily_summaries,
            trigger=CronTrigger(hour=1, minute=0),
            id="daily_summary",
            name="Daily Summary Computation",
            replace_existing=True,
        )

        # Check alerts every 15 minutes
        self.scheduler.add_job(
            check_alerts,
            trigger=IntervalTrigger(minutes=15),
            id="check_alerts",
            name="Alert Checking",
            replace_existing=True,
        )

        logger.info("Scheduled jobs configured")

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler is already running")
            return

        self.configure_jobs()
        self.scheduler.start()
        self._running = True
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Scheduler stopped")

    def run_job_now(self, job_id: str) -> None:
        """Manually trigger a job.

        Args:
            job_id: ID of the job to run
        """
        job = self.scheduler.get_job(job_id)
        if job:
            job.func()
            logger.info(f"Manually triggered job: {job_id}")
        else:
            logger.error(f"Job not found: {job_id}")

    def get_job_status(self) -> list[dict]:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger),
                }
            )
        return jobs

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


def run_scheduler() -> None:
    """Run the scheduler in the foreground.

    This is the main entry point for the scheduler service.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = SchedulerRunner()

    def signal_handler(signum: int, frame: object) -> None:
        logger.info("Received shutdown signal")
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting scheduler service")
    runner.start()

    # Run initial jobs
    logger.info("Running initial jobs...")
    ingest_all_sources()
    analyze_pending_articles()

    # Keep running
    try:
        while runner.is_running:
            import time

            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        runner.stop()


if __name__ == "__main__":
    run_scheduler()
