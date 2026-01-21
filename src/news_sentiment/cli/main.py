"""CLI application entry point using Typer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from news_sentiment import __version__

# Initialize Typer app
app = typer.Typer(
    name="news-sentiment",
    help="Analyze sentiment across news sources spanning the political spectrum.",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"news-sentiment version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-V", help="Enable verbose logging."),
    ] = False,
) -> None:
    """News Sentiment Analyzer - Analyze news across the political spectrum."""
    setup_logging(verbose)


@app.command()
def ingest(
    source: Annotated[
        Optional[str],
        typer.Option("--source", "-s", help="Specific source slug to ingest."),
    ] = None,
    all_sources: Annotated[
        bool,
        typer.Option("--all", "-a", help="Ingest from all active sources."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum articles per source."),
    ] = 50,
) -> None:
    """Ingest articles from RSS feeds.

    Either specify a source with --source or use --all for all sources.
    """
    from news_sentiment.ingestion.deduplication import DeduplicationService
    from news_sentiment.ingestion.rss_parser import RSSParser
    from news_sentiment.ingestion.source_registry import get_registry
    from news_sentiment.storage.database import init_database
    from news_sentiment.storage.repositories.article_repo import (
        ArticleRepository,
        SourceRepository,
    )

    # Initialize database
    db = init_database()

    # Get source registry
    registry = get_registry()

    # Get sources to process
    if source:
        try:
            sources = [registry.get_source(source)]
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    elif all_sources:
        sources = registry.get_active_sources()
    else:
        console.print("[yellow]Specify --source or --all to ingest articles.[/yellow]")
        raise typer.Exit(1)

    if not sources:
        console.print("[yellow]No sources found.[/yellow]")
        raise typer.Exit(0)

    # Initialize services
    parser = RSSParser()
    article_repo = ArticleRepository(db)
    source_repo = SourceRepository(db)
    dedup_service = DeduplicationService(article_repo)

    # Process each source
    total_articles = 0
    total_saved = 0

    with console.status("[bold green]Ingesting articles...") as status:
        for src in sources:
            status.update(f"[bold green]Fetching from {src.name}...")

            # Save source to database
            source_repo.save(src)

            try:
                articles = parser.fetch_feed(src)
                fetched_count = len(articles)

                # Limit articles
                articles = articles[:limit]

                # Filter duplicates
                unique_articles = dedup_service.filter_duplicates(articles)

                # Save to database
                saved = article_repo.save_batch(unique_articles)

                total_articles += fetched_count
                total_saved += len(saved)

                console.print(
                    f"  [green]✓[/green] {src.name}: {len(saved)} new / {fetched_count} fetched"
                )

            except Exception as e:
                console.print(f"  [red]✗[/red] {src.name}: {e}")
                continue

    console.print()
    console.print(f"[bold]Total:[/bold] {total_saved} articles saved ({total_articles} fetched)")


@app.command()
def status() -> None:
    """Show current database and ingestion status."""
    from news_sentiment.ingestion.circuit_breaker import get_circuit_breaker_registry
    from news_sentiment.ingestion.source_registry import get_registry
    from news_sentiment.storage.database import init_database
    from news_sentiment.storage.repositories.article_repo import (
        ArticleRepository,
        SourceRepository,
    )
    from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository

    db = init_database()
    article_repo = ArticleRepository(db)
    source_repo = SourceRepository(db)
    sentiment_repo = SentimentRepository(db)
    registry = get_registry()
    cb_registry = get_circuit_breaker_registry()

    # Database stats
    console.print("[bold]Database Status[/bold]")
    console.print(f"  Sources: {source_repo.count_total()}")
    console.print(f"  Articles: {article_repo.count_total()}")
    console.print(f"  Analyzed: {sentiment_repo.count_total()}")

    # Unanalyzed articles
    unanalyzed = article_repo.list_unanalyzed(limit=1000)
    console.print(f"  Pending analysis: {len(unanalyzed)}")

    # Average sentiment
    avg = sentiment_repo.get_average_sentiment(days=7)
    if avg is not None:
        console.print(f"  7-day avg sentiment: {avg:.3f}")

    console.print()

    # Source status table
    console.print("[bold]Configured Sources[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Source", style="cyan")
    table.add_column("Bias", style="magenta")
    table.add_column("Articles", justify="right")
    table.add_column("Analyzed", justify="right")
    table.add_column("Status", style="green")

    sources = registry.get_active_sources()
    for src in sources:
        article_count = article_repo.count_by_source(src.id)
        analyzed_count = sentiment_repo.count_by_source(src.id)
        # Check circuit breaker
        cb = cb_registry.get_breaker(src.slug)
        cb_status = cb.state.value.upper()
        status_style = "green" if cb_status == "CLOSED" else "red"

        table.add_row(
            src.name,
            src.bias_label.value.replace("_", " ").title(),
            str(article_count),
            str(analyzed_count),
            f"[{status_style}]{cb_status}[/{status_style}]",
        )

    console.print(table)

    # Circuit breaker status
    open_circuits = cb_registry.get_open_circuits()
    if open_circuits:
        console.print()
        console.print(f"[yellow]Warning: {len(open_circuits)} circuit(s) open[/yellow]")
        for circuit in open_circuits:
            console.print(f"  - {circuit}")


@app.command()
def sources() -> None:
    """List all configured news sources."""
    from news_sentiment.ingestion.source_registry import get_registry

    registry = get_registry()
    sources = registry.get_all_sources()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Slug", style="cyan")
    table.add_column("Name")
    table.add_column("Bias", style="magenta")
    table.add_column("Active", justify="center")
    table.add_column("Feed URL", style="dim")

    for src in sources:
        table.add_row(
            src.slug,
            src.name,
            src.bias_label.value.replace("_", " ").title(),
            "[green]Yes[/green]" if src.is_active else "[red]No[/red]",
            src.feed_url[:50] + "..." if len(src.feed_url) > 50 else src.feed_url,
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {len(sources)} sources configured")


@app.command()
def analyze(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Analyze articles from the last N days."),
    ] = 1,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum articles to analyze."),
    ] = 100,
    source: Annotated[
        Optional[str],
        typer.Option("--source", "-s", help="Only analyze articles from this source."),
    ] = None,
) -> None:
    """Analyze sentiment of unanalyzed articles."""
    from news_sentiment.analysis.pipeline import AnalysisPipeline
    from news_sentiment.storage.database import init_database
    from news_sentiment.storage.repositories.article_repo import ArticleRepository
    from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository

    db = init_database()
    article_repo = ArticleRepository(db)
    sentiment_repo = SentimentRepository(db)

    # Get unanalyzed articles
    articles = article_repo.list_unanalyzed(limit=limit)

    if not articles:
        console.print("[yellow]No unanalyzed articles found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Found {len(articles)} articles to analyze.[/bold]")

    # Create analysis pipeline
    pipeline = AnalysisPipeline()

    # Analyze with progress
    analyzed_count = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    with console.status("[bold green]Analyzing articles...") as status:
        for i, article in enumerate(articles):
            status.update(f"[bold green]Analyzing {i + 1}/{len(articles)}...")

            try:
                result = pipeline.analyze_article(article)
                sentiment_repo.save(result)
                analyzed_count += 1

                # Count sentiment categories
                label = result.sentiment_score.sentiment_label
                if label == "positive":
                    positive_count += 1
                elif label == "negative":
                    negative_count += 1
                else:
                    neutral_count += 1

            except Exception as e:
                console.print(f"  [red]✗[/red] Error analyzing article: {e}")
                continue

    console.print()
    console.print("[bold]Analysis Complete[/bold]")
    console.print(f"  Analyzed: {analyzed_count} articles")
    console.print(f"  Positive: [green]{positive_count}[/green]")
    console.print(f"  Negative: [red]{negative_count}[/red]")
    console.print(f"  Neutral:  [yellow]{neutral_count}[/yellow]")

    # Show recent results
    if analyzed_count > 0:
        console.print()
        console.print("[bold]Recent Results[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Title", style="cyan", max_width=40)
        table.add_column("Sentiment", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Loaded", justify="right")

        recent = sentiment_repo.list_recent(days=1, limit=5)
        for result in recent:
            label = result.sentiment_score.sentiment_label
            label_style = "green" if label == "positive" else "red" if label == "negative" else "yellow"
            loaded_ratio = f"{result.loaded_language.loaded_ratio:.1%}" if result.loaded_language else "-"

            title = result.article_title or ""
            table.add_row(
                title[:37] + "..." if len(title) > 40 else title,
                f"[{label_style}]{label}[/{label_style}]",
                f"{result.sentiment_score.compound:.3f}",
                loaded_ratio,
            )

        console.print(table)


@app.command()
def export(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path."),
    ] = Path("data/export.json"),
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Export format (json, csv)."),
    ] = "json",
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Export articles from the last N days."),
    ] = 7,
) -> None:
    """Export articles and analysis results.

    This command will be fully implemented in Phase 5.
    """
    console.print("[yellow]Export functionality coming in Phase 5.[/yellow]")
    console.print(f"  Output: {output}")
    console.print(f"  Format: {format}")
    console.print(f"  Days: {days}")


if __name__ == "__main__":
    app()
