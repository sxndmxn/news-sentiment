# News Sentiment Analyzer

Analyze sentiment and loaded language across news sources spanning the political
spectrum. This project provides a CLI for ingestion and analysis plus a
Streamlit dashboard for exploration.

## What it does
- Ingests RSS feeds into a local SQLite database.
- Runs sentiment scoring (VADER by default).
- Detects loaded language (hedges, intensifiers, weasel words).
- Shows trends and source comparisons in a dashboard.

## Requirements

Use Python 3.10 or newer.
Run the setup commands from the repository root.

## Quickstart (uv)
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Inspect sources:
   ```bash
   uv run news-sentiment sources
   ```
3. Ingest and analyze:
   ```bash
   uv run news-sentiment ingest --all
   uv run news-sentiment analyze
   uv run news-sentiment status
   ```
4. Launch the dashboard:
   ```bash
   uv run streamlit run src/news_sentiment/dashboard/app.py
   ```

## Quickstart (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
news-sentiment --help
```

## CLI commands (common)
- `news-sentiment sources` list configured feeds
- `news-sentiment ingest --all` fetch RSS content
- `news-sentiment analyze` score unanalyzed articles
- `news-sentiment status` show database stats
- `news-sentiment export` placeholder (not implemented yet)

## Config and data
- Feeds live in `config/feeds.yaml`.
- Alerts live in `config/alerts.yaml`.
- Loaded-language lexicons live in `config/lexicons/`.
- Data is stored in `data/news_sentiment.db` by default.
- Override locations:
  - `NEWS_SENTIMENT_CONFIG_DIR` for config root
  - `NEWS_SENTIMENT_DATA_DIR` for data root

## Adding feeds
Edit `config/feeds.yaml` and add a source entry with:
`slug`, `name`, `feed_url`, `website_url`, `bias_label`, `description`.
Valid `bias_label` values: `far_left`, `left`, `center_left`, `center`,
`center_right`, `right`, `far_right`.

Example:
```yaml
- slug: example
  name: Example News
  feed_url: https://example.com/rss
  website_url: https://example.com
  bias_label: center
  description: Example description
```

## Optional extras
- Transformers: `uv sync --extra transformers`
- Vector store: `uv sync --extra vector`

## Scheduler (optional)
Run periodic ingestion, analysis, and alerts:
```bash
uv run python -m news_sentiment.scheduler.runner
```

## Notes
- RSS ingestion requires network access.
