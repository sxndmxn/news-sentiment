# News Sentiment Analyzer

Build a Python application that analyzes sentiment and linguistic patterns across news sources spanning the political spectrum.

## Goal
Create a CLI + dashboard application that:
1. Ingests news from RSS feeds across the political spectrum
2. Analyzes sentiment using VADER (fast) + optional transformers
3. Detects loaded language (hedges, intensifiers, weasel words)
4. Tracks historical trends and generates alerts
5. Provides a Streamlit dashboard for visualization

## Project Location
All code should be created in this directory. Use a `src/news_sentiment/` layout.

## Architectural Principles (MUST FOLLOW)

### Domain-Driven Design
- Use bounded contexts: Ingestion, Analysis, Reporting, Alerting
- Create aggregates: Article, SentimentResult, Source
- Use value objects: BiasLabel, SentimentScore, ContentHash

### Design Patterns
- **Strategy Pattern**: Analyzer interface with swappable implementations (VADER, Transformer)
- **Factory Pattern**: Create analyzers based on content domain
- **Repository Pattern**: Abstract data access behind clean interfaces
- **Circuit Breaker**: Protect against RSS feed failures (use tenacity)

### Data Engineering
- **Data Quality**: Validate at ingestion (content length, language, duplicates)
- **Data Lineage**: Track source → transformation → output for every record
- **Model Versioning**: Track which model version produced each sentiment score

## Technology Stack
- Python 3.10+ with src layout
- SQLAlchemy 2.0 for persistence (SQLite)
- Pydantic for domain models and validation
- feedparser for RSS ingestion
- vaderSentiment for fast sentiment
- spaCy for NER and text processing
- Typer + Rich for CLI
- Streamlit for dashboard
- APScheduler for job scheduling
- Apprise for alerts

## Project Structure

```
src/news_sentiment/
├── __init__.py
├── __main__.py
├── domain/                      # DDD: Core domain models
│   ├── models.py                # Article, SentimentResult, Entity
│   ├── events.py                # Domain events
│   ├── value_objects.py         # BiasLabel, SentimentScore, etc.
│   └── exceptions.py            # Domain-specific exceptions
├── ingestion/                   # Bounded Context: Data Ingestion
│   ├── rss_parser.py            # feedparser wrapper
│   ├── source_registry.py       # YAML-based feed definitions
│   ├── deduplication.py         # URL/content hash dedup
│   ├── quality.py               # Data quality validators
│   └── circuit_breaker.py       # Resilience for feed fetching
├── analysis/                    # Bounded Context: NLP Analysis
│   ├── pipeline.py              # Orchestrator with stage pattern
│   ├── preprocessor.py          # Text cleaning, chunking
│   ├── analyzers/               # Strategy pattern implementations
│   │   ├── base.py              # Abstract analyzer interface
│   │   ├── vader.py             # Fast lexicon-based
│   │   ├── transformer.py       # Hugging Face models
│   │   └── factory.py           # Analyzer factory
│   ├── lexicon.py               # Loaded language detection
│   ├── embeddings.py            # Sentence embeddings
│   └── lineage.py               # Track processing provenance
├── storage/                     # Infrastructure: Persistence
│   ├── database.py              # Connection management
│   ├── repositories/            # Repository pattern
│   │   ├── article_repo.py
│   │   ├── sentiment_repo.py
│   │   └── analytics_repo.py
│   ├── vector_store.py          # ChromaDB for embeddings
│   └── migrations/              # Alembic migrations
├── analytics/                   # Bounded Context: Reporting
│   ├── aggregator.py            # Daily/weekly summaries
│   ├── trends.py                # Trend detection
│   ├── statistics.py            # Statistical analysis
│   └── exports.py               # CSV, JSON exports
├── alerts/                      # Bounded Context: Alerting
│   ├── detector.py              # Threshold monitoring
│   ├── notifier.py              # Multi-channel dispatch
│   └── rules.py                 # Alert rule definitions
├── dashboard/                   # Interface: Web UI
│   ├── app.py                   # Streamlit entry
│   └── pages/
│       ├── overview.py
│       ├── trends.py
│       ├── sources.py
│       └── explorer.py
├── cli/                         # Interface: Command Line
│   ├── main.py                  # Typer app
│   └── commands/
│       ├── ingest.py
│       ├── analyze.py
│       ├── query.py
│       └── export.py
└── scheduler/                   # Infrastructure: Job Scheduling
    ├── jobs.py
    └── runner.py

config/
├── feeds.yaml                   # RSS sources with bias labels
├── alerts.yaml                  # Alert configuration
├── lexicons/                    # Loaded language word lists
│   ├── hedges.txt
│   ├── intensifiers.txt
│   └── weasel_words.txt
└── settings.yaml                # App settings

data/                            # Local data (gitignored)
├── news_sentiment.db
├── chroma/
└── exports/

tests/
├── unit/
├── integration/
└── fixtures/
```

## Quality Standards
- Type hints on all functions
- Docstrings on public methods
- Unit tests for core logic
- Use ruff for linting

## Task Tracking
Follow the phased implementation in @fix_plan.md. Complete each phase fully before moving to the next.

## Exit Condition
The project is complete when:
1. All phases in @fix_plan.md are checked off
2. CLI commands work: `news-sentiment ingest`, `news-sentiment analyze`, `news-sentiment status`
3. Dashboard launches: `streamlit run src/news_sentiment/dashboard/app.py`
4. At least 5 news sources are configured and working

## 🎯 Status Reporting (CRITICAL - Ralph needs this!)

**IMPORTANT**: At the end of your response, ALWAYS include this status block:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of what to do next>
---END_RALPH_STATUS---
```

### When to set EXIT_SIGNAL: true

Set EXIT_SIGNAL to **true** when ALL of these conditions are met:
1. ✅ All items in @fix_plan.md are marked [x]
2. ✅ All tests are passing (or no tests exist for valid reasons)
3. ✅ No errors or warnings in the last execution
4. ✅ CLI commands work and dashboard launches
5. ✅ You have nothing meaningful left to implement
