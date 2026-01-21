# News Sentiment Analyzer - Fix Plan

## Phase 1: Foundation (Domain + Infrastructure)
- [x] Create pyproject.toml with dependencies
- [x] Create src/news_sentiment/ package structure
- [x] Implement domain/models.py (Article, Source aggregates)
- [x] Implement domain/value_objects.py (BiasLabel, SentimentScore)
- [x] Implement domain/exceptions.py
- [x] Implement storage/database.py (SQLAlchemy engine, session)
- [x] Implement storage/repositories/article_repo.py
- [x] Create config/feeds.yaml with 10 RSS sources
- [x] Implement ingestion/rss_parser.py with feedparser
- [x] Implement ingestion/quality.py (validators)
- [x] Implement ingestion/circuit_breaker.py (tenacity)
- [x] Implement cli/main.py with Typer (ingest, status commands)
- [x] Test: ingest articles from one source successfully

## Phase 2: Analysis Pipeline (Strategy + Lineage)
- [x] Implement analysis/analyzers/base.py (Protocol)
- [x] Implement analysis/analyzers/vader.py
- [x] Implement analysis/analyzers/factory.py
- [x] Implement analysis/lineage.py (processing tracking)
- [x] Implement analysis/lexicon.py (loaded language detection)
- [x] Create config/lexicons/ with word lists (hedges.txt, intensifiers.txt, weasel_words.txt)
- [x] Implement analysis/pipeline.py (orchestrator)
- [x] Implement analysis/preprocessor.py (text cleaning)
- [x] Add cli analyze command
- [x] Implement storage/repositories/sentiment_repo.py
- [x] Test: analyze articles with lineage output

## Phase 3: Persistence + Aggregation
- [x] Implement analytics/aggregator.py (daily summaries)
- [x] Implement domain/events.py (domain events)
- [x] Implement scheduler/jobs.py
- [x] Implement scheduler/runner.py (APScheduler)
- [x] Implement dashboard/app.py (basic Streamlit)
- [x] Implement dashboard/pages/overview.py
- [x] Implement dashboard/pages/trends.py
- [x] Implement dashboard/pages/sources.py
- [x] Implement dashboard/pages/explorer.py
- [x] Test: scheduler runs, dashboard shows data

## Phase 4: Semantic Search (Optional - skip if time constrained)
- [ ] Add sentence-transformers dependency
- [ ] Implement analysis/embeddings.py
- [ ] Implement storage/vector_store.py (ChromaDB)
- [ ] Test: semantic search works

## Phase 5: Alerts + Polish
- [ ] Implement alerts/detector.py
- [ ] Implement alerts/notifier.py (Apprise)
- [ ] Create config/alerts.yaml
- [ ] Implement analytics/exports.py (CSV, JSON)
- [ ] Add more sources (total 18+)
- [ ] Final testing and documentation

## Completed
- [x] Project initialization
- [x] Phase 1: Foundation (Domain + Infrastructure)
- [x] Phase 2: Analysis Pipeline (Strategy + Lineage)
- [x] Phase 3: Persistence + Aggregation
