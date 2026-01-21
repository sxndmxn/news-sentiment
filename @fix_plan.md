# News Sentiment Analyzer - Fix Plan

## Phase 1: Foundation (Domain + Infrastructure)
- [ ] Create pyproject.toml with dependencies
- [ ] Create src/news_sentiment/ package structure
- [ ] Implement domain/models.py (Article, Source aggregates)
- [ ] Implement domain/value_objects.py (BiasLabel, SentimentScore)
- [ ] Implement domain/exceptions.py
- [ ] Implement storage/database.py (SQLAlchemy engine, session)
- [ ] Implement storage/repositories/article_repo.py
- [ ] Create config/feeds.yaml with 10 RSS sources
- [ ] Implement ingestion/rss_parser.py with feedparser
- [ ] Implement ingestion/quality.py (validators)
- [ ] Implement ingestion/circuit_breaker.py (tenacity)
- [ ] Implement cli/main.py with Typer (ingest, status commands)
- [ ] Test: ingest articles from one source successfully

## Phase 2: Analysis Pipeline (Strategy + Lineage)
- [ ] Implement analysis/analyzers/base.py (Protocol)
- [ ] Implement analysis/analyzers/vader.py
- [ ] Implement analysis/analyzers/factory.py
- [ ] Implement analysis/lineage.py (processing tracking)
- [ ] Implement analysis/lexicon.py (loaded language detection)
- [ ] Create config/lexicons/ with word lists (hedges.txt, intensifiers.txt, weasel_words.txt)
- [ ] Implement analysis/pipeline.py (orchestrator)
- [ ] Implement analysis/preprocessor.py (text cleaning)
- [ ] Add cli analyze command
- [ ] Implement storage/repositories/sentiment_repo.py
- [ ] Test: analyze articles with lineage output

## Phase 3: Persistence + Aggregation
- [ ] Implement analytics/aggregator.py (daily summaries)
- [ ] Implement domain/events.py (domain events)
- [ ] Implement scheduler/jobs.py
- [ ] Implement scheduler/runner.py (APScheduler)
- [ ] Implement dashboard/app.py (basic Streamlit)
- [ ] Implement dashboard/pages/overview.py
- [ ] Test: scheduler runs, dashboard shows data

## Phase 4: Semantic Search (Optional - skip if time constrained)
- [ ] Add sentence-transformers dependency
- [ ] Implement analysis/embeddings.py
- [ ] Implement storage/vector_store.py (ChromaDB)
- [ ] Implement dashboard/pages/explorer.py
- [ ] Test: semantic search works

## Phase 5: Alerts + Polish
- [ ] Implement alerts/detector.py
- [ ] Implement alerts/notifier.py (Apprise)
- [ ] Create config/alerts.yaml
- [ ] Implement analytics/exports.py (CSV, JSON)
- [ ] Add more sources (total 18+)
- [ ] Implement dashboard/pages/trends.py
- [ ] Implement dashboard/pages/sources.py
- [ ] Final testing and documentation

## Completed
- [x] Project initialization
