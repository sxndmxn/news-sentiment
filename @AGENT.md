# Build and Run Instructions

## Setup
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Run CLI
```bash
# Show help
news-sentiment --help

# Ingest from a specific source
news-sentiment ingest --source nytimes

# Ingest from all sources
news-sentiment ingest --all

# Analyze recent articles
news-sentiment analyze --days 1

# Show status
news-sentiment status
```

## Run Dashboard
```bash
streamlit run src/news_sentiment/dashboard/app.py
```

## Run Tests
```bash
pytest tests/
```

## Run Linting
```bash
ruff check src/
ruff format src/
```

## Project Structure
- `src/news_sentiment/` - Main package
- `config/` - Configuration files (feeds.yaml, alerts.yaml, lexicons/)
- `data/` - SQLite database and exports (gitignored)
- `tests/` - Test suite

## Key Dependencies
- pydantic>=2.0 - Domain models
- sqlalchemy>=2.0 - Database ORM
- feedparser>=6.0 - RSS parsing
- vaderSentiment>=3.3 - Sentiment analysis
- spacy>=3.7 - NLP processing
- typer>=0.9 - CLI framework
- streamlit>=1.28 - Dashboard
- apscheduler>=3.10 - Job scheduling
- apprise>=1.6 - Notifications

## Feature Development Quality Standards

**CRITICAL**: All new features MUST meet the following mandatory requirements before being considered complete.

### Testing Requirements

- **Test Pass Rate**: 100% - all tests must pass, no exceptions
- **Test Types Required**:
  - Unit tests for domain logic
  - Integration tests for ingestion and analysis pipelines
- **Test Quality**: Tests must validate behavior, not just achieve coverage metrics

### Git Workflow Requirements

Before moving to the next feature, ALL changes must be:

1. **Committed with Clear Messages**:
   ```bash
   git add .
   git commit -m "feat(module): descriptive message following conventional commits"
   ```

2. **Branch Hygiene**:
   - Work on feature branches when appropriate
   - Create pull requests for significant changes

3. **Ralph Integration**:
   - Update @fix_plan.md with new tasks before starting work
   - Mark items complete in @fix_plan.md upon completion

### Feature Completion Checklist

Before marking ANY feature as complete, verify:

- [ ] All tests pass
- [ ] Code formatted with ruff
- [ ] Type hints on all functions
- [ ] All changes committed
- [ ] @fix_plan.md task marked as complete
