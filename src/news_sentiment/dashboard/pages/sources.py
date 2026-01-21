"""Sources dashboard page."""

import streamlit as st

from news_sentiment.storage.database import init_database
from news_sentiment.storage.repositories.article_repo import ArticleRepository, SourceRepository
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository


def render_sources() -> None:
    """Render the sources page."""
    st.header("News Sources")

    db = init_database()
    article_repo = ArticleRepository(db)
    source_repo = SourceRepository(db)
    sentiment_repo = SentimentRepository(db)

    sources = source_repo.list_all()

    if sources:
        import pandas as pd

        data = []
        for src in sources:
            article_count = article_repo.count_by_source(src.id)
            analyzed_count = sentiment_repo.count_by_source(src.id)
            avg = sentiment_repo.get_average_sentiment(days=30, source_id=src.id)

            data.append(
                {
                    "Name": src.name,
                    "Slug": src.slug,
                    "Bias": src.bias_label.value.replace("_", " ").title(),
                    "Articles": article_count,
                    "Analyzed": analyzed_count,
                    "Avg Sentiment": f"{avg:.3f}" if avg is not None else "-",
                    "Active": "✓" if src.is_active else "✗",
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Breakdown by bias
        st.subheader("Sources by Bias")
        bias_counts = {}
        for src in sources:
            bias = src.bias_label.value.replace("_", " ").title()
            bias_counts[bias] = bias_counts.get(bias, 0) + 1

        cols = st.columns(min(len(bias_counts), 4))
        for i, (bias, count) in enumerate(sorted(bias_counts.items())):
            with cols[i % len(cols)]:
                st.metric(bias, count)
    else:
        st.info("No sources configured. Check config/feeds.yaml.")
