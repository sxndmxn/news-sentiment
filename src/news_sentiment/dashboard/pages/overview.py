"""Overview dashboard page."""

import streamlit as st

from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import init_database
from news_sentiment.storage.repositories.article_repo import ArticleRepository, SourceRepository
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository


def render_overview() -> None:
    """Render the overview page."""
    st.header("Overview")

    # Initialize repos
    db = init_database()
    article_repo = ArticleRepository(db)
    source_repo = SourceRepository(db)
    sentiment_repo = SentimentRepository(db)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_articles = article_repo.count_total()
        st.metric("Total Articles", total_articles)

    with col2:
        total_analyzed = sentiment_repo.count_total()
        st.metric("Analyzed", total_analyzed)

    with col3:
        pending = len(article_repo.list_unanalyzed(limit=1000))
        st.metric("Pending Analysis", pending)

    with col4:
        sources = source_repo.list_active()
        st.metric("Active Sources", len(sources))

    st.divider()

    # Sentiment distribution
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Sentiment Distribution (Last 7 Days)")
        distribution = sentiment_repo.get_sentiment_distribution(days=7)

        import pandas as pd
        df = pd.DataFrame([
            {"Sentiment": "Positive", "Count": distribution["positive"]},
            {"Sentiment": "Negative", "Count": distribution["negative"]},
            {"Sentiment": "Neutral", "Count": distribution["neutral"]},
        ])
        st.bar_chart(df.set_index("Sentiment"))

    with col2:
        st.subheader("By Bias Category")
        for bias in [BiasLabel.LEFT, BiasLabel.CENTER, BiasLabel.RIGHT]:
            avg = sentiment_repo.get_average_sentiment(days=7, bias=bias)
            if avg is not None:
                st.metric(bias.value.replace("_", " ").title(), f"{avg:.3f}")

    # Recent articles
    st.divider()
    st.subheader("Recent Analysis Results")

    recent = sentiment_repo.list_recent(days=7, limit=10)
    if recent:
        import pandas as pd
        data = []
        for r in recent:
            data.append({
                "Title": (r.article_title or "")[:60],
                "Sentiment": r.sentiment_score.sentiment_label,
                "Score": f"{r.sentiment_score.compound:.3f}",
                "Bias": r.source_bias.value.replace("_", " ").title() if r.source_bias else "-",
                "Loaded %": f"{r.loaded_language.loaded_ratio:.1%}" if r.loaded_language else "-",
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No analyzed articles yet. Run `news-sentiment analyze` to analyze articles.")
