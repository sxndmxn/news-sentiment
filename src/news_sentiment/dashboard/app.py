"""Streamlit dashboard for News Sentiment Analyzer."""

from __future__ import annotations

import streamlit as st

from news_sentiment.storage.database import init_database
from news_sentiment.storage.repositories.article_repo import ArticleRepository, SourceRepository
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository
from news_sentiment.domain.value_objects import BiasLabel


# Page config
st.set_page_config(
    page_title="News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database and repositories
@st.cache_resource
def get_db():
    return init_database()

db = get_db()
article_repo = ArticleRepository(db)
source_repo = SourceRepository(db)
sentiment_repo = SentimentRepository(db)


def main() -> None:
    """Main dashboard entry point."""
    st.title("📰 News Sentiment Analyzer")
    st.markdown("Analyze sentiment across news sources spanning the political spectrum")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Overview", "Trends", "Sources", "Explorer"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats
        st.subheader("Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Articles", article_repo.count_total())
        with col2:
            st.metric("Analyzed", sentiment_repo.count_total())

        # Average sentiment
        avg_sentiment = sentiment_repo.get_average_sentiment(days=7)
        if avg_sentiment is not None:
            st.metric("7-Day Avg Sentiment", f"{avg_sentiment:.3f}")

    # Main content based on page selection
    if page == "Overview":
        show_overview()
    elif page == "Trends":
        show_trends()
    elif page == "Sources":
        show_sources()
    elif page == "Explorer":
        show_explorer()


def show_overview() -> None:
    """Display overview dashboard."""
    st.header("Overview")

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


def show_trends() -> None:
    """Display sentiment trends."""
    st.header("Sentiment Trends")
    st.info("Trend visualization will be fully implemented in Phase 5.")

    # Show what data we have
    recent = sentiment_repo.list_recent(days=30, limit=100)
    if recent:
        import pandas as pd
        from datetime import datetime

        # Group by date
        data = []
        for r in recent:
            if r.analyzed_at:
                data.append({
                    "date": r.analyzed_at.date(),
                    "sentiment": r.sentiment_score.compound,
                    "bias": r.source_bias.value if r.source_bias else "unknown",
                })

        if data:
            df = pd.DataFrame(data)
            daily = df.groupby("date")["sentiment"].mean().reset_index()
            daily.columns = ["Date", "Average Sentiment"]

            st.line_chart(daily.set_index("Date"))

            st.subheader("By Bias Category")
            for bias in [BiasLabel.LEFT, BiasLabel.CENTER, BiasLabel.RIGHT]:
                bias_df = df[df["bias"] == bias.value]
                if not bias_df.empty:
                    avg = bias_df["sentiment"].mean()
                    st.metric(bias.value.replace("_", " ").title(), f"{avg:.3f}")
    else:
        st.info("No data available for trends. Ingest and analyze articles first.")


def show_sources() -> None:
    """Display source management."""
    st.header("News Sources")

    sources = source_repo.list_all()

    if sources:
        import pandas as pd
        data = []
        for src in sources:
            article_count = article_repo.count_by_source(src.id)
            analyzed_count = sentiment_repo.count_by_source(src.id)
            avg = sentiment_repo.get_average_sentiment(days=30, source_id=src.id)

            data.append({
                "Name": src.name,
                "Slug": src.slug,
                "Bias": src.bias_label.value.replace("_", " ").title(),
                "Articles": article_count,
                "Analyzed": analyzed_count,
                "Avg Sentiment": f"{avg:.3f}" if avg is not None else "-",
                "Active": "✓" if src.is_active else "✗",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Breakdown by bias
        st.subheader("Sources by Bias")
        bias_counts = {}
        for src in sources:
            bias = src.bias_label.value.replace("_", " ").title()
            bias_counts[bias] = bias_counts.get(bias, 0) + 1

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i, (bias, count) in enumerate(sorted(bias_counts.items())):
            with cols[i % 3]:
                st.metric(bias, count)
    else:
        st.info("No sources configured. Check config/feeds.yaml.")


def show_explorer() -> None:
    """Display article explorer."""
    st.header("Article Explorer")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        days = st.slider("Days", 1, 30, 7)

    with col2:
        bias_options = ["All"] + [b.value.replace("_", " ").title() for b in BiasLabel]
        selected_bias = st.selectbox("Bias Filter", bias_options)

    with col3:
        limit = st.slider("Max Results", 10, 100, 25)

    # Get articles
    bias_filter = None
    if selected_bias != "All":
        bias_filter = BiasLabel.from_string(selected_bias.lower().replace(" ", "_"))

    articles = article_repo.list_recent(days=days, bias=bias_filter, limit=limit)

    if articles:
        import pandas as pd
        data = []
        for article in articles:
            # Get sentiment if available
            sentiment = sentiment_repo.get_by_article_id(article.id)

            data.append({
                "Title": article.title[:80],
                "Source": article.source_name or "-",
                "Bias": article.source_bias.value.replace("_", " ").title() if article.source_bias else "-",
                "Words": article.word_count,
                "Sentiment": f"{sentiment.sentiment_score.compound:.3f}" if sentiment else "-",
                "Published": article.published_at.strftime("%Y-%m-%d") if article.published_at else "-",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Article detail view
        st.subheader("Article Detail")
        selected_title = st.selectbox("Select article", [a.title[:80] for a in articles])

        if selected_title:
            selected = next((a for a in articles if a.title.startswith(selected_title[:80])), None)
            if selected:
                st.markdown(f"**{selected.title}**")
                st.markdown(f"*{selected.source_name}* | {selected.published_at}")
                st.markdown(f"URL: {selected.url}")

                with st.expander("Content Preview"):
                    st.text(selected.content[:1000] + "..." if len(selected.content) > 1000 else selected.content)

                sentiment = sentiment_repo.get_by_article_id(selected.id)
                if sentiment:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Compound", f"{sentiment.sentiment_score.compound:.3f}")
                    with col2:
                        st.metric("Positive", f"{sentiment.sentiment_score.positive:.3f}")
                    with col3:
                        st.metric("Negative", f"{sentiment.sentiment_score.negative:.3f}")

                    if sentiment.loaded_language:
                        st.subheader("Loaded Language")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Hedges", sentiment.loaded_language.hedges_count)
                        with col2:
                            st.metric("Intensifiers", sentiment.loaded_language.intensifiers_count)
                        with col3:
                            st.metric("Weasel Words", sentiment.loaded_language.weasel_words_count)
                        with col4:
                            st.metric("Loaded %", f"{sentiment.loaded_language.loaded_ratio:.1%}")
    else:
        st.info("No articles found matching the filters.")


if __name__ == "__main__":
    main()
