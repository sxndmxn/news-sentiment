"""Trends dashboard page."""

import streamlit as st

from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import init_database
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository


def render_trends() -> None:
    """Render the trends page."""
    st.header("Sentiment Trends")

    db = init_database()
    sentiment_repo = SentimentRepository(db)

    # Show what data we have
    recent = sentiment_repo.list_recent(days=30, limit=100)
    if recent:
        import pandas as pd

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
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, bias in enumerate([BiasLabel.LEFT, BiasLabel.CENTER, BiasLabel.RIGHT]):
                bias_df = df[df["bias"] == bias.value]
                if not bias_df.empty:
                    avg = bias_df["sentiment"].mean()
                    with cols[i]:
                        st.metric(bias.value.replace("_", " ").title(), f"{avg:.3f}")
    else:
        st.info("No data available for trends. Ingest and analyze articles first.")
