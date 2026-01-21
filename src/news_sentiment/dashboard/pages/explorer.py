"""Article explorer dashboard page."""

import streamlit as st

from news_sentiment.domain.value_objects import BiasLabel
from news_sentiment.storage.database import init_database
from news_sentiment.storage.repositories.article_repo import ArticleRepository
from news_sentiment.storage.repositories.sentiment_repo import SentimentRepository


def render_explorer() -> None:
    """Render the article explorer page."""
    st.header("Article Explorer")

    db = init_database()
    article_repo = ArticleRepository(db)
    sentiment_repo = SentimentRepository(db)

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

            data.append(
                {
                    "Title": article.title[:80],
                    "Source": article.source_name or "-",
                    "Bias": article.source_bias.value.replace("_", " ").title()
                    if article.source_bias
                    else "-",
                    "Words": article.word_count,
                    "Sentiment": f"{sentiment.sentiment_score.compound:.3f}" if sentiment else "-",
                    "Published": article.published_at.strftime("%Y-%m-%d")
                    if article.published_at
                    else "-",
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Article detail view
        st.subheader("Article Detail")
        titles = [a.title[:80] for a in articles]
        selected_idx = st.selectbox(
            "Select article", range(len(titles)), format_func=lambda i: titles[i]
        )

        if selected_idx is not None:
            selected = articles[selected_idx]
            st.markdown(f"**{selected.title}**")
            st.markdown(f"*{selected.source_name}* | {selected.published_at}")
            st.markdown(f"URL: {selected.url}")

            with st.expander("Content Preview"):
                content = selected.content or ""
                st.text(content[:1000] + "..." if len(content) > 1000 else content)

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
