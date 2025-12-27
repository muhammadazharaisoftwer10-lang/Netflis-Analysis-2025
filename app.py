import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from pathlib import Path
from collections import Counter

# ----------------------------- STREAMLIT CONFIG -----------------------------
st.set_page_config(
    page_title="Netflix Executive Dashboard",
    layout="wide",
    page_icon="🎬"
)

# ----------------------------- CUSTOM CSS -----------------------------
st.markdown("""
<style>
.stApp { background-color: #111111; color: white; }
.metric-label { font-size: 16px; color: #FF4136; font-weight: bold; }
.metric-value { font-size: 28px; color: white; font-weight: bold; }
h2, h3 { color: #FF4136; }
.stButton>button { background-color: #FF4136; color: white; border-radius: 8px; padding: 8px 20px; }
[data-testid="stSidebar"] { background-color: #1C1C1C; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------
st.title("🎬 Netflix Executive Dashboard")
st.markdown("Analyze Netflix content trends, ratings, and generate AI-driven genre insights.")

# ----------------------------- LOAD CSV -----------------------------
st.subheader("Upload Netflix CSV file")
uploaded_file = st.file_uploader("Upload Netflix CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ----------------------------- DATA CLEANING -----------------------------
    df['director'] = df['director'].fillna('Unknown')
    df['description'] = df['description'].fillna('')
    df['country'] = df['country'].fillna('Not Specified')
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month_name()
    df['rating'] = df['rating'].fillna('Not Rated')

    # ----------------------------- LOAD ML MODELS -----------------------------
    tfidf_path = Path("models/tfidf_vectorizer.pkl")
    nn_path = Path("models/genre_decision_model.pkl")
    if tfidf_path.exists() and nn_path.exists():
        tfidf = joblib.load(tfidf_path)
        nn = joblib.load(nn_path)
    else:
        tfidf = nn = None

    # ----------------------------- METRICS -----------------------------
    m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
    m1.metric("Total Content", len(df))
    m2.metric("Total Movies", len(df[df['type'] == "Movie"]))
    m3.metric("Total TV Shows", len(df[df['type'] == "TV Show"]))
    m4.metric("Unique Directors", df['director'].nunique())
    m5.metric("Unique Countries", df['country'].nunique())

    st.divider()

    # ----------------------------- SIDEBAR FILTERS -----------------------------
    with st.sidebar:
        st.header("🔍 Filters")
        content_type = st.multiselect("Content Type", ["Movie", "TV Show"], default=["Movie", "TV Show"])
        country_filter = st.multiselect("Country", sorted(df['country'].unique()), default=sorted(df['country'].unique()))
        year_filter = st.slider("Year Added",
                                min_value=int(df['year_added'].min()),
                                max_value=int(df['year_added'].max()),
                                value=(int(df['year_added'].min()), int(df['year_added'].max())))
        if st.button("Reset Filters"):
            content_type = ["Movie", "TV Show"]
            country_filter = sorted(df['country'].unique())
            year_filter = (int(df['year_added'].min()), int(df['year_added'].max()))

    # ----------------------------- FILTER DATA -----------------------------
    df_filtered = df[
        (df['type'].isin(content_type)) &
        (df['country'].isin(country_filter)) &
        (df['year_added'] >= year_filter[0]) &
        (df['year_added'] <= year_filter[1])
    ]

    # ----------------------------- CONTENT TREND -----------------------------
    st.subheader("📈 Content Trend (Movies vs TV Shows)")
    trend = df_filtered.groupby(['year_added', 'type']).size().reset_index(name='count')
    fig_trend = px.line(trend, x='year_added', y='count', color='type',
                        markers=True, title="Content Added Over Years",
                        color_discrete_map={'Movie':'#FF4136','TV Show':'#FF851B'})
    fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # ----------------------------- GENRE DISTRIBUTION -----------------------------
    st.subheader("🎭 Genre Distribution")
    df_genres = df_filtered.copy()
    df_genres['listed_in'] = df_genres['listed_in'].str.split(',')
    df_genres = df_genres.explode('listed_in')
    df_genres['listed_in'] = df_genres['listed_in'].str.strip()
    genre_counts = df_genres['listed_in'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig_genre = px.bar(genre_counts, x='Genre', y='Count', text='Count', color='Count',
                       color_continuous_scale='Reds', title="Most Frequent Genres")
    fig_genre.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_genre, use_container_width=True)

    st.divider()

    # ----------------------------- RELEASES BY MONTH -----------------------------
    st.subheader("🔥 Releases by Month")
    month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    month_counts = df_filtered['month_added'].value_counts().reindex(month_order).fillna(0)
    fig_month = px.imshow([month_counts.values], x=month_order, y=["Releases"], text_auto=True,
                          color_continuous_scale='Reds')
    fig_month.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_month, use_container_width=True)

    st.divider()

    # ----------------------------- RATING PATTERNS -----------------------------
    st.subheader("⭐ Rating Patterns")
    rating_counts = df_filtered['rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    fig_rating = px.pie(
        rating_counts,
        names='Rating',
        values='Count',
        color_discrete_sequence=px.colors.sequential.Reds,
        title="Content Ratings Distribution",
        text_auto=True  # fixed textinfo error
    )
    st.plotly_chart(fig_rating, use_container_width=True)

    st.divider()

    # ----------------------------- GENRE RECOMMENDER -----------------------------
    st.subheader("🤖 Genre Decision AI")
    if tfidf and nn:
        idea = st.text_input("Enter content idea (e.g., sci-fi crime, AI documentary)")
        if st.button("🔮 Recommend Genres"):
            vec = tfidf.transform([idea])
            _, idx = nn.kneighbors(vec)
            genres = df.iloc[idx[0]]['listed_in']
            genre_list = [g.strip() for g in ",".join(genres).split(",")]
            top_genres = [g for g, _ in Counter(genre_list).most_common(3)]
            st.markdown(f"<h3 style='color:#FF4136;'>🎯 Recommended Genres: {', '.join(top_genres)}</h3>", unsafe_allow_html=True)
    else:
        st.warning("Model not trained yet! Run: python train_model.py")

    st.divider()

    # ----------------------------- DOWNLOAD DATA -----------------------------
    st.subheader("⬇ Cleaned Dataset Download")
    st.download_button(
        "Download CSV",
        df_filtered.to_csv(index=False).encode('utf-8'),
        "cleaned_netflix_data.csv",
        "text/csv"
    )

    st.success("🚀 Dashboard Updated — Ready for Executive Use!")

else:
    st.warning("Please upload the Netflix CSV file to continue.")
