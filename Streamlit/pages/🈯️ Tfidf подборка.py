import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse

CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "films_data.csv")

@st.cache_data
def load_csv_data(CSV_FILE):
    return pd.read_csv(CSV_FILE)

class MovieSearchEngine:
    def __init__(self, df: pd.DataFrame, cache_dir="tfidf_cache"):
        self.df = df.copy()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.texts_path = os.path.join(cache_dir, "texts.joblib")
        self.vectorizer_path = os.path.join(cache_dir, "vectorizer.joblib")
        self.matrix_path = os.path.join(cache_dir, "tfidf_matrix.npz")

        self._prepare_text()
        self._load_or_fit()

    def _prepare_text(self):
        self.df['search_text'] = (
            self.df['movie_title'].fillna('').str.lower().apply(self._clean_text) + " " +
            self.df['genre'].fillna('').str.lower() + " " +
            self.df['description'].fillna('').str.lower().apply(self._clean_text) + " " +
            self.df['director'].fillna('').str.lower() + " " +
            self.df['actors'].fillna('').str.lower()
        )
        joblib.dump(self.df['search_text'].tolist(), self.texts_path)

    def _clean_text(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join([w for w in text.split() if len(w) > 2])

    def _load_or_fit(self):
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.matrix_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.tfidf_matrix = sparse.load_npz(self.matrix_path)
        else:
            self.vectorizer = TfidfVectorizer(max_features=50000)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_text'])
            joblib.dump(self.vectorizer, self.vectorizer_path)
            sparse.save_npz(self.matrix_path, self.tfidf_matrix)

    def search(self, query: str, top_n=10):
        query_vec = self.vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-top_n:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['similarity'] = scores[top_indices]
        return results[results.similarity > 0.1]

# Интерфейс Streamlit
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    html, body, .block-container {
        font-size: 22px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #d4a5a5;'>🎬 Поиск фильмов с помощью TF-IDF</h1>", unsafe_allow_html=True)

data_path = os.path.join(os.path.dirname(__file__), "..", "films_data.csv")
df = load_csv_data(data_path)

engine = MovieSearchEngine(df)

query = st.text_input("Введите ваш запрос")

if query:
    with st.spinner("🔎 Ищем похожие фильмы..."):
        results = engine.search(query)

    st.markdown(f"<h4>🔍 Найдено результатов: {len(results)}</h4>", unsafe_allow_html=True)

    for idx, row in results.iterrows():
        st.markdown(f"<h3 style='color:#d4a5a5'>{row['movie_title']} ({row['year']})</h3>", unsafe_allow_html=True)

        cols = st.columns([2, 7])
        with cols[0]:
            if isinstance(row['image_url'], str) and row['image_url'].startswith("http"):
                st.image(row['image_url'], width=300)

        with cols[1]:
            st.markdown(f"**Жанр:** {row['genre']}")
            st.markdown(f"**Режиссер:** {row['director']}")
            st.markdown(f"**Актеры:** {row['actors']}")
            st.markdown(f"**Описание:** {row['description']}")
            st.markdown(f"**Сходство:** `{row['similarity']:.3f}`")

        st.divider()
