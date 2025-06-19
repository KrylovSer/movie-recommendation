import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image
from io import BytesIO

CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "films_data.csv")
st.write("Ищу файл по пути:", CSV_FILE)
st.write("CSV существует?", os.path.exists(CSV_FILE))

@st.cache_resource
def load_data():
    if not os.path.exists(CSV_FILE):
        st.error(f"Файл `{CSV_FILE}` не найден. Загрузите его или проверьте путь.")
        return pd.DataFrame()
    return pd.read_csv(CSV_FILE, encoding="utf-8")

@st.cache_data(show_spinner=False)
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: #d4a5a5; font-size: 48px;'>🎬 10 случайных фильмов</h1>", unsafe_allow_html=True)

    with st.spinner("Загружаем данные..."):
        df = load_data()


    if df.empty:
        st.warning("Файл пустой или не найден.")
        return
    
    if st.button("🎥 Показать подборку фильмов", type='primary', help="Нажми, чтобы отобразить 10 случайных фильмов"):
        random_samples = df.sample(10)

        for idx, row in random_samples.iterrows():
            with st.container():
                st.markdown(
    f"<h3 style='text-decoration: underline; color: white; margin-bottom: 0.5em;'>"
    f"<a href='{row['page_url']}' style='text-decoration: underline; color: #d4a5a5; font-size: 35px;'>"
    f"{row['movie_title']}</a></h3>",
    unsafe_allow_html=True
)
                cols = st.columns([2, 7])
                with cols[0]:
                    if pd.notna(row['image_url']) and row['image_url'].startswith("http"):
                        img = load_image_from_url(row['image_url'])
                        if img:
                            st.image(img, width=300)
                        else:
                            st.write("❌ Не удалось загрузить изображение")
                with cols[1]:
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Год:</b> {row['year']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Жанр:</b> {row['genre']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Режиссер:</b> {row['director']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Актеры:</b> {row['actors']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Рейтинг IMDb:</b> {row['rating']}</p>", unsafe_allow_html=True)
                    st.markdown(f"""<div style='max-height: 200px; overflow-y: auto; font-size:22px; color:#ffecda; line-height:1.2; margin:0.2'>{row['description']}</div>""", unsafe_allow_html=True)

                st.divider()

if __name__ == "__main__":
    main()