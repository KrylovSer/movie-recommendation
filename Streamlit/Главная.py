import streamlit as st
import os
from PIL import Image

BASE_DIR = os.path.dirname(__file__)  # Папка, где лежит Главная.py
image_path = os.path.join(BASE_DIR, "images", "title_page.png")

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    /* Увеличиваем базовый размер шрифта для всего приложения */
    html, body, .block-container {
        font-size: 26px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    img = Image.open(image_path)
    st.image(img, width=600)
    st.write("Добро пожаловать! 👋🏻 Выберите интересующую вас страницу в меню слева.")