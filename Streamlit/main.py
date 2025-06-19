import streamlit as st
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
st.title("Главная страница")
st.write("Добро пожаловать! Выберите страницу в меню слева.")