import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.models import Filter, FieldCondition, Range
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    html, body, .block-container {
        font-size: 26px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path="../db/qdrant_db")

client = get_qdrant_client()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

hf = load_embeddings()

@st.cache_data(show_spinner=False)
def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

@st.cache_data
def load_dict(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

dict_filtr = load_dict('dict_filtr.json')

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=hf
)

llm = ChatGroq(
    api_key="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",
    model="deepseek-r1-distill-llama-70b",
    temperature=1.3,
    max_tokens=700
)

film_prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты — Кристофер Торрантино 🎬: страстный кинокритик. Ты получаешь описание фильма и запрос пользователя, и сразу пишешь короткий, но живой и яркий комментарий.

🎯 Твой анализ должен:
- Избегай вводных размышлений ("хм...", "может быть...")
- Включать краткий разбор сюжета
- Учитывать актёрский состав, жанр, режиссёра и год
- Делать забавные и умные параллели с другими фильмами, сериалами, книгами
- Подмечать необычности или клише
- Вставлять киношные мемы или шутки
- В конце — сказать, стоит ли смотреть, и кому фильм может понравиться
- Отвечать на русском, необычно но связно, живо, с эмодзи и кинолюбовью
     
Важно: отвечай не как бот, а как человек разбирающийся в кино и с хорошим юмором! 🎥🍿"""),

    ("human", "🎥 Запрос пользователя: {question}\n\n📽️ Фильм: {metadata}\n")
])

def main():
    st.markdown("<h1 style='text-align: center; color: #d4a5a5; font-size: 48px;'>🧙🏻‍♂️ Киноаналитик AI</h1>", unsafe_allow_html=True)

    st.markdown("### 🔍 Введите ваш запрос:")
    query = st.text_input("Например: фэнтези про магию и путешествия", "")

    with st.expander("⚙️🎞️ Дополнительные фильтры"):
        selected_genres = st.multiselect("Жанры", dict_filtr['genres'])
        selected_directors = st.multiselect("Режиссеры", dict_filtr['directors'])
        selected_actors = st.multiselect("Актеры", dict_filtr['actors'])
        selected_years = st.slider("Выберите диапазон годов", min_value=min(dict_filtr['years']), max_value=max(dict_filtr['years']), value=(min(dict_filtr['years']), max(dict_filtr['years'])))
        selected_ratings = st.slider("Выберите диапазон рейтинга", min_value=min(dict_filtr['ratings']), max_value=10.0, value=(min(dict_filtr['ratings']), 10.0))

    should_conditions = []
    must_conditions = []

    if selected_genres:
        should_conditions.append(FieldCondition(key="metadata.genre", match={"any": selected_genres}))
    if selected_directors:
        should_conditions.append(FieldCondition(key="metadata.director", match={"any": selected_directors}))
    if selected_actors:
        should_conditions.append(FieldCondition(key="metadata.actors", match={"any": selected_actors}))

    if selected_years:
        must_conditions.append(FieldCondition(key="metadata.year", range=Range(gte=selected_years[0], lte=selected_years[1])))
    if selected_ratings:
        must_conditions.append(FieldCondition(key="metadata.rating", range=Range(gte=selected_ratings[0], lte=selected_ratings[1])))

    filter_obj = None
    if should_conditions or must_conditions:
        filter_obj = Filter(
            should=should_conditions if should_conditions else None,
            must=must_conditions if must_conditions else None
        )

    if query:
        with st.spinner('Ищем лучшие рекомендации...'):
            results = vector_store.similarity_search(query, k=2, filter=filter_obj)

        st.markdown(f"Найдено результатов: {len(results)}")

        film_chain = ({
                        "question": RunnablePassthrough(),
                        "metadata": lambda md: json.dumps(md, ensure_ascii=False, indent=2)
                    }
                    | film_prompt
                    | llm
                    | StrOutputParser())
        
        for i, doc in enumerate(results):
            metadata = doc.metadata

            with st.container():
                st.markdown(
    f"<h3 style='text-decoration: underline; color: white; margin-bottom: 0.5em;'>"
    f"<a href='{metadata.get('page_url', '')}' style='text-decoration: underline; color: #d4a5a5; font-size: 35px;'>"
    f"{metadata.get('movie_title', '')}</a></h3>",
    unsafe_allow_html=True)

                cols = st.columns([2, 7])
                with cols[0]:
                    image_url = metadata.get('image_url', '🎞️ Нет постера')
                    if image_url and image_url.startswith('http'):
                        img = load_image_from_url(image_url)
                        if img:
                            st.image(img, width=300)
                        else:
                            st.write('❌ Не удалось загрузить изображение')

                with cols[1]:
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Год:</b> {metadata.get('year', '-')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Жанр:</b> {', '.join(metadata.get('genre', '-'))}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Режиссер:</b> {', '.join(metadata.get('director', '-'))}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Актеры:</b> {', '.join(metadata.get('actors', '-'))}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:24px; color:#a6d0e4; line-height:1.0; margin:0.2'><b>Рейтинг IMDb:</b> {metadata.get('rating', '-')}</p>", unsafe_allow_html=True)

                    commentary = film_chain.invoke({
                    "question": query,
                    "metadata": metadata
                })

                st.markdown(
                    f"""<div style='margin-top:10px; font-size:22px; color:#ffe5b4; background:#1c1c1c; padding:10px; border-radius:12px'>
                    🎙️ <i>{commentary}</i>
                    </div>""", unsafe_allow_html=True
                )

                st.divider()

if __name__ == "__main__":
    main()
