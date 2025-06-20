# 🎬 Movie Recommendation System — Streamlit App

Добро пожаловать в проект по интеллектуальному подбору фильмов, объединяющий традиционные и нейросетевые подходы!  
Приложение уже развернуто и доступно по ссылке:  
👉 **[https://krylovser-movie-recommendation-streamlit-o5evkz.streamlit.app/](https://krylovser-movie-recommendation-streamlit-o5evkz.streamlit.app/)**

---

## 📌 Возможности

### 🧠 1. TF-IDF Поиск
- Пользовательский текстовый запрос обрабатывается через **TF-IDF векторизацию**.
- Подбираются наиболее релевантные фильмы по описанию, жанру, актёрам и режиссёру.
- Отображение результатов с сортировкой по **сходству**.

---

### 🎲 2. Случайные фильмы 
- Раздел для **случайного вдохновения** — нажмите кнопку и получите уникальную подборку фильмов.
- Работает на основе случайной выборки из DataFrame.

---

### 🔍 4. Семантический поиск (Qdrant + LangChain)
- Используются:
  - `LangChain`
  - `Qdrant`
  - `langchain_huggingface`
  - Модель: **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
- Реализован **семантический векторный поиск** с использованием `QdrantVectorStore` и гибкой фильтрацией:
  - По жанрам
  - По актёрам
  - По режиссёрам
  - По рейтингу и году
- Результаты выводятся в красивом интерактивном интерфейсе.

---

### 🤖 5. AI RAG Ассистент по фильмам
- На основе:
  - `LangChain`
  - `Qdrant`
  - `langchain_huggingface`
  - Модель **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
  - Chat-модель от **Groq**: **deepseek-r1-distill-llama-70b**
- Реализован **AI-ассистент** в формате RAG (**Retrieval-Augmented Generation**), который:
  - Подбирает фильмы по запросу
  - Делает **интеллектуальные резюме и анализ** на основе семантики результатов

---

## 🚀 Деплой

Приложение развернуто на **Streamlit Cloud** и доступно онлайн:
👉 [https://krylovser-movie-recommendation-streamlit-o5evkz.streamlit.app/](https://krylovser-movie-recommendation-streamlit-o5evkz.streamlit.app/)

---

## 🗝️ Использование AI ассистента (ChatGroq)

Для работы с AI ассистентом требуется **API-ключ** от Groq:

1. Перейдите на сайт [https://console.groq.com/keys](https://console.groq.com/keys)
2. Зарегистрируйтесь или войдите
3. Сгенерируйте **Groq API key**
4. Вставьте ключ в соответствующее поле на странице RAG-ассистента

---

## 📦 Установка локально (опционально)

```bash
git clone https://github.com/yourusername/movie-recommendation.git
cd movie-recommendation
pip install -r requirements.txt
streamlit run Главная.py
