import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
import os
import requests

MODEL_ID = "1z94IUYb-eqlbKtjo4VWkY1rXXslsUaVr"
VECTORIZER_ID = "1MjVVvAomop18KmZI2HO_HlBex24bUb30"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def download_file_from_gdrive(file_id, destination):
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"{destination} downloaded successfully!")
    except requests.RequestException as e:
        st.error(f"Failed to download {destination}: {e}")
        st.stop()

# Download the model and vectorizer if they don't exist locally
if not os.path.exists(MODEL_PATH):
    download_file_from_gdrive(MODEL_ID, MODEL_PATH)

if not os.path.exists(VECTORIZER_PATH):
    download_file_from_gdrive(VECTORIZER_ID, VECTORIZER_PATH)

# Download NLTK stopwords if not present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def output_label(n):
    return "True News" if n == 0 else "Fake News"

def predict_fake_news(news, model, vectorizer):
    news = preprocess_text(news)
    news_list = [news]
    news_vector = vectorizer.transform(news_list).toarray()
    prediction = model.predict(news_vector)
    return output_label(prediction[0])

# Load the model and vectorizer
try:
    tfidf = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title('Fake News Detection')

news_article = st.text_area('Enter the news article:', '')

if st.button('Predict'):
    if news_article.strip():
        prediction = predict_fake_news(news_article, model, tfidf)
        st.header(prediction)
    else:
        st.warning('Please enter a news article for prediction')
