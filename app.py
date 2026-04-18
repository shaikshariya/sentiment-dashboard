import streamlit as st
import joblib
import re

# Load model & vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Text cleaning (same as training)
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

# UI
st.set_page_config(page_title="Sentiment Dashboard", page_icon="📊")

st.title("📊 AI Sentiment Analysis Dashboard")
st.write("Enter text to analyze sentiment")

text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        result = "Positive 😊" if pred == 1 else "Negative 😡"

        st.success(f"Result: {result}")
