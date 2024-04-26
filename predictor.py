import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load your KNN model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("knn_model.pkl")  # Load your trained KNN model
    return model

# Function to preprocess input text
def preprocess_text(text):
    # Preprocess your text data here (e.g., tokenization, stop word removal, etc.)
    return text

def main():
    st.title('Phishing Email Detector')

    # Load your KNN model
    model = load_model()

    # Input text for prediction
    text_input = st.text_area('Enter email text:', 'Type here...')
    if st.button('Predict'):
        # Preprocess input text
        preprocessed_text = preprocess_text(text_input)

        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load your TF-IDF vectorizer

        # Transform preprocessed text into TF-IDF features
        input_tfidf = tfidf_vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(input_tfidf)[0]
        if prediction == 'legitimate':
            st.success('Prediction: Legitimate Email')
        else:
            st.error('Prediction: Phishing Email')

if __name__ == "__main__":
    main()
