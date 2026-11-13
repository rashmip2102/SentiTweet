import os
import re
import pickle
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

port_stem = PorterStemmer()


def preprocess_text(content: str) -> str:
    """Clean, tokenize, remove stopwords, and stem input text (matches training pipeline)."""
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word) for word in stemmed_content if word not in STOP_WORDS
    ]
    return " ".join(stemmed_content)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts(
    model_path: str = os.path.join(BASE_DIR, "model.sav"),
    vectorizer_path: str = os.path.join(BASE_DIR, "vectorizer.sav"),
):
    """Load the trained Logistic Regression model and TF-IDF vectorizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Place it in the same directory as app.py."
        )
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer file '{vectorizer_path}' not found. "
            "In your notebook, save it with: pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))"
        )

    with open(model_path, "rb") as f_model:
        model = pickle.load(f_model)
    with open(vectorizer_path, "rb") as f_vec:
        vectorizer = pickle.load(f_vec)
    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer):
    """Preprocess, vectorize, and predict sentiment for a single tweet/text."""
    processed = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed])
    prediction = model.predict(vectorized_text)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized_text)[0]
        confidence = float(probs[prediction])

    label = "Positive" if int(prediction) == 1 else "Negative"
    return label, int(prediction), confidence, processed


def main():
    st.set_page_config(
        page_title="Twitter Sentiment Analyzer",
        page_icon="💬",
        layout="centered",
    )

    st.title("💬 Twitter Sentiment Analysis")
    st.caption(
        "Predicts whether a tweet or short text is **Positive** or **Negative** "
        "using TF-IDF + Logistic Regression."
    )

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as err:
        st.error(str(err))
        st.info(
            "**Quick Setup Checklist:**\n"
            "1. Ensure `model.sav` is in the same folder as `app.py`.\n"
            "2. Save your fitted `TfidfVectorizer` from your Jupyter notebook:\n"
            "```python\n"
            "pickle.dump(vectorizer, open('vectorizer.sav', 'wb'))\n"
            "```\n"
            "3. Refresh this page."
        )
        st.stop()

    tweet_input = st.text_area(
        "Enter a tweet or text to analyze:",
        placeholder="e.g., I love how smooth and fast this new update is!",
        height=130,
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not tweet_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            label, pred_class, confidence, processed = predict_sentiment(
                tweet_input, model, vectorizer
            )

            st.divider()
            if pred_class == 1:
                st.success(f"### Sentiment: {label} 😊")
            else:
                st.error(f"### Sentiment: {label} 😞")

            if confidence is not None:
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(confidence)

            with st.expander("View Preprocessed Tokens"):
                st.code(processed if processed else "(no valid tokens after stopword removal)")


if __name__ == "__main__":
    main()
