
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND WORD INDEX
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_rnn.h5")


@st.cache_resource
def load_word_index():
    return imdb.get_word_index()


model = load_model()
word_index = load_word_index()


# ============================================================
# CONSTANTS
# ============================================================

MAX_LENGTH = 200


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def encode_review(review):
    words = review.lower().split()

    encoded = []

    for word in words:
        if word in word_index:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)

    return encoded


def preprocess_review(review):
    encoded_review = encode_review(review)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH
    )

    return padded_review


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🎬 Movie Sentiment Analyzer")

st.write(
    "Analyze whether a movie review is positive or negative "
    "using a trained Recurrent Neural Network (RNN)."
)

st.divider()

review = st.text_area(
    "📝 Enter your movie review",
    placeholder="Example: This movie was absolutely amazing!",
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review first.")

    else:
        padded_review = preprocess_review(review)

        prediction = model.predict(
            padded_review,
            verbose=0
        )[0][0]

        # Positive
        if prediction >= 0.5:

            confidence = prediction * 100

            st.success("😊 Positive Review")

            st.metric(
                label="Positive Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(prediction))

        # Negative
        else:

            confidence = (1 - prediction) * 100

            st.error("😞 Negative Review")

            st.metric(
                label="Negative Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(1 - prediction))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built using TensorFlow, RNN and Streamlit"
)
