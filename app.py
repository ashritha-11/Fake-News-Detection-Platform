import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import gdown
import os

from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fake News Detection Platform",
    page_icon="📰",
    layout="wide"
)

# =====================================================
# FILE PATHS
# =====================================================

MODEL_FILE = "fake_news_lstm_model.h5"
TOKENIZER_FILE = "tokenizer.pkl"

# =====================================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# =====================================================

if not os.path.exists(MODEL_FILE):

    with st.spinner("Downloading AI Model..."):

        file_id = "11lHhuDarxYmFDH3tT2zlqWxHkwlrEI2H"

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url,
            MODEL_FILE,
            quiet=False
        )

# =====================================================
# CHECK TOKENIZER
# =====================================================

if not os.path.exists(TOKENIZER_FILE):

    st.error(
        "tokenizer.pkl file not found."
    )

    st.stop()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

@st.cache_resource
def load_tokenizer():

    return joblib.load(
        TOKENIZER_FILE
    )

try:

    model = load_model()

    tokenizer = load_tokenizer()

except Exception as e:

    st.error(
        f"Error Loading Files: {e}"
    )

    st.stop()

# =====================================================
# SETTINGS
# =====================================================

MAX_LEN = 300

# =====================================================
# TITLE
# =====================================================

st.title("📰 Fake News Detection Platform")

st.markdown(
    """
    Detect whether a news article is **REAL** or **FAKE**
    using an LSTM Deep Learning Model.
    """
)

# =====================================================
# INPUT
# =====================================================

article = st.text_area(
    "Enter News Article",
    height=250
)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Detect News"):

    if article.strip() == "":

        st.warning(
            "Please enter a news article."
        )

    else:

        sequence = tokenizer.texts_to_sequences(
            [article]
        )

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post"
        )

        prediction = model.predict(
            padded,
            verbose=0
        )[0][0]

        confidence = max(
            prediction,
            1 - prediction
        )

        st.subheader("Prediction")

        if prediction >= 0.5:

            st.success(
                "✅ REAL NEWS"
            )

        else:

            st.error(
                "❌ FAKE NEWS"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2%}"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Deep Learning Assignment 7 - Fake News Detection using LSTM"
)
