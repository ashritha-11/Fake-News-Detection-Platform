
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

model = tf.keras.models.load_model(
    "fake_news_lstm_model.h5",
    compile=False
)

tokenizer = joblib.load(
    "tokenizer.pkl"
)

MAX_LEN = 300

st.title(
    "📰 Fake News Detection Platform"
)

article = st.text_area(
    "Enter News Article"
)

if st.button(
    "Detect"
):

    seq = tokenizer.texts_to_sequences(
        [article]
    )

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post"
    )

    pred = model.predict(
        padded,
        verbose=0
    )[0][0]

    if pred > 0.5:
        st.success("✅ REAL NEWS")
    else:
        st.error("❌ FAKE NEWS")

    st.write(
        f"Confidence: {max(pred,1-pred):.2%}"
    )
