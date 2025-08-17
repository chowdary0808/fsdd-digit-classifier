import streamlit as st
import soundfile as sf
import numpy as np
from src.infer import predict_wav
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="FSDD Digit Classifier", layout="centered")

st.title("🎤 Spoken Digit → Classifier (FSDD)")
st.caption("Upload a WAV or record audio. Model: MFCC + Logistic Regression")

tab1, tab2 = st.tabs(["Upload WAV", "Record"])

with tab1:
    up = st.file_uploader("Upload a .wav file", type=["wav"])
    if up is not None:
        with open("tmp.wav", "wb") as f:
            f.write(up.read())
        pred = predict_wav("tmp.wav")
        st.success(f"Predicted digit: **{pred}**")

with tab2:
    st.write("Record ~1 second saying a digit (0–9).")
    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=44100)
    if audio_bytes:
        with open("tmp_rec.wav", "wb") as f:
            f.write(audio_bytes)
        pred = predict_wav("tmp_rec.wav")
        st.info(f"Predicted digit: **{pred}**")

st.divider()
st.markdown("**Latency tip:** Using light features + linear models keeps response instant.")
