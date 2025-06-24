import os
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/app/.cache"
os.environ["XDG_CONFIG_HOME"] = "/app/.streamlit"

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

@st.cache_resource
def load_models():
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    text_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
    tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
    return whisper_processor, whisper_model, text_model, tokenizer

whisper_processor, whisper_model, text_model, tokenizer = load_models()

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    input_features = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    prediction = outputs.logits.argmax(dim=1).item()
    return prediction

def predict_hate_speech(audio_path=None, text=None):
    if text:
        text_input = text
    elif audio_path:
        transcription = transcribe(audio_path)
        text_input = transcription
    else:
        return "Please provide either audio or text input."

    prediction = extract_text_features(text_input)
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

st.title("Hate Speech Detector with Audio and Text")
audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac, ogg, opus)", type=["wav", "mp3", "flac", "ogg", "opus"])
text_input = st.text_input("Optional text input")

if st.button("Predict"):
    if audio_file is not None:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_file.read())
        prediction = predict_hate_speech(temp_path, text_input)
        st.success(prediction)
    elif text_input:
        prediction = predict_hate_speech(text=text_input)
        st.success(prediction)
    else:
        st.warning("Please provide at least audio or text input.")
