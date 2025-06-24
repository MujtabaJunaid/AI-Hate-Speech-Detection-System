import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import os

os.environ["TRANSFORMERS_CACHE"] = "/app/.cache"
os.environ["HF_HOME"] = "/app/.cache"


def load_models():
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return whisper_processor, whisper_model, text_model, tokenizer

whisper_processor, whisper_model, text_model, tokenizer = load_models()

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    input_features = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    return outputs.logits.argmax(dim=1).item()

def predict_hate_speech(audio_path, text):
    transcription = transcribe(audio_path)
    text_input = text if text else transcription
    prediction = extract_text_features(text_input)
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

st.title("Hate Speech Detector with Audio and Text")
audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac, ogg, opus)", type=["wav", "mp3", "flac", "ogg", "opus"])
text_input = st.text_input("Optional text input")
if st.button("Predict"):
    if audio_file is not None or text_input:
        audio_path = None
        if audio_file is not None:
            with open("temp_audio_input", "wb") as f:
                f.write(audio_file.read())
            audio_path = "temp_audio_input"

        prediction = predict_hate_speech(audio_path, text_input) if audio_path else extract_text_features(text_input)
        st.success(prediction)
    else:
        st.warning("Please provide either audio or text input.")
