
import os
import torch
import torchaudio
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    waveform, sample_rate = torchaudio.load(tmp_path)
    input_features = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    os.remove(tmp_path)
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    return outputs.logits.argmax(dim=1).item()

def predict_hate_speech(audio_bytes, text):
    if audio_bytes:
        transcription = transcribe(audio_bytes)
        text_input = text if text else transcription
    elif text:
        text_input = text
    else:
        return "Please provide audio or text"
    prediction = extract_text_features(text_input)
    return "Hate Speech" if prediction == 1 else "Not Hate Speech"

st.title("Hate Speech Detection")
audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg", "opus"])
text_input = st.text_input("Or enter text")

if st.button("Predict"):
    if audio_file is not None or text_input:
        audio_bytes = audio_file.read() if audio_file else None
        result = predict_hate_speech(audio_bytes, text_input)
        st.success(result)
    else:
        st.warning("Please provide either audio or text input")
