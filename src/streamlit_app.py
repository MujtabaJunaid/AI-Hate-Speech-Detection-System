import torch
import torchaudio
import os
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
hf_token = os.getenv("HateSpeechMujtabatoken")

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import streamlit as st

whisper_processor = WhisperProcessor.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
text_model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

label_map = {0: "Not Hate Speech", 1: "Hate Speech"}

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    input_features = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = text_model(**inputs)
    pred_label = outputs.logits.argmax(dim=1).item()
    return label_map.get(pred_label, "Unknown")

def predict_hate_speech(audio_path=None, text=None):
    if audio_path:
        transcription = transcribe(audio_path)
        text_input = text if text else transcription
    elif text:
        text_input = text
    else:
        return "No input provided"

    prediction = extract_text_features(text_input)
    return prediction

st.title("Hate Speech Detector with Audio and Text")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg", "opus"])
text_input = st.text_input("Optional text input")
if st.button("Predict"):
    if audio_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.read())
        prediction = predict_hate_speech("temp_audio.wav", text_input)
        st.success(prediction)
    elif text_input:
        prediction = predict_hate_speech(text=text_input)
        st.success(prediction)
    else:
        st.warning("Please upload an audio file or enter text.")
