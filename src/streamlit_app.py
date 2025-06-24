import torch
import torchaudio
import os
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
os.environ["HF_TOKEN"] = st.secrets["Hf_token"]

hf_token = st.secrets["Hf_token"]

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", token=hf_token)
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", token=hf_token)
text_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", token=hf_token)

def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    input_features = whisper_processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return "Hate Speech" if predicted_class == 1 else "Not Hate Speech"

def predict(audio_file, text_input):
    if not audio_file and not text_input:
        return "Please provide either an audio file or some text."
    if audio_file:
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        transcribed_text = transcribe(audio_path)
        prediction = extract_text_features(text_input or transcribed_text)
        if text_input:
            return f"Predicted: {prediction}"
        else:
            return f"Predicted: {prediction} \n\n(Transcribed: {transcribed_text})"
    else:
        prediction = extract_text_features(text_input)
        return f"Predicted: {prediction}"

st.title("Hate Speech Detector")
uploaded_audio = st.file_uploader("Upload Audio File (.mp3, .wav, .ogg, .flac, .opus)", type=["mp3", "wav", "ogg", "flac", "opus"])
text_input = st.text_input("Or enter text:")
if st.button("Predict"):
    result = predict(uploaded_audio, text_input)
    st.success(result)
