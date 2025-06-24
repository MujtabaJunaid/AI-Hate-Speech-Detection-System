import torch
import os
import streamlit as st
from pydub import AudioSegment
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
hf_token = os.getenv("HateSpeechMujtabatoken")

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", token=hf_token)
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", token=hf_token)
text_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", token=hf_token)

def transcribe(audio_path):
    audio = AudioSegment.from_file(audio_path, format="opus")
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
    input_features = whisper_processor(samples, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return "Hate Speech" if predicted_class >= 1 else "Not Hate Speech"

def predict(text_input):
    audio_path = "input.opus"
    transcribed_text = transcribe(audio_path)
    prediction = extract_text_features(text_input or transcribed_text)
    if text_input:
        return f"Predicted: {prediction}"
    else:
        return f"Predicted: {prediction} \n\n(Transcribed: {transcribed_text})"

st.title("Hate Speech Detector")
text_input = st.text_input("Enter text (optional):")
if st.button("Run Prediction"):
    result = predict(text_input)
    st.success(result)
