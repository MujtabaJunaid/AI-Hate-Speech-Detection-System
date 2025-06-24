import torch
import torchaudio
import os
import streamlit as st
import sounddevice as sd
import soundfile as sf
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

def record_audio(duration, filename, samplerate=16000):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, samplerate)

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
    return "Hate Speech" if predicted_class >= 1 else "Not Hate Speech"

def predict(text_input):
    audio_path = "mic_input.wav"
    record_audio(5, audio_path)
    transcribed_text = transcribe(audio_path)
    prediction = extract_text_features(text_input or transcribed_text)
    if text_input:
        return f"Predicted: {prediction}"
    else:
        return f"Predicted: {prediction} \n\n(Transcribed: {transcribed_text})"

st.title("Hate Speech Detector")
text_input = st.text_input("Enter text (optional):")
if st.button("Start Recording and Predict"):
    result = predict(text_input)
    st.success(result)
