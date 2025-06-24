import torch
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TORCH_HOME"] = "/app/.cache/torch"
hf_token = os.getenv("HateSpeechMujtabatoken")

text_model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT", token=hf_token)

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    return "Hate Speech" if predicted_class >= 1 else "Not Hate Speech"

def predict(text_input):
    if not text_input:
        return "Please enter some text."
    prediction = extract_text_features(text_input)
    return f"Predicted: {prediction}"

st.title("Hate Speech Detector")
text_input = st.text_input("Enter text:")
if st.button("Predict"):
    result = predict(text_input)
    st.success(result)
