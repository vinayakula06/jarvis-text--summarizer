import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from sklearn.pipeline import Pipeline  # Import Pipeline from scikit-learn
import joblib
from flask import Flask, request, jsonify, send_from_directory

# Load the full model
full_model_path = "full_model.pth"

# Safely load the model with weights_only=False (if you trust the source)
saved_data = torch.load(full_model_path, map_location=torch.device('cpu'), weights_only=False)

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(saved_data['bert_model'], strict=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load LSA-based classifier
Pipeline = saved_data['Pipeline']

# Streamlit interface
st.title("NLP Text Summarizer")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text:
        # Tokenization
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # BERT Model Output
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Classifier prediction
        prediction = Pipeline.predict(cls_embedding)
        
        st.write("Summary:", prediction.tolist())
    else:
        st.write("Please enter some text to summarize.")
