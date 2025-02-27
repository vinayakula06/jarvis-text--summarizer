import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import joblib

# Load the full model
full_model_path = "bert_text_summarizer/full_model.pth"

# Load the model safely with weights_only=False (ensure you trust the source)
try:
    saved_data = torch.load(full_model_path, map_location=torch.device('cpu'), weights_only=False)
except Exception:
    st.write("Error loading model. Please check the model file.")
    st.stop()

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert_text_summarizer')
try:
    model.load_state_dict(saved_data['bert_model'], strict=False)
    tokenizer = BertTokenizer.from_pretrained('bert_text_summarizer')
except Exception:
    st.write("Error loading BERT components. Please check the model file.")
    st.stop()

# Load LSA-based classifier
classifier_path = "bert_text_summarizer/lsa_classifier.pkl"
try:
    classifier = joblib.load(classifier_path)
    if not hasattr(classifier, 'predict'):
        st.write("Loaded object is not a valid classifier.")
        st.stop()
    else:
        st.write("Classifier loaded successfully.")
except Exception:
    st.write("Error loading LSA-based classifier from the saved model data.")
    st.stop()

# Streamlit interface
st.title("NLP Text Summarizer")

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text:
        # Tokenization
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        except Exception:
            st.write("Error during tokenization.")
            st.stop()

        # BERT Model Output
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception:
            st.write("Error during BERT model inference.")
            st.stop()

        # Extract CLS token embedding
        try:
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            st.write(f"CLS Embedding shape: {cls_embedding.shape}")
        except Exception:
            st.write("Error extracting CLS token embedding.")
            st.stop()

        # Classifier prediction
        try:
            prediction = classifier.predict(cls_embedding)
            st.write("Summary:", prediction.tolist())
        except Exception:
            st.write("Error during classification.")
    else:
        st.write("Please enter some text to summarize.")
