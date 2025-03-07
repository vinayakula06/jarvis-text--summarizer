from flask import Flask, render_template, request
import numpy as np
from nltk.data import find
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer, BertModel
import torch
import joblib
import os
import logging
import gc
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Initialize Flask app with increased timeout
app = Flask(__name__)

# Set the nltk_data directory to a location inside your virtual environment
nltk_data_dir = os.path.join(os.getenv('VIRTUAL_ENV', os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)  # Add to beginning of path for priority

# Download punkt resource more reliably
try:
    # Try to find the punkt resource
    find('tokenizers/punkt')
    logger.info("NLTK punkt resource already exists")
except LookupError:
    # If not found, download it
    logger.info(f"Downloading NLTK punkt resource to {nltk_data_dir}")
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
    # Verify download was successful
    try:
        find('tokenizers/punkt')
        logger.info("NLTK punkt resource successfully downloaded")
    except LookupError:
        logger.error("WARNING: Failed to download NLTK punkt resource")

# Paths for saving/loading models
MODEL_DIR = "models"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "full_model.pth")
LSA_PIPELINE_PATH = os.path.join(MODEL_DIR, "lsa_classifier.pkl")

# Create the models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize variables to hold models
tokenizer = None
model = None
lsa_pipeline = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def segment_text(text):
    """Segment the input text into sentences using NLTK's sent_tokenize."""
    try:
        return sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error in sentence tokenization: {e}")
        # Return a simple split as fallback
        return [s.strip() for s in text.split('.') if s.strip()]

def load_bert_model():
    """Load BERT model on demand to save memory"""
    global tokenizer, model
    
    try:
        logger.info("Loading BERT model and tokenizer")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.to(device)
        return True
    except Exception as e:
        logger.error(f"Failed to load BERT model: {e}")
        return False

def generate_batch_embeddings(sentences, batch_size=8):
    """
    Generate sentence embeddings for a batch of sentences using a pre-trained BERT model.
    Uses batching to reduce memory usage.
    """
    # Ensure BERT model is loaded
    if model is None or tokenizer is None:
        success = load_bert_model()
        if not success:
            raise RuntimeError("Failed to load BERT model")
    
    try:
        model.eval()  # Set model to evaluation mode
        all_embeddings = []
        
        # Process in smaller batches to reduce memory usage
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            inputs = tokenizer(
                batch, 
                return_tensors='pt', 
                max_length=256,  # Reduced from 512 to save memory
                truncation=True, 
                padding='max_length'
            )
            
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():  # Disable gradient calculation
                outputs = model(**inputs)
                
            attention_mask = inputs['attention_mask']
            batch_embeddings = (
                outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            ).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
            
            all_embeddings.append(batch_embeddings.cpu().detach().numpy())
            
            # Clean up GPU memory
            del inputs, outputs, attention_mask, batch_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return np.vstack(all_embeddings)
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def compute_similarity_matrix(embeddings):
    """Compute the similarity matrix using cosine similarity."""
    try:
        return cosine_similarity(embeddings)
    except Exception as e:
        logger.error(f"Error computing similarity matrix: {e}")
        # Return identity matrix as fallback
        return np.eye(embeddings.shape[0])

def rank_units(similarity_matrix, units, num_units=10):
    """Rank units (e.g., sentences) based on similarity scores."""
    try:
        # Ensure we don't try to get more units than we have
        num_units = min(num_units, len(units))
        unit_ranks = np.argsort(-similarity_matrix.sum(axis=1))[:num_units]
        return [units[i] for i in unit_ranks]
    except Exception as e:
        logger.error(f"Error ranking units: {e}")
        # Return first few units as fallback
        return units[:min(len(units), num_units)]

def lsa_summarizer(text, num_sentences=10):
    """Generate an LSA-based summary using TF-IDF and SVD."""
    try:
        sentences = segment_text(text)
        
        # Handle case with very few sentences
        if len(sentences) <= 1:
            return sentences  # Just return the original sentence(s)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Make sure n_components is at least 1 and at most len(sentences)-1
        n_components = min(max(1, num_sentences), len(sentences) - 1)
        svd = TruncatedSVD(n_components=n_components)
        
        lsa_matrix = svd.fit_transform(tfidf_matrix)
        sentence_scores = np.sum(lsa_matrix, axis=1)
        
        # Only get as many sentences as we have
        requested_sentences = min(num_sentences, len(sentences))
        ranked_indices = np.argsort(-sentence_scores)[:requested_sentences]
        
        return [sentences[i] for i in np.sort(ranked_indices)]  # Sort to preserve original order
    
    except Exception as e:
        logger.error(f"Error in LSA summarization: {e}")
        # Return first few sentences as fallback
        sentences = segment_text(text)
        return sentences[:min(len(sentences), num_sentences)]

def save_model(m, t, pipeline):
    """
    Save the BERT model, tokenizer, and LSA pipeline.
    """
    try:
        # Save BERT model and tokenizer
        m.save_pretrained(MODEL_DIR)
        t.save_pretrained(MODEL_DIR)
        
        # Save BERT model's state dictionary
        torch.save(m.state_dict(), FULL_MODEL_PATH)
        
        # Save LSA pipeline
        joblib.dump(pipeline, LSA_PIPELINE_PATH)
        
        logger.info(f"BERT model and tokenizer saved in directory: {MODEL_DIR}/")
        logger.info(f"BERT model state dictionary saved as: {FULL_MODEL_PATH}")
        logger.info(f"LSA pipeline saved as: {LSA_PIPELINE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return False

def load_saved_model():
    """
    Load the BERT model, tokenizer, and LSA pipeline.
    """
    global model, tokenizer, lsa_pipeline
    
    try:
        # Load BERT model and tokenizer
        logger.info("Loading saved BERT model and tokenizer")
        model = BertModel.from_pretrained(MODEL_DIR)
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        
        # Load BERT model's state dictionary - fixed the weights_only parameter
        model.load_state_dict(torch.load(FULL_MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        
        # Load LSA pipeline
        lsa_pipeline = joblib.load(LSA_PIPELINE_PATH)
        
        logger.info("Models and pipeline loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# Create the LSA pipeline
lsa_pipeline = Pipeline([
    ('segmentation', segment_text),
    ('lsa_summarization', lsa_summarizer)
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()
        error_message = None
        bert_summary = None
        lsa_summary = None
        
        try:
            # Get input text from the form
            text = request.form['text']
            max_length = 10000  # Set a reasonable limit
            
            if len(text) > max_length:
                logger.warning(f"Input text truncated from {len(text)} to {max_length} characters")
                text = text[:max_length]
            
            logger.info(f"Processing text of length {len(text)}")
            
            # Step 1: Segment the text into sentences
            sentences = segment_text(text)
            if not sentences:
                return render_template('index.html', error="No valid sentences found in the input text.")
            
            # Limit number of sentences to process
            max_sentences = 100
            if len(sentences) > max_sentences:
                logger.warning(f"Limiting sentences from {len(sentences)} to {max_sentences}")
                sentences = sentences[:max_sentences]
            
            # Step 2: Generate BERT embeddings with a smaller batch size
            bert_embeddings = generate_batch_embeddings(sentences, batch_size=8)
            
            # Step 3: Compute similarity matrix using BERT embeddings
            similarity_matrix = compute_similarity_matrix(bert_embeddings)
            
            # Step 4: Rank sentences based on similarity scores (BERT)
            bert_ranked_sentences = rank_units(similarity_matrix, sentences, num_units=min(10, len(sentences)))
            
            # Step 5: Generate BERT summary
            bert_summary = '. '.join(bert_ranked_sentences)
            
            # Step 6: Generate LSA summary
            lsa_ranked_sentences = lsa_summarizer(text, num_sentences=min(10, len(sentences)))
            lsa_summary = '. '.join(lsa_ranked_sentences)
            
            # Clean up memory
            del bert_embeddings, similarity_matrix
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            execution_time = time.time() - start_time
            logger.info(f"Request processed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            error_message = f"An error occurred: {str(e)}"
            
            # Clean up memory on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Render the results in the template
        return render_template('index.html', 
                              bert_summary=bert_summary, 
                              lsa_summary=lsa_summary, 
                              error=error_message)

    # Render the initial form
    return render_template('index.html')

if __name__ == '__main__':
    # Check if models exist, and load them if they do
    if os.path.exists(FULL_MODEL_PATH) and os.path.exists(LSA_PIPELINE_PATH):
        load_saved_model()
    
    # Run with increased timeout and better error handling
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
