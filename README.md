
# Text Summarization Application

A sophisticated Flask-based web application that delivers high-quality text summarization utilizing both BERT-based and LSA-based algorithms for optimal extraction of key information from documents.

## Features

- Dual-algorithm extractive summarization:
  - BERT (Bidirectional Encoder Representations from Transformers) leveraging contextual understanding
  - LSA (Latent Semantic Analysis) utilizing TF-IDF vectorization and SVD decomposition
- Intuitive web interface with responsive design
- Memory-optimized processing through intelligent batching
- Robust sentence segmentation with fallback mechanisms
- Persistent model architecture with save/load functionality
- Comprehensive error handling and logging
- Interactive sample text functionality for immediate demonstration

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd text-summarization-app
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

1. Launch the Flask application:
   ```
   python app.py
   ```

2. Access the application via web browser:
   ```
   http://localhost:5000
   ```

3. Input your text or utilize the "Load Sample Text" functionality for demonstration purposes.

4. Process the text by selecting "Summarize" to initiate the dual-algorithm analysis.

5. Review both BERT and LSA generated summaries for comprehensive information extraction.

## Technical Implementation

### BERT Summarization Pipeline
1. Document segmentation into semantic units (sentences)
2. Neural encoding via BERT to generate contextual embeddings
3. Semantic similarity computation using cosine distance metrics
4. Importance ranking based on aggregated similarity scores
5. Strategic selection of key sentences based on ranking algorithm

### LSA Summarization Pipeline
1. Document segmentation into semantic units (sentences)
2. Statistical analysis via TF-IDF vectorization
3. Dimensionality reduction through Truncated SVD
4. Mathematical scoring of sentence importance
5. Algorithmic selection of sentences for optimal summary generation

## System Requirements

- Python 3.6+
- Minimum 4GB RAM (8GB recommended for larger documents)
- CUDA-compatible GPU (optional but recommended for performance)
- 500MB free disk space for model storage

## Dependencies

The application leverages industry-standard libraries:
- Flask (2.0.3): Enterprise-grade web framework
- PyTorch (1.10.2): Production-level deep learning framework
- Transformers (4.17.0): State-of-the-art NLP models
- NLTK (3.9.1): Comprehensive natural language processing toolkit
- Scikit-learn (1.0.2): Professional machine learning library
- NumPy (1.21.0): Advanced numerical computing
- Werkzeug (2.0.3): WSGI web application toolkit
- Joblib (1.1.0): Efficient model serialization

## Performance Considerations

- Optimized for documents with up to 100 sentences
- Input limitation of 10,000 characters for browser stability
- Extractive methodology focuses on key content selection
- Processing time scales with document length and complexity

## Troubleshooting

- Memory constraints: Adjust batch_size parameter in generate_batch_embeddings()
- NLTK resources: Verify punkt tokenizer installation
- Performance issues: Check GPU availability with torch.cuda.is_available()
- Diagnostic information: Review detailed logs in app.log

## Future Development Roadmap

- Integration of abstractive summarization capabilities
- Implementation of domain-specific language models
- Document format support expansion (PDF, DOCX, HTML)
- REST API implementation for enterprise integration
- Multilingual processing capabilities



## Development

Designed and developed by VINAY AKULA
