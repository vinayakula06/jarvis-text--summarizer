import torch
from transformers import BertTokenizer, BertModel
import joblib
from flask import Flask, request, jsonify, send_from_directory

# Load the full model
full_model_path = "full_model.pth"
saved_data = torch.load(full_model_path, map_location=torch.device('cpu'))

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(saved_data['bert_model'], strict=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load LSA-based classifier
Pipeline = saved_data['Pipeline']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    
    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # BERT Model Output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Classifier prediction
    prediction = Pipeline.predict(cls_embedding)
    
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
