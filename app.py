from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from pathlib import Path
import os

app = Flask(__name__)
CORS(app)

MODEL_ID = "google/siglip2-base-patch16-naflex"
DATA_DIR = "data"
DATASET_ROOT = Path("F:/Documents 1/Programing/test_kaggle/dataset")

print("Loading SigLIP 2 embeddings and index...")
data = np.load(f"{DATA_DIR}/embeddings.npz", allow_pickle=True)
image_ids = data['image_ids'].tolist()
image_paths = data['image_paths'].tolist()
siglip_index = faiss.read_index(f"{DATA_DIR}/faiss_index.bin")
print(f"Loaded {len(image_ids)} images for text search")

print("Loading DINOv3 embeddings and index...")
dinov3_data = np.load(f"{DATA_DIR}/dinov3_embeddings.npz", allow_pickle=True)
dinov3_embeddings = dinov3_data['embeddings']
dinov3_ids = dinov3_data['image_ids'].tolist()
dinov3_index = faiss.read_index(f"{DATA_DIR}/dinov3_faiss_index.bin")
print(f"Loaded {len(dinov3_ids)} images for image search")

# Create mapping from image_id to dinov3 embedding index
dinov3_id_to_idx = {img_id: idx for idx, img_id in enumerate(dinov3_ids)}

print("Loading SigLIP 2 model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
device = "cpu"
print("SigLIP 2 model loaded and ready!")

def embed_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features.cpu().numpy()

print("Ready for searches!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    query_embedding = embed_text(query)
    scores, indices = siglip_index.search(query_embedding.astype(np.float32), top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        # image_paths[idx] = /kaggle/input/datasets/chetankv/dogs-cats-images/dataset/test_set/cats/cat.4085.jpg
        # Need: dataset/test_set/cats/cat.4085.jpg
        img_relative = image_paths[idx].split('dogs-cats-images/')[-1]
        results.append({
            'rank': rank + 1,
            'image_id': image_ids[idx],
            'image_url': f"/images/{img_relative}",
            'score': float(score)
        })
    
    return jsonify({'results': results})

@app.route('/api/search-similar', methods=['POST'])
def search_similar():
    data = request.json
    image_id = data.get('image_id', '')
    top_k = data.get('top_k', 10)
    
    if not image_id:
        return jsonify({'error': 'image_id is required'}), 400
    
    if image_id not in dinov3_id_to_idx:
        return jsonify({'error': 'Image not found'}), 404
    
    idx = dinov3_id_to_idx[image_id]
    query_embedding = dinov3_embeddings[idx:idx+1]
    
    scores, indices = dinov3_index.search(query_embedding.astype(np.float32), top_k + 1)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        result_id = dinov3_ids[idx]
        if result_id == image_id:
            continue
        
        img_relative = f"dataset/test_set/cats/{result_id}"
        results.append({
            'rank': len(results) + 1,
            'image_id': result_id,
            'image_url': f"/images/{img_relative}",
            'score': float(score)
        })
        if len(results) >= top_k:
            break
    
    return jsonify({'results': results})

@app.route('/images/<path:filepath>')
def serve_image(filepath):
    # filepath = dataset/test_set/cats/cat.4414.jpg
    # DATASET_ROOT already includes 'dataset', so strip it from filepath
    if filepath.startswith('dataset/'):
        filepath = filepath[8:]  # Remove 'dataset/' prefix
    full_path = DATASET_ROOT / filepath
    if not full_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(full_path.parent, full_path.name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
