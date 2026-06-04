"""
Script to prepare embeddings for image search
Run this ONCE to generate embeddings, then use app.py for search
"""
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import faiss

# Config
DATASET_PATH = "F:\\Documents 1\\Programing\\test_kaggle\\dataset"
MODEL_ID = "google/siglip2-base-patch16-naflex"
EMBEDDING_DIM = 768
BATCH_SIZE = 4
OUTPUT_DIR = "data"

device = "cpu"
print(f"Using device: {device}")

# Find all images
def find_images(root_path: str) -> List[Tuple[str, str]]:
    image_files = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    root = Path(root_path)
    
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() in extensions and img_path.is_file():
            img_id = str(img_path.relative_to(root))
            image_files.append((img_id, str(img_path)))
    
    return image_files

# Load model
print(f"Loading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = model.to(device)
model.eval()
print("Model loaded")

# Find images
print(f"Scanning images in: {DATASET_PATH}")
image_list = find_images(DATASET_PATH)
print(f"Found {len(image_list)} images")

image_ids = [img_id for img_id, _ in image_list]
image_paths = [img_path for _, img_path in image_list]

# Embed images
print(f"Embedding {len(image_paths)} images...")
all_embeddings = []

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Embedding"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_images = []
    
    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            batch_images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            batch_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    
    inputs = processor(images=batch_images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        image_features = F.normalize(image_features, p=2, dim=-1)
    
    all_embeddings.append(image_features.cpu().numpy())

image_embeddings = np.vstack(all_embeddings)
print(f"Embeddings shape: {image_embeddings.shape}")

# Verify normalization
norms = np.linalg.norm(image_embeddings, axis=1)
print(f"L2 norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")

# Save embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(
    f"{OUTPUT_DIR}/embeddings.npz",
    embeddings=image_embeddings,
    image_ids=image_ids,
    image_paths=image_paths
)
print(f"Embeddings saved to {OUTPUT_DIR}/embeddings.npz")

# Build FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(image_embeddings.astype(np.float32))
faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index.bin")
print(f"FAISS index saved to {OUTPUT_DIR}/faiss_index.bin")

print("\n=== Preparation Complete ===")
print(f"Total images: {len(image_paths)}")
print(f"Embeddings: {OUTPUT_DIR}/embeddings.npz ({os.path.getsize(f'{OUTPUT_DIR}/embeddings.npz') / 1024**2:.2f} MB)")
print(f"FAISS index: {OUTPUT_DIR}/faiss_index.bin ({os.path.getsize(f'{OUTPUT_DIR}/faiss_index.bin') / 1024**2:.2f} MB)")
print("\nNow you can run: python app.py")
