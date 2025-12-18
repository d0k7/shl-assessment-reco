import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from app.schemas.catalog import CatalogItem
from tqdm import tqdm
import os

# Load the pre-trained model (you can swap to any other model if you prefer)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_catalog_data(file_path: str) -> List[CatalogItem]:
    """Loads catalog data from a JSONL file."""
    catalog_items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item_data = json.loads(line)
                catalog_items.append(CatalogItem(**item_data))
            except json.JSONDecodeError:
                continue
    return catalog_items

def create_embedding_index(catalog_items: List[CatalogItem], output_path: str):
    """Generates embeddings for catalog items and stores them in a FAISS index."""
    texts = [item.name + " " + item.description for item in catalog_items]  # You can concatenate title + description
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Create a FAISS index
    dimension = embeddings.shape[1]  # Embedding dimension (e.g., 384 for 'all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric (Euclidean)

    # Add embeddings to the FAISS index
    index.add(embeddings)

    # Save the index to disk
    os.makedirs(output_path, exist_ok=True)
    faiss.write_index(index, os.path.join(output_path, 'catalog_index.faiss'))

    # Optionally save the metadata (for reverse lookup) to a file
    with open(os.path.join(output_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump([item.dict() for item in catalog_items], f, ensure_ascii=False, indent=2)

    print(f"Indexing complete. Saved FAISS index and metadata to {output_path}")

def main():
    catalog_file = "data/catalog.jsonl"  # Path to your catalog data (JSONL)
    output_path = "data/index"  # Directory to save the FAISS index and metadata

    print("Loading catalog data...")
    catalog_items = load_catalog_data(catalog_file)
    
    print(f"Creating embedding index for {len(catalog_items)} items...")
    create_embedding_index(catalog_items, output_path)

if __name__ == "__main__":
    main()
