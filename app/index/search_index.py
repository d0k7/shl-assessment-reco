import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from app.schemas.catalog import CatalogItem
from typing import List

# Load the pre-trained model (same as used for indexing)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_index(index_path: str):
    """Loads the FAISS index from the specified path."""
    return faiss.read_index(index_path)

def load_metadata(metadata_path: str) -> List[CatalogItem]:
    """Loads the metadata (item details) from the JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return [CatalogItem(**item_data) for item_data in json.load(f)]

def recommend(query: str, index, metadata: List[CatalogItem], top_k: int = 10):
    """Find the most similar items based on the query."""
    embedding = model.encode([query])[0]  # Get the embedding for the query
    distances, indices = index.search(np.array([embedding]), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # Skip invalid results
        item = metadata[idx]
        results.append({
            'rank': i + 1,
            'item_url': item.url,
            'name': item.name,
            'description': item.description,
            'distance': float(distances[0][i])
        })
    return results

def main():
    index_path = "data/index/catalog_index.faiss"
    metadata_path = "data/index/metadata.json"

    query = input("Enter your query: ")

    print("Loading FAISS index and metadata...")
    index = load_index(index_path)
    metadata = load_metadata(metadata_path)

    print("Generating recommendations...")
    recommendations = recommend(query, index, metadata)

    for rec in recommendations:
        print(f"Rank {rec['rank']}: {rec['name']} - {rec['description']} (Distance: {rec['distance']})")

if __name__ == "__main__":
    main()
