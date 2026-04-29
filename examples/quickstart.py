from local_vectors import Embedder
import numpy as np

# 1. Initialize (will automatically use CUDA or MPS if available)
print("Loading model...")
client = Embedder("sentence-transformers/all-MiniLM-L6-v2")

# 2. Single embedding
text = "The quick brown fox jumps over the lazy dog."
vector = client.embed(text)
print(f"Dimension: {vector.shape}")

# 3. Semantic similarity example
sentences = [
    "Machine learning is fascinating.",
    "I love artificial intelligence.",
    "The weather is nice today."
]
embeddings = [client.embed(s) for s in sentences]

# Calculate cosine similarity between first two
sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
print(f"Similarity (ML vs AI): {sim:.4f}")