from local_vectors import LocalEmbedder, detect_device, LanceDBConnection
import numpy as np

# 1. Initialize (will automatically use CUDA or MPS if available)
device = detect_device()
print(f"Using device: {device}")

print("Loading model...")
client = LocalEmbedder(
    "sentence-transformers/all-MiniLM-L6-v2", 
    device=device
)

# 2. Single embedding
text = "The quick brown fox jumps over the lazy dog."
embedding_dict = client.embed_text(text)
assert len(embedding_dict) != 0, "Embedding dictionary should not be empty."
assert all([emb_dict["vector_full"].shape[0] == client.model_metadata["dims"] for emb_dict in embedding_dict]), \
    "All vectors should have the correct dimensionality."
print(f"Dimension: {client.model_metadata['dims']}")

# 3. Semantic similarity example
sentences = [
    "Machine learning is fascinating.",
    "I love artificial intelligence.",
    "The weather is nice today."
]
embedding_dict = [client.embed_text(sentence)[0] for sentence in sentences]
assert len(embedding_dict) != 0, "Embedding dictionary should not be empty."
assert all([emb_dict["vector_full"].shape[0] == client.model_metadata["dims"] for emb_dict in embedding_dict]), \
    "All vectors should have the correct dimensionality."

# Calculate cosine similarity between first two
embeddings = [emb_dict["vector_full"] for emb_dict in embedding_dict]
sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
print(f"Similarity (ML vs AI): {sim:.4f}")