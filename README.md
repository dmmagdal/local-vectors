# local-vectors


A python package for generating text vector embeddings locally with huggingface transformers.


## Usage

`local-vectors` is designed for high-performance, local-first embedding tasks. It automatically detects your hardware (supporting **MPS** for Apple Silicon and **CUDA** for NVIDIA GPUs) to ensure optimal performance without manual configuration.

### 1. Basic Embedding
Generate high-quality FP32 embeddings using standard sentence-transformer models.

```python
from local_vectors import LocalEmbedder, detect_device

# Automatically selects CUDA, MPS, or CPU
device = detect_device()
client = LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2", device=device)

text = "The quick brown fox jumps over the lazy dog."
embeddings = client.embed_text(text)

print(f"Dimension: {client.model_metadata['dims']}")
# Output: Dimension: 384
```

### 2. Binary Quantization
For large-scale retrieval, `local-vectors` supports binary embeddings. This reduces storage requirements significantly while maintaining high search accuracy via Hamming distance.

```python
# Generate bit-packed binary vectors
binary_dict = client.embed_text(text, to_binary=True)

print(f"Binary dimension: {client.model_metadata['binary_dims']}")
# This returns a uint8 array suitable for Hamming distance search
```

### 3. Vector Database Integration (LanceDB)
The package includes a built-in wrapper for **LanceDB**, allowing you to manage and search your embeddings locally without a server.

```python
from local_vectors import LanceDBConnection
import pyarrow as pa

# Initialize local database
db = LanceDBConnection("./my_vectors")

# Define a schema for FP32 vectors
schema = pa.schema([
    pa.field("text", pa.string()),
    pa.field("vector_full", pa.list_(pa.float32(), 384)),
])

db.create_table("documents", schema=schema)

# Batch embed and update
sentences = ["Machine learning is fascinating.", "I love AI."]
data = [
    {"text": s, "vector_full": client.embed_text(s)[0]["vector_full"]} 
    for s in sentences
]

db.update_table("documents", data=data, mode="append")
```

### 4. Efficient Semantic Search
You can perform semantic searches using standard cosine similarity or lightning-fast Hamming distance for binary vectors.

```python
# Search FP32 table
results = db.search_table(
    table_name="documents",
    query_vector=client.embed_text("AI and tech")[0]["vector_full"],
    metric="cosine",
    top_k=2
)

for r in results:
    print(f"Text: {r['text']}, Score: {r['_distance']}")
```

---

### Pro-Tips for Production
* **Lazy Loading:** Models are downloaded on-the-fly and cached locally in `~/.cache/local-vectors`. 
* **Hardware Acceleration:** If you are running on a machine with a dedicated GPU, `local-vectors` will prioritize it automatically to speed up batch processing.
* **Data Types:** Use the `to_binary=True` flag when dealing with millions of documents to keep your memory footprint low.

### Cache & Model Management
To keep performance high, `local-vectors` caches model weights and metadata locally. If you need to switch a model version or clear the cache, use the `refresh_model` method:

```python
# This will delete local cached data for the current model and re-initialize
client.refresh_model()
```

> **Warning:** If you refresh a model and the vector dimensions change (e.g., switching from a 384-dim to a 768-dim model), existing LanceDB tables using the old dimensions will become incompatible. You will need to create a new table or migrate your data.

---

### Critical Check: LanceDB Schema Persistence
Because your project emphasizes local AI infrastructure and embedded databases like LanceDB and SQLite, there is a functional constraint to consider. 

When a user calls `refresh_model()` and changes the `model_id`, the `LocalEmbedder` instance remains valid, but any downstream LanceDB tables created with the previous model's dimensions will throw an error during the next `update_table` or `search_table` call. 

To make the package more robust before the PyPI release, consider adding a **dimension check** in the `LanceDBConnection.update_table` method:
* Before appending data, compare the `vector_full` length in the new data against the existing table's schema.
* Raise a clear `ValueError` if they don't match, suggesting the user create a new table.

### Wildcard Recommendation: Model Validation
To prevent users from accidentally "breaking" their setup with a typo in a `model_id`, you could add a small utility that checks if a `model_id` exists on the Hugging Face Hub before attempting a full download. This prevents the `refresh_model()` call from deleting a working cache only to fail on a non-existent new model.

### Final Step: The `uv` Polish
Since you are using `uv`, ensure your `lock` file is excluded from the PyPI upload (this happens by default with `uv build`), but keep it in your GitHub repository. This ensures that while users get the most compatible versions, your own development environment remains reproducible on both **macOS (M2)** and **Linux (NVIDIA)** setups.

Are you planning to include a "benchmarking" script in the examples to show off the performance differences between the FP32 and binary search modes?

### Wildcard: Collaborative Filtering or RAG?
While this package is perfect for **Retrieval-Augmented Generation (RAG)**, you can also use these local embeddings for **Recommendation Systems**. By embedding user interaction history alongside document content, you can calculate similarities locally to suggest related content without ever sending user data to an external API.
