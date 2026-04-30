# local-vectors


A python package for generating text vector embeddings locally with huggingface transformers.


This is a solid foundation for your documentation. To make the `README.md` effective for PyPI, you want to move away from assertion-heavy test code and toward clear, modular examples that highlight the unique features of your project—specifically **local inference** and **efficient quantization**.

Here is a structured "Usage" section for your `README.md` based on your script.

---

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

### Wildcard: Collaborative Filtering or RAG?
While this package is perfect for **Retrieval-Augmented Generation (RAG)**, you can also use these local embeddings for **Recommendation Systems**. By embedding user interaction history alongside document content, you can calculate similarities locally to suggest related content without ever sending user data to an external API.

How are you planning to handle versioning for the model metadata—will you hardcode the dimensions for specific model IDs or allow the library to inspect the model config files dynamically?