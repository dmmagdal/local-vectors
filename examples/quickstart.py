
import os
import shutil

import numpy as np
import pyarrow as pa


from local_vectors import LocalEmbedder, detect_device, LanceDBConnection

def main():
    # 1. Initialize (will automatically use CUDA or MPS if available)
    device = detect_device()
    print(f"Using device: {device}")

    print("Loading model...")
    client = LocalEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2", 
        device=device
    )

    print()

    # 2. Single embedding.
    text = "The quick brown fox jumps over the lazy dog."
    embedding_dict = client.embed_text(text)
    assert len(embedding_dict) != 0, "Embedding dictionary should not be empty."
    assert all([emb_dict["vector_full"].shape[0] == client.model_metadata["dims"] for emb_dict in embedding_dict]), \
        "All vectors should have the correct dimensionality."
    print(f"Dimension: {client.model_metadata['dims']}")

    print()

    # 3. Semantic similarity example with batch text.
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

    print()

    # 4. Single embedding with binary embeddings.
    embedding_dict = client.embed_text(text, to_binary=True)
    assert len(embedding_dict) != 0, "Embedding dictionary should not be empty."
    assert all([emb_dict["vector_binary"].shape[0] == client.model_metadata["binary_dims"] for emb_dict in embedding_dict]), \
        "All binary vectors should have the correct dimensionality."
    print(f"Binary embedding dimension: {client.model_metadata['binary_dims']}")
    
    print()

    # 5. Using the vector database connection (LanceDB).
    lance_db_uri = "./quickstart_lancedb"
    if os.path.exists(lance_db_uri):
        shutil.rmtree(lance_db_uri)

    db = LanceDBConnection(lance_db_uri)
    current_tables = db.table_names()
    assert len(current_tables) == 0, "LanceDB should start with no tables."

    fp32_schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("text_idx", pa.int32()),
        pa.field("text_len", pa.int32()),
        pa.field("vector_full", pa.list_(pa.float32(), client.model_metadata["dims"])),
    ])
    binary_schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("text_idx", pa.int32()),
        pa.field("text_len", pa.int32()),
        pa.field("vector_binary", pa.list_(pa.uint8(), client.model_metadata["binary_dims"])),
    ])

    db.create_table(
        "embeddings", 
        schema=fp32_schema
    )
    db.create_table(
        "binary_embeddings", 
        schema=binary_schema
    )
    assert set(db.table_names()) == {"embeddings", "binary_embeddings"}, "Tables should be created successfully."

    # Generate embeddings list.
    embedding_dict = [
        client.embed_text(sentence, to_binary=True)[0]
        for sentence in sentences
    ]

    # Update metadata in embeddings list with text.
    embedding_dict = [
        {**emb_dict, "text": sentences[idx]}
        for idx, emb_dict in enumerate(embedding_dict)
    ]

    db.update_table(
        "embeddings", 
        data=[{
            k: v for k, v in emb_dict.items() if k != "vector_binary"
        } for emb_dict in embedding_dict],
        mode="append"
    )
    db.update_table(
        "binary_embeddings", 
        data=[{
            k: v for k, v in emb_dict.items() if k != "vector_full"
        } for emb_dict in embedding_dict],
        mode="append"
    )

    limit = 3
    search_results = db.search_table(
        table_name="embeddings",
        query_vector=embedding_dict[0]["vector_full"],
        metric="cosine",
        top_k=limit
    )
    assert len(search_results) == limit, "Should return top_k results."
    print("Search results (FP32):")
    print(f"Query text: {embedding_dict[0]['text']}")
    for result in search_results:
        print(f"Text: {result['text']}, Distance: {result['_distance']}")

    print()

    search_results = db.search_table(
        table_name="binary_embeddings",
        query_vector=embedding_dict[0]["vector_binary"],
        metric="hamming",
        top_k=limit
    )
    assert len(search_results) == limit, "Should return top_k results."
    print("Search results (Binary):")
    print(f"Query text: {embedding_dict[0]['text']}")
    for result in search_results:
        print(f"Text: {result['text']}, Distance: {result['_distance']}")

    print()

    db.delete_table("embeddings")
    db.delete_table("binary_embeddings")
    current_tables = db.table_names()
    assert len(current_tables) == 0, "LanceDB should end with no tables."

    db.delete_all_tables()
    current_tables = db.table_names()
    assert len(current_tables) == 0, "LanceDB should end with no tables."

    shutil.rmtree(lance_db_uri, ignore_errors=True)  # Clean up any existing DB for a fresh start

    # Exit the program.
    exit(0)


if __name__ == "__main__":
    main()