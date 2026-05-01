import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, call
import os
import shutil
from pathlib import Path
import pyarrow as pa
import lancedb
from datetime import timedelta

# Import all relevant modules from local_vectors
from local_vectors import LocalEmbedder, LanceDBConnection, detect_device


# --- Fixtures ---

@pytest.fixture
def mock_tokenizer():
    """Mocks a HuggingFace AutoTokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0 # Example pad token ID

    def mock_encode(text, add_special_tokens=False, padding=None, max_length=None, return_tensors=None):
        if isinstance(text, list): # Batch encoding
            encoded_texts = []
            max_len_batch = max(len(t) for t in text) if text else 0
            for t in text:
                ids = [ord(c) for c in t] # Simple char-based tokenization
                if padding == "max_length" and max_length:
                    ids = ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(ids))
                elif padding == "max_length": # If max_length not specified, pad to longest in batch
                    ids = ids + [tokenizer.pad_token_id] * (max_len_batch - len(ids))
                encoded_texts.append(ids)
            
            if return_tensors == "pt":
                input_ids = torch.tensor(encoded_texts)
                attention_mask = torch.tensor([[1 if x != tokenizer.pad_token_id else 0 for x in ids_list] for ids_list in encoded_texts])
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            return encoded_texts # Return list of lists for non-pt
        else: # Single text encoding
            ids = [ord(c) for c in text]
            if padding == "max_length" and max_length:
                ids = ids[:max_length] + [tokenizer.pad_token_id] * (max_length - len(ids))
            if return_tensors == "pt":
                input_ids = torch.tensor([ids])
                attention_mask = torch.tensor([[1 if x != tokenizer.pad_token_id else 0 for x in ids]])
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            return ids # Return list of ints for non-pt

    tokenizer.encode.side_effect = mock_encode
    tokenizer.decode.side_effect = lambda tokens: "".join(chr(t) for t in tokens if t != tokenizer.pad_token_id)
    return tokenizer

@pytest.fixture
def mock_model():
    """Mocks a HuggingFace AutoModel."""
    model = MagicMock()
    # Mock the __call__ method for direct text input
    def mock_call(**kwargs):
        input_ids = kwargs.get('input_ids')
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        seq_len = input_ids.shape[1] if input_ids is not None else 10 # Default seq_len
        
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, 384) # Batch, seq_len, hidden_size
        
        # Mock the mean pooling directly on the tensor
        mock_output.last_hidden_state.mean.return_value = torch.randn(batch_size, 384)
        return mock_output
    
    model.side_effect = mock_call
    return model

@pytest.fixture
def mock_config():
    """Mocks a HuggingFace AutoConfig."""
    config = MagicMock()
    config._name_or_path = "test-model"
    config.max_position_embeddings = 512
    config.hidden_size = 384
    return config

@pytest.fixture
def mock_lancedb_table():
    """Mocks a LanceDB Table object."""
    table = MagicMock(spec=lancedb.Table)
    table.add.return_value = None
    table.optimize.return_value = None
    table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = []
    table.to_pandas.return_value = MagicMock() # Mock pandas DataFrame
    return table

@pytest.fixture
def mock_lancedb_connect(mock_lancedb_table):
    """Mocks lancedb.connect to return a mock database."""
    mock_db = MagicMock()
    mock_db.table_names.return_value = []
    mock_db.create_table.return_value = None
    mock_db.open_table.return_value = mock_lancedb_table
    mock_db.drop_table.return_value = None
    mock_db.drop_all_tables.return_value = None
    
    with patch('lancedb.connect', return_value=mock_db) as mock_connect:
        yield mock_connect

@pytest.fixture
def mock_requests_get():
    """Mocks requests.get for network checks."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get

@pytest.fixture
def mock_local_embedder_instance(mock_tokenizer, mock_model, mock_config):
    """Provides a LocalEmbedder instance with mocked internal components."""
    # We patch the internal providers called by LocalEmbedder.__init__
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModel.from_pretrained', return_value=mock_model), \
         patch('transformers.AutoConfig.from_pretrained', return_value=mock_config), \
         patch('local_vectors.embedders.get_model_metadata', return_value={
             "model_id": "test-model",
             "max_tokens": 128,
             "dims": 384, # Matching the mock model's output
             "binary_dims": 48 # ceil(384/8)
         }):
        embedder = LocalEmbedder("test-model", device="cpu")
        yield embedder

@pytest.fixture
def mock_lancedb_connection_instance(tmp_path, mock_lancedb_connect):
    """Provides a LanceDBConnection instance with mocked internal components."""
    db_path = tmp_path / "test_db"
    conn = LanceDBConnection(str(db_path))
    yield conn


# --- Test `detect_device` (from providers.py) ---

def test_detect_device_cpu():
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        assert detect_device() == "cpu"

def test_detect_device_cuda():
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.backends.mps.is_available', return_value=False):
        assert detect_device() == "cuda"

def test_detect_device_mps():
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=True):
        assert detect_device() == "mps"

def test_detect_device_force_cpu():
    with patch('torch.cuda.is_available', return_value=True):
        assert detect_device(force_cpu=True) == "cpu"

def test_detect_device_get_count_cuda():
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=2):
        result = detect_device(get_count=True)
        assert isinstance(result, tuple)
        assert result[0] == "cuda"
        assert result[1] == 2

# --- Test `LocalEmbedder` ---

def test_local_embedder_init(mock_tokenizer, mock_model, mock_config):
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModel.from_pretrained', return_value=mock_model), \
         patch('transformers.AutoConfig.from_pretrained', return_value=mock_config), \
         patch('local_vectors.embedders.get_model_metadata', return_value={
             "model_id": "test-model", "max_tokens": 128, "dims": 384, "binary_dims": 48
         }):
        embedder = LocalEmbedder("test-model", device="cpu")
    
    assert embedder.model_id == "test-model"
    assert embedder.device == "cpu"
    assert embedder.overlap == 128
    assert embedder.batch_size == 8
    assert embedder.model_metadata["dims"] == 384
    assert embedder.model_metadata["binary_dims"] == 48

def test_local_embedder_set_device(mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    embedder.set_device("cuda")
    assert embedder.device == "cuda"
    embedder.model.to.assert_called_once_with("cuda")

def test_local_embedder_set_batch_size(mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    embedder.set_batch_size(16)
    assert embedder.batch_size == 16

def test_local_embedder_refresh_model(tmp_path, mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    model_id = embedder.model_id
    model_save_root = tmp_path / "cache"
    # We set this manually because the class implementation lacks this attribute
    embedder.model_save_root = model_save_root
    
    model_path = model_save_root / model_id.replace("/", "_")
    os.makedirs(model_path, exist_ok=True)
    (model_path / "config.json").touch()

    with patch('shutil.rmtree') as mock_rmtree, \
         patch('local_vectors.embedders.load_model', return_value=(embedder.tokenizer, embedder.model)) as mock_load_model, \
         patch('local_vectors.embedders.get_model_metadata', return_value=embedder.model_metadata):
        
        embedder.refresh_model()

        mock_rmtree.assert_called_once_with(model_path)
        mock_load_model.assert_called_once()

def test_local_embedder_embed_text_single(mock_local_embedder_instance, mock_tokenizer):
    embedder = mock_local_embedder_instance
    text = "Hello world"

    with patch('local_vectors.embedders.vector_preprocessing') as mock_vp, \
         patch('local_vectors.embedders.batch_embed_text') as mock_bet:
        
        mock_vp.return_value = [{
            "tokens": [1] * embedder.model_metadata["max_tokens"],
            "text_idx": 0,
            "text_len": len(text)
        }]
        mock_bet.return_value = (np.random.rand(384),)
        
        results = embedder.embed_text(text)
        
        assert len(results) == 1
        assert "vector_full" in results[0]
        assert results[0]["vector_full"].shape == (embedder.model_metadata["dims"],)

def test_local_embedder_embed_text_to_binary(mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    with patch('local_vectors.embedders.vector_preprocessing') as mock_vp, \
         patch('local_vectors.embedders.batch_embed_text') as mock_bet:
        
        mock_vp.return_value = [{"tokens": [1]*128, "text_idx": 0, "text_len": 11}]
        mock_bet.return_value = (np.random.rand(384), np.random.randint(0, 255, (48), dtype=np.uint8))
        
        results = embedder.embed_text("binary test", to_binary=True)
        assert "vector_full" in results[0]
        assert "vector_binary" in results[0]
        assert results[0]["vector_binary"].shape == (embedder.model_metadata["binary_dims"],)

def test_local_embedder_embed_text_vectors_only(mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    with patch('local_vectors.embedders.vector_preprocessing') as mock_vp, \
         patch('local_vectors.embedders.batch_embed_text') as mock_bet:
        
        mock_vp.return_value = [{"tokens": [1]*128, "text_idx": 0, "text_len": 12}]
        mock_bet.return_value = (np.random.rand(1, 384),)
        
        results = embedder.embed_text("vectors only", vectors_only=True)
        assert "vector_full" in results[0]
        assert "text_idx" not in results[0]

def test_local_embedder_embed_text_empty_string(mock_local_embedder_instance):
    embedder = mock_local_embedder_instance
    results = embedder.embed_text("")
    assert len(results) == 0


# --- Test `LanceDBConnection` ---

def test_lancedb_connection_init(tmp_path, mock_lancedb_connect):
    db_path = tmp_path / "test_db"
    conn = LanceDBConnection(str(db_path))
    mock_lancedb_connect.assert_called_once_with(str(db_path))
    assert "cosine" in conn.valid_metrics

def test_lancedb_connection_table_names(mock_lancedb_connection_instance, mock_lancedb_connect):
    mock_lancedb_connect.return_value.table_names.return_value = ["table1"]
    names = mock_lancedb_connection_instance.table_names()
    assert names == ["table1"]
    mock_lancedb_connect.return_value.table_names.assert_called_once()

def test_lancedb_connection_create_table(mock_lancedb_connection_instance, mock_lancedb_connect):
    schema = pa.schema([pa.field("id", pa.int32())])
    mock_lancedb_connection_instance.create_table("new_table", schema=schema)
    mock_lancedb_connect.return_value.create_table.assert_called_once_with("new_table", schema=schema)

def test_lancedb_connection_open_table(mock_lancedb_connection_instance, mock_lancedb_connect, mock_lancedb_table):
    table = mock_lancedb_connection_instance.open_table("existing_table")
    assert table == mock_lancedb_table
    mock_lancedb_connect.return_value.open_table.assert_called_once_with("existing_table")

def test_lancedb_connection_delete_table(mock_lancedb_connection_instance, mock_lancedb_connect):
    mock_lancedb_connection_instance.delete_table("table_to_delete")
    mock_lancedb_connect.return_value.drop_table.assert_called_once_with("table_to_delete")

def test_lancedb_connection_delete_all_tables(mock_lancedb_connection_instance, mock_lancedb_connect):
    mock_lancedb_connection_instance.delete_all_tables()
    mock_lancedb_connect.return_value.drop_all_tables.assert_called_once()

def test_lancedb_connection_update_table_append(mock_lancedb_connection_instance, mock_lancedb_table):
    data = [{"id": 1, "vector": [0.1, 0.2]}]
    mock_lancedb_connection_instance.update_table("my_table", data, mode="append")
    mock_lancedb_table.add.assert_called_once_with(data, mode="append")
    mock_lancedb_table.optimize.assert_called_once_with(cleanup_older_than=timedelta(seconds=30))

def test_lancedb_connection_update_table_overwrite(mock_lancedb_connection_instance, mock_lancedb_table):
    data = [{"id": 1, "vector": [0.1, 0.2]}]
    mock_lancedb_connection_instance.update_table("my_table", data, mode="overwrite")
    mock_lancedb_table.add.assert_called_once_with(data, mode="overwrite")

def test_lancedb_connection_search_table_cosine(mock_lancedb_connection_instance, mock_lancedb_table):
    query_vector = [0.1, 0.2, 0.3]
    mock_lancedb_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [
        {"id": 1, "vector": [0.1, 0.2, 0.3], "_distance": 0.9},
        {"id": 2, "vector": [0.4, 0.5, 0.6], "_distance": 0.8}
    ]
    
    results = mock_lancedb_connection_instance.search_table("my_table", query_vector, top_k=2, metric="cosine")
    
    mock_lancedb_table.search.assert_called_once_with(query_vector)
    mock_lancedb_table.search.return_value.metric.assert_called_once_with("cosine")
    mock_lancedb_table.search.return_value.metric.return_value.limit.assert_called_once_with(2)
    assert len(results) == 2
    assert results[0]["id"] == 1

def test_lancedb_connection_search_table_invalid_top_k(mock_lancedb_connection_instance):
    query_vector = [0.1, 0.2]
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        mock_lancedb_connection_instance.search_table("my_table", query_vector, top_k=0)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        mock_lancedb_connection_instance.search_table("my_table", query_vector, top_k=-1)

def test_lancedb_connection_search_table_invalid_metric(mock_lancedb_connection_instance):
    query_vector = [0.1, 0.2]
    with pytest.raises(ValueError, match="Invalid metric: invalid_metric"):
        mock_lancedb_connection_instance.search_table("my_table", query_vector, metric="invalid_metric")

def test_lancedb_connection_download_table_parquet(mock_lancedb_connection_instance, mock_lancedb_table, tmp_path):
    output_path = tmp_path / "output.parquet"
    mock_df = mock_lancedb_table.to_pandas.return_value
    mock_lancedb_connection_instance.download_table("my_table", str(output_path))
    mock_lancedb_table.to_pandas.assert_called_once()
    mock_df.to_parquet.assert_called_once_with(str(output_path))

def test_lancedb_connection_download_table_csv(mock_lancedb_connection_instance, mock_lancedb_table, tmp_path):
    output_path = tmp_path / "output.csv"
    mock_df = mock_lancedb_table.to_pandas.return_value
    mock_lancedb_connection_instance.download_table("my_table", str(output_path))
    mock_df.to_csv.assert_called_once_with(str(output_path))

def test_lancedb_connection_download_table_db(mock_lancedb_connection_instance, mock_lancedb_table, tmp_path):
    output_path = tmp_path / "output.db"
    mock_df = mock_lancedb_table.to_pandas.return_value
    
    with patch('sqlite3.connect') as mock_sqlite_connect:
        mock_conn = MagicMock()
        mock_sqlite_connect.return_value = mock_conn
        mock_lancedb_connection_instance.download_table("my_table", str(output_path))
        mock_df.to_sql.assert_called_once_with("my_table", mock_conn, if_exists="replace", index=False)
        mock_conn.close.assert_called_once()

def test_lancedb_connection_download_table_unsupported_format(mock_lancedb_connection_instance, tmp_path):
    output_path = tmp_path / "output.txt"
    with pytest.raises(ValueError, match="output_path must end with one of the supported formats"):
        mock_lancedb_connection_instance.download_table("my_table", str(output_path))


# --- Workflow Test ---

def test_quickstart_workflow(tmp_path, mock_tokenizer, mock_model, mock_config, mock_lancedb_connect, mock_lancedb_table):
    """End-to-end workflow test using only public APIs."""
    model_metadata = {"model_id": "test-model", "max_tokens": 128, "dims": 384, "binary_dims": 48}
    
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModel.from_pretrained', return_value=mock_model), \
         patch('transformers.AutoConfig.from_pretrained', return_value=mock_config), \
         patch('local_vectors.embedders.get_model_metadata', return_value=model_metadata):
        
        # 1. Initialize LocalEmbedder
        device = detect_device(force_cpu=True)
        client = LocalEmbedder("test-model", device=device)
        
        # Mock batch_embed_text to return consistent dummy embeddings
        dummy_full = np.random.rand(384).astype(np.float32)
        dummy_bin = np.random.randint(0, 255, (48), dtype=np.uint8)
        
        with patch('local_vectors.embedders.batch_embed_text', side_effect=lambda *args, **kwargs: (dummy_full, dummy_bin) if kwargs.get('to_binary') else (dummy_full,)), \
            patch('local_vectors.embedders.vector_preprocessing', return_value=[
                {"tokens": [0]*128, "text_idx": 0, "text_len": 10}
            ]): # Mock preprocessing to always return one chunk
            
            # 2. Single embedding
            text = "The quick brown fox jumps over the lazy dog."
            embedding_dict = client.embed_text(text)
            assert len(embedding_dict) == 1
            assert embedding_dict[0]["vector_full"].shape[0] == client.model_metadata["dims"]

            # 3. Semantic similarity example with batch text (mocked)
            sentences = ["Machine learning is fascinating.", "I love artificial intelligence.", "The weather is nice today."]
            embedding_dicts = [client.embed_text(sentence)[0] for sentence in sentences]
            assert len(embedding_dicts) == len(sentences)
            assert all(emb_dict["vector_full"].shape[0] == client.model_metadata["dims"] for emb_dict in embedding_dicts)

            # 4. Single embedding with binary embeddings
            binary_embedding_dict = client.embed_text(text, to_binary=True)
            assert len(binary_embedding_dict) == 1
            assert binary_embedding_dict[0]["vector_binary"].shape[0] == client.model_metadata["binary_dims"]

            # 5. Using the vector database connection (LanceDB).
            lance_db_uri = tmp_path / "quickstart_lancedb"
            
            # Ensure no tables initially
            mock_lancedb_connect.return_value.table_names.return_value = []
            db = LanceDBConnection(str(lance_db_uri))
            assert db.table_names() == []

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

            db.create_table("embeddings", schema=fp32_schema)
            db.create_table("binary_embeddings", schema=binary_schema)
            mock_lancedb_connect.return_value.table_names.return_value = ["embeddings", "binary_embeddings"]
            assert set(db.table_names()) == {"embeddings", "binary_embeddings"}
            
            # 6. Fill and Search
            embedding_data_for_db = []
            for idx, sentence in enumerate(sentences):
                emb_dict = client.embed_text(sentence, to_binary=True)[0]
                embedding_data_for_db.append({**emb_dict, "text": sentence, "text_idx": 0, "text_len": len(sentence)})

            db.update_table( 
                "embeddings", 
                data=[{k: v for k, v in emb_dict.items() if k != "vector_binary"} for emb_dict in embedding_data_for_db],
                mode="append"
            )
            db.update_table(
                "binary_embeddings", 
                data=[{k: v for k, v in emb_dict.items() if k != "vector_full"} for emb_dict in embedding_data_for_db],
                mode="append"
            )
            
            # Perform search (mocked result)
            mock_lancedb_table.search.return_value.metric.return_value.limit.return_value.to_list.return_value = [{"text": sentences[0]}]
            
            limit = 3
            search_results_fp32 = db.search_table(
                table_name="embeddings",
                query_vector=embedding_data_for_db[0]["vector_full"],
                metric="cosine",
                top_k=limit
            )
            assert len(search_results_fp32) == 1
            assert search_results_fp32[0]["text"] == sentences[0]