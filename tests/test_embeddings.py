import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
# Replace 'local_vectors' with the actual name of your module if different
from local_vectors import LocalEmbedder, detect_device, LanceDBConnection


@pytest.fixture
def model_name():
    return "sentence-transformers/all-MiniLM-L6-v2"


@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_embedder_initialization(mock_tokenizer, mock_model, model_name):
    """Verify the embedder starts up and picks a device."""
    embedder = LocalEmbedder(model_name)
    assert embedder.model_name == model_name
    assert isinstance(embedder.device, torch.device)


@patch("transformers.AutoModel.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_embedding_output_format(mock_tokenizer, mock_model, model_name):
    """Ensure the output is a numpy array (standard for vector DBs)."""
    embedder = LocalEmbedder(model_name)

    # Mock the internal inference to return a dummy tensor
    dummy_output = torch.randn(1, 384) 
    with patch.object(embedder, '_run_inference', return_value=dummy_output):
        vec = embedder.embed("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)


def test_invalid_input():
    """Fail gracefully on bad types."""
    # This shouldn't need mocks if your validation happens early
    embedder = LocalEmbedder(model_name, model_loading=False) # Assuming a 'lazy' or test mode
    with pytest.raises(TypeError):
        embedder.embed(12345)