import numpy as np
import pytest


def test_initialization(mock_embedder):
    """Test if the class initializes with the correct attributes."""
    assert mock_embedder.model_name == "test-model"
    assert mock_embedder.device in ["cpu", "cuda", "mps"]


def test_input_validation(mock_embedder):
    """Test that the embedder handles invalid inputs gracefully."""
    with pytest.raises(ValueError):
        mock_embedder.embed(None)
    with pytest.raises(TypeError):
        mock_embedder.embed(12345)


def test_output_shape(mocker, mock_embedder):
    """Test that the output is a numpy array of the correct shape."""
    # Mock the return value of the internal embedding logic
    fake_vector = np.random.rand(1, 384)
    mocker.patch.object(mock_embedder, '_run_inference', return_value=fake_vector)
    
    result = mock_embedder.embed("Hello world")
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,) # Assuming a 384-dim model