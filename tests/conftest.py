import pytest
from unittest.mock import MagicMock
from local_vectors import LocalEmbedder # Change to your class name


@pytest.fixture
def mock_embedder(mocker):
    # Mock the transformer/sentence-transformer loading
    mocker.patch("local_vectors.core.AutoModel.from_pretrained")
    mocker.patch("local_vectors.core.AutoTokenizer.from_pretrained")
    
    # Return an instance with the mocked backends
    return LocalEmbedder(model_name="test-model")