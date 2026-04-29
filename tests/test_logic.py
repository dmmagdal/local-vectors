import numpy as np


def test_batch_vs_single_consistency(mock_embedder, mocker):
    """Ensure embedding 1 sentence is the same as embedding it in a batch."""
    fake_vecs = np.array([[0.1, 0.2], [0.3, 0.4]])
    mocker.patch.object(mock_embedder, '_run_inference', return_value=fake_vecs)
    
    batch_result = mock_embedder.embed_batch(["text1", "text2"])
    single_result = mock_embedder.embed("text1")
    
    # Check if the first vector in the batch matches the single call
    assert np.allclose(batch_result[0], single_result)