import pytest
from app.retrieval.vector import get_embedding

def test_get_embedding():
    text = "This is a test document."
    embedding = get_embedding(text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert isinstance(embedding[0], float)
