import pytest
from openended_philosophy.utils import semantic_similarity

def test_jaccard_similarity():
    c1 = {"features": ["apple", "banana", "orange"]}
    c2 = {"features": ["apple", "grape", "orange"]}
    assert semantic_similarity(c1, c2, method="jaccard") == pytest.approx(2/4)

    c3 = {"features": []}
    c4 = {"features": ["apple"]}
    assert semantic_similarity(c3, c4, method="jaccard") == 0.0

def test_cosine_similarity():
    c1 = {"features": ["apple", "banana", "orange"]}
    c2 = {"features": ["apple", "grape", "orange"]}
    # Expected value for cosine similarity of [1,1,1,0] and [1,0,1,1]
    # dot product = 1*1 + 1*0 + 1*1 + 0*1 = 2
    # norm1 = sqrt(1^2+1^2+1^2+0^2) = sqrt(3)
    # norm2 = sqrt(1^2+0^2+1^2+1^2) = sqrt(3)
    # similarity = 2 / sqrt(3)*sqrt(3) = 2/3
    assert semantic_similarity(c1, c2, method="cosine") == pytest.approx(2/3)

    c3 = {"features": []}
    c4 = {"features": ["apple"]}
    assert semantic_similarity(c3, c4, method="cosine") == 0.0

def test_wittgenstein_similarity():
    c1 = {"features": ["a", "b"], "contexts": ["x"], "uses": ["p"]}
    c2 = {"features": ["b", "c"], "contexts": ["x", "y"], "uses": ["p"]}
    # features: 1/3
    # contexts: 1/2
    # uses: 1/1
    # avg = (1/3 + 1/2 + 1)/3 = (0.333 + 0.5 + 1)/3 = 1.833/3 = 0.611
    assert semantic_similarity(c1, c2, method="wittgenstein") == pytest.approx((1/3 + 1/2 + 1)/3)

    c3 = {}
    c4 = {"features": ["a"]}
    assert semantic_similarity(c3, c4, method="wittgenstein") == 0.0

def test_embedding_similarity():
    c1 = {"content": "The quick brown fox jumps over the lazy dog."}
    c2 = {"content": "A fast, dark-colored fox leaps over a sleepy canine."}
    c3 = {"content": "The cat sat on the mat."}

    # Expect high similarity for c1 and c2
    sim12 = semantic_similarity(c1, c2, method="embedding")

    # Expect low similarity for c1 and c3
    sim13 = semantic_similarity(c1, c3, method="embedding")

    # Expect sim12 to be significantly higher than sim13
    assert sim12 > sim13 + 0.2  # Ensure a clear difference

    # Test with empty content
    c4 = {"content": ""}
    c5 = {"content": "Some text"}
    assert semantic_similarity(c4, c5, method="embedding") == 0.0

    # Test with only features
    c6 = {"features": ["apple", "banana"]}
    c7 = {"features": ["fruit", "banana"]}
    sim67 = semantic_similarity(c6, c7, method="embedding")
    assert sim67 > 0.5 # Should be moderately similar
