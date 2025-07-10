import pytest
import pytest_asyncio
from openended_philosophy.core import FallibilisticInference

@pytest.fixture
def inference_engine():
    return FallibilisticInference()

@pytest.fixture
def sample_patterns():
    return [
        {"id": "p1", "content": "The cat is on the mat.", "confidence": 0.8, "concepts": ["cat", "mat"]},
        {"id": "p2", "content": "The feline rests on the rug.", "confidence": 0.7, "concepts": ["feline", "rug"]},
        {"id": "p3", "content": "Dogs are loyal animals.", "confidence": 0.9, "concepts": ["dog", "animal"]},
        {"id": "p4", "content": "Loyalty is a virtue.", "confidence": 0.6, "concepts": ["loyalty", "virtue"]},
    ]

@pytest.mark.asyncio
async def test_group_patterns(inference_engine, sample_patterns):
    # This test relies on the semantic_similarity function, which is tested separately.
    # Here, we primarily check if grouping logic correctly forms groups.
    groups = inference_engine._group_patterns(sample_patterns)

    # Expecting two main groups based on semantic content (cat/mat, dog/loyalty)
    assert len(groups) == 2
    # Check if patterns are distributed into expected groups (order might vary)
    group_contents = [sorted([p['id'] for p in g]) for g in groups]
    assert ['p1', 'p2'] in group_contents or ['p2', 'p1'] in group_contents
    assert ['p3', 'p4'] in group_contents or ['p4', 'p3'] in group_contents

@pytest.mark.asyncio
async def test_create_single_pattern_synthesis(inference_engine, sample_patterns):
    pattern = sample_patterns[0] # "The cat is on the mat."
    synthesis = inference_engine._create_single_pattern_synthesis(pattern)

    assert "content" in synthesis
    assert "evidence_summary" in synthesis
    assert synthesis['synthesis_type'] == "single_pattern"
    assert synthesis['synthesis_confidence'] == pytest.approx(pattern['confidence'])

@pytest.mark.asyncio
async def test_create_multi_pattern_synthesis(inference_engine, sample_patterns):
    # Use a group that should cohere well
    pattern_group = [sample_patterns[0], sample_patterns[1]] # cat/mat group
    synthesis = inference_engine._create_multi_pattern_synthesis(pattern_group)

    assert "content" in synthesis
    assert "evidence_summary" in synthesis
    assert synthesis['synthesis_type'] == "multi_pattern"
    assert synthesis['pattern_count'] == 2
    assert synthesis['synthesis_confidence'] > 0.0 # Should be calculated
    assert "semantic_coherence" in synthesis

@pytest.mark.asyncio
async def test_filter_and_rank_syntheses(inference_engine, sample_patterns):
    # Create some mock syntheses with varying coherence and explanatory breadth
    mock_syntheses = [
        {"content": "High coherence insight", "coherence_score": 0.9, "explanatory_breadth": 0.8},
        {"content": "Medium coherence insight", "coherence_score": 0.6, "explanatory_breadth": 0.9},
        {"content": "Low coherence insight", "coherence_score": 0.2, "explanatory_breadth": 0.5},
    ]
    # This method relies on _filter_by_global_coherence_real and _rank_by_explanatory_value_real
    # which are internal and might need more specific mocking or direct testing if they are complex.
    # For now, we assume they work as intended and test the overall flow.
    filtered_and_ranked = inference_engine._filter_and_rank_syntheses(mock_syntheses)

    # Expect the highest coherence/explanatory breadth to be ranked first
    assert len(filtered_and_ranked) <= len(mock_syntheses)
    # The exact order might depend on internal logic, but we can check for presence
    assert mock_syntheses[0] in filtered_and_ranked # High coherence should likely be kept

@pytest.mark.asyncio
async def test_synthesize_evidence_patterns_overall(inference_engine, sample_patterns):
    # Test the main entry point after refactoring
    insights = await inference_engine.derive_insights(sample_patterns, confidence_threshold=0.0)

    assert len(insights) > 0
    # Check if insights have expected structure
    for insight in insights:
        assert isinstance(insight.content, str)
        assert insight.confidence is not None
        assert insight.identified_limitations is not None
        assert insight.revision_triggers is not None
