"""
Basic Tests for OpenEnded Philosophy Components
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run with: python -m pytest tests/
"""

import pytest
import asyncio
from openended_philosophy import (
    EmergentCoherenceNode,
    DynamicPluralismFramework,
    LanguageGameProcessor,
    calculate_epistemic_uncertainty,
    semantic_similarity
)

def test_coherence_node_creation():
    """Test basic coherence node creation."""
    node = EmergentCoherenceNode(
        initial_pattern={'concept': 'test', 'domain': 'testing'},
        confidence=0.5,
        context_sensitivity=0.8
    )
    
    assert node.pattern.confidence == 0.5
    assert node.pattern.context_sensitivity == 0.8
    assert 'concept' in node.pattern.content
    assert node.pattern.content['concept'] == 'test'

def test_epistemic_uncertainty_calculation():
    """Test uncertainty calculation."""
    # High evidence, high coherence should give low uncertainty
    uncertainty1 = calculate_epistemic_uncertainty(
        evidence_count=10,
        coherence_score=0.9,
        temporal_factor=1.0,
        domain_complexity=0.3
    )
    assert uncertainty1 < 0.5
    
    # Low evidence, low coherence should give high uncertainty
    uncertainty2 = calculate_epistemic_uncertainty(
        evidence_count=1,
        coherence_score=0.3,
        temporal_factor=1.5,
        domain_complexity=0.9
    )
    assert uncertainty2 > 0.5
    assert uncertainty2 > uncertainty1

def test_semantic_similarity():
    """Test semantic similarity calculations."""
    concept1 = {
        'features': ['awareness', 'experience', 'subjectivity'],
        'contexts': ['philosophy', 'neuroscience']
    }
    
    concept2 = {
        'features': ['awareness', 'consciousness', 'qualia'],
        'contexts': ['philosophy', 'psychology']
    }
    
    concept3 = {
        'features': ['computation', 'algorithm', 'data'],
        'contexts': ['computer_science', 'mathematics']
    }
    
    # Similar concepts should have higher similarity
    sim12 = semantic_similarity(concept1, concept2, method='jaccard')
    sim13 = semantic_similarity(concept1, concept3, method='jaccard')
    
    assert sim12 > sim13
    assert 0 <= sim12 <= 1
    assert 0 <= sim13 <= 1

def test_language_game_processor():
    """Test language game processing."""
    game = LanguageGameProcessor(
        game_type='test_game',
        grammatical_rules={'rule1': True, 'rule2': False}
    )
    
    # Test rule compliance checking
    pattern_fit = game.assess_pattern_fit({'rule1': 'value', 'rule3': 'value'})
    assert 0 <= pattern_fit <= 1

def test_pluralism_framework():
    """Test dynamic pluralism framework."""
    framework = DynamicPluralismFramework(openness_coefficient=0.9)
    
    # Add perspectives
    schema1 = {'id': 'perspective1', 'concepts': ['A', 'B']}
    schema2 = {'id': 'perspective2', 'concepts': ['B', 'C']}
    
    id1 = framework.integrate_perspective(schema1)
    id2 = framework.integrate_perspective(schema2)
    
    assert id1 in framework.interpretive_schemas
    assert id2 in framework.interpretive_schemas
    
    # Test weight distribution
    total_weight = sum(
        s['weight'] for s in framework.interpretive_schemas.values()
    )
    assert abs(total_weight - 1.0) < 0.01  # Weights should sum to 1

@pytest.mark.asyncio
async def test_coherence_node_adaptation():
    """Test node adaptation to feedback."""
    node = EmergentCoherenceNode(
        initial_pattern={'value': 0.5},
        confidence=0.6
    )
    
    initial_confidence = node.pattern.confidence
    
    # Provide positive feedback
    node.adapt_to_feedback(
        feedback={'value': 0.8, 'quality': 0.9},
        learning_rate=0.2
    )
    
    # Confidence should increase with positive feedback
    assert node.pattern.confidence > initial_confidence
    assert node.pattern.revision_count == 1

if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    test_coherence_node_creation()
    print("✅ Coherence node creation test passed")
    
    test_epistemic_uncertainty_calculation()
    print("✅ Epistemic uncertainty test passed")
    
    test_semantic_similarity()
    print("✅ Semantic similarity test passed")
    
    test_language_game_processor()
    print("✅ Language game processor test passed")
    
    test_pluralism_framework()
    print("✅ Pluralism framework test passed")
    
    # Run async test
    asyncio.run(test_coherence_node_adaptation())
    print("✅ Coherence node adaptation test passed")
    
    print("\n✨ All tests passed!")
