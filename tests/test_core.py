#!/usr/bin/env python3
"""
Test Suite for OpenEnded Philosophy Framework
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Testing Methodology

This test suite verifies core framework functionality through:
- **Unit Testing**: Individual component verification
- **Integration Testing**: Inter-component interaction validation
- **Philosophical Consistency**: Epistemic commitment verification
"""

import pytest
import asyncio
from datetime import datetime
import numpy as np

from openended_philosophy.core import (
    EmergentCoherenceNode,
    DynamicPluralismFramework,
    LanguageGameProcessor,
    CoherenceLandscape,
    FallibilisticInference,
    SemanticPattern
)

from openended_philosophy.utils import (
    calculate_epistemic_uncertainty,
    semantic_similarity,
    coherence_metrics
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEmergentCoherenceNode:
    """Test suite for EmergentCoherenceNode functionality."""
    
    def test_node_creation(self):
        """Test basic node instantiation."""
        node = EmergentCoherenceNode(
            initial_pattern={"concept": "identity", "domain": "metaphysics"},
            confidence=0.7,
            context_sensitivity=0.8
        )
        
        assert node.pattern.content["concept"] == "identity"
        assert node.pattern.confidence == 0.7
        assert node.pattern.context_sensitivity == 0.8
        assert len(node.revision_history) == 0
    
    def test_contextualization(self):
        """Test meaning contextualization within language games."""
        node = EmergentCoherenceNode(
            initial_pattern={"concept": "causation", "features": ["necessity", "regularity"]},
            confidence=0.8
        )
        
        # Create test language game
        game = LanguageGameProcessor(
            "scientific",
            {"empirical_verification": True, "mathematical_formalism": True}
        )
        
        # Contextualize
        result = node.contextualize_meaning(game)
        
        assert "provisional_meaning" in result
        assert "confidence" in result
        assert result["confidence"] <= node.pattern.confidence
        assert result["revision_openness"] == 1.0 - result["confidence"]
    
    def test_adaptive_feedback(self):
        """Test pattern adaptation through feedback."""
        node = EmergentCoherenceNode(
            initial_pattern={"strength": 0.5, "relevance": 0.7}
        )
        
        initial_confidence = node.pattern.confidence
        
        # Apply feedback
        feedback = {
            "strength": 0.8,
            "relevance": 0.9,
            "quality": 0.85
        }
        
        node.adapt_to_feedback(feedback, learning_rate=0.1)
        
        # Verify adaptation
        assert node.pattern.content["strength"] > 0.5
        assert node.pattern.content["relevance"] > 0.7
        assert node.pattern.confidence > initial_confidence
        assert node.pattern.revision_count == 1
        assert len(node.revision_history) == 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDynamicPluralismFramework:
    """Test suite for DynamicPluralismFramework."""
    
    def test_perspective_integration(self):
        """Test integration of interpretive schemas."""
        framework = DynamicPluralismFramework(openness_coefficient=0.9)
        
        # Integrate perspectives
        schema1 = {
            'id': 'empiricist',
            'concepts': ['observation', 'verification', 'data']
        }
        
        schema2 = {
            'id': 'rationalist',
            'concepts': ['reason', 'deduction', 'axioms']
        }
        
        id1 = framework.integrate_perspective(schema1)
        id2 = framework.integrate_perspective(schema2)
        
        assert len(framework.interpretive_schemas) == 2
        assert framework.interpretive_schemas[id1]['weight'] <= 0.5
        assert framework.interpretive_schemas[id2]['weight'] <= 0.5
    
    def test_schema_dialogue(self):
        """Test dialogue between interpretive schemas."""
        framework = DynamicPluralismFramework()
        
        # Add schemas
        s1_id = framework.integrate_perspective({
            'id': 'schema1',
            'concepts': ['A', 'B', 'C']
        })
        
        s2_id = framework.integrate_perspective({
            'id': 'schema2',
            'concepts': ['B', 'C', 'D']
        })
        
        # Enable dialogue
        topic = {'content': 'test_topic', 'domain': 'philosophy'}
        result = framework.dialogue_between_schemas(s1_id, s2_id, topic)
        
        assert 'agreements' in result
        assert 'tensions' in result
        assert 'emergent_insights' in result
        assert result['interaction_quality'] >= 0.0
    
    def test_diversity_maintenance(self):
        """Test that framework maintains interpretive diversity."""
        framework = DynamicPluralismFramework(openness_coefficient=0.8)
        
        # Add dominant perspective
        dominant_schema = {'id': 'dominant', 'concepts': ['X', 'Y', 'Z']}
        framework.integrate_perspective(dominant_schema, weight=0.9)
        
        # Verify diversity maintenance
        max_weight = max(s['weight'] for s in framework.interpretive_schemas.values())
        assert max_weight <= (1.0 - framework.openness)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFallibilisticInference:
    """Test suite for FallibilisticInference engine."""
    
    @pytest.mark.asyncio
    async def test_insight_generation(self):
        """Test fallibilistic insight generation."""
        engine = FallibilisticInference()
        
        # Create test evidence
        evidence_patterns = [
            {'content': 'Pattern A', 'concepts': ['X', 'Y'], 'confidence': 0.8},
            {'content': 'Pattern B', 'concepts': ['Y', 'Z'], 'confidence': 0.7},
            {'content': 'Pattern C', 'concepts': ['X', 'Z'], 'confidence': 0.9}
        ]
        
        # Generate insights
        insights = await engine.derive_insights(
            evidence_patterns,
            confidence_threshold=0.6
        )
        
        assert len(insights) > 0
        
        for insight in insights:
            assert insight.confidence >= 0.6
            assert len(insight.revision_triggers) > 0
            assert len(insight.identified_limitations) > 0
            assert 'temporal' in insight.expiration_conditions
    
    def test_uncertainty_propagation(self):
        """Test uncertainty calculation and propagation."""
        engine = FallibilisticInference()
        
        # Test synthesis with varying evidence
        synthesis = {
            'content': 'Test synthesis',
            'pattern_count': 3,
            'coherence_score': 0.8
        }
        
        evidence = [{'timestamp': datetime.now()} for _ in range(3)]
        
        confidence = engine._calculate_inference_confidence(synthesis, evidence)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence < synthesis['coherence_score']  # Uncertainty reduces confidence

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCoherenceLandscape:
    """Test suite for CoherenceLandscape dynamics."""
    
    @pytest.mark.asyncio
    async def test_domain_mapping(self):
        """Test coherence landscape mapping."""
        landscape = CoherenceLandscape(dimensionality='variable')
        
        # Map domain
        state = await landscape.map_domain(
            domain="ethics",
            depth=2,
            allow_revision=True
        )
        
        assert isinstance(state.global_coherence, float)
        assert 0.0 <= state.global_coherence <= 1.0
        assert len(state.coherence_regions) > 0
        assert state.calculate_emergence_potential() >= 0.0
    
    def test_landscape_evolution(self):
        """Test landscape evolution with new experiences."""
        landscape = CoherenceLandscape()
        
        # Initialize test region
        region = landscape.CoherenceRegion(
            id="test_region",
            central_concepts=["concept1", "concept2"],
            stability_score=0.8,
            connection_count=3
        )
        landscape.coherence_regions["test_region"] = region
        
        # Apply perturbation
        new_experience = {
            'impact': 0.5,
            'concepts': ["concept1", "concept3"],
            'type': 'discovery'
        }
        
        initial_stability = region.stability_score
        landscape.evolve_landscape(new_experience)
        
        # Verify evolution
        assert region.stability_score < initial_stability
        assert len(landscape.phase_transitions) >= 0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_epistemic_uncertainty_calculation(self):
        """Test epistemic uncertainty quantification."""
        # Low evidence, low coherence = high uncertainty
        u1 = calculate_epistemic_uncertainty(
            evidence_count=1,
            coherence_score=0.3,
            temporal_factor=1.0,
            domain_complexity=0.8
        )
        
        # High evidence, high coherence = low uncertainty
        u2 = calculate_epistemic_uncertainty(
            evidence_count=10,
            coherence_score=0.9,
            temporal_factor=1.0,
            domain_complexity=0.2
        )
        
        assert u1 > u2
        assert 0.0 <= u1 <= 1.0
        assert 0.0 <= u2 <= 1.0
    
    def test_semantic_similarity_methods(self):
        """Test different semantic similarity calculations."""
        concept1 = {
            'features': ['A', 'B', 'C'],
            'contexts': ['X', 'Y'],
            'uses': ['U1', 'U2']
        }
        
        concept2 = {
            'features': ['B', 'C', 'D'],
            'contexts': ['Y', 'Z'],
            'uses': ['U2', 'U3']
        }
        
        # Test Jaccard similarity
        jaccard_sim = semantic_similarity(concept1, concept2, method="jaccard")
        assert 0.0 <= jaccard_sim <= 1.0
        
        # Test cosine similarity
        cosine_sim = semantic_similarity(concept1, concept2, method="cosine")
        assert 0.0 <= cosine_sim <= 1.0
        
        # Test Wittgensteinian family resemblance
        family_sim = semantic_similarity(concept1, concept2, method="wittgenstein")
        assert 0.0 <= family_sim <= 1.0
    
    def test_coherence_metrics(self):
        """Test coherence metric calculations."""
        propositions = [
            {'id': 0, 'content': 'P1'},
            {'id': 1, 'content': 'P2'},
            {'id': 2, 'content': 'P3'}
        ]
        
        relations = [
            (0, 1, 'supports'),
            (1, 2, 'explains'),
            (0, 2, 'contradicts')
        ]
        
        metrics = coherence_metrics(propositions, relations)
        
        assert 'constraint_satisfaction' in metrics
        assert 'explanatory_breadth' in metrics
        assert 'analogical_fit' in metrics
        assert 'overall_coherence' in metrics
        
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_semantic_pattern_stability():
    """Test SemanticPattern stability calculation."""
    pattern = SemanticPattern(
        pattern_id="test_pattern",
        content={"test": "data"},
        confidence=0.8,
        revision_count=2
    )
    
    stability = pattern.calculate_stability()
    
    assert 0.0 <= stability <= 1.0
    assert stability < pattern.confidence  # Revisions reduce stability

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    """Execute test suite with verbose output."""
    pytest.main([__file__, "-v", "--tb=short"])
