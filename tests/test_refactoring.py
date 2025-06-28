#!/usr/bin/env python3
"""
Test script to verify the philosophical operations refactoring.
"""

import asyncio
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.operations import PhilosophicalOperations
from openended_philosophy.core import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    FallibilisticInference,
    LanguageGameProcessor,
)
from openended_philosophy.nars import NARSManager, NARSMemory, NARSReasoning

async def test_operations_refactoring():
    """Test that the operations module works correctly."""
    print("Testing philosophical operations refactoring...")
    
    try:
        # Initialize the components needed for operations
        pluralism_framework = DynamicPluralismFramework(openness_coefficient=0.9)
        coherence_landscape = CoherenceLandscape(dimensionality='variable')
        inference_engine = FallibilisticInference()
        
        # Initialize NARS components
        nars_manager = NARSManager()
        nars_memory = NARSMemory(
            memory_file=Path("test_philosophy_nars_memory.json"),
            attention_size=30,
            recency_size=10
        )
        nars_reasoning = NARSReasoning(
            nars_manager=nars_manager,
            nars_memory=nars_memory
        )
        
        # Initialize language games
        language_games = {
            "scientific_discourse": LanguageGameProcessor(
                "scientific",
                {
                    "empirical_verification": True,
                    "mathematical_formalism": True,
                    "peer_review": True
                }
            ),
            "ethical_deliberation": LanguageGameProcessor(
                "ethical",
                {
                    "normative_reasoning": True,
                    "value_pluralism": True,
                    "moral_imagination": True
                }
            ),
            "ordinary_language": LanguageGameProcessor(
                "ordinary",
                {
                    "pragmatic_success": True,
                    "family_resemblance": True,
                    "contextual_meaning": True
                }
            )
        }
        
        # Create philosophical operations instance
        operations = PhilosophicalOperations(
            pluralism_framework=pluralism_framework,
            coherence_landscape=coherence_landscape,
            inference_engine=inference_engine,
            language_games=language_games,
            nars_manager=nars_manager,
            nars_memory=nars_memory,
            nars_reasoning=nars_reasoning
        )
        
        print("✓ Successfully created PhilosophicalOperations instance")
        
        # Test concept analysis
        result = await operations.analyze_concept(
            concept="consciousness",
            context="neuroscience",
            confidence_threshold=0.5
        )
        
        print("✓ Successfully executed concept analysis")
        print(f"  Analysis result keys: {list(result.keys())}")
        print(f"  Concept: {result.get('concept', 'N/A')}")
        print(f"  Context: {result.get('context', 'N/A')}")
        print(f"  Number of analyses: {len(result.get('analyses', []))}")
        print(f"  Epistemic status: {result.get('epistemic_status', 'N/A')}")
        
        # Test coherence exploration
        coherence_result = await operations.explore_coherence(
            domain="ethics",
            depth=2
        )
        
        print("✓ Successfully executed coherence exploration")
        print(f"  Coherence result keys: {list(coherence_result.keys())}")
        print(f"  Domain: {coherence_result.get('domain', 'N/A')}")
        print(f"  Exploration depth: {coherence_result.get('exploration_depth', 'N/A')}")
        
        # Test meaning contextualization
        meaning_result = await operations.contextualize_meaning(
            expression="justice",
            language_game="ethical_deliberation"
        )
        
        print("✓ Successfully executed meaning contextualization")
        print(f"  Meaning result keys: {list(meaning_result.keys())}")
        print(f"  Expression: {meaning_result.get('expression', 'N/A')}")
        print(f"  Language game: {meaning_result.get('language_game', 'N/A')}")
        
        print("\n🎉 All tests passed! Refactoring is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_operations_refactoring())
    sys.exit(0 if success else 1)
