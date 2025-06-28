#!/usr/bin/env python3
"""
Test script to verify that the fixed philosophical tools work correctly.
"""

import asyncio
import json
import logging
from openended_philosophy.operations import PhilosophicalOperations
from openended_philosophy.core import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    FallibilisticInference,
    LanguageGameProcessor,
)
from openended_philosophy.nars import NARSManager, NARSMemory, NARSReasoning
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_fixed_tools():
    """Test the fixed philosophical tools."""
    
    try:
        # Initialize components
        pluralism_framework = DynamicPluralismFramework(openness_coefficient=0.9)
        coherence_landscape = CoherenceLandscape(dimensionality='variable')
        inference_engine = FallibilisticInference()

        # Initialize NARS components
        nars_manager = NARSManager()
        nars_memory = NARSMemory(
            memory_file=Path("test_philosophy_memory.json"),
            attention_size=30,
            recency_size=10
        )
        nars_reasoning = NARSReasoning(
            nars_manager=nars_manager,
            nars_memory=nars_memory
        )

        # Language game registry
        language_games = {
            "scientific_discourse": LanguageGameProcessor(
                "scientific",
                {"empirical_verification": True, "mathematical_formalism": True}
            ),
            "ethical_deliberation": LanguageGameProcessor(
                "ethical",
                {"normative_reasoning": True, "value_pluralism": True}
            ),
            "aesthetic_judgment": LanguageGameProcessor(
                "aesthetic",
                {"subjective_validity": True, "cultural_context": True}
            ),
            "ordinary_language": LanguageGameProcessor(
                "ordinary",
                {"pragmatic_success": True, "family_resemblance": True}
            )
        }

        # Initialize philosophical operations
        operations = PhilosophicalOperations(
            pluralism_framework=pluralism_framework,
            coherence_landscape=coherence_landscape,
            inference_engine=inference_engine,
            language_games=language_games,
            nars_manager=nars_manager,
            nars_memory=nars_memory,
            nars_reasoning=nars_reasoning
        )

        print("🧠 Testing OpenEnded Philosophy Tools...")
        print("=" * 50)

        # Test 1: analyze_concept
        print("\n🎯 Test 1: analyze_concept")
        try:
            result = await operations.analyze_concept(
                concept="consciousness",
                context="philosophy_of_mind",
                perspectives=["materialist", "phenomenological"],
                confidence_threshold=0.7
            )
            print(f"✅ analyze_concept: SUCCESS")
            print(f"   Concept: {result.get('concept')}")
            print(f"   Confidence: {result.get('confidence_assessment', {}).get('overall', 'N/A')}")
        except Exception as e:
            print(f"❌ analyze_concept: FAILED - {e}")

        # Test 2: explore_coherence  
        print("\n🎯 Test 2: explore_coherence")
        try:
            result = await operations.explore_coherence(
                domain="metaphysics",
                depth=3,
                allow_revision=True
            )
            print(f"✅ explore_coherence: SUCCESS")
            print(f"   Domain: {result.get('domain')}")
            print(f"   Concepts analyzed: {len(result.get('concepts_analyzed', []))}")
        except Exception as e:
            print(f"❌ explore_coherence: FAILED - {e}")

        # Test 3: generate_insights
        print("\n🎯 Test 3: generate_insights")
        try:
            result = await operations.generate_insights(
                phenomenon="emergence",
                perspectives=["materialist", "enactivist"],
                depth=3,
                include_contradictions=True
            )
            print(f"✅ generate_insights: SUCCESS")
            print(f"   Phenomenon: {result.get('phenomenon')}")
            print(f"   Insights generated: {len(result.get('substantive_insights', []))}")
        except Exception as e:
            print(f"❌ generate_insights: FAILED - {e}")

        # Test 4: contextualize_meaning
        print("\n🎯 Test 4: contextualize_meaning")
        try:
            result = await operations.contextualize_meaning(
                expression="justice",
                language_game="ethical_deliberation",
                trace_genealogy=False
            )
            print(f"✅ contextualize_meaning: SUCCESS")
            print(f"   Expression: {result.get('expression')}")
            print(f"   Language game: {result.get('language_game')}")
        except Exception as e:
            print(f"❌ contextualize_meaning: FAILED - {e}")

        # Test 5: test_philosophical_hypothesis
        print("\n🎯 Test 5: test_philosophical_hypothesis")
        try:
            result = await operations.test_philosophical_hypothesis(
                hypothesis="Free will is incompatible with determinism",
                test_domains=["metaphysics", "neuroscience"],
                criteria={"logical_consistency": 0.8, "empirical_support": 0.6}
            )
            print(f"✅ test_philosophical_hypothesis: SUCCESS")
            print(f"   Hypothesis: {result.get('hypothesis', '')[:50]}...")
            print(f"   Posterior confidence: {result.get('posterior_confidence', 'N/A')}")
        except Exception as e:
            print(f"❌ test_philosophical_hypothesis: FAILED - {e}")

        # Test 6: recursive_self_analysis
        print("\n🎯 Test 6: recursive_self_analysis")
        try:
            # Use the result from analyze_concept for recursive analysis
            test_analysis = {
                "concept": "consciousness",
                "analysis_type": "concept_analysis",
                "results": {"confidence": 0.7}
            }
            result = await operations.recursive_self_analysis(
                analysis_result=test_analysis,
                analysis_type="concept_analysis",
                meta_depth=2
            )
            print(f"✅ recursive_self_analysis: SUCCESS")
            print(f"   Meta depth: {result.get('meta_depth')}")
            print(f"   Recursive insights: {len(result.get('recursive_insights', []))}")
        except Exception as e:
            print(f"❌ recursive_self_analysis: FAILED - {e}")

        print("\n" + "=" * 50)
        print("🎉 Testing completed!")

    except Exception as e:
        logger.error(f"Error in testing: {e}", exc_info=True)
        print(f"❌ Overall test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_fixed_tools())
