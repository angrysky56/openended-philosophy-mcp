#!/usr/bin/env python3
"""
Test Enhanced OpenEnded Philosophy System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script tests the enhanced modules for the OpenEnded Philosophy framework.
"""

import asyncio
import json
import logging

# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.enhanced import (
    EnhancedInsightSynthesis,
    EnhancedNARSMemory,
    LLMSemanticProcessor,
    PhilosophicalContext,
    PhilosophicalNARSReasoning,
)
from openended_philosophy.nars import NARSManager, NARSMemory
from openended_philosophy.utils import setup_logging

# Configure logging
logger = setup_logging(__name__)


async def test_llm_semantic_processor():
    """Test the LLM semantic processor."""
    print("\n" + "="*80)
    print("Testing LLM Semantic Processor")
    print("="*80)

    processor = LLMSemanticProcessor()

    # Test philosophical statement analysis
    statement = "Consciousness emerges from complex information integration in neural networks"
    context = PhilosophicalContext(
        domain="philosophy of mind",
        inquiry_type="emergence",
        depth_requirements=3
    )

    analysis = await processor.analyze_statement(statement, context)

    print(f"\nStatement: '{statement}'")
    print(f"Primary concepts: {[c.term for c in analysis.primary_concepts]}")
    print(f"Epistemic uncertainty: {analysis.epistemic_uncertainty:.2f}")
    print(f"Philosophical categories: {list(analysis.philosophical_categorization.keys())}")
    print(f"Revision triggers: {analysis.revision_triggers[:2]}")

    return analysis


async def test_enhanced_nars_memory(llm_processor):
    """Test enhanced NARS memory."""
    print("\n" + "="*80)
    print("Testing Enhanced NARS Memory")
    print("="*80)

    # Initialize base NARS components
    nars_manager = NARSManager()
    base_memory = NARSMemory()

    # Create enhanced memory
    enhanced_memory = EnhancedNARSMemory(base_memory, llm_processor)

    # Process philosophical statements
    statements = [
        ("Knowledge requires justified true belief", "epistemology"),
        ("Consciousness is irreducible to physical processes", "philosophy of mind"),
        ("Moral truths are objective and universal", "ethics")
    ]

    beliefs = []
    for statement, domain in statements:
        belief = await enhanced_memory.process_philosophical_statement(
            statement=statement,
            context={"domain": domain},
            perspective="analytical"
        )
        beliefs.append(belief)
        print(f"\nProcessed: '{statement}'")
        print(f"  Truth: F={belief.truth.frequency:.2f}, C={belief.truth.confidence:.2f}")
        print(f"  Temporal scope: {belief.temporal_scope}")

    # Test belief revision
    print("\n\nTesting belief revision...")
    new_evidence = {
        "type": "challenging",
        "strength": 0.7,
        "content": "Gettier cases show JTB is insufficient for knowledge"
    }

    # Find belief to revise
    belief_id = list(enhanced_memory.philosophical_beliefs.keys())[0]
    revision_event = await enhanced_memory.revise_belief(
        belief_id=belief_id,
        new_evidence=new_evidence,
        revision_type="evidence_based"
    )

    if revision_event:
        print(f"Revised belief: '{revision_event.original_belief.statement}'")
        print(f"  Original truth: F={revision_event.original_belief.truth.frequency:.2f}")
        print(f"  Revised truth: F={revision_event.revised_belief.truth.frequency:.2f}")
        print(f"  Confidence change: {revision_event.confidence_change:.2f}")

    # Get coherence landscape
    landscape = enhanced_memory.get_coherence_landscape()
    print("\nCoherence landscape:")
    print(f"  Total beliefs: {landscape['total_beliefs']}")
    print(f"  Clusters: {len(landscape['clusters'])}")
    print(f"  Tensions: {len(landscape['tensions'])}")

    return enhanced_memory


async def test_philosophical_reasoning(enhanced_memory, llm_processor):
    """Test philosophical NARS reasoning."""
    print("\n" + "="*80)
    print("Testing Philosophical NARS Reasoning")
    print("="*80)

    nars_manager = NARSManager()
    reasoning = PhilosophicalNARSReasoning(nars_manager, enhanced_memory, llm_processor)

    # Test philosophical inference
    query = "What is the nature of consciousness?"
    context = {"domain": "philosophy of mind", "perspectives": ["phenomenological", "functionalist"]}

    print(f"\nQuery: '{query}'")
    inference_result = await reasoning.philosophical_inference(
        query=query,
        context=context,
        reasoning_type="mixed"
    )

    print(f"Reasoning type: {inference_result['reasoning_type']}")
    print(f"Relevant beliefs found: {len(inference_result['relevant_beliefs'])}")

    if inference_result['inference_result'].get('multi_pattern_analysis'):
        print(f"Patterns applied: {inference_result['inference_result']['patterns_applied']}")
        print(f"Integrated confidence: {inference_result['inference_result']['integrated_confidence']:.2f}")

    return reasoning


async def test_insight_synthesis(enhanced_memory, llm_processor):
    """Test enhanced insight synthesis."""
    print("\n" + "="*80)
    print("Testing Enhanced Insight Synthesis")
    print("="*80)

    synthesis_engine = EnhancedInsightSynthesis(enhanced_memory, llm_processor)

    # Test multi-perspectival synthesis
    inquiry = "the relationship between mind and body"
    perspectives = ["analytical", "phenomenological", "functionalist"]

    print(f"\nInquiry: '{inquiry}'")
    print(f"Perspectives: {perspectives}")

    insights = await synthesis_engine.synthesize_insights(
        inquiry_focus=inquiry,
        available_perspectives=perspectives,
        depth_level=3,
        synthesis_strategy="auto"
    )

    print(f"\nGenerated {len(insights)} insights:")
    for i, insight in enumerate(insights[:3]):  # Show first 3
        print(f"\n{i+1}. {insight.insight_type.upper()} insight:")
        print(f"   {insight.content}")
        print(f"   Confidence: {insight.confidence:.2f}")
        print(f"   Supporting perspectives: {insight.supporting_perspectives}")

    return insights


async def test_philosophical_concept():
    """Test a specific philosophical concept through the system."""
    print("\n" + "="*80)
    print("Testing Complete Philosophical Analysis: 'Emergence'")
    print("="*80)

    # Initialize components
    llm_processor = LLMSemanticProcessor()
    base_memory = NARSMemory()
    enhanced_memory = EnhancedNARSMemory(base_memory, llm_processor)

    # Analyze concept of emergence
    context = PhilosophicalContext(
        domain="metaphysics",
        inquiry_type="conceptual_analysis",
        depth_requirements=3
    )

    # Process relevant statements
    emergence_statements = [
        "Emergence occurs when complex systems exhibit properties not present in their parts",
        "Consciousness emerges from neural activity but cannot be reduced to it",
        "Emergent properties are ontologically novel",
        "Weak emergence is epistemological, strong emergence is metaphysical"
    ]

    for statement in emergence_statements:
        await enhanced_memory.process_philosophical_statement(
            statement=statement,
            context={"domain": "emergence", "perspective": "systems_theory"},
            perspective="analytical"
        )

    # Synthesize insights
    synthesis_engine = EnhancedInsightSynthesis(enhanced_memory, llm_processor)
    insights = await synthesis_engine.synthesize_insights(
        inquiry_focus="the nature of emergence",
        available_perspectives=["analytical", "phenomenological", "pragmatist"],
        depth_level=3
    )

    print("\nEmergence Analysis Results:")
    print(f"Beliefs stored: {len(enhanced_memory.philosophical_beliefs)}")
    print(f"Insights generated: {len(insights)}")

    # Show key insight
    if insights:
        key_insight = max(insights, key=lambda i: i.confidence)
        print(f"\nKey insight (confidence {key_insight.confidence:.2f}):")
        print(f"{key_insight.content}")
        print("\nPractical implications:")
        for imp in key_insight.practical_implications:
            print(f"  - {imp}")


async def main():
    """Run all tests."""
    print("OpenEnded Philosophy Enhanced Modules Test Suite")
    print("=" * 80)

    try:
        # Test LLM processor
        semantic_analysis = await test_llm_semantic_processor()

        # Test enhanced memory
        llm_processor = LLMSemanticProcessor()
        enhanced_memory = await test_enhanced_nars_memory(llm_processor)

        # Test philosophical reasoning
        reasoning = await test_philosophical_reasoning(enhanced_memory, llm_processor)

        # Test insight synthesis
        insights = await test_insight_synthesis(enhanced_memory, llm_processor)

        # Test complete philosophical analysis
        await test_philosophical_concept()

        print("\n" + "="*80)
        print("All tests completed successfully!")
        print("="*80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nTest failed with error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
