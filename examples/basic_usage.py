#!/usr/bin/env python3
"""
Example: Using OpenEnded Philosophy Framework
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This example demonstrates basic usage of the framework
for philosophical analysis without the MCP server.
"""

import asyncio
import sys

sys.path.append('..')  # Add parent directory to path

from openended_philosophy import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    EmergentCoherenceNode,
    FallibilisticInference,
    LanguageGameProcessor,
    calculate_epistemic_uncertainty,
    format_philosophical_output,
)


async def analyze_consciousness():
    """Example: Analyzing the concept of consciousness."""

    print("═" * 60)
    print("OpenEnded Philosophy: Analyzing 'Consciousness'")
    print("═" * 60)

    # 1. Create a coherence node for the concept
    consciousness_node = EmergentCoherenceNode(
        initial_pattern={
            'term': 'consciousness',
            'domain': 'philosophy_of_mind',
            'features': ['awareness', 'subjective_experience', 'intentionality'],
            'contexts': ['neuroscience', 'phenomenology', 'cognitive_science']
        },
        confidence=0.6,  # Moderate initial confidence
        context_sensitivity=0.9  # Highly context-sensitive concept
    )

    # 2. Set up language games
    scientific_game = LanguageGameProcessor(
        game_type='scientific_discourse',
        grammatical_rules={
            'empirical_verification': True,
            'operational_definitions': True,
            'causal_explanation': True
        }
    )

    phenomenological_game = LanguageGameProcessor(
        game_type='phenomenological_inquiry',
        grammatical_rules={
            'first_person_validity': True,
            'lived_experience': True,
            'intentional_structure': True
        }
    )

    # 3. Contextualize meaning in different games
    print("\n### Scientific Context")
    scientific_meaning = consciousness_node.contextualize_meaning(
        scientific_game,
        form_of_life={'research_paradigm': 'neuroscience'}
    )
    print(f"Confidence: {scientific_meaning['confidence']:.2f}")
    print(f"Meaning: {scientific_meaning['provisional_meaning']}")

    print("\n### Phenomenological Context")
    phenomenological_meaning = consciousness_node.contextualize_meaning(
        phenomenological_game,
        form_of_life={'tradition': 'continental_philosophy'}
    )
    print(f"Confidence: {phenomenological_meaning['confidence']:.2f}")
    print(f"Meaning: {phenomenological_meaning['provisional_meaning']}")

    # 4. Create pluralist framework
    pluralism = DynamicPluralismFramework(openness_coefficient=0.85)

    # Add multiple perspectives
    neuroscience_schema = {
        'id': 'neuroscience',
        'concepts': ['neural_correlates', 'information_integration', 'global_workspace'],
        'evaluate': lambda x: 0.8 if 'empirical' in str(x) else 0.4
    }

    philosophy_schema = {
        'id': 'philosophy',
        'concepts': ['qualia', 'hard_problem', 'phenomenal_consciousness'],
        'evaluate': lambda x: 0.8 if 'experience' in str(x) else 0.4
    }

    pluralism.integrate_perspective(neuroscience_schema)
    pluralism.integrate_perspective(philosophy_schema)

    # 5. Enable dialogue between perspectives
    print("\n### Inter-perspective Dialogue")
    dialogue = pluralism.dialogue_between_schemas(
        'neuroscience',
        'philosophy',
        topic={'concept': 'consciousness', 'aspect': 'subjective_experience'}
    )

    print(f"Agreements: {dialogue['agreements']}")
    print(f"Tensions: {dialogue['tensions']}")
    print(f"Interaction Quality: {dialogue['interaction_quality']:.2f}")

    # 6. Generate fallibilistic insights
    print("\n### Fallibilistic Insights")
    inference_engine = FallibilisticInference()

    evidence_patterns = [
        {
            'content': 'Neural activity correlates with reported conscious states',
            'concepts': ['neural_correlates', 'consciousness'],
            'confidence': 0.85
        },
        {
            'content': 'Subjective experience has qualitative properties irreducible to physical description',
            'concepts': ['qualia', 'hard_problem'],
            'confidence': 0.75
        },
        {
            'content': 'Information integration in brain networks relates to conscious awareness',
            'concepts': ['information_integration', 'global_workspace'],
            'confidence': 0.80
        }
    ]

    insights = await inference_engine.derive_insights(
        evidence_patterns,
        confidence_threshold=0.7
    )

    for i, insight in enumerate(insights, 1):
        print(f"\nInsight {i}:")
        print(f"Content: {insight.content}")
        print(f"Confidence: {insight.confidence:.2f}")
        print(f"Limitations: {', '.join(insight.identified_limitations[:2])}")
        print(f"Revision Triggers: {insight.revision_triggers[0]}")

    # 7. Calculate overall epistemic uncertainty
    landscape = CoherenceLandscape()
    landscape_state = await landscape.map_domain(
        domain='consciousness',
        depth=2,
        allow_revision=True
    )

    uncertainty = calculate_epistemic_uncertainty(
        evidence_count=len(evidence_patterns),
        coherence_score=landscape_state.global_coherence,
        temporal_factor=1.0,
        domain_complexity=0.9  # Consciousness is highly complex
    )

    print("\n### Epistemic Assessment")
    print(f"Overall Uncertainty: {uncertainty:.2f}")
    print(f"Recommendation: {'High confidence' if uncertainty < 0.5 else 'Proceed with caution'}")

def main():
    """Run examples."""
    print("\n🔮 OpenEnded Philosophy Framework Example\n")

    # Run async example
    asyncio.run(analyze_consciousness())

    print("\n\n✨ Example complete. The inquiry continues...\n")

if __name__ == "__main__":
    main()
