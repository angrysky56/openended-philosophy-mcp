#!/usr/bin/env python3
"""
Example: Analyzing the Concept of Truth
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This example demonstrates how to use the OpenEnded Philosophy framework
to analyze a fundamental philosophical concept through multiple lenses.

### Conceptual Architecture

The analysis employs:
- **Multi-perspectival investigation**: Examining truth from various philosophical traditions
- **Coherence mapping**: Identifying stable conceptual regions
- **Epistemic quantification**: Measuring confidence with uncertainty bounds
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openended_philosophy import (
    OpenEndedPhilosophyServer,
    format_philosophical_output
)

async def analyze_truth_concept():
    """
    Comprehensive analysis of the concept of truth.
    
    ### Methodological Framework
    
    1. **Contextual Grounding**: Situate concept within epistemological discourse
    2. **Perspectival Analysis**: Apply correspondence, coherence, and pragmatist lenses
    3. **Tension Identification**: Map contradictions and complementarities
    4. **Synthesis Generation**: Integrate insights with uncertainty metrics
    """
    # Initialize server
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("OpenEnded Philosophy: Analyzing the Concept of Truth")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Analyze concept
    result = await server._analyze_concept(
        concept="truth",
        context="epistemology",
        perspectives=[
            "correspondence",
            "coherentist", 
            "pragmatist",
            "deflationist",
            "phenomenological"
        ],
        confidence_threshold=0.6
    )
    
    # Format output
    print("\n### Analysis Results ###\n")
    
    # Academic format
    academic_output = format_philosophical_output(result, style="academic")
    print(academic_output)
    
    # Epistemic metrics
    print("\n### Epistemic Metrics ###")
    print(f"Epistemic Status: {result['epistemic_status']}")
    print(f"Analysis Timestamp: {result['timestamp']}")
    
    # Identified tensions
    if result['tensions']:
        print("\n### Conceptual Tensions ###")
        for tension in result['tensions']:
            print(f"- {tension}")
    
    # Revision conditions
    print("\n### Revision Conditions ###")
    for condition in result['revision_conditions']:
        print(f"- {condition}")
    
    # Further questions
    print("\n### Further Inquiry Paths ###")
    for question in result['further_questions']:
        print(f"- {question}")
    
    # Save results
    output_path = Path("truth_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Full results saved to: {output_path}")
    
    return result

async def explore_truth_coherence():
    """
    Map coherence landscape for truth-related concepts.
    
    ### Topological Analysis
    
    Explores the conceptual topology surrounding truth, identifying:
    - **Stable regions**: Areas of conceptual consensus
    - **Phase transitions**: Boundaries between interpretations
    - **Emergent patterns**: Novel conceptual connections
    """
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    
    print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Exploring Coherence Landscape: Truth and Related Concepts")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Explore coherence
    landscape = await server._explore_coherence(
        domain="epistemology",
        depth=3,
        allow_revision=True
    )
    
    print("\n### Coherence Landscape Analysis ###")
    
    # Global metrics
    print(f"\nGlobal Coherence: {landscape['stability_metrics']['overall_coherence']:.3f}")
    print(f"Fragmentation Index: {landscape['stability_metrics']['fragmentation_index']:.3f}")
    print(f"Emergence Potential: {landscape['stability_metrics']['emergence_potential']:.3f}")
    
    # Coherence regions
    print("\n### Identified Coherence Regions ###")
    for region in landscape['regions'][:5]:  # Top 5 regions
        print(f"\nRegion: {region['id']}")
        print(f"  Central Concepts: {', '.join(region['central_concepts'])}")
        print(f"  Stability: {region['stability']:.3f}")
        print(f"  Semantic Density: {region['semantic_density']:.3f}")
    
    # Phase transitions
    if landscape['transitions']:
        print("\n### Phase Transitions ###")
        for transition in landscape['transitions'][:3]:
            print(f"- {transition}")
    
    # Emergent patterns
    if landscape['emergent_patterns']:
        print("\n### Emergent Patterns ###")
        for pattern in landscape['emergent_patterns']:
            print(f"- Type: {pattern['type']}")
            print(f"  Emergence Score: {pattern.get('emergence_score', 'N/A')}")
    
    return landscape

async def contextualize_truth_meaning():
    """
    Analyze meaning of 'truth' across different language games.
    
    ### Wittgensteinian Analysis
    
    Examines how the concept of truth functions within different
    discursive contexts, revealing family resemblances and variations.
    """
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    
    print("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Contextualizing 'Truth' Across Language Games")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    language_games = [
        "scientific_discourse",
        "ethical_deliberation",
        "aesthetic_judgment",
        "ordinary_language"
    ]
    
    for game in language_games:
        print(f"\n### {game.replace('_', ' ').title()} ###")
        
        result = await server._contextualize_meaning(
            expression="truth",
            language_game=game,
            trace_genealogy=False
        )
        
        print(f"\nPrimary Meaning: {result['primary_meaning']['primary_sense']}")
        print(f"Meaning Stability: {result['meaning_stability']:.3f}")
        
        print("\nUsage Patterns:")
        for pattern in result['usage_patterns'][:2]:
            print(f"  - {pattern['pattern']}: {pattern['frequency']:.2f}")
        
        print("\nFamily Resemblances:")
        for concept in result['family_resemblances'][:3]:
            print(f"  - {concept}")
        
        print("\nCanonical Uses:")
        for use in result['canonical_uses'][:2]:
            print(f"  - {use}")

async def main():
    """
    Execute comprehensive truth analysis demonstration.
    
    ### Execution Flow
    
    1. Multi-perspectival concept analysis
    2. Coherence landscape mapping
    3. Cross-game semantic analysis
    4. Integration and synthesis
    """
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║              OpenEnded Philosophy Framework Demonstration                 ║")
    print("║                    Analyzing the Concept of Truth                         ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # Run analyses
    await analyze_truth_concept()
    await explore_truth_coherence()
    await contextualize_truth_meaning()
    
    print("\n\n✓ Demonstration complete!")
    print("\nKey Insights:")
    print("- Truth exhibits contextual semantic variation across language games")
    print("- Multiple stable coherence regions exist within epistemological space")
    print("- Tensions between perspectives reveal productive inquiry paths")
    print("- All conclusions remain provisional and open to revision")

if __name__ == "__main__":
    asyncio.run(main())
