#!/usr/bin/env python3
"""
Demonstrate NARS integration features in OpenEnded Philosophy MCP
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.nars import NARSManager, NARSMemory, NARSReasoning, Truth
from openended_philosophy.nars.types import TruthValue


async def demonstrate_nars_features():
    """Show key NARS integration capabilities."""
    print("NARS Integration Demonstration")
    print("=" * 80)

    # Initialize NARS components
    nars_manager = NARSManager()
    nars_memory = NARSMemory()
    nars_reasoning = NARSReasoning(nars_manager, nars_memory)

    print("\n1. Truth Value Operations")
    print("-" * 40)

    # Demonstrate truth values
    tv1 = TruthValue(0.8, 0.9)  # High confidence belief
    tv2 = TruthValue(0.6, 0.7)  # Moderate confidence belief

    print(f"Belief 1: frequency={tv1.frequency}, confidence={tv1.confidence}")
    print(f"Belief 2: frequency={tv2.frequency}, confidence={tv2.confidence}")

    # Revision
    revised = Truth.revision(tv1, tv2)
    print(f"Revised: frequency={revised.frequency:.3f}, confidence={revised.confidence:.3f}")

    # Different inference types
    deduced = Truth.deduction(tv1, tv2)
    print(f"Deduced: frequency={deduced.frequency:.3f}, confidence={deduced.confidence:.3f}")

    induced = Truth.induction(tv1, tv2)
    print(f"Induced: frequency={induced.frequency:.3f}, confidence={induced.confidence:.3f}")

    print("\n2. Memory System")
    print("-" * 40)

    # Add beliefs to memory
    nars_memory.add_belief(
        term="<consciousness --> emergent>",
        truth=TruthValue(0.7, 0.8),
        occurrence_time="eternal"
    )

    nars_memory.add_belief(
        term="<consciousness --> computational>",
        truth=TruthValue(0.6, 0.7),
        occurrence_time="eternal"
    )

    nars_memory.add_belief(
        term="<emergent --> [complex]>",
        truth=TruthValue(0.9, 0.9),
        occurrence_time="eternal"
    )

    # Get coherence landscape
    landscape = nars_memory.get_coherence_landscape()
    print("Coherence Landscape:")
    for category, info in landscape.items():
        print(f"  {category}: {info['belief_count']} beliefs, coherence={info['semantic_coherence']:.2f}")

    # Attention buffer demonstration
    attention = nars_memory.get_attention_buffer(query="consciousness")
    print(f"\nAttention Buffer (query='consciousness'): {len(attention)} items")
    for item in attention[:3]:
        print(f"  - {item.term} ({item.truth.expectation:.2f})")

    print("\n3. Reasoning Patterns")
    print("-" * 40)

    # Show different reasoning patterns available
    reasoning_patterns = [
        "Deductive: From general principles to specific conclusions",
        "Inductive: From specific instances to general rules",
        "Abductive: Inference to the best explanation",
        "Analogical: Transfer properties based on similarity",
        "Dialectical: Synthesis through opposing views"
    ]

    for pattern in reasoning_patterns:
        print(f"  • {pattern}")

    print("\n4. Philosophical Enhancement")
    print("-" * 40)

    print("NARS provides:")
    print("  • Non-axiomatic reasoning (no fixed axioms)")
    print("  • Truth maintenance (automatic belief revision)")
    print("  • Temporal reasoning (beliefs change over time)")
    print("  • Evidence tracking (traceable inference chains)")
    print("  • Uncertainty quantification (epistemic humility)")

    print("\n5. Integration Benefits")
    print("-" * 40)

    benefits = [
        "Combines philosophical analysis with formal reasoning",
        "Tracks evidential support for all claims",
        "Handles contradictions through revision",
        "Provides confidence metrics for insights",
        "Enables learning from experience"
    ]

    for benefit in benefits:
        print(f"  ✓ {benefit}")

    print("\n" + "=" * 80)
    print("NARS + Philosophy = Enhanced epistemic reasoning!")


if __name__ == "__main__":
    asyncio.run(demonstrate_nars_features())
