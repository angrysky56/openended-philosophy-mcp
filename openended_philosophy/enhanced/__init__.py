"""
Enhanced OpenEnded Philosophy Modules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This package contains enhanced modules that provide deep LLM and NARS integration
for the OpenEnded Philosophy framework.
"""

from .enhanced_nars_integration import (
    BeliefRevisionEvent,
    EnhancedNARSMemory,
    PhilosophicalBelief,
    PhilosophicalNARSReasoning,
)
from .insight_synthesis import (
    DialecticalTension,
    EnhancedInsightSynthesis,
    PerspectivalAnalysis,
    SubstantiveInsight,
    SynthesisPathway,
)
from .llm_semantic_processor import (
    LLMSemanticProcessor,
    PhilosophicalConcept,
    PhilosophicalContext,
    SemanticAnalysis,
)

__all__ = [
    # LLM Semantic Processor
    'LLMSemanticProcessor',
    'PhilosophicalConcept',
    'SemanticAnalysis',
    'PhilosophicalContext',

    # Enhanced NARS Integration
    'EnhancedNARSMemory',
    'PhilosophicalBelief',
    'BeliefRevisionEvent',
    'PhilosophicalNARSReasoning',

    # Insight Synthesis
    'EnhancedInsightSynthesis',
    'SubstantiveInsight',
    'PerspectivalAnalysis',
    'DialecticalTension',
    'SynthesisPathway'
]
