#!/usr/bin/env python3

import sys
sys.path.append('/home/ty/Repositories/ai_workspace/openended-philosophy-mcp')

from openended_philosophy.operations import PhilosophicalOperations
from openended_philosophy.core import (
    CoherenceLandscape,
    DynamicPluralismFramework, 
    FallibilisticInference,
    LanguageGameProcessor
)
from openended_philosophy.nars import NARSManager, NARSMemory, NARSReasoning

# Create minimal objects for testing
pluralism = DynamicPluralismFramework()
coherence = CoherenceLandscape()
inference = FallibilisticInference()
language_games = {"scientific": LanguageGameProcessor("scientific", [])}
nars_manager = NARSManager()
nars_memory = NARSMemory()
nars_reasoning = NARSReasoning(nars_manager, nars_memory)

# Create operations instance
ops = PhilosophicalOperations(
    pluralism_framework=pluralism,
    coherence_landscape=coherence,
    inference_engine=inference,
    language_games=language_games,
    nars_manager=nars_manager,
    nars_memory=nars_memory,
    nars_reasoning=nars_reasoning
)

# Test if methods exist
methods_to_test = [
    'analyze_concept',
    'generate_insights', 
    'explore_coherence',
    'contextualize_meaning',
    'test_philosophical_hypothesis',
    'recursive_self_analysis'
]

print("Testing method existence:")
for method_name in methods_to_test:
    if hasattr(ops, method_name):
        method = getattr(ops, method_name)
        print(f"✓ {method_name}: {type(method)}")
    else:
        print(f"✗ {method_name}: NOT FOUND")

print(f"\nAll attributes: {[attr for attr in dir(ops) if not attr.startswith('_')]}")
