"""
NARS Integration Module for OpenEnded Philosophy MCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Integrates Non-Axiomatic Reasoning System (NARS) capabilities into the
philosophical framework, providing:

- Non-axiomatic reasoning with uncertainty quantification
- Truth maintenance and belief revision
- Evidence tracking and coherence analysis
- Temporal reasoning capabilities
"""

from .nars_manager import NARSManager
from .nars_memory import NARSMemory
from .nars_reasoning import NARSReasoning
from .truth_functions import Truth, TruthValue

__all__ = [
    "NARSManager",
    "NARSMemory", 
    "NARSReasoning",
    "Truth",
    "TruthValue"
]
