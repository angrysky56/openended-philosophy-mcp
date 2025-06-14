"""
OpenEnded Philosophy MCP Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A Model Context Protocol server implementing an open-ended philosophical framework
that embraces epistemic humility, contextual semantics, and dynamic pluralism.

### Theoretical Foundation

This framework operationalizes non-foundationalist philosophy through:
- **Fallibilistic Inference**: All conclusions carry uncertainty metrics
- **Language Games**: Context-dependent semantic processing
- **Dynamic Coherence**: Evolving conceptual landscapes
- **Pragmatic Orientation**: Efficacy-based evaluation

### Usage

```python
from openended_philosophy import OpenEndedPhilosophyServer

# Initialize server
server = OpenEndedPhilosophyServer()
server.setup_handlers()

# Run the server
await server.run()
```
"""

from .core import (
    CoherenceLandscape,
    CoherenceRegion,
    DynamicPluralismFramework,
    EmergentCoherenceNode,
    FallibilisticInference,
    FallibilisticInsight,
    LandscapeState,
    LanguageGameProcessor,
    MetaLearningEngine,
    SemanticPattern,
)
from .server import OpenEndedPhilosophyServer, PhilosophicalContext
from .utils import (
    calculate_epistemic_uncertainty,
    coherence_metrics,
    format_philosophical_output,
    pragmatic_evaluation,
    semantic_similarity,
    setup_logging,
)

__version__ = "0.1.0"
__author__ = "angrysky56"

__all__ = [
    # Core classes
    "EmergentCoherenceNode",
    "DynamicPluralismFramework",
    "LanguageGameProcessor",
    "CoherenceLandscape",
    "FallibilisticInference",
    "SemanticPattern",
    "CoherenceRegion",
    "LandscapeState",
    "FallibilisticInsight",
    "MetaLearningEngine",

    # Server components
    "OpenEndedPhilosophyServer",
    "PhilosophicalContext",

    # Utilities
    "setup_logging",
    "calculate_epistemic_uncertainty",
    "semantic_similarity",
    "coherence_metrics",
    "pragmatic_evaluation",
    "format_philosophical_output"
]
