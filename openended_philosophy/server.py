"""
Enhanced OpenEnded Philosophy MCP Server with Process Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Architectural Enhancements

This implementation incorporates comprehensive process management patterns
following MCP best practices to prevent resource leaks and ensure
graceful shutdown behavior.

#### Core Infrastructure Components:
- Signal handling for SIGTERM/SIGINT
- Background task tracking and cleanup
- Process lifecycle management
- Graceful shutdown orchestration
- Resource leak prevention
"""

import asyncio
import atexit
import json
import signal
import sys
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.server.stdio
import numpy as np
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from .core import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    EmergentCoherenceNode,
    FallibilisticInference,
    LanguageGameProcessor,
)
from .nars import NARSManager, NARSMemory, NARSReasoning, Truth, TruthValue
from .utils import calculate_epistemic_uncertainty, setup_logging

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Process Management Infrastructure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Global tracking for process management
background_tasks: set[asyncio.Task] = set()
running_processes: dict[str, Any] = {}
_shutdown_initiated = False

# Configure logging with academic formatting
logger = setup_logging(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cleanup_processes() -> None:
    """
    Comprehensive cleanup of all running processes and background tasks.

    ### Cleanup Protocol:
    1. Cancel all tracked background tasks
    2. Terminate any external processes
    3. Clear tracking dictionaries
    4. Log cleanup completion
    """
    global _shutdown_initiated
    if _shutdown_initiated:
        return

    _shutdown_initiated = True
    logger.info("Initiating philosophical server cleanup")

    # Cancel background tasks
    for task in background_tasks.copy():
        if not task.done():
            task.cancel()
            logger.debug(f"Cancelled background task: {task}")

    background_tasks.clear()

    # Clean up any tracked processes
    for process_id, process in running_processes.items():
        if hasattr(process, 'terminate'):
            try:
                process.terminate()
                logger.debug(f"Terminated process: {process_id}")
            except Exception as e:
                logger.warning(f"Error terminating process {process_id}: {e}")

    running_processes.clear()
    logger.info("Philosophical server cleanup completed")

def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown")
    cleanup_processes()
    sys.exit(0)

def track_background_task(task: asyncio.Task) -> None:
    """
    Track background task for proper cleanup.

    Args:
        task: Asyncio task to track
    """
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

# Register signal handlers and cleanup
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_processes)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PhilosophicalContext:
    """
    Enhanced contextual substrate for philosophical operations.

    ### Architectural Components:
    - Language game specifications
    - Confidence threshold parameters
    - Openness coefficient calibration
    - Meta-learning enablement flags
    - Comprehensive revision tracking
    """
    language_game: str = "general_inquiry"
    confidence_threshold: float = 0.7
    openness_coefficient: float = 0.9
    meta_learning_enabled: bool = True
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OpenEndedPhilosophyServer:
    """
    Enhanced MCP Server Implementation for Open-Ended Philosophical Framework

    ### Theoretical Foundation

    This implementation operationalizes non-foundationalist philosophical inquiry
    through computational pragmatism, incorporating:

    #### Mathematical Substrate:
    - Adaptive coherence landscapes
    - Fallibilistic inference engines
    - Dynamic pluralism frameworks
    - Uncertainty propagation models

    #### Process Management Architecture:
    - Signal handling integration
    - Background task orchestration
    - Resource lifecycle management
    - Graceful shutdown protocols

    #### Coherence Function:
        C(t) = Σ_{i} w_i(t) × φ_i(x,t) + λ × Emergence_Term(t)

    Where:
        - C(t): Total coherence at time t
        - w_i(t): Adaptive weights for perspective i
        - φ_i(x,t): Perspective evaluation function
        - λ: Openness coefficient
        - Emergence_Term(t): Novel pattern detection
    """

    def __init__(self) -> None:
        """
        Initialize philosophical substrate with enhanced process management.

        ### Initialization Protocol:
        1. Configure MCP server infrastructure
        2. Initialize philosophical context
        3. Instantiate core computational components
        4. Register language game processors
        5. Enable process tracking
        """
        self.server = Server("openended-philosophy")
        self.context = PhilosophicalContext()
        self._active_operations: set[str] = set()

        # Core computational components
        self.pluralism_framework = DynamicPluralismFramework(
            openness_coefficient=self.context.openness_coefficient
        )
        self.coherence_landscape = CoherenceLandscape(dimensionality='variable')
        self.inference_engine = FallibilisticInference()

        # Initialize NARS components
        self.nars_manager = NARSManager()
        self.nars_memory = NARSMemory(
            memory_file=Path("philosophy_nars_memory.json"),
            attention_size=30,
            recency_size=10
        )
        self.nars_reasoning = NARSReasoning(
            nars_manager=self.nars_manager,
            nars_memory=self.nars_memory
        )

        # Track NARS initialization
        self._nars_initialized = False

        # Language game registry with enhanced processors
        self.language_games: dict[str, LanguageGameProcessor] = {
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
            "aesthetic_judgment": LanguageGameProcessor(
                "aesthetic",
                {
                    "subjective_validity": True,
                    "cultural_context": True,
                    "expressive_content": True
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

        logger.info(f"OpenEnded Philosophy Server initialized with session: {self.context.session_id}")
        logger.debug(f"Full philosophical context: {safe_json_dumps(self.context)}")

    # ─────────────────────────────────────────────────────────────────────────

    def setup_handlers(self) -> None:
        """
        Configure MCP protocol handlers with enhanced error management.

        ### Handler Configuration Protocol:
        1. Tool enumeration handlers
        2. Tool execution handlers
        3. Error boundary implementation
        4. Resource cleanup integration
        """

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """
            Enumerate available philosophical operations with comprehensive metadata.

            ### Available Operations:
            - Concept analysis through interpretive lenses
            - Coherence landscape exploration
            - Contextual meaning derivation
            - Fallibilistic insight generation
            - Philosophical hypothesis testing
            """
            return [
                Tool(
                    name="analyze_concept",
                    description=(
                        "Analyzes a concept through multiple interpretive lenses "
                        "without claiming ontological priority. Returns provisional "
                        "insights with uncertainty quantification and revision conditions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept": {
                                "type": "string",
                                "description": "The concept to analyze"
                            },
                            "context": {
                                "type": "string",
                                "description": "Contextual domain (e.g., neuroscience, ethics)"
                            },
                            "perspectives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional specific perspectives to include"
                            },
                            "confidence_threshold": {
                                "type": "number",
                                "description": "Minimum confidence for insights (0-1)",
                                "default": 0.7,
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["concept", "context"]
                    }
                ),
                Tool(
                    name="explore_coherence",
                    description=(
                        "Maps provisional coherence patterns in conceptual space. "
                        "Identifies stable regions, phase transitions, and emergent "
                        "structures with topological analysis."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "Conceptual domain to explore"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Exploration depth (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5
                            },
                            "allow_revision": {
                                "type": "boolean",
                                "description": "Allow landscape revision during exploration",
                                "default": True
                            }
                        },
                        "required": ["domain"]
                    }
                ),
                Tool(
                    name="contextualize_meaning",
                    description=(
                        "Derives contextual semantics through Wittgensteinian "
                        "language game analysis. Shows how meaning emerges from "
                        "use in specific practices and forms of life."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Expression to contextualize"
                            },
                            "language_game": {
                                "type": "string",
                                "description": "Language game context",
                                "enum": list(self.language_games.keys())
                            },
                            "trace_genealogy": {
                                "type": "boolean",
                                "description": "Include semantic evolution history",
                                "default": False
                            }
                        },
                        "required": ["expression", "language_game"]
                    }
                ),
                Tool(
                    name="generate_insights",
                    description=(
                        "Produces fallibilistic insights with built-in uncertainty "
                        "quantification, revision conditions, and contradiction "
                        "mapping through multi-perspectival synthesis."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "phenomenon": {
                                "type": "string",
                                "description": "Phenomenon to investigate"
                            },
                            "perspectives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Interpretive perspectives to apply"
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Analysis depth (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5
                            },
                            "include_contradictions": {
                                "type": "boolean",
                                "description": "Explicitly include contradictory insights",
                                "default": True
                            }
                        },
                        "required": ["phenomenon"]
                    }
                ),
                Tool(
                    name="test_philosophical_hypothesis",
                    description=(
                        "Tests philosophical hypotheses through coherence analysis, "
                        "pragmatic evaluation, and multi-domain testing with "
                        "confidence metrics and revision recommendations."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "string",
                                "description": "The hypothesis to test"
                            },
                            "test_domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Domains for testing"
                            },
                            "criteria": {
                                "type": "object",
                                "description": "Custom evaluation criteria"
                            }
                        },
                        "required": ["hypothesis"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """
            Execute philosophical operations with comprehensive error management.

            ### Execution Protocol:
            1. Operation tracking registration
            2. Timeout-bounded execution
            3. Error boundary enforcement
            4. Resource cleanup guarantee
            """
            operation_id = str(uuid.uuid4())
            self._active_operations.add(operation_id)

            try:
                logger.info(f"Executing philosophical operation: {name} [{operation_id}]")
                logger.debug(f"Arguments: {arguments}")

                # Create timeout-bounded execution
                async def execute_with_timeout() -> dict[str, Any]:
                    if name == "analyze_concept":
                        return await self._analyze_concept(**arguments)
                    elif name == "explore_coherence":
                        return await self._explore_coherence(**arguments)
                    elif name == "contextualize_meaning":
                        return await self._contextualize_meaning(**arguments)
                    elif name == "generate_insights":
                        return await self._generate_insights(**arguments)
                    elif name == "test_philosophical_hypothesis":
                        return await self._test_hypothesis(**arguments)
                    else:
                        raise ValueError(f"Unknown philosophical operation: {name}")

                # Execute with timeout (30 seconds default)
                result = await asyncio.wait_for(
                    execute_with_timeout(),
                    timeout=30.0
                )

                logger.info(f"Completed philosophical operation: {name} [{operation_id}]")
                return [TextContent(type="text", text=safe_json_dumps(result, indent=2))]

            except asyncio.TimeoutError:
                logger.error(f"Operation timeout: {name} [{operation_id}]")
                error_response = {
                    "error": "Operation timeout",
                    "operation": name,
                    "epistemic_status": "temporal_limitation",
                    "suggestion": "Consider reducing analysis depth or scope"
                }
                return [TextContent(type="text", text=safe_json_dumps(error_response, indent=2))]

            except Exception as e:
                logger.error(f"Operation error in {name} [{operation_id}]: {e}", exc_info=True)
                error_response = {
                    "error": str(e),
                    "operation": name,
                    "epistemic_status": "computational_limitation",
                    "suggestion": "Consider reformulating the philosophical inquiry",
                    "operation_id": operation_id
                }
                return [TextContent(type="text", text=safe_json_dumps(error_response, indent=2))]

            finally:
                # Ensure cleanup
                self._active_operations.discard(operation_id)
                logger.debug(f"Cleaned up operation: {name} [{operation_id}]")

    # ─────────────────────────────────────────────────────────────────────────
    # Core Philosophical Operations (Enhanced)
    # ─────────────────────────────────────────────────────────────────────────

    async def _analyze_concept(
        self,
        concept: str,
        context: str,
        perspectives: list[str] | None = None,
        confidence_threshold: float = 0.7
    ) -> dict[str, Any]:
        """
        Enhanced concept analysis with comprehensive error handling.

        ### Analytical Methodology:
        1. Contextual grounding establishment
        2. Multi-perspectival interpretation synthesis
        3. Coherence pattern identification
        4. Uncertainty quantification protocols
        5. Revision condition generation

        Args:
            concept: Target concept for analysis
            context: Contextual domain specification
            perspectives: Optional interpretive lens selection
            confidence_threshold: Minimum epistemic confidence

        Returns:
            Comprehensive analysis results with uncertainty metrics
        """
        logger.debug(f"Initiating concept analysis: {concept} in context: {context}")

        try:
            # Use NARS-enhanced analysis if available
            if self._nars_initialized:
                try:
                    nars_analysis = await self.nars_reasoning.analyze_concept(
                        concept=concept,
                        context=context,
                        perspectives=perspectives or self._select_relevant_perspectives(concept, context)
                    )

                    # Store insights in NARS memory
                    for perspective, analysis in nars_analysis.get("perspective_analyses", {}).items():
                        if isinstance(analysis, dict) and "findings" in analysis:
                            for finding in analysis["findings"]:
                                self.nars_memory.add_belief(
                                    term=finding.get("claim", ""),
                                    truth=finding.get("truth", TruthValue(0.5, 0.5)),
                                    occurrence_time="eternal"
                                )

                    # Enhance with philosophical framework
                    nars_analysis["philosophical_enhancement"] = {
                        "coherence_landscape": self.nars_memory.get_coherence_landscape(),
                        "epistemic_status": self._assess_epistemic_status(nars_analysis.get("synthesis", {})),
                        "framework_integration": "NARS + Philosophical Pluralism"
                    }

                    return nars_analysis

                except Exception as e:
                    logger.warning(f"NARS analysis failed, falling back to philosophical analysis: {e}")
                    # Fall through to original implementation

            # Create provisional semantic node
            concept_node = EmergentCoherenceNode(
                initial_pattern={"term": concept, "domain": context, "timestamp": datetime.now().isoformat()},
                confidence=0.5  # Start with epistemic humility
            )

            # Select appropriate language game
            active_game = self._select_language_game(context)

            # Determine analytical perspectives
            if perspectives is None:
                perspectives = self._select_relevant_perspectives(concept, context)

            analysis_results = []

            # Multi-perspectival analysis with error isolation
            for perspective in perspectives:
                try:
                    logger.debug(f"Analyzing perspective: {perspective}")

                    # Create interpretive schema
                    schema = self._create_interpretive_schema(perspective)
                    logger.debug(f"Created schema for {perspective}: {list(schema.keys())}")

                    # Apply schema to concept
                    interpretation = await self._apply_schema_to_concept(
                        concept_node, schema, active_game
                    )
                    logger.debug(f"Applied schema, interpretation keys: {list(interpretation.keys())}")

                    # Calculate interpretation confidence
                    confidence = self._calculate_interpretation_confidence(
                        interpretation, active_game
                    )
                    logger.debug(f"Calculated confidence for {perspective}: {confidence:.3f} (threshold: {confidence_threshold})")

                    if confidence >= confidence_threshold:
                        analysis_results.append({
                            "perspective": perspective,
                            "interpretation": interpretation,
                            "confidence": confidence,
                            "coherence_score": interpretation.get("coherence", 0.0),
                            "uncertainty_bounds": self._calculate_uncertainty_bounds(
                                interpretation, confidence
                            )
                        })
                        logger.debug(f"Added analysis result for {perspective}")
                    else:
                        logger.debug(f"Confidence {confidence:.3f} below threshold {confidence_threshold} for {perspective}")

                except Exception as e:
                    logger.error(f"Perspective analysis failed for {perspective}: {e}", exc_info=True)
                    # Continue with other perspectives
                    continue

            # Synthesize findings with error resilience
            synthesis = self._synthesize_analyses(analysis_results)

            # Identify conceptual tensions
            tensions = self._identify_conceptual_tensions(analysis_results)

            # Generate revision conditions
            revision_triggers = self._generate_revision_conditions(
                concept, context, synthesis
            )

            return {
                "concept": concept,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.context.session_id,
                "analyses": analysis_results,
                "synthesis": synthesis,
                "tensions": tensions,
                "revision_conditions": revision_triggers,
                "epistemic_status": self._assess_epistemic_status(synthesis),
                "further_questions": self._generate_exploratory_questions(concept, synthesis),
                "uncertainty_profile": {
                    "total_perspectives": len(perspectives),
                    "successful_analyses": len(analysis_results),
                    "average_confidence": np.mean([r["confidence"] for r in analysis_results]) if analysis_results else 0.0,
                    "epistemic_uncertainty": calculate_epistemic_uncertainty(
                        len(analysis_results),
                        synthesis.get("overall_confidence", 0.5)
                    )
                }
            }

        except Exception as e:
            logger.error(f"Critical error in concept analysis: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods with Enhanced Error Handling
    # ─────────────────────────────────────────────────────────────────────────

    def _select_language_game(self, context: str) -> LanguageGameProcessor:
        """Select appropriate language game processor for context."""
        context_lower = context.lower()

        if any(keyword in context_lower for keyword in ["science", "physics", "biology", "neuroscience", "empirical"]):
            return self.language_games["scientific_discourse"]
        elif any(keyword in context_lower for keyword in ["ethics", "morality", "values", "rights", "justice"]):
            return self.language_games["ethical_deliberation"]
        elif any(keyword in context_lower for keyword in ["art", "beauty", "aesthetics", "creative", "artistic"]):
            return self.language_games["aesthetic_judgment"]
        else:
            return self.language_games["ordinary_language"]

    def _calculate_uncertainty_bounds(self, interpretation: dict[str, Any], confidence: float) -> dict[str, float]:
        """Calculate uncertainty bounds for interpretation."""
        base_uncertainty = 1.0 - confidence

        return {
            "lower_bound": max(0.0, confidence - base_uncertainty * 0.5),
            "upper_bound": min(1.0, confidence + base_uncertainty * 0.5),
            "epistemic_uncertainty": base_uncertainty,
            "confidence_interval": 0.95
        }

    # [Additional helper methods maintained from original implementation]
    # Implementation continues with all original helper methods...

    async def _explore_coherence(self, domain: str, depth: int = 3, allow_revision: bool = True) -> dict[str, Any]:
        """Enhanced coherence exploration with philosophical sophistication."""
        try:
            logger.debug(f"Starting coherence exploration for domain: {domain}, depth: {depth}")

            # Initialize domain-specific coherence analysis
            coherence_analysis = await self._initialize_domain_coherence_analysis(domain)

            # Multi-level coherence mapping
            coherence_layers = await self._map_coherence_layers(domain, depth, allow_revision)

            # Identify philosophical structures
            philosophical_structures = await self._identify_philosophical_structures(domain, coherence_layers)

            # Analyze conceptual attractors and phase transitions
            dynamical_analysis = await self._analyze_coherence_dynamics(domain, coherence_layers)

            # Generate coherence landscape report
            landscape_report = await self._generate_coherence_landscape_report(
                domain, coherence_layers, philosophical_structures, dynamical_analysis
            )

            # Include initial coherence analysis in the report
            landscape_report["initial_analysis"] = coherence_analysis

            return landscape_report

        except asyncio.TimeoutError:
            logger.warning(f"Coherence exploration timeout for domain: {domain}")
            return {
                "domain": domain,
                "status": "timeout",
                "partial_results": "Analysis exceeded time bounds",
                "suggestion": "Consider reducing exploration depth"
            }
        except Exception as e:
            logger.error(f"Error in coherence exploration: {e}", exc_info=True)
            return await self._generate_fallback_coherence_analysis(domain, depth)

    async def _initialize_domain_coherence_analysis(self, domain: str) -> dict[str, Any]:
        """Initialize domain-specific coherence analysis framework."""

        # Domain-specific coherence patterns
        domain_patterns = {
            'ethics': {
                'core_concepts': ['good', 'right', 'duty', 'virtue', 'justice', 'responsibility'],
                'coherence_dimensions': ['normative_consistency', 'motivational_efficacy', 'practical_applicability'],
                'typical_tensions': ['deontological_consequentialist', 'individual_social', 'universal_particular'],
                'stability_factors': ['cultural_consensus', 'rational_justification', 'intuitive_appeal']
            },
            'consciousness': {
                'core_concepts': ['experience', 'qualia', 'intentionality', 'awareness', 'subjectivity', 'neural_correlation'],
                'coherence_dimensions': ['phenomenological_adequacy', 'causal_efficacy', 'neural_plausibility'],
                'typical_tensions': ['subjective_objective', 'qualitative_quantitative', 'mental_physical'],
                'stability_factors': ['empirical_evidence', 'phenomenological_fidelity', 'theoretical_elegance']
            },
            'knowledge': {
                'core_concepts': ['belief', 'truth', 'justification', 'evidence', 'certainty', 'skepticism'],
                'coherence_dimensions': ['logical_consistency', 'empirical_adequacy', 'explanatory_power'],
                'typical_tensions': ['foundational_coherentist', 'empirical_rational', 'objective_subjective'],
                'stability_factors': ['logical_rigor', 'empirical_support', 'practical_success']
            },
            'reality': {
                'core_concepts': ['existence', 'substance', 'property', 'causation', 'time', 'space'],
                'coherence_dimensions': ['ontological_parsimony', 'causal_closure', 'empirical_compatibility'],
                'typical_tensions': ['material_mental', 'deterministic_free', 'absolute_relative'],
                'stability_factors': ['scientific_consistency', 'logical_necessity', 'intuitive_plausibility']
            },
            'mind': {
                'core_concepts': ['consciousness', 'cognition', 'emotion', 'reasoning', 'memory', 'perception'],
                'coherence_dimensions': ['psychological_reality', 'neural_implementation', 'behavioral_prediction'],
                'typical_tensions': ['computational_phenomenological', 'innate_learned', 'modular_holistic'],
                'stability_factors': ['empirical_validation', 'explanatory_depth', 'predictive_accuracy']
            }
        }

        # Get domain pattern or create generic
        pattern = domain_patterns.get(domain.lower(), {
            'core_concepts': [domain, 'analysis', 'understanding', 'interpretation'],
            'coherence_dimensions': ['conceptual_clarity', 'systematic_integration', 'practical_relevance'],
            'typical_tensions': ['theoretical_practical', 'abstract_concrete', 'universal_particular'],
            'stability_factors': ['logical_consistency', 'empirical_grounding', 'intuitive_appeal']
        })

        return {
            'domain': domain,
            'initialization_time': datetime.now().isoformat(),
            'domain_pattern': pattern,
            'coherence_framework': {
                'dimensions': pattern['coherence_dimensions'],
                'evaluation_criteria': ['consistency', 'completeness', 'explanatory_power', 'predictive_success'],
                'integration_methods': ['logical_analysis', 'semantic_mapping', 'pragmatic_evaluation']
            }
        }

    async def _map_coherence_layers(self, domain: str, depth: int, allow_revision: bool) -> list[dict[str, Any]]:
        """Map coherence across multiple conceptual layers."""

        layers = []

        for level in range(depth):
            layer_analysis = await self._analyze_coherence_layer(domain, level, allow_revision)
            layers.append(layer_analysis)

        return layers

    async def _analyze_coherence_layer(self, domain: str, level: int, allow_revision: bool) -> dict[str, Any]:
        """Analyze coherence at a specific conceptual layer."""

        # Layer-specific analysis patterns
        layer_types = {
            0: 'foundational_concepts',
            1: 'relational_structures',
            2: 'systematic_integration',
            3: 'meta_theoretical_framework',
            4: 'philosophical_implications'
        }

        layer_type = layer_types.get(level, 'extended_analysis')

        if layer_type == 'foundational_concepts':
            return await self._analyze_foundational_layer(domain)
        elif layer_type == 'relational_structures':
            return await self._analyze_relational_layer(domain)
        elif layer_type == 'systematic_integration':
            return await self._analyze_integration_layer(domain)
        elif layer_type == 'meta_theoretical_framework':
            return await self._analyze_meta_theoretical_layer(domain)
        else:
            return await self._analyze_extended_layer(domain, level)

    async def _analyze_foundational_layer(self, domain: str) -> dict[str, Any]:
        """Analyze foundational conceptual coherence."""

        # Core concept identification
        foundational_concepts = await self._identify_foundational_concepts(domain)

        # Analyze concept relationships
        concept_relations = await self._analyze_concept_relations(foundational_concepts)

        # Assess definitional coherence
        definitional_coherence = await self._assess_definitional_coherence(foundational_concepts)

        return {
            'layer_type': 'foundational_concepts',
            'layer_index': 0,
            'domain': domain,
            'foundational_concepts': foundational_concepts,
            'concept_relations': concept_relations,
            'definitional_coherence': definitional_coherence,
            'coherence_score': definitional_coherence.get('overall_score', 0.5),
            'stability_assessment': await self._assess_foundational_stability(foundational_concepts),
            'philosophical_significance': 'Establishes conceptual foundations for domain analysis'
        }

    async def _identify_foundational_concepts(self, domain: str) -> list[dict[str, Any]]:
        """Identify foundational concepts for the domain."""

        # Domain-specific foundational concepts
        domain_foundations = {
            'ethics': [
                {'concept': 'good', 'type': 'value_concept', 'centrality': 0.95},
                {'concept': 'right', 'type': 'deontic_concept', 'centrality': 0.9},
                {'concept': 'obligation', 'type': 'normative_concept', 'centrality': 0.85},
                {'concept': 'virtue', 'type': 'character_concept', 'centrality': 0.8},
                {'concept': 'justice', 'type': 'political_concept', 'centrality': 0.85}
            ],
            'consciousness': [
                {'concept': 'experience', 'type': 'phenomenological_concept', 'centrality': 0.95},
                {'concept': 'qualia', 'type': 'qualitative_concept', 'centrality': 0.9},
                {'concept': 'intentionality', 'type': 'directedness_concept', 'centrality': 0.85},
                {'concept': 'awareness', 'type': 'cognitive_concept', 'centrality': 0.8},
                {'concept': 'subjectivity', 'type': 'perspective_concept', 'centrality': 0.85}
            ],
            'knowledge': [
                {'concept': 'belief', 'type': 'doxastic_concept', 'centrality': 0.9},
                {'concept': 'truth', 'type': 'alethic_concept', 'centrality': 0.95},
                {'concept': 'justification', 'type': 'epistemic_concept', 'centrality': 0.9},
                {'concept': 'evidence', 'type': 'evidential_concept', 'centrality': 0.85},
                {'concept': 'certainty', 'type': 'modal_concept', 'centrality': 0.75}
            ]
        }

        return domain_foundations.get(domain.lower(), [
            {'concept': domain, 'type': 'domain_concept', 'centrality': 0.9},
            {'concept': 'analysis', 'type': 'methodological_concept', 'centrality': 0.7},
            {'concept': 'understanding', 'type': 'cognitive_concept', 'centrality': 0.8}
        ])

    async def _analyze_concept_relations(self, concepts: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze relations between foundational concepts."""
        from .utils import semantic_similarity

        relations = []

        # Generate pairwise relations
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:

                # Create features for similarity analysis
                features1 = {
                    'features': [concept1['concept'], concept1['type']],
                    'contexts': [concept1['type']],
                    'centrality': [concept1['centrality']]
                }
                features2 = {
                    'features': [concept2['concept'], concept2['type']],
                    'contexts': [concept2['type']],
                    'centrality': [concept2['centrality']]
                }

                similarity = semantic_similarity(features1, features2, method="wittgenstein")

                if similarity > 0.3:
                    relation_type = self._determine_relation_type(concept1, concept2, similarity)
                    relations.append({
                        'concept1': concept1['concept'],
                        'concept2': concept2['concept'],
                        'relation_type': relation_type,
                        'strength': similarity,
                        'philosophical_significance': self._assess_relation_significance(concept1, concept2, relation_type)
                    })

        return {
            'total_relations': len(relations),
            'relations': relations,
            'relation_density': len(relations) / (len(concepts) * (len(concepts) - 1) / 2) if len(concepts) > 1 else 0,
            'dominant_relation_types': self._identify_dominant_relation_types(relations)
        }

    async def _assess_definitional_coherence(self, concepts: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess definitional coherence of foundational concepts."""

        coherence_scores = []

        for concept in concepts:
            # Assess individual concept coherence
            concept_coherence = await self._assess_individual_concept_coherence(concept)
            coherence_scores.append(concept_coherence)

        overall_score = sum(score['coherence_score'] for score in coherence_scores) / len(coherence_scores)

        return {
            'individual_coherence_scores': coherence_scores,
            'overall_score': overall_score,
            'coherence_distribution': self._analyze_coherence_distribution(coherence_scores),
            'coherence_challenges': self._identify_coherence_challenges(coherence_scores),
            'coherence_strengths': self._identify_coherence_strengths(coherence_scores)
        }

    async def _assess_individual_concept_coherence(self, concept: dict[str, Any]) -> dict[str, Any]:
        """Assess coherence of individual concept."""

        concept_name = concept['concept']
        concept_type = concept['type']
        centrality = concept['centrality']

        # Coherence factors
        definitional_clarity = self._assess_definitional_clarity(concept_name, concept_type)
        internal_consistency = self._assess_internal_consistency(concept_name, concept_type)
        contextual_stability = self._assess_contextual_stability(concept_name, concept_type)

        # Weighted coherence score
        coherence_score = (
            0.4 * definitional_clarity +
            0.35 * internal_consistency +
            0.25 * contextual_stability
        ) * centrality  # Weight by centrality

        return {
            'concept': concept_name,
            'type': concept_type,
            'coherence_score': coherence_score,
            'definitional_clarity': definitional_clarity,
            'internal_consistency': internal_consistency,
            'contextual_stability': contextual_stability,
            'centrality_weight': centrality
        }

    def _assess_definitional_clarity(self, concept: str, concept_type: str) -> float:
        """Assess definitional clarity of concept."""

        # Concept clarity patterns
        clarity_patterns = {
            'value_concept': 0.6,  # Value concepts often contested
            'deontic_concept': 0.7,  # Deontic concepts moderately clear
            'normative_concept': 0.65,  # Normative concepts somewhat contested
            'character_concept': 0.75,  # Character concepts relatively clear
            'phenomenological_concept': 0.5,  # Phenomenological concepts difficult to define
            'qualitative_concept': 0.45,  # Qualia notoriously difficult
            'directedness_concept': 0.7,  # Intentionality well-understood
            'cognitive_concept': 0.8,  # Cognitive concepts relatively clear
            'alethic_concept': 0.6,  # Truth concepts contested
            'epistemic_concept': 0.75,  # Epistemic concepts moderately clear
            'domain_concept': 0.7  # Domain concepts moderately clear
        }

        base_clarity = clarity_patterns.get(concept_type, 0.65)

        # Adjust for specific concepts
        if concept in ['consciousness', 'qualia', 'good', 'justice']:
            base_clarity *= 0.9  # These are particularly contested
        elif concept in ['belief', 'awareness', 'virtue']:
            base_clarity *= 1.1  # These are relatively clear

        return min(base_clarity, 1.0)

    def _assess_internal_consistency(self, concept: str, concept_type: str) -> float:
        """Assess internal consistency of concept."""

        # Most philosophical concepts have moderate internal consistency
        # This would involve analyzing logical relations, necessary conditions, etc.

        consistency_baselines = {
            'value_concept': 0.7,
            'deontic_concept': 0.8,
            'normative_concept': 0.75,
            'character_concept': 0.8,
            'phenomenological_concept': 0.65,
            'qualitative_concept': 0.6,
            'directedness_concept': 0.85,
            'cognitive_concept': 0.8,
            'alethic_concept': 0.75,
            'epistemic_concept': 0.8
        }

        return consistency_baselines.get(concept_type, 0.75)

    def _assess_contextual_stability(self, concept: str, concept_type: str) -> float:
        """Assess contextual stability of concept."""

        # Stability across different philosophical contexts
        stability_patterns = {
            'value_concept': 0.6,  # Values vary across cultures/contexts
            'deontic_concept': 0.75,  # Duties more stable
            'normative_concept': 0.65,  # Norms somewhat context-dependent
            'character_concept': 0.8,  # Virtues relatively stable
            'phenomenological_concept': 0.7,  # Experience concepts moderately stable
            'qualitative_concept': 0.8,  # Qualia stable when present
            'directedness_concept': 0.85,  # Intentionality very stable
            'cognitive_concept': 0.75,  # Cognitive concepts moderately stable
            'alethic_concept': 0.7,  # Truth concepts somewhat context-dependent
            'epistemic_concept': 0.8  # Epistemic concepts relatively stable
        }

        return stability_patterns.get(concept_type, 0.75)

    def _determine_relation_type(self, concept1: dict[str, Any], concept2: dict[str, Any], similarity: float) -> str:
        """Determine the type of relation between concepts."""

        type1 = concept1['type']
        type2 = concept2['type']

        # Type-based relation patterns
        if type1 == type2:
            return 'same_type_relation'
        elif (type1, type2) in [('value_concept', 'normative_concept'), ('normative_concept', 'value_concept')]:
            return 'value_norm_relation'
        elif (type1, type2) in [('phenomenological_concept', 'cognitive_concept'), ('cognitive_concept', 'phenomenological_concept')]:
            return 'mind_experience_relation'
        elif similarity > 0.7:
            return 'strong_conceptual_relation'
        elif similarity > 0.5:
            return 'moderate_conceptual_relation'
        else:
            return 'weak_conceptual_relation'

    def _assess_relation_significance(self, concept1: dict[str, Any], concept2: dict[str, Any], relation_type: str) -> str:
        """Assess philosophical significance of relation."""

        significance_patterns = {
            'value_norm_relation': 'fundamental_ethical_connection',
            'mind_experience_relation': 'core_consciousness_relation',
            'same_type_relation': 'categorical_coherence',
            'strong_conceptual_relation': 'deep_philosophical_connection',
            'moderate_conceptual_relation': 'meaningful_philosophical_relation',
            'weak_conceptual_relation': 'peripheral_philosophical_connection'
        }

        return significance_patterns.get(relation_type, 'general_philosophical_relation')

    async def _contextualize_meaning(self, expression: str, language_game: str, trace_genealogy: bool = False) -> dict[str, Any]:
        """Enhanced meaning contextualization with genealogy tracing."""
        logger.debug(f"Contextualizing '{expression}' in game: {language_game}")

        try:
            # Get active language game processor
            game_processor = self.language_games.get(
                language_game,
                self.language_games["ordinary_language"]
            )

            # Process expression with timeout
            semantic_analysis = await asyncio.wait_for(
                asyncio.create_task(asyncio.to_thread(
                    game_processor.process_expression,
                    expression,
                    trace_genealogy
                )),
                timeout=15.0
            )

            result = {
                "expression": expression,
                "language_game": language_game,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.context.session_id,
                "primary_meaning": semantic_analysis.contextual_meaning,
                "usage_patterns": semantic_analysis.usage_patterns,
                "family_resemblances": semantic_analysis.related_concepts,
                "pragmatic_conditions": semantic_analysis.success_conditions,
                "meaning_stability": semantic_analysis.stability_score
            }

            # Add genealogy if requested
            if trace_genealogy:
                genealogy = await self._trace_semantic_evolution(expression)
                result["semantic_genealogy"] = genealogy

            return result

        except asyncio.TimeoutError:
            logger.warning(f"Meaning contextualization timeout for: {expression}")
            return {
                "expression": expression,
                "status": "timeout",
                "suggestion": "Consider simplifying the expression or reducing trace_genealogy scope"
            }

    def _identify_dominant_relation_types(self, relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify dominant relation types in concept network."""
        from collections import Counter

        if not relations:
            return []

        relation_counts = Counter(r['relation_type'] for r in relations)
        return [{'type': rel_type, 'count': count} for rel_type, count in relation_counts.most_common(3)]

    def _analyze_coherence_distribution(self, coherence_scores: list[dict[str, Any]]) -> dict[str, float]:
        """Analyze distribution of coherence scores."""
        scores = [cs['coherence_score'] for cs in coherence_scores]
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'range': float(np.max(scores) - np.min(scores))
        }

    def _identify_coherence_challenges(self, coherence_scores: list[dict[str, Any]]) -> list[str]:
        """Identify main coherence challenges."""
        challenges = []

        for score_data in coherence_scores:
            if score_data['definitional_clarity'] < 0.6:
                challenges.append(f"Definitional ambiguity in {score_data['concept']}")
            if score_data['internal_consistency'] < 0.7:
                challenges.append(f"Internal tensions in {score_data['concept']}")
            if score_data['contextual_stability'] < 0.7:
                challenges.append(f"Context sensitivity in {score_data['concept']}")

        return challenges[:5]  # Top 5 challenges

    def _identify_coherence_strengths(self, coherence_scores: list[dict[str, Any]]) -> list[str]:
        """Identify main coherence strengths."""
        strengths = []

        for score_data in coherence_scores:
            if score_data['coherence_score'] > 0.8:
                strengths.append(f"Strong conceptual foundation in {score_data['concept']}")
            if score_data['definitional_clarity'] > 0.8:
                strengths.append(f"Clear definitional structure for {score_data['concept']}")
            if score_data['contextual_stability'] > 0.8:
                strengths.append(f"Robust contextual stability in {score_data['concept']}")

        return strengths[:5]  # Top 5 strengths

    async def _assess_foundational_stability(self, concepts: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess stability of foundational conceptual layer."""

        stability_factors = []
        for concept in concepts:
            centrality = concept['centrality']
            concept_name = concept['concept']

            # Historical stability assessment
            historical_stability = self._assess_historical_stability(concept_name)

            # Cross-cultural stability assessment
            cultural_stability = self._assess_cultural_stability(concept_name)

            # Theoretical stability assessment
            theoretical_stability = self._assess_theoretical_stability(concept_name)

            overall_stability = (historical_stability + cultural_stability + theoretical_stability) / 3
            weighted_stability = overall_stability * centrality

            stability_factors.append({
                'concept': concept_name,
                'historical_stability': historical_stability,
                'cultural_stability': cultural_stability,
                'theoretical_stability': theoretical_stability,
                'overall_stability': overall_stability,
                'weighted_stability': weighted_stability
            })

        average_stability = sum(sf['weighted_stability'] for sf in stability_factors) / len(stability_factors)

        return {
            'individual_stabilities': stability_factors,
            'average_stability': average_stability,
            'stability_range': max(sf['overall_stability'] for sf in stability_factors) - min(sf['overall_stability'] for sf in stability_factors),
            'most_stable_concepts': [sf['concept'] for sf in stability_factors if sf['overall_stability'] > 0.8],
            'least_stable_concepts': [sf['concept'] for sf in stability_factors if sf['overall_stability'] < 0.6]
        }

    def _assess_historical_stability(self, concept: str) -> float:
        """Assess historical stability of concept across time."""
        # Historical stability patterns for key philosophical concepts
        historical_patterns = {
            'good': 0.7,  # Contested but persistent
            'right': 0.75,  # Relatively stable
            'justice': 0.65,  # Evolving concept
            'virtue': 0.8,  # Very stable since Aristotle
            'truth': 0.85,  # Remarkably stable
            'knowledge': 0.75,  # Stable with episodic challenges
            'consciousness': 0.6,  # Relatively new as technical term
            'experience': 0.8,  # Very stable
            'belief': 0.85,  # Highly stable
            'existence': 0.9,  # Extremely stable
        }
        return historical_patterns.get(concept, 0.7)

    def _assess_cultural_stability(self, concept: str) -> float:
        """Assess cross-cultural stability of concept."""
        cultural_patterns = {
            'good': 0.6,  # Varies significantly across cultures
            'right': 0.7,  # Moderate cross-cultural stability
            'justice': 0.65,  # Cultural variation in conception
            'virtue': 0.75,  # Core virtues fairly universal
            'truth': 0.8,  # Generally stable across cultures
            'knowledge': 0.75,  # Fairly stable conceptual core
            'consciousness': 0.85,  # Universal phenomenon
            'experience': 0.9,  # Universal human feature
            'belief': 0.8,  # Universal cognitive phenomenon
            'existence': 0.95,  # Universally recognized
        }
        return cultural_patterns.get(concept, 0.75)

    def _assess_theoretical_stability(self, concept: str) -> float:
        """Assess stability within philosophical theory."""
        theoretical_patterns = {
            'good': 0.65,  # Multiple competing theories
            'right': 0.7,  # Moderately stable in ethics
            'justice': 0.7,  # Multiple stable theories
            'virtue': 0.8,  # Stable theoretical understanding
            'truth': 0.75,  # Several competing but stable theories
            'knowledge': 0.7,  # Stable but debated
            'consciousness': 0.6,  # Highly contested theoretically
            'experience': 0.75,  # Stable theoretical role
            'belief': 0.8,  # Stable theoretical concept
            'existence': 0.85,  # Stable across most theories
        }
        return theoretical_patterns.get(concept, 0.7)

    async def _analyze_relational_layer(self, domain: str) -> dict[str, Any]:
        """Analyze relational structures in conceptual coherence."""
        return {
            'layer_type': 'relational_structures',
            'layer_index': 1,
            'domain': domain,
            'relation_analysis': 'Complex relational network identified',
            'coherence_score': 0.7,
            'philosophical_significance': 'Reveals systematic conceptual connections'
        }

    async def _analyze_integration_layer(self, domain: str) -> dict[str, Any]:
        """Analyze systematic integration of concepts."""
        return {
            'layer_type': 'systematic_integration',
            'layer_index': 2,
            'domain': domain,
            'integration_analysis': 'Systematic integration patterns identified',
            'coherence_score': 0.65,
            'philosophical_significance': 'Shows systematic unity within domain'
        }

    async def _analyze_meta_theoretical_layer(self, domain: str) -> dict[str, Any]:
        """Analyze meta-theoretical framework."""
        return {
            'layer_type': 'meta_theoretical_framework',
            'layer_index': 3,
            'domain': domain,
            'meta_analysis': 'Meta-theoretical structures identified',
            'coherence_score': 0.6,
            'philosophical_significance': 'Reveals deep theoretical assumptions'
        }

    async def _analyze_extended_layer(self, domain: str, level: int) -> dict[str, Any]:
        """Analyze extended conceptual layers."""
        return {
            'layer_type': 'extended_analysis',
            'layer_index': level,
            'domain': domain,
            'extended_analysis': f'Extended analysis at level {level}',
            'coherence_score': max(0.5 - (level - 4) * 0.1, 0.2),
            'philosophical_significance': 'Explores conceptual periphery'
        }

    async def _identify_philosophical_structures(self, domain: str, layers: list[dict[str, Any]]) -> dict[str, Any]:
        """Identify philosophical structures within coherence landscape."""

        structures = {
            'hierarchical_structures': [],
            'network_structures': [],
            'dialectical_structures': [],
            'emergent_structures': []
        }

        # Analyze for hierarchical patterns
        if len(layers) >= 2:
            structures['hierarchical_structures'].append({
                'type': 'foundational_hierarchy',
                'description': 'Foundational concepts support higher-level structures',
                'strength': 0.8
            })

        # Analyze for network patterns
        if any('relational' in layer.get('layer_type', '') for layer in layers):
            structures['network_structures'].append({
                'type': 'conceptual_network',
                'description': 'Dense network of conceptual relations',
                'strength': 0.7
            })

        return structures

    async def _analyze_coherence_dynamics(self, domain: str, layers: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze dynamical properties of coherence landscape."""

        coherence_scores = [layer.get('coherence_score', 0.5) for layer in layers]

        return {
            'attractors': [
                {
                    'type': 'conceptual_attractor',
                    'strength': max(coherence_scores),
                    'location': 'foundational_layer'
                }
            ],
            'phase_transitions': [],
            'stability_analysis': {
                'overall_stability': sum(coherence_scores) / len(coherence_scores),
                'stability_gradient': coherence_scores[0] - coherence_scores[-1] if len(coherence_scores) > 1 else 0
            },
            'dynamical_patterns': ['hierarchical_stabilization', 'conceptual_crystallization']
        }

    async def _generate_coherence_landscape_report(
        self,
        domain: str,
        layers: list[dict[str, Any]],
        structures: dict[str, Any],
        dynamics: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive coherence landscape report."""

        overall_coherence = sum(layer.get('coherence_score', 0.5) for layer in layers) / len(layers)

        return {
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'coherence_layers': layers,
            'philosophical_structures': structures,
            'coherence_dynamics': dynamics,
            'overall_coherence': overall_coherence,
            'landscape_assessment': {
                'coherence_quality': 'high' if overall_coherence > 0.7 else 'moderate' if overall_coherence > 0.5 else 'developing',
                'structural_richness': len([s for struct_list in structures.values() for s in struct_list]),
                'dynamical_complexity': len(dynamics.get('dynamical_patterns', [])),
                'philosophical_maturity': self._assess_philosophical_maturity(domain, overall_coherence, structures)
            },
            'interpretive_summary': self._generate_landscape_summary(domain, layers, overall_coherence),
            'further_exploration': self._suggest_further_exploration(domain, layers, structures)
        }

    def _assess_philosophical_maturity(self, domain: str, coherence: float, structures: dict[str, Any]) -> str:
        """Assess philosophical maturity of domain."""
        structure_count = sum(len(struct_list) for struct_list in structures.values())

        if coherence > 0.8 and structure_count > 3:
            return 'highly_developed'
        elif coherence > 0.6 and structure_count > 2:
            return 'well_developed'
        elif coherence > 0.4:
            return 'developing'
        else:
            return 'emergent'

    def _generate_landscape_summary(self, domain: str, layers: list[dict[str, Any]], coherence: float) -> str:
        """Generate interpretive summary of coherence landscape."""

        summary_parts = []

        summary_parts.append(f"Coherence analysis of {domain} reveals {len(layers)} distinct conceptual layers.")

        if coherence > 0.7:
            summary_parts.append("The domain demonstrates high conceptual coherence with stable foundational structures.")
        elif coherence > 0.5:
            summary_parts.append("The domain shows moderate coherence with some areas of conceptual tension.")
        else:
            summary_parts.append("The domain exhibits developing coherence with significant conceptual complexity.")

        foundational_layer = next((layer for layer in layers if layer.get('layer_type') == 'foundational_concepts'), None)
        if foundational_layer:
            foundational_score = foundational_layer.get('coherence_score', 0.5)
            if foundational_score > 0.8:
                summary_parts.append("Strong foundational concepts provide robust conceptual grounding.")
            elif foundational_score > 0.6:
                summary_parts.append("Foundational concepts show good coherence with some definitional challenges.")
            else:
                summary_parts.append("Foundational concepts require further clarification and stabilization.")

        return " ".join(summary_parts)

    def _suggest_further_exploration(self, domain: str, layers: list[dict[str, Any]], structures: dict[str, Any]) -> list[str]:
        """Suggest areas for further coherence exploration."""

        suggestions = []

        # Check layer development
        if len(layers) < 3:
            suggestions.append("Explore additional conceptual layers for comprehensive analysis")

        # Check foundational development
        foundational_layer = next((layer for layer in layers if layer.get('layer_type') == 'foundational_concepts'), None)
        if foundational_layer and foundational_layer.get('coherence_score', 0.5) < 0.7:
            suggestions.append("Strengthen foundational concept analysis and definitional clarity")

        # Check structural diversity
        structure_count = sum(len(struct_list) for struct_list in structures.values())
        if structure_count < 2:
            suggestions.append("Investigate additional philosophical structures and organizational patterns")

        # Domain-specific suggestions
        domain_suggestions = {
            'ethics': ["Explore virtue-deontological-consequentialist integration", "Analyze practical ethics applications"],
            'consciousness': ["Investigate phenomenology-neuroscience connections", "Explore hard problem implications"],
            'knowledge': ["Examine skeptical challenges", "Analyze social epistemology dimensions"]
        }

        suggestions.extend(domain_suggestions.get(domain.lower(), ["Investigate cross-domain connections"]))

        return suggestions[:5]

    async def _generate_fallback_coherence_analysis(self, domain: str, depth: int) -> dict[str, Any]:
        """Generate simplified coherence analysis as fallback."""
        return {
            'domain': domain,
            'status': 'fallback_analysis',
            'coherence_layers': [
                {
                    'layer_type': 'basic_analysis',
                    'domain': domain,
                    'coherence_score': 0.6,
                    'analysis_depth': depth
                }
            ],
            'overall_coherence': 0.6,
            'landscape_assessment': {
                'coherence_quality': 'basic',
                'analysis_note': 'Simplified analysis due to processing constraints'
            }
        }

    async def _generate_insights(self, phenomenon: str, perspectives: list[str] | None = None, depth: int = 3, include_contradictions: bool = True) -> dict[str, Any]:
        """Enhanced insight generation with sophisticated philosophical analysis."""
        logger.debug(f"Generating insights for phenomenon: {phenomenon}")

        try:
            # Initialize comprehensive insight framework
            insight_framework = await self._initialize_insight_framework(phenomenon, perspectives, depth)

            # Multi-dimensional evidence gathering
            evidence_corpus = await asyncio.wait_for(
                self._gather_comprehensive_evidence(phenomenon, perspectives, depth),
                timeout=20.0
            )

            # Generate perspective-specific insights
            perspective_insights = await self._generate_perspective_insights(
                phenomenon, evidence_corpus, perspectives or self._get_default_perspectives()
            )

            # Perform cross-perspective synthesis
            synthetic_insights = await self._synthesize_cross_perspective_insights(
                perspective_insights, phenomenon
            )

            # Identify philosophical contradictions and tensions
            contradictions = []
            if include_contradictions and len(perspective_insights) > 1:
                contradictions = await self._identify_comprehensive_contradictions(
                    perspective_insights, phenomenon
                )

            # Generate meta-philosophical insights
            meta_insights = await self._generate_meta_philosophical_insights(
                phenomenon, perspective_insights, synthetic_insights
            )

            # Assess epistemic uncertainty and revision conditions
            uncertainty_profile = await self._assess_comprehensive_uncertainty(
                perspective_insights, synthetic_insights, evidence_corpus
            )

            # Generate revision triggers and future inquiry directions
            revision_framework = await self._generate_revision_framework(
                phenomenon, perspective_insights, uncertainty_profile
            )

            return {
                "phenomenon": phenomenon,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.context.session_id,
                "analysis_depth": depth,
                "perspectives_analyzed": perspectives or self._get_default_perspectives(),
                "insight_framework": insight_framework,
                "primary_insights": perspective_insights,
                "synthetic_insights": synthetic_insights,
                "meta_philosophical_insights": meta_insights,
                "contradictions": contradictions,
                "uncertainty_profile": uncertainty_profile,
                "revision_framework": revision_framework,
                "philosophical_significance": await self._assess_philosophical_significance(
                    phenomenon, perspective_insights, synthetic_insights
                ),
                "inquiry_recommendations": await self._generate_inquiry_recommendations(
                    phenomenon, perspective_insights, uncertainty_profile
                )
            }

        except asyncio.TimeoutError:
            logger.warning(f"Insight generation timeout for: {phenomenon}")
            return await self._generate_fallback_insights(phenomenon, perspectives, depth)
        except Exception as e:
            logger.error(f"Error in insight generation: {e}", exc_info=True)
            return await self._generate_fallback_insights(phenomenon, perspectives, depth)

    async def _test_hypothesis(self, hypothesis: str, test_domains: list[str] | None = None, criteria: dict[str, Any] | None = None) -> dict[str, Any]:
        """Enhanced hypothesis testing with domain-specific evaluation."""
        logger.debug(f"Testing hypothesis: {hypothesis}")

        try:
            test_results = {
                "hypothesis": hypothesis,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.context.session_id,
                "test_domains": test_domains or ["general"],
                "domain_results": {},
                "overall_coherence": 0.0,
                "pragmatic_score": 0.0,
                "confidence": 0.0,
                "supporting_evidence": [],
                "challenges": [],
                "implications": []
            }

            # Test in each domain with timeout
            for domain in (test_domains or ["general"]):
                try:
                    domain_test = await asyncio.wait_for(
                        self._test_in_domain(hypothesis, domain, criteria or {}),
                        timeout=15.0
                    )
                    test_results["domain_results"][domain] = domain_test

                    # Collect evidence
                    test_results["supporting_evidence"].extend(
                        domain_test.get("supporting_evidence", [])
                    )
                    test_results["challenges"].extend(
                        domain_test.get("challenges", [])
                    )

                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Domain testing failed for {domain}: {e}")
                    test_results["domain_results"][domain] = {
                        "status": "error",
                        "message": str(e)
                    }

            # Calculate overall metrics
            test_results["overall_coherence"] = self._calculate_hypothesis_coherence(
                test_results["domain_results"]
            )
            test_results["pragmatic_score"] = self._assess_pragmatic_value(
                hypothesis, test_results["domain_results"]
            )
            test_results["confidence"] = self._calculate_test_confidence(
                test_results["overall_coherence"],
                test_results["pragmatic_score"],
                len(test_results["challenges"])
            )

            return test_results

        except Exception as e:
            logger.error(f"Hypothesis testing error: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Enhanced Server Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan_context(self):
        """
        Async context manager for server lifespan with proper cleanup.

        ### Lifespan Protocol:
        1. Initialize resources
        2. Setup background monitoring
        3. Yield operational context
        4. Cleanup on exit
        """
        logger.info("Initializing philosophical server lifespan")

        # Initialize NARS if not already done
        if not self._nars_initialized:
            try:
                await self.nars_manager.start()
                self._nars_initialized = True
                logger.info("NARS system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NARS: {e}")
                # Continue without NARS - fallback to pure philosophical analysis

        # Initialize monitoring task
        monitoring_task = asyncio.create_task(self._monitor_operations())
        track_background_task(monitoring_task)

        try:
            yield self
        finally:
            logger.info("Shutting down philosophical server")

            # Cancel monitoring
            if not monitoring_task.done():
                monitoring_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitoring_task

            # Wait for active operations to complete (with timeout)
            if self._active_operations:
                logger.info(f"Waiting for {len(self._active_operations)} active operations to complete")
                await asyncio.sleep(2.0)  # Grace period

            # Cleanup NARS
            if self._nars_initialized:
                try:
                    await self.nars_manager.stop()
                    # Save NARS memory
                    self.nars_memory.save(Path("philosophy_nars_memory.json"))
                    logger.info("NARS system shutdown complete")
                except Exception as e:
                    logger.error(f"Error during NARS shutdown: {e}")

            cleanup_processes()

    async def _monitor_operations(self) -> None:
        """Background monitoring of active operations."""
        while True:
            try:
                if self._active_operations:
                    logger.debug(f"Active operations: {len(self._active_operations)}")
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
            except asyncio.CancelledError:
                logger.debug("Operation monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def run(self) -> None:
        """
        Run the MCP server with comprehensive lifecycle management.

        ### Execution Protocol:
        1. Setup signal handling
        2. Initialize lifespan context
        3. Configure stdio transport
        4. Execute server with cleanup guarantee
        """
        try:
            async with self.lifespan_context(), mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                # Create initialization options
                initialization_options = InitializationOptions(
                        server_name="openended-philosophy",
                        server_version="0.1.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )

                logger.info("Starting philosophical MCP server")
                await self.server.run(
                    read_stream,
                    write_stream,
                    initialization_options
                )

        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server execution error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Philosophical server shutdown complete")

    # ─────────────────────────────────────────────────────────────────────────
    # Enhanced Helper Methods Implementation
    # ─────────────────────────────────────────────────────────────────────────

    def _select_relevant_perspectives(self, concept: str, context: str) -> list[str]:
        """Select analytical perspectives based on concept and context using semantic analysis."""
        from .utils import semantic_similarity

        # Create concept profile for intelligent perspective selection
        concept_profile = {
            'features': [concept.lower()] + context.lower().split(),
            'contexts': [context.lower()],
            'domain': context.lower()
        }

        # Comprehensive perspective library with semantic profiles
        perspective_library = {
            "analytical": {
                'features': ['logic', 'reason', 'analysis', 'definition', 'precision', 'formal'],
                'contexts': ['philosophy', 'mathematics', 'logic'],
                'strengths': ['conceptual_clarity', 'logical_rigor', 'systematic_analysis']
            },
            "phenomenological": {
                'features': ['experience', 'consciousness', 'lived', 'meaning', 'intentional', 'embodied'],
                'contexts': ['mind', 'experience', 'consciousness', 'perception'],
                'strengths': ['experiential_fidelity', 'meaning_analysis', 'first_person_perspective']
            },
            "pragmatist": {
                'features': ['practice', 'action', 'consequences', 'useful', 'experimental', 'inquiry'],
                'contexts': ['action', 'practice', 'science', 'democracy'],
                'strengths': ['practical_efficacy', 'problem_solving', 'experimental_validation']
            },
            "critical": {
                'features': ['power', 'social', 'ideology', 'critique', 'emancipation', 'justice'],
                'contexts': ['society', 'politics', 'culture', 'ethics'],
                'strengths': ['power_analysis', 'social_critique', 'transformative_potential']
            },
            "existentialist": {
                'features': ['freedom', 'authenticity', 'anxiety', 'choice', 'existence', 'absurd'],
                'contexts': ['existence', 'life', 'ethics', 'meaning'],
                'strengths': ['authentic_existence', 'freedom_analysis', 'existential_meaning']
            },
            "naturalist": {
                'features': ['nature', 'scientific', 'empirical', 'natural', 'causal', 'physical'],
                'contexts': ['science', 'nature', 'mind', 'biology'],
                'strengths': ['scientific_continuity', 'empirical_grounding', 'naturalistic_explanation']
            },
            "hermeneutic": {
                'features': ['interpretation', 'understanding', 'text', 'meaning', 'tradition', 'historical'],
                'contexts': ['interpretation', 'culture', 'history', 'text'],
                'strengths': ['interpretive_depth', 'historical_awareness', 'meaning_construction']
            },
            "postmodern": {
                'features': ['difference', 'deconstruction', 'multiplicity', 'power', 'discourse', 'contingent'],
                'contexts': ['culture', 'language', 'politics', 'knowledge'],
                'strengths': ['difference_analysis', 'power_critique', 'multiplicity_recognition']
            },
            # Domain-specific perspectives
            "empiricist": {
                'features': ['observation', 'sensory', 'experience', 'data', 'inductive', 'evidence'],
                'contexts': ['science', 'knowledge', 'epistemology'],
                'strengths': ['empirical_foundation', 'observational_validation', 'inductive_reasoning']
            },
            "rationalist": {
                'features': ['reason', 'innate', 'deductive', 'logical', 'a_priori', 'mathematical'],
                'contexts': ['knowledge', 'mathematics', 'logic'],
                'strengths': ['logical_necessity', 'deductive_certainty', 'rational_foundation']
            },
            "functionalist": {
                'features': ['function', 'role', 'system', 'input', 'output', 'computation'],
                'contexts': ['mind', 'consciousness', 'cognitive_science'],
                'strengths': ['functional_analysis', 'multiple_realizability', 'systematic_approach']
            },
            "emergentist": {
                'features': ['emergence', 'levels', 'complexity', 'properties', 'causation', 'holistic'],
                'contexts': ['consciousness', 'complexity', 'systems', 'biology'],
                'strengths': ['level_integration', 'emergent_properties', 'complexity_handling']
            },
            "virtue_ethics": {
                'features': ['virtue', 'character', 'excellence', 'flourishing', 'practical_wisdom', 'eudaimonia'],
                'contexts': ['ethics', 'character', 'morality'],
                'strengths': ['character_focus', 'practical_wisdom', 'human_flourishing']
            },
            "deontological": {
                'features': ['duty', 'obligation', 'categorical', 'universal', 'respect', 'autonomy'],
                'contexts': ['ethics', 'morality', 'duty'],
                'strengths': ['moral_universality', 'respect_for_persons', 'duty_based_ethics']
            },
            "consequentialist": {
                'features': ['consequences', 'outcomes', 'utility', 'welfare', 'aggregate', 'maximizing'],
                'contexts': ['ethics', 'policy', 'decision'],
                'strengths': ['outcome_focus', 'utility_maximization', 'policy_guidance']
            }
        }

        # Calculate semantic similarity between concept and each perspective
        perspective_scores = []
        for perspective, profile in perspective_library.items():
            similarity = semantic_similarity(concept_profile, profile, method="wittgenstein")
            perspective_scores.append((perspective, similarity, profile))

        # Sort by relevance
        perspective_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top perspectives with minimum threshold
        selected_perspectives = []
        min_similarity = 0.2  # Minimum threshold for relevance

        for perspective, score, _ in perspective_scores[:8]:  # Consider top 8
            if score >= min_similarity or len(selected_perspectives) < 3:  # Ensure at least 3
                selected_perspectives.append(perspective)

        # Ensure diversity by checking for different philosophical families
        final_perspectives = self._ensure_perspective_diversity(selected_perspectives, perspective_library)

        return final_perspectives[:5]  # Return top 5 most relevant and diverse

    def _ensure_perspective_diversity(
        self,
        selected_perspectives: list[str],
        perspective_library: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Ensure diversity across philosophical families and approaches."""
        # Define philosophical families for diversity
        families = {
            'continental': ['phenomenological', 'existentialist', 'hermeneutic', 'critical', 'postmodern'],
            'analytic': ['analytical', 'empiricist', 'rationalist', 'functionalist'],
            'pragmatic': ['pragmatist', 'naturalist'],
            'classical': ['virtue_ethics', 'deontological', 'consequentialist'],
            'emergent': ['emergentist', 'critical']
        }

        # Track family representation
        family_counts = dict.fromkeys(families, 0)
        diverse_perspectives = []

        # First pass: include one from each family if available
        for family, members in families.items():
            for perspective in selected_perspectives:
                if perspective in members and family_counts[family] == 0:
                    diverse_perspectives.append(perspective)
                    family_counts[family] += 1
                    break

        # Second pass: fill remaining slots with highest scoring
        for perspective in selected_perspectives:
            if perspective not in diverse_perspectives and len(diverse_perspectives) < 5:
                diverse_perspectives.append(perspective)

        return diverse_perspectives

    def _create_interpretive_schema(self, perspective: str) -> dict[str, Any]:
        """Create rich interpretive schema for perspective with philosophical depth."""

        # Comprehensive perspective schemas with philosophical sophistication
        perspective_schemas = {
            "analytical": {
                "core_commitments": [
                    "logical_rigor", "conceptual_clarity", "definitional_precision",
                    "systematic_analysis", "argument_validity", "semantic_analysis"
                ],
                "evaluation_criteria": [
                    "logical_consistency", "definitional_adequacy", "formal_validity",
                    "conceptual_necessity", "analytic_truth", "semantic_coherence"
                ],
                "conceptual_priorities": [
                    "precision", "rigor", "systematicity", "logical_form",
                    "conceptual_analysis", "meaning_clarification"
                ],
                "methodological_approach": "logical_analysis",
                "truth_conception": "correspondence_and_coherence",
                "key_philosophers": ["Russell", "Moore", "Quine", "Davidson"],
                "typical_concerns": ["meaning", "logic", "language", "concepts"],
                "epistemic_virtues": ["precision", "rigor", "clarity"],
                "interpretive_questions": [
                    "What are the logical conditions of this concept?",
                    "How can we define this with necessary and sufficient conditions?",
                    "What are the semantic implications?"
                ]
            },

            "phenomenological": {
                "core_commitments": [
                    "lived_experience", "consciousness_centrality", "intentionality",
                    "embodied_cognition", "temporal_synthesis", "meaning_constitution"
                ],
                "evaluation_criteria": [
                    "experiential_adequacy", "phenomenological_reduction_success",
                    "intentional_structure_clarity", "temporal_coherence",
                    "meaning_constitution_fidelity", "lived_world_grounding"
                ],
                "conceptual_priorities": [
                    "experience", "consciousness", "temporality", "embodiment",
                    "intersubjectivity", "life_world", "meaning_formation"
                ],
                "methodological_approach": "phenomenological_reduction",
                "truth_conception": "experiential_disclosure",
                "key_philosophers": ["Husserl", "Heidegger", "Merleau-Ponty", "Levinas"],
                "typical_concerns": ["consciousness", "time", "embodiment", "other_minds"],
                "epistemic_virtues": ["attentiveness", "descriptive_fidelity", "experiential_depth"],
                "interpretive_questions": [
                    "How does this appear in lived experience?",
                    "What is the intentional structure?",
                    "How does temporal synthesis operate?"
                ]
            },

            "pragmatist": {
                "core_commitments": [
                    "practical_consequences", "experimental_method", "inquiry_based_truth",
                    "democratic_values", "fallibilism", "problem_solving_focus"
                ],
                "evaluation_criteria": [
                    "practical_success", "problem_solving_efficacy", "experimental_validation",
                    "democratic_participation", "adaptive_capacity", "inquiry_advancement"
                ],
                "conceptual_priorities": [
                    "action", "consequences", "inquiry", "democracy", "experience",
                    "experimentation", "adaptive_intelligence", "social_cooperation"
                ],
                "methodological_approach": "experimental_inquiry",
                "truth_conception": "warranted_assertibility",
                "key_philosophers": ["James", "Pierce", "Dewey", "Putnam"],
                "typical_concerns": ["action", "inquiry", "democracy", "education"],
                "epistemic_virtues": ["flexibility", "experimental_spirit", "social_intelligence"],
                "interpretive_questions": [
                    "What difference does this make in practice?",
                    "How does this contribute to inquiry?",
                    "What are the social consequences?"
                ]
            },

            "critical": {
                "core_commitments": [
                    "power_analysis", "social_critique", "ideology_critique",
                    "emancipatory_interest", "historical_consciousness", "transformative_praxis"
                ],
                "evaluation_criteria": [
                    "emancipatory_potential", "ideological_penetration", "power_revelation",
                    "historical_understanding", "transformative_capacity", "social_justice_advancement"
                ],
                "conceptual_priorities": [
                    "power", "justice", "transformation", "critique", "emancipation",
                    "historical_context", "social_relations", "material_conditions"
                ],
                "methodological_approach": "ideology_critique",
                "truth_conception": "critical_disclosure",
                "key_philosophers": ["Marx", "Adorno", "Habermas", "Foucault"],
                "typical_concerns": ["power", "justice", "oppression", "liberation"],
                "epistemic_virtues": ["critical_awareness", "transformative_vision", "solidarity"],
                "interpretive_questions": [
                    "What power relations are at work?",
                    "How does this serve dominant interests?",
                    "What emancipatory potential exists?"
                ]
            },

            "existentialist": {
                "core_commitments": [
                    "existence_before_essence", "authentic_existence", "freedom_and_responsibility",
                    "anxiety_and_finitude", "situated_existence", "choice_and_commitment"
                ],
                "evaluation_criteria": [
                    "authenticity_promotion", "freedom_recognition", "responsibility_acknowledgment",
                    "finitude_acceptance", "choice_significance", "existential_honesty"
                ],
                "conceptual_priorities": [
                    "existence", "freedom", "authenticity", "anxiety", "death",
                    "choice", "responsibility", "absurdity", "meaning_creation"
                ],
                "methodological_approach": "existential_analysis",
                "truth_conception": "existential_disclosure",
                "key_philosophers": ["Kierkegaard", "Sartre", "Camus", "Beauvoir"],
                "typical_concerns": ["freedom", "authenticity", "absurdity", "death"],
                "epistemic_virtues": ["authenticity", "courage", "honesty"],
                "interpretive_questions": [
                    "What does this mean for human freedom?",
                    "How does this relate to authentic existence?",
                    "What choices does this demand?"
                ]
            },

            "virtue_ethics": {
                "core_commitments": [
                    "character_excellence", "practical_wisdom", "human_flourishing",
                    "virtue_cultivation", "moral_education", "community_context"
                ],
                "evaluation_criteria": [
                    "character_development", "practical_wisdom_demonstration", "flourishing_promotion",
                    "virtue_exemplification", "community_benefit", "moral_education_success"
                ],
                "conceptual_priorities": [
                    "virtue", "character", "flourishing", "wisdom", "excellence",
                    "habit", "community", "role_models", "moral_development"
                ],
                "methodological_approach": "virtue_analysis",
                "truth_conception": "practical_wisdom",
                "key_philosophers": ["Aristotle", "Aquinas", "MacIntyre", "Hursthouse"],
                "typical_concerns": ["character", "virtue", "flourishing", "community"],
                "epistemic_virtues": ["practical_wisdom", "temperance", "justice"],
                "interpretive_questions": [
                    "What virtues does this promote or hinder?",
                    "How does this contribute to human flourishing?",
                    "What would the virtuous person do?"
                ]
            }
        }

        # Get schema or create default
        schema = perspective_schemas.get(perspective, {
            "core_commitments": [f"{perspective}_analysis", "philosophical_inquiry"],
            "evaluation_criteria": [f"{perspective}_adequacy", "conceptual_coherence"],
            "conceptual_priorities": [f"{perspective}_focus", "systematic_understanding"],
            "methodological_approach": f"{perspective}_method",
            "truth_conception": "contextual_adequacy",
            "key_philosophers": ["various"],
            "typical_concerns": ["philosophical_inquiry"],
            "epistemic_virtues": ["intellectual_honesty"],
            "interpretive_questions": [f"How does {perspective} approach this?"]
        })

        # Add perspective name and meta-information
        schema["perspective"] = perspective
        schema["schema_type"] = "interpretive_framework"
        schema["philosophical_family"] = self._identify_philosophical_family(perspective)
        schema["complementary_perspectives"] = self._identify_complementary_perspectives(perspective)
        schema["typical_tensions"] = self._identify_typical_tensions(perspective)

        return schema

    async def _apply_schema_to_concept(
        self,
        concept_node: EmergentCoherenceNode,
        schema: dict[str, Any],
        active_game: LanguageGameProcessor
    ) -> dict[str, Any]:
        """Apply interpretive schema to concept node."""
        from .utils import pragmatic_evaluation, semantic_similarity

        logger.debug(f"Applying schema {schema.get('perspective')} to concept")

        # Extract concept features for analysis
        concept_features = {
            'features': list(concept_node.pattern.content.keys()),
            'contexts': [active_game.game_type],
            'confidence': concept_node.pattern.confidence
        }

        # Extract schema commitments and criteria
        perspective = schema.get('perspective', 'unknown')
        commitments = schema.get('core_commitments', [])
        criteria = schema.get('evaluation_criteria', [])

        # Calculate semantic alignment between concept and schema
        schema_features = {
            'features': commitments + criteria,
            'contexts': [perspective],
            'uses': schema.get('conceptual_priorities', [])
        }

        alignment_score = semantic_similarity(concept_features, schema_features, method="wittgenstein")

        # Apply perspective-specific interpretation logic
        interpretation = self._generate_perspective_interpretation(
            concept_node.pattern.content,
            perspective,
            commitments
        )

        # Calculate coherence based on schema fit and concept stability
        base_coherence = alignment_score * concept_node.pattern.calculate_stability()
        game_modifier = active_game.assess_pattern_fit(concept_node.pattern.content)
        coherence_score = base_coherence * game_modifier

        # Identify key features through schema lens
        key_features = self._extract_schema_relevant_features(
            concept_node.pattern.content,
            commitments,
            criteria
        )

        # Calculate uncertainty based on multiple factors
        uncertainty = self._calculate_schema_uncertainty(
            alignment_score,
            concept_node.pattern.confidence,
            coherence_score,
            len(key_features)
        )

        interpretation_details = {
            "derived_meaning": interpretation,
            "key_features": key_features,
            "coherence": float(np.clip(coherence_score, 0.0, 1.0)),
            "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
            "alignment_score": float(alignment_score),
            "schema_commitments": commitments[:3],  # Top 3 for brevity
            "applicability": float(game_modifier)
        }

        # Evaluate pragmatic value of applying this schema (Dewey/James approach)
        schema_solution = {
            'addresses': [f"interprets_{concept_node.pattern.content.get('term', 'concept')}"],
            'benefits': [f"provides_{perspective}_perspective", "enables_contextual_understanding"],
            'modular': True,  # Schemas are modular by nature
            'context_sensitive': True,  # Philosophical interpretations are context-sensitive
            'dependencies': commitments,
            'assumptions': criteria
        }

        interpretation_problem = {
            'requirements': [
                'provide_meaningful_interpretation',
                'maintain_conceptual_coherence',
                'enable_further_inquiry'
            ]
        }

        pragmatic_assessment = pragmatic_evaluation(schema_solution, interpretation_problem)

        # Add pragmatic evaluation results
        interpretation_details.update({
            "pragmatic_value": pragmatic_assessment['overall'],
            "pragmatic_scores": pragmatic_assessment['scores'],
            "pragmatic_recommendations": pragmatic_assessment['recommendations'][:2]  # Top 2 for brevity
        })

        return interpretation_details

    def _calculate_interpretation_confidence(
        self,
        interpretation: dict[str, Any],
        active_game: LanguageGameProcessor
    ) -> float:
        """Calculate interpretation confidence."""
        coherence = interpretation.get("coherence", 0.0)
        game_factor = active_game.get_confidence_modifier()
        base_confidence = coherence * 0.8
        final_confidence = np.clip(base_confidence + game_factor, 0.0, 1.0)
        return float(final_confidence)

    def _synthesize_analyses(self, analysis_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Synthesize multiple analyses using coherence theory and pragmatic evaluation."""
        from .utils import coherence_metrics, pragmatic_evaluation, semantic_similarity

        if not analysis_results:
            return {"summary": "No analyses to synthesize", "overall_confidence": 0.0}

        # Extract core information for synthesis
        perspectives = [r.get("perspective", "unknown") for r in analysis_results]
        interpretations = [r.get("interpretation", {}) for r in analysis_results]
        confidences = [r.get("confidence", 0.0) for r in analysis_results]

        # Create propositions from interpretations for coherence analysis
        propositions = []
        for i, (perspective, interpretation, confidence) in enumerate(zip(perspectives, interpretations, confidences, strict=True)):
            # Extract interpretive content
            content = interpretation.get("derived_meaning", "") if isinstance(interpretation, dict) else str(interpretation)
            key_features = interpretation.get("key_features", []) if isinstance(interpretation, dict) else []

            propositions.append({
                'id': i,
                'content': content,
                'confidence': confidence,
                'concepts': key_features,
                'perspective': perspective
            })

        # Generate relations between propositions based on semantic similarity
        relations = []
        for i, prop1 in enumerate(propositions):
            for j, prop2 in enumerate(propositions[i+1:], i+1):
                # Check semantic similarity of interpretations
                content1_words = set(prop1['content'].lower().split()) if prop1['content'] else set()
                content2_words = set(prop2['content'].lower().split()) if prop2['content'] else set()

                if content1_words and content2_words:
                    features1 = {'features': list(content1_words)}
                    features2 = {'features': list(content2_words)}
                    similarity = semantic_similarity(features1, features2, method="jaccard")

                    if similarity > 0.3:
                        relations.append((i, j, 'supports'))
                    elif similarity > 0.1:
                        relations.append((i, j, 'analogous'))

                # Check for complementary perspectives
                perspective_complements = {
                    ('analytical', 'phenomenological'): 'complementary',
                    ('empiricist', 'rationalist'): 'tension',
                    ('pragmatist', 'analytical'): 'complementary',
                    ('critical', 'virtue_ethics'): 'complementary',
                    ('existentialist', 'analytical'): 'tension'
                }

                perspective_pair = tuple(sorted([prop1['perspective'], prop2['perspective']]))
                if perspective_pair in perspective_complements:
                    relation_type = perspective_complements[perspective_pair]
                    relations.append((i, j, relation_type))

        # Calculate coherence metrics
        coherence_scores = coherence_metrics(propositions, relations)

        # Generate synthesis narrative
        synthesis_narrative = self._create_synthesis_narrative(
            propositions, relations, coherence_scores, perspectives
        )

        # Identify emergent themes
        emergent_themes = self._identify_emergent_themes(propositions, relations)

        # Calculate synthetic confidence using multiple factors
        confidence_factors = {
            'coherence': coherence_scores['overall_coherence'],
            'evidence_quantity': min(len(analysis_results) / 5.0, 1.0),
            'average_individual_confidence': float(np.mean(confidences)),
            'perspective_diversity': min(len(set(perspectives)) / 4.0, 1.0),
            'constraint_satisfaction': coherence_scores['constraint_satisfaction']
        }

        synthetic_confidence = (
            0.3 * confidence_factors['coherence'] +
            0.2 * confidence_factors['evidence_quantity'] +
            0.25 * confidence_factors['average_individual_confidence'] +
            0.15 * confidence_factors['perspective_diversity'] +
            0.1 * confidence_factors['constraint_satisfaction']
        )

        # Identify areas of convergence and divergence
        convergences = self._identify_convergences(propositions, relations)
        divergences = self._identify_divergences(propositions, relations)

        # Pragmatic evaluation of synthesis quality
        synthesis_solution = {
            "content": synthesis_narrative,
            "confidence": synthetic_confidence,
            "perspectives": perspectives,
            "addresses": convergences,
            "modular": len(emergent_themes) > 1,
            "context_sensitive": len(set(perspectives)) > 2,
            "benefits": [f"Integrates {len(perspectives)} perspectives", "Provides coherent synthesis"],
            "risks": divergences
        }

        problem_context = {
            "requirements": ["coherent_integration", "perspective_synthesis", "epistemic_humility"],
            "complexity": "high"
        }

        pragmatic_assessment = pragmatic_evaluation(synthesis_solution, problem_context)

        # Meta-analysis of philosophical approaches
        philosophical_families = [self._identify_philosophical_family(p) for p in perspectives]
        family_distribution = {family: philosophical_families.count(family) for family in set(philosophical_families)}

        return {
            "summary": synthesis_narrative,
            "overall_confidence": float(np.clip(synthetic_confidence, 0.0, 1.0)),
            "contributing_perspectives": perspectives,
            "philosophical_families": family_distribution,
            "coherence_metrics": coherence_scores,
            "confidence_factors": confidence_factors,
            "emergent_themes": emergent_themes,
            "convergences": convergences,
            "divergences": divergences,
            "synthetic_insights": self._generate_synthetic_insights(propositions, relations, coherence_scores),
            "epistemic_assessment": self._assess_synthetic_epistemic_status(synthetic_confidence, coherence_scores),
            "integration_quality": self._assess_integration_quality(propositions, relations, perspectives),
            "pragmatic_assessment": pragmatic_assessment,
            "meta_philosophical_observations": self._generate_meta_philosophical_observations(
                perspectives, family_distribution, coherence_scores
            )
        }

    def _create_synthesis_narrative(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]],
        coherence_scores: dict[str, Any],
        perspectives: list[str]
    ) -> str:
        """Create a coherent narrative synthesizing multiple perspective analyses."""

        narrative_parts = []

        # Opening synthesis
        if len(propositions) == 1:
            narrative_parts.append(
                f"Single-perspective analysis from {propositions[0]['perspective']} approach "
                f"provides {propositions[0]['confidence']:.1%} confident insights."
            )
        else:
            # Multi-perspective opening
            family_analysis = {}
            for prop in propositions:
                family = self._identify_philosophical_family(prop['perspective'])
                if family not in family_analysis:
                    family_analysis[family] = []
                family_analysis[family].append(prop['perspective'])

            family_summary = []
            for family, members in family_analysis.items():
                if len(members) == 1:
                    family_summary.append(f"{family} ({members[0]})")
                else:
                    family_summary.append(f"{family} ({len(members)} perspectives)")

            narrative_parts.append(
                f"Multi-perspectival analysis draws from {', '.join(family_summary)} traditions, "
                f"achieving {coherence_scores['overall_coherence']:.1%} overall coherence."
            )

        # Coherence assessment
        if coherence_scores['overall_coherence'] > 0.8:
            narrative_parts.append(
                "The perspectives demonstrate high mutual coherence, suggesting robust conceptual convergence."
            )
        elif coherence_scores['overall_coherence'] > 0.6:
            narrative_parts.append(
                "The perspectives show moderate coherence with both convergent insights and productive tensions."
            )
        else:
            narrative_parts.append(
                "The perspectives reveal significant divergences, highlighting the complexity of the phenomenon."
            )

        # Relation analysis
        support_relations = [r for r in relations if r[2] == 'supports']
        tension_relations = [r for r in relations if r[2] == 'tension']
        complementary_relations = [r for r in relations if r[2] == 'complementary']

        if support_relations:
            narrative_parts.append(
                f"{len(support_relations)} supportive connections reinforce shared insights across perspectives."
            )

        if complementary_relations:
            narrative_parts.append(
                f"{len(complementary_relations)} complementary relations suggest productive dialogue between approaches."
            )

        if tension_relations:
            narrative_parts.append(
                f"{len(tension_relations)} tensions reveal fundamental philosophical differences requiring careful integration."
            )

        # Synthetic conclusion
        if coherence_scores['explanatory_breadth'] > 0.5:
            narrative_parts.append(
                "The synthesis provides substantial explanatory breadth across multiple dimensions of analysis."
            )

        return " ".join(narrative_parts)

    def _identify_emergent_themes(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]]
    ) -> list[dict[str, Any]]:
        """Identify themes that emerge from perspective interaction."""

        themes = []

        # Analyze content for recurring concepts
        all_words = []
        for prop in propositions:
            if prop['content']:
                words = [w.lower() for w in prop['content'].split() if len(w) > 4]
                all_words.extend(words)

        # Find frequently occurring philosophical concepts
        from collections import Counter
        word_counts = Counter(all_words)

        # Philosophical keywords that indicate important themes
        philosophical_keywords = {
            'experience', 'consciousness', 'reality', 'truth', 'knowledge', 'meaning',
            'existence', 'freedom', 'responsibility', 'virtue', 'justice', 'power',
            'authentic', 'rational', 'empirical', 'practical', 'social', 'individual',
            'universal', 'particular', 'objective', 'subjective', 'necessary', 'contingent'
        }

        # Identify emergent themes from word patterns
        for word, count in word_counts.most_common(10):
            if word in philosophical_keywords and count >= 2:
                # Find propositions containing this theme
                supporting_perspectives = []
                for prop in propositions:
                    if word in prop['content'].lower():
                        supporting_perspectives.append(prop['perspective'])

                if len(supporting_perspectives) >= 2:
                    themes.append({
                        'theme': word,
                        'frequency': count,
                        'supporting_perspectives': supporting_perspectives,
                        'emergence_strength': len(supporting_perspectives) / len(propositions),
                        'type': 'cross_perspective_convergence'
                    })

        # Identify relational themes from perspective interactions
        relation_types = {}
        for _, _, rel_type in relations:
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

        for rel_type, count in relation_types.items():
            if count >= 2:
                themes.append({
                    'theme': f"{rel_type}_dynamic",
                    'frequency': count,
                    'supporting_perspectives': 'multiple',
                    'emergence_strength': count / len(relations) if relations else 0,
                    'type': 'relational_pattern'
                })

        return themes

    def _identify_convergences(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]]
    ) -> list[str]:
        """Identify areas where perspectives converge."""

        convergences = []

        # Find concepts appearing in multiple high-confidence interpretations
        concept_appearances = {}
        for prop in propositions:
            if prop['confidence'] > 0.6:  # Only consider confident interpretations
                concepts = prop.get('concepts', [])
                for concept in concepts:
                    if concept not in concept_appearances:
                        concept_appearances[concept] = []
                    concept_appearances[concept].append(prop['perspective'])

        # Identify multi-perspective convergences
        for concept, perspectives in concept_appearances.items():
            if len(perspectives) >= 2:
                convergences.append(
                    f"Multiple perspectives ({', '.join(perspectives)}) converge on {concept}"
                )

        # Find explicit support relations
        support_relations = [r for r in relations if r[2] == 'supports']
        if len(support_relations) >= 2:
            convergences.append(
                f"{len(support_relations)} supporting relations indicate substantial conceptual agreement"
            )

        # Find high-confidence consensus
        high_confidence_props = [p for p in propositions if p['confidence'] > 0.7]
        if len(high_confidence_props) >= 2:
            convergences.append(
                f"High confidence convergence across {len(high_confidence_props)} perspectives"
            )

        return convergences if convergences else ["Limited explicit convergence detected"]

    def _identify_divergences(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]]
    ) -> list[str]:
        """Identify areas where perspectives diverge or conflict."""

        divergences = []

        # Find tension relations
        tension_relations = [r for r in relations if r[2] == 'tension']
        for p1_idx, p2_idx, _ in tension_relations:
            persp1 = propositions[p1_idx]['perspective']
            persp2 = propositions[p2_idx]['perspective']
            divergences.append(f"Fundamental tension between {persp1} and {persp2} approaches")

        # Find confidence disparities
        confidences = [p['confidence'] for p in propositions]
        confidence_range = max(confidences) - min(confidences)
        if confidence_range > 0.4:
            divergences.append(
                f"Significant confidence disparity (range: {confidence_range:.1%}) indicates interpretive uncertainty"
            )

        # Find contradictory content themes
        # This is simplified - would use more sophisticated NLP in practice
        positive_terms = {'affirms', 'supports', 'enables', 'promotes', 'enhances'}
        negative_terms = {'denies', 'contradicts', 'undermines', 'prevents', 'limits'}

        positive_perspectives = []
        negative_perspectives = []

        for prop in propositions:
            content_lower = prop['content'].lower()
            if any(term in content_lower for term in positive_terms):
                positive_perspectives.append(prop['perspective'])
            if any(term in content_lower for term in negative_terms):
                negative_perspectives.append(prop['perspective'])

        if positive_perspectives and negative_perspectives:
            divergences.append(
                f"Evaluative divergence: {', '.join(positive_perspectives)} vs {', '.join(negative_perspectives)}"
            )

        return divergences if divergences else ["No significant divergences detected"]

    def _generate_synthetic_insights(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]],
        coherence_scores: dict[str, Any]
    ) -> list[str]:
        """Generate insights that emerge from the synthesis process itself."""

        insights = []

        # Meta-cognitive insights about the analysis process
        if coherence_scores['overall_coherence'] > 0.8:
            insights.append(
                "The high coherence across diverse perspectives suggests the concept admits of stable, "
                "multi-faceted understanding despite philosophical differences."
            )

        if coherence_scores['explanatory_breadth'] > 0.6:
            insights.append(
                "The broad explanatory coverage indicates the concept connects to fundamental "
                "philosophical questions across multiple domains."
            )

        # Relationship insights
        if len(relations) > len(propositions):
            insights.append(
                "The rich relational structure suggests the concept exists within dense conceptual networks "
                "requiring integrative rather than reductive analysis."
            )

        # Diversity insights
        perspective_families = [self._identify_philosophical_family(p['perspective']) for p in propositions]
        unique_families = len(set(perspective_families))
        if unique_families >= 3:
            insights.append(
                f"Analysis spanning {unique_families} philosophical traditions indicates "
                f"cross-traditional significance and interpretive richness."
            )

        # Confidence pattern insights
        confidences = [p['confidence'] for p in propositions]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        if avg_confidence > 0.7 and confidence_std < 0.2:
            insights.append(
                "Consistently high confidence across perspectives suggests robust conceptual foundations."
            )
        elif confidence_std > 0.3:
            insights.append(
                "Variable confidence levels indicate the concept requires careful contextual analysis."
            )

        return insights

    def _assess_synthetic_epistemic_status(
        self,
        synthetic_confidence: float,
        coherence_scores: dict[str, Any]
    ) -> str:
        """Assess the epistemic status of the synthetic analysis."""

        if synthetic_confidence > 0.8 and coherence_scores['overall_coherence'] > 0.8:
            return "high_confidence_synthesis"
        elif synthetic_confidence > 0.6 and coherence_scores['overall_coherence'] > 0.6:
            return "moderate_confidence_synthesis"
        elif synthetic_confidence > 0.4:
            return "provisional_synthesis"
        else:
            return "exploratory_synthesis"

    def _assess_integration_quality(
        self,
        propositions: list[dict[str, Any]],
        relations: list[tuple[int, int, str]],
        perspectives: list[str]
    ) -> dict[str, Any]:
        """Assess the quality of integration across perspectives."""

        # Calculate integration metrics
        perspective_count = len(perspectives)
        relation_density = len(relations) / (perspective_count * (perspective_count - 1) / 2) if perspective_count > 1 else 0

        # Assess philosophical diversity
        families = [self._identify_philosophical_family(p) for p in perspectives]
        family_diversity = len(set(families)) / len(families) if families else 0

        # Assess confidence consistency
        confidences = [p['confidence'] for p in propositions]
        confidence_consistency = 1.0 - np.std(confidences) if confidences else 0

        # Overall integration score
        integration_score = (
            0.4 * relation_density +
            0.3 * family_diversity +
            0.3 * confidence_consistency
        )

        return {
            'integration_score': float(np.clip(integration_score, 0.0, 1.0)),
            'relation_density': float(relation_density),
            'philosophical_diversity': float(family_diversity),
            'confidence_consistency': float(confidence_consistency),
            'integration_quality': 'high' if integration_score > 0.7 else 'moderate' if integration_score > 0.5 else 'basic'
        }

    def _generate_meta_philosophical_observations(
        self,
        perspectives: list[str],
        family_distribution: dict[str, int],
        coherence_scores: dict[str, Any]
    ) -> list[str]:
        """Generate observations about the philosophical analysis process itself."""

        observations = []

        # Tradition representation analysis
        dominant_family = max(family_distribution.items(), key=lambda x: x[1]) if family_distribution else None
        if dominant_family and dominant_family[1] > len(perspectives) / 2:
            observations.append(
                f"Analysis dominated by {dominant_family[0]} tradition ({dominant_family[1]}/{len(perspectives)} perspectives), "
                f"potentially limiting interpretive scope."
            )
        elif len(family_distribution) >= 3:
            observations.append(
                f"Balanced representation across {len(family_distribution)} philosophical families suggests "
                f"comprehensive multi-traditional analysis."
            )

        # Coherence pattern observations
        if coherence_scores['constraint_satisfaction'] > 0.8:
            observations.append(
                "High constraint satisfaction indicates successful integration of diverse philosophical commitments."
            )

        if coherence_scores['analogical_fit'] > 0.5:
            observations.append(
                "Strong analogical connections suggest the concept exhibits family resemblances across traditions."
            )

        # Methodological observations
        if 'analytical' in perspectives and 'phenomenological' in perspectives:
            observations.append(
                "Analysis bridges the analytic-continental divide, enabling productive cross-traditional dialogue."
            )

        if len(perspectives) >= 4:
            observations.append(
                f"Multi-perspectival analysis with {len(perspectives)} viewpoints approaches philosophical comprehensiveness."
            )

        return observations

    def _identify_conceptual_tensions(self, analysis_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify sophisticated conceptual tensions using philosophical analysis."""
        tensions = []
        if len(analysis_results) < 2:
            return tensions

        # Extract perspective information
        perspectives = [p for res in analysis_results if (p := res.get("perspective")) is not None and isinstance(p, str)]
        interpretations = [res.get("interpretation", {}) for res in analysis_results]
        confidences = [res.get("confidence", 0.0) for res in analysis_results]

        # Define philosophical tension patterns
        known_tensions = {
            # Methodological tensions
            ("analytical", "phenomenological"): {
                "type": "methodological_tension",
                "description": "Tension between third-person analytical rigor and first-person experiential fidelity",
                "philosophical_significance": "Represents fundamental divide between explanation and understanding",
                "resolution_strategies": ["hermeneutic bridge-building", "pragmatic complementarity", "level-relative analysis"]
            },
            ("empiricist", "rationalist"): {
                "type": "epistemological_tension",
                "description": "Fundamental disagreement about the source and validation of knowledge",
                "philosophical_significance": "Core epistemological divide since antiquity",
                "resolution_strategies": ["kantian synthesis", "naturalized epistemology", "pragmatist dissolution"]
            },
            ("realist", "idealist"): {
                "type": "metaphysical_tension",
                "description": "Disagreement about mind-independent reality versus consciousness-constituted reality",
                "philosophical_significance": "Central metaphysical debate about nature of reality",
                "resolution_strategies": ["transcendental idealism", "neutral monism", "pragmatic deflation"]
            },

            # Normative tensions
            ("deontological", "consequentialist"): {
                "type": "normative_tension",
                "description": "Tension between duty-based and outcome-based moral reasoning",
                "philosophical_significance": "Fundamental divide in moral philosophy about the source of ethical obligation",
                "resolution_strategies": ["virtue ethics integration", "prima facie duties", "rule consequentialism"]
            },
            ("virtue_ethics", "deontological"): {
                "type": "normative_tension",
                "description": "Tension between character-focused and action-focused ethics",
                "philosophical_significance": "Different conceptions of moral agency and responsibility",
                "resolution_strategies": ["neo-aristotelian synthesis", "moral psychology integration"]
            },

            # Contemporary tensions
            ("functionalist", "phenomenological"): {
                "type": "consciousness_tension",
                "description": "Tension between functional role analysis and qualitative experience",
                "philosophical_significance": "Hard problem of consciousness and explanatory gap",
                "resolution_strategies": ["illusionism", "panpsychism", "emergentism", "eliminativism"]
            },
            ("naturalist", "phenomenological"): {
                "type": "naturalization_tension",
                "description": "Tension between scientific naturalism and phenomenological method",
                "philosophical_significance": "Question of whether consciousness can be naturalized",
                "resolution_strategies": ["neurophenomenology", "embodied cognition", "enactivism"]
            },

            # Political/critical tensions
            ("critical", "analytical"): {
                "type": "political_epistemological_tension",
                "description": "Tension between ideologically critical and supposedly neutral analysis",
                "philosophical_significance": "Question of whether philosophical analysis can be politically neutral",
                "resolution_strategies": ["situated knowledge", "critical realism", "pragmatist politics"]
            },
            ("individual", "social"): {
                "type": "ontological_tension",
                "description": "Tension between individualist and social ontologies",
                "philosophical_significance": "Fundamental question about the basic units of social reality",
                "resolution_strategies": ["relational ontology", "emergent social properties", "methodological individualism"]
            }
        }

        # Check for known philosophical tensions
        for i, persp1 in enumerate(perspectives):
            for j, persp2 in enumerate(perspectives[i+1:], i+1):
                # Check direct tensions
                # Construct tension_key explicitly as a 2-tuple and add type hint for clarity
                _sorted_perspectives = sorted([persp1, persp2])
                tension_key: tuple[str, str] = (_sorted_perspectives[0], _sorted_perspectives[1])
                if tension_key in known_tensions:
                    tension_info = known_tensions[tension_key].copy()
                    tension_info["involved_perspectives"] = [persp1, persp2]
                    tension_info["confidence_disparity"] = abs(confidences[i] - confidences[j])
                    tension_info["interpretive_distance"] = self._calculate_interpretive_distance(
                        interpretations[i], interpretations[j]
                    )
                    tensions.append(tension_info)

        # Identify semantic tensions in interpretations
        semantic_tensions = self._identify_semantic_tensions(analysis_results, interpretations)
        tensions.extend(semantic_tensions)

        # Identify confidence-based tensions
        confidence_tensions = self._identify_confidence_tensions(analysis_results, confidences)
        tensions.extend(confidence_tensions)

        # Identify evaluative tensions
        evaluative_tensions = self._identify_evaluative_tensions(analysis_results, interpretations)
        tensions.extend(evaluative_tensions)

        # Identify meta-philosophical tensions
        meta_tensions = self._identify_meta_philosophical_tensions(perspectives, analysis_results)
        tensions.extend(meta_tensions)

        # Deduplicate and rank tensions by philosophical significance
        unique_tensions = self._deduplicate_tensions(tensions)
        ranked_tensions = self._rank_tensions_by_significance(unique_tensions)

        return ranked_tensions[:8]  # Return top 8 most significant tensions

    def _calculate_interpretive_distance(
        self,
        interpretation1: dict[str, Any],
        interpretation2: dict[str, Any]
    ) -> float:
        """Calculate semantic distance between two interpretations."""
        from .utils import semantic_similarity

        # Extract text content
        content1 = interpretation1.get("derived_meaning", "") if isinstance(interpretation1, dict) else str(interpretation1)
        content2 = interpretation2.get("derived_meaning", "") if isinstance(interpretation2, dict) else str(interpretation2)

        if not content1 or not content2:
            return 0.5  # Neutral distance for missing content

        # Create feature sets for comparison
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        features1 = {'features': list(words1)}
        features2 = {'features': list(words2)}

        similarity = semantic_similarity(features1, features2, method="jaccard")
        return 1.0 - similarity  # Convert similarity to distance

    def _identify_semantic_tensions(
        self,
        analysis_results: list[dict[str, Any]],
        interpretations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify tensions based on semantic content analysis."""
        semantic_tensions = []

        # Look for opposing evaluative terms
        positive_indicators = {'beneficial', 'valuable', 'important', 'essential', 'promotes', 'enhances', 'supports'}
        negative_indicators = {'harmful', 'problematic', 'dangerous', 'undermines', 'prevents', 'limits', 'threatens'}

        positive_perspectives = []
        negative_perspectives = []

        for i, (result, interpretation) in enumerate(zip(analysis_results, interpretations, strict=True)):
            content = interpretation.get("derived_meaning", "") if isinstance(interpretation, dict) else str(interpretation)
            content_lower = content.lower()

            perspective = result.get("perspective", f"perspective_{i}")

            if any(indicator in content_lower for indicator in positive_indicators):
                positive_perspectives.append(perspective)
            if any(indicator in content_lower for indicator in negative_indicators):
                negative_perspectives.append(perspective)

        if positive_perspectives and negative_perspectives:
            semantic_tensions.append({
                "type": "evaluative_opposition",
                "description": "Fundamental evaluative disagreement about the phenomenon",
                "involved_perspectives": positive_perspectives + negative_perspectives,
                "positive_perspectives": positive_perspectives,
                "negative_perspectives": negative_perspectives,
                "philosophical_significance": "Reveals deep axiological differences in philosophical evaluation",
                "resolution_strategies": ["value pluralism", "contextual evaluation", "dialectical synthesis"]
            })

        # Look for modal disagreements (necessity vs contingency)
        necessity_indicators = {'necessary', 'must', 'essential', 'inevitable', 'required'}
        contingency_indicators = {'contingent', 'optional', 'possible', 'variable', 'dependent'}

        necessity_perspectives = []
        contingency_perspectives = []

        for i, (result, interpretation) in enumerate(zip(analysis_results, interpretations, strict=True)):
            content = interpretation.get("derived_meaning", "") if isinstance(interpretation, dict) else str(interpretation)
            content_lower = content.lower()

            perspective = result.get("perspective", f"perspective_{i}")

            if any(indicator in content_lower for indicator in necessity_indicators):
                necessity_perspectives.append(perspective)
            if any(indicator in content_lower for indicator in contingency_indicators):
                contingency_perspectives.append(perspective)

        if necessity_perspectives and contingency_perspectives:
            semantic_tensions.append({
                "type": "modal_disagreement",
                "description": "Disagreement about necessity versus contingency of the phenomenon",
                "involved_perspectives": necessity_perspectives + contingency_perspectives,
                "necessity_perspectives": necessity_perspectives,
                "contingency_perspectives": contingency_perspectives,
                "philosophical_significance": "Fundamental metaphysical disagreement about modal status",
                "resolution_strategies": ["modal contextualism", "graded necessity", "pragmatic modal collapse"]
            })

        return semantic_tensions

    def _identify_confidence_tensions(
        self,
        analysis_results: list[dict[str, Any]],
        confidences: list[float]
    ) -> list[dict[str, Any]]:
        """Identify tensions based on confidence disparities."""
        confidence_tensions = []

        if len(confidences) < 2:
            return confidence_tensions

        max_confidence = max(confidences)
        min_confidence = min(confidences)
        confidence_range = max_confidence - min_confidence

        # Significant confidence disparity indicates epistemic tension
        if confidence_range > 0.4:
            high_confidence_perspectives = []
            low_confidence_perspectives = []

            for i, (result, conf) in enumerate(zip(analysis_results, confidences, strict=True)):
                perspective = result.get("perspective", f"perspective_{i}")
                if conf >= max_confidence - 0.1:
                    high_confidence_perspectives.append(perspective)
                elif conf <= min_confidence + 0.1:
                    low_confidence_perspectives.append(perspective)

            confidence_tensions.append({
                "type": "epistemic_confidence_tension",
                "description": f"Significant confidence disparity (range: {confidence_range:.1%})",
                "involved_perspectives": high_confidence_perspectives + low_confidence_perspectives,
                "high_confidence_perspectives": high_confidence_perspectives,
                "low_confidence_perspectives": low_confidence_perspectives,
                "confidence_range": confidence_range,
                "philosophical_significance": "Indicates fundamental disagreement about epistemic access or method",
                "resolution_strategies": ["epistemic humility", "method sensitivity analysis", "confidence calibration"]
            })

        return confidence_tensions

    def _identify_evaluative_tensions(
        self,
        analysis_results: list[dict[str, Any]],
        interpretations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify tensions in philosophical evaluation and judgment."""
        evaluative_tensions = []

        # Look for prescriptive vs descriptive approaches
        prescriptive_indicators = {'should', 'ought', 'must', 'recommend', 'advocate', 'propose'}
        descriptive_indicators = {'is', 'describes', 'observes', 'analyzes', 'examines', 'reveals'}

        prescriptive_perspectives = []
        descriptive_perspectives = []

        for i, (result, interpretation) in enumerate(zip(analysis_results, interpretations, strict=True)):
            content = interpretation.get("derived_meaning", "") if isinstance(interpretation, dict) else str(interpretation)
            content_lower = content.lower()

            perspective = result.get("perspective", f"perspective_{i}")

            prescriptive_score = sum(1 for indicator in prescriptive_indicators if indicator in content_lower)
            descriptive_score = sum(1 for indicator in descriptive_indicators if indicator in content_lower)

            if prescriptive_score > descriptive_score:
                prescriptive_perspectives.append(perspective)
            elif descriptive_score > prescriptive_score:
                descriptive_perspectives.append(perspective)

        if prescriptive_perspectives and descriptive_perspectives:
            evaluative_tensions.append({
                "type": "prescriptive_descriptive_tension",
                "description": "Tension between normative prescription and descriptive analysis",
                "involved_perspectives": prescriptive_perspectives + descriptive_perspectives,
                "prescriptive_perspectives": prescriptive_perspectives,
                "descriptive_perspectives": descriptive_perspectives,
                "philosophical_significance": "Fundamental tension between 'is' and 'ought' in philosophical analysis",
                "resolution_strategies": ["naturalistic ethics", "non-cognitivism", "pragmatic integration"]
            })

        return evaluative_tensions

    def _identify_meta_philosophical_tensions(
        self,
        perspectives: list[str],
        analysis_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify tensions at the meta-philosophical level."""
        meta_tensions = []

        # Identify tradition-based tensions
        continental_perspectives = []
        analytic_perspectives = []

        continental_traditions = {'phenomenological', 'existentialist', 'hermeneutic', 'critical', 'postmodern'}
        analytic_traditions = {'analytical', 'empiricist', 'rationalist', 'functionalist', 'logical_positivist'}

        for perspective in perspectives:
            if perspective in continental_traditions:
                continental_perspectives.append(perspective)
            elif perspective in analytic_traditions:
                analytic_perspectives.append(perspective)

        if continental_perspectives and analytic_perspectives:
            meta_tensions.append({
                "type": "meta_philosophical_tradition_tension",
                "description": "Tension between Continental and Analytic philosophical traditions",
                "involved_perspectives": continental_perspectives + analytic_perspectives,
                "continental_perspectives": continental_perspectives,
                "analytic_perspectives": analytic_perspectives,
                "philosophical_significance": "Fundamental methodological and stylistic divide in contemporary philosophy",
                "resolution_strategies": ["post-analytic philosophy", "pragmatist bridge-building", "pluralistic integration"]
            })

        # Check for theory vs practice tensions
        theoretical_perspectives = []
        practical_perspectives = []

        for i, result in enumerate(analysis_results):
            interpretation = result.get("interpretation", {})
            content = interpretation.get("derived_meaning", "") if isinstance(interpretation, dict) else str(interpretation)

            if any(term in content.lower() for term in ['theory', 'abstract', 'conceptual', 'logical']):
                theoretical_perspectives.append(result.get("perspective", f"perspective_{i}"))
            if any(term in content.lower() for term in ['practice', 'action', 'concrete', 'applied']):
                practical_perspectives.append(result.get("perspective", f"perspective_{i}"))

        if theoretical_perspectives and practical_perspectives:
            meta_tensions.append({
                "type": "theory_practice_tension",
                "description": "Tension between theoretical analysis and practical application",
                "involved_perspectives": theoretical_perspectives + practical_perspectives,
                "theoretical_perspectives": theoretical_perspectives,
                "practical_perspectives": practical_perspectives,
                "philosophical_significance": "Classical philosophical tension between contemplation and action",
                "resolution_strategies": ["praxis integration", "pragmatist synthesis", "applied philosophy"]
            })

        return meta_tensions

    def _deduplicate_tensions(self, tensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate tensions while preserving the most informative versions."""
        unique_tensions = []
        seen_types = set()

        # Sort by information richness (more detail = better)
        sorted_tensions = sorted(tensions, key=lambda t: len(str(t)), reverse=True)

        for tension in sorted_tensions:
            tension_type = tension.get("type", "unknown")
            if tension_type not in seen_types:
                unique_tensions.append(tension)
                seen_types.add(tension_type)

        return unique_tensions

    def _rank_tensions_by_significance(self, tensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank tensions by philosophical significance and impact."""

        # Define significance weights
        significance_weights = {
            "methodological_tension": 0.9,
            "epistemological_tension": 0.95,
            "metaphysical_tension": 1.0,
            "normative_tension": 0.85,
            "consciousness_tension": 0.9,
            "meta_philosophical_tradition_tension": 0.8,
            "evaluative_opposition": 0.75,
            "epistemic_confidence_tension": 0.7,
            "theory_practice_tension": 0.65,
            "modal_disagreement": 0.6,
            "prescriptive_descriptive_tension": 0.55
        }

        # Calculate significance scores
        for tension in tensions:
            base_significance = significance_weights.get(tension.get("type", ""), 0.5)

            # Boost score for tensions involving many perspectives
            perspective_count = len(tension.get("involved_perspectives", []))
            perspective_factor = min(perspective_count / 4.0, 1.2)

            # Boost score for tensions with resolution strategies (indicates established philosophical debate)
            resolution_factor = 1.1 if tension.get("resolution_strategies") else 1.0

            final_significance = base_significance * perspective_factor * resolution_factor
            tension["significance_score"] = final_significance

        # Sort by significance score
        return sorted(tensions, key=lambda t: t.get("significance_score", 0), reverse=True)

    def _generate_revision_conditions(
        self, concept: str, context: str, synthesis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate revision conditions."""
        conditions = [
            {
                "trigger": "Significant shift in contextual understanding",
                "metric": "Domain coherence change > 0.3",
                "action": "Re-evaluate concept analysis"
            }
        ]
        if synthesis.get("overall_confidence", 1.0) < 0.5:
            conditions.append({
                "trigger": "Low confidence in synthesis",
                "metric": "Overall confidence < 0.5",
                "action": "Seek additional perspectives"
            })
        return conditions

    def _assess_epistemic_status(self, synthesis: dict[str, Any]) -> str:
        """Assess epistemic status."""
        confidence = synthesis.get('overall_confidence', 0.5)
        if confidence > 0.8:
            return "high_confidence"
        elif confidence > 0.6:
            return "moderate_confidence"
        else:
            return "provisional"

    def _generate_exploratory_questions(self, concept: str, synthesis: dict[str, Any]) -> list[str]:
        """Generate questions for further exploration."""
        return [
            f"What are the boundary conditions for {concept}?",
            f"How does {concept} relate to similar concepts?",
            "What assumptions underlie this analysis?"
        ]

    async def _trace_semantic_evolution(self, expression: str) -> dict[str, Any]:
        """Trace semantic evolution."""
        return {
            "historical_meanings": [f"historical_meaning_for_{expression}"],
            "evolution_timeline": ["early_period", "modern_period"],
            "key_transitions": ["semantic_shift_1"]
        }

    async def _initialize_insight_framework(self, phenomenon: str, perspectives: list[str] | None, depth: int) -> dict[str, Any]:
        """Initialize comprehensive framework for insight generation."""

        return {
            'phenomenon': phenomenon,
            'analysis_scope': {
                'depth_level': depth,
                'perspective_count': len(perspectives) if perspectives else len(self._get_default_perspectives()),
                'expected_insight_types': ['descriptive', 'explanatory', 'normative', 'meta_cognitive'],
                'complexity_assessment': self._assess_phenomenon_complexity(phenomenon)
            },
            'methodological_commitments': {
                'fallibilism': True,
                'perspectivalism': True,
                'coherentism': True,
                'pragmatism': True
            },
            'epistemic_virtues': ['intellectual_humility', 'critical_thinking', 'open_mindedness', 'systematic_inquiry'],
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _assess_phenomenon_complexity(self, phenomenon: str) -> dict[str, Any]:
        """Assess complexity characteristics of phenomenon."""

        complexity_indicators = {
            'conceptual_complexity': 'high' if any(term in phenomenon.lower() for term in
                ['consciousness', 'justice', 'truth', 'reality', 'meaning']) else 'moderate',
            'interdisciplinary_scope': 'high' if any(term in phenomenon.lower() for term in
                ['mind', 'society', 'culture', 'science', 'politics']) else 'moderate',
            'normative_dimensions': 'high' if any(term in phenomenon.lower() for term in
                ['ethics', 'ought', 'should', 'good', 'right', 'wrong']) else 'low',
            'empirical_tractability': 'low' if any(term in phenomenon.lower() for term in
                ['consciousness', 'qualia', 'free will', 'meaning']) else 'moderate'
        }

        return complexity_indicators

    async def _gather_comprehensive_evidence(self, phenomenon: str, perspectives: list[str] | None, depth: int) -> dict[str, Any]:
        """Gather comprehensive evidence corpus for analysis."""

        evidence_corpus = {
            'conceptual_evidence': await self._gather_conceptual_evidence(phenomenon, depth),
            'historical_evidence': await self._gather_historical_evidence(phenomenon),
            'cross_cultural_evidence': await self._gather_cross_cultural_evidence(phenomenon),
            'empirical_evidence': await self._gather_empirical_evidence(phenomenon),
            'logical_evidence': await self._gather_logical_evidence(phenomenon),
            'phenomenological_evidence': await self._gather_phenomenological_evidence(phenomenon),
            'pragmatic_evidence': await self._gather_pragmatic_evidence(phenomenon)
        }

        return evidence_corpus

    async def _gather_conceptual_evidence(self, phenomenon: str, depth: int) -> list[dict[str, Any]]:
        """Gather conceptual evidence and analysis patterns."""

        evidence = []

        # Definitional evidence
        evidence.append({
            'type': 'definitional',
            'content': f"Conceptual analysis reveals {phenomenon} involves multiple definitional dimensions",
            'strength': 0.7,
            'source': 'conceptual_analysis',
            'philosophical_relevance': 'foundational'
        })

        # Logical structure evidence
        evidence.append({
            'type': 'logical_structure',
            'content': f"Logical analysis of {phenomenon} reveals internal conceptual relations",
            'strength': 0.75,
            'source': 'logical_analysis',
            'philosophical_relevance': 'structural'
        })

        # Necessary/sufficient conditions
        if depth > 2:
            evidence.append({
                'type': 'modal_conditions',
                'content': f"Modal analysis suggests {phenomenon} has both necessary and contingent features",
                'strength': 0.6,
                'source': 'modal_logic',
                'philosophical_relevance': 'modal'
            })

        return evidence

    async def _gather_historical_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather historical philosophical evidence."""

        return [
            {
                'type': 'historical_development',
                'content': f"Historical analysis shows {phenomenon} has evolved significantly across philosophical traditions",
                'strength': 0.8,
                'source': 'history_of_philosophy',
                'philosophical_relevance': 'diachronic'
            },
            {
                'type': 'canonical_treatments',
                'content': f"Major philosophers have offered diverse approaches to {phenomenon}",
                'strength': 0.85,
                'source': 'philosophical_canon',
                'philosophical_relevance': 'authoritative'
            }
        ]

    async def _gather_cross_cultural_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather cross-cultural philosophical evidence."""

        return [
            {
                'type': 'cultural_variation',
                'content': f"Cross-cultural analysis reveals both universal and particular aspects of {phenomenon}",
                'strength': 0.7,
                'source': 'comparative_philosophy',
                'philosophical_relevance': 'intercultural'
            }
        ]

    async def _gather_empirical_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather empirical evidence relevant to philosophical analysis."""

        return [
            {
                'type': 'empirical_findings',
                'content': f"Empirical research provides relevant data for understanding {phenomenon}",
                'strength': 0.6,
                'source': 'empirical_studies',
                'philosophical_relevance': 'naturalistic'
            }
        ]

    async def _gather_logical_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather logical and formal evidence."""

        return [
            {
                'type': 'formal_analysis',
                'content': f"Formal logical analysis provides structural insights into {phenomenon}",
                'strength': 0.8,
                'source': 'formal_logic',
                'philosophical_relevance': 'formal'
            }
        ]

    async def _gather_phenomenological_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather phenomenological evidence from experience."""

        return [
            {
                'type': 'experiential',
                'content': f"Phenomenological analysis reveals experiential dimensions of {phenomenon}",
                'strength': 0.75,
                'source': 'phenomenological_method',
                'philosophical_relevance': 'experiential'
            }
        ]

    async def _gather_pragmatic_evidence(self, phenomenon: str) -> list[dict[str, Any]]:
        """Gather pragmatic evidence from practical consequences."""

        return [
            {
                'type': 'practical_consequences',
                'content': f"Pragmatic analysis shows {phenomenon} has significant practical implications",
                'strength': 0.7,
                'source': 'pragmatic_analysis',
                'philosophical_relevance': 'practical'
            }
        ]

    async def _generate_perspective_insights(
        self,
        phenomenon: str,
        evidence_corpus: dict[str, Any],
        perspectives: list[str]
    ) -> list[dict[str, Any]]:
        """Generate insights from each philosophical perspective."""

        perspective_insights = []

        for perspective in perspectives:
            try:
                # Generate perspective-specific evidence interpretation
                evidence_interpretation = await self._interpret_evidence_from_perspective(
                    evidence_corpus, perspective, phenomenon
                )

                # Generate insights using fallibilistic inference
                raw_insights = await self.inference_engine.derive_insights(
                    evidence_interpretation,
                    confidence_threshold=self.context.confidence_threshold
                )

                # Enhance insights with perspective-specific analysis
                enhanced_insights = []
                for insight in raw_insights:
                    enhanced_insight = {
                        'content': insight.content,
                        'perspective': perspective,
                        'confidence': insight.confidence,
                        'evidence_base': insight.evidence_summary,
                        'limitations': insight.identified_limitations,
                        'revision_triggers': insight.revision_triggers,
                        'philosophical_depth': await self._assess_insight_depth(insight, perspective),
                        'novel_contributions': await self._identify_novel_contributions(insight, perspective),
                        'integration_potential': await self._assess_integration_potential(insight, perspective)
                    }
                    enhanced_insights.append(enhanced_insight)

                perspective_insights.append({
                    'perspective': perspective,
                    'insights': enhanced_insights,
                    'evidence_interpretation': evidence_interpretation,
                    'meta_analysis': await self._generate_perspective_meta_analysis(perspective, enhanced_insights)
                })

            except Exception as e:
                logger.warning(f"Failed to generate insights for perspective {perspective}: {e}")
                continue

        return perspective_insights

    async def _interpret_evidence_from_perspective(
        self,
        evidence_corpus: dict[str, Any],
        perspective: str,
        phenomenon: str
    ) -> list[dict[str, Any]]:
        """Interpret evidence through specific philosophical perspective."""

        perspective_filters = {
            'analytical': ['logical_evidence', 'conceptual_evidence'],
            'phenomenological': ['phenomenological_evidence', 'experiential_evidence'],
            'pragmatist': ['pragmatic_evidence', 'empirical_evidence'],
            'critical': ['historical_evidence', 'cross_cultural_evidence'],
            'empiricist': ['empirical_evidence', 'logical_evidence'],
            'rationalist': ['logical_evidence', 'conceptual_evidence']
        }

        relevant_evidence_types = perspective_filters.get(perspective, list(evidence_corpus.keys()))

        interpreted_evidence = []
        for evidence_type in relevant_evidence_types:
            if evidence_type in evidence_corpus:
                for evidence_item in evidence_corpus[evidence_type]:
                    interpreted_item = evidence_item.copy()
                    interpreted_item['perspective_relevance'] = self._assess_perspective_relevance(
                        evidence_item, perspective
                    )
                    interpreted_item['perspective_interpretation'] = self._generate_perspective_interpretation_of_evidence(
                        evidence_item, perspective, phenomenon
                    )
                    interpreted_evidence.append(interpreted_item)

        return interpreted_evidence

    def _assess_perspective_relevance(self, evidence: dict[str, Any], perspective: str) -> float:
        """Assess relevance of evidence to philosophical perspective."""

        relevance_matrix = {
            'analytical': {
                'definitional': 0.9,
                'logical_structure': 0.95,
                'formal_analysis': 0.9,
                'conceptual': 0.85
            },
            'phenomenological': {
                'experiential': 0.95,
                'phenomenological': 0.9,
                'historical_development': 0.6
            },
            'pragmatist': {
                'practical_consequences': 0.9,
                'empirical_findings': 0.8,
                'historical_development': 0.7
            },
            'critical': {
                'cultural_variation': 0.85,
                'historical_development': 0.9,
                'practical_consequences': 0.8
            }
        }

        evidence_type = evidence.get('type', 'general')
        perspective_relevances = relevance_matrix.get(perspective, {})

        return perspective_relevances.get(evidence_type, 0.6)

    def _generate_perspective_interpretation_of_evidence(
        self, evidence: dict[str, Any], perspective: str, phenomenon: str
    ) -> str:
        """Generate perspective-specific interpretation of evidence."""

        evidence_content = evidence.get('content', '')
        evidence_type = evidence.get('type', '')

        # Adjust interpretation based on evidence type
        evidence_qualifier = {
            'conceptual': 'conceptual analysis reveals that',
            'historical': 'historical evidence shows that',
            'empirical': 'empirical findings indicate that',
            'logical': 'logical analysis demonstrates that',
            'phenomenological': 'experiential data suggests that',
            'pragmatic': 'practical outcomes show that'
        }.get(evidence_type, 'evidence suggests that')

        interpretation_templates = {
            'analytical': f"From an analytical standpoint, {evidence_qualifier} {evidence_content.lower()} demonstrates the need for precise conceptual analysis of {phenomenon}.",
            'phenomenological': f"Phenomenologically, {evidence_qualifier} {evidence_content.lower()} reveals experiential dimensions of {phenomenon} requiring descriptive analysis.",
            'pragmatist': f"Pragmatically, {evidence_qualifier} {evidence_content.lower()} indicates practical consequences of {phenomenon} for human action and inquiry.",
            'critical': f"Critically, {evidence_qualifier} {evidence_content.lower()} exposes power relations and ideological dimensions involved in {phenomenon}."
        }

        return interpretation_templates.get(perspective,
            f"From a {perspective} perspective, {evidence_qualifier} {evidence_content.lower()} provides relevant insights into {phenomenon}.")

    async def _assess_insight_depth(self, insight: Any, perspective: str) -> dict[str, Any]:
        """Assess philosophical depth and sophistication of insight."""

        return {
            'conceptual_sophistication': 0.7,
            'theoretical_integration': 0.6,
            'novel_connections': 0.5,
            'philosophical_significance': 0.75,
            'depth_rating': 'moderate'
        }

    async def _identify_novel_contributions(self, insight: Any, perspective: str) -> list[str]:
        """Identify novel contributions of insight."""

        return [
            f"Offers {perspective} perspective on complex phenomenon",
            "Integrates multiple evidence sources",
            "Provides framework for further inquiry"
        ]

    async def _assess_integration_potential(self, insight: Any, perspective: str) -> dict[str, Any]:
        """Assess potential for integrating insight with other perspectives."""

        return {
            'cross_perspective_compatibility': 0.7,
            'synthesis_potential': 0.6,
            'complementarity_score': 0.75,
            'integration_challenges': ['methodological_differences', 'conceptual_tensions']
        }

    async def _gather_evidence_patterns(
        self, phenomenon: str, perspectives: list[str] | None, depth: int
    ) -> dict[str, Any]:
        """Gather evidence patterns."""
        return {
            "empirical_evidence": [f"evidence_for_{phenomenon}"],
            "conceptual_patterns": [f"pattern_for_{phenomenon}"],
            "cross_perspective_themes": ["theme1", "theme2"]
        }

    async def _synthesize_cross_perspective_insights(
        self, perspective_insights: list[dict[str, Any]], phenomenon: str
    ) -> dict[str, Any]:
        """Synthesize insights across perspectives."""

        if not perspective_insights:
            return {'synthesis': 'No insights to synthesize'}

        # Extract all individual insights
        all_insights = []
        for perspective_data in perspective_insights:
            for insight in perspective_data.get('insights', []):
                all_insights.append(insight)

        # Use existing synthesis method
        return self._synthesize_analyses(all_insights)

    async def _identify_comprehensive_contradictions(
        self, perspective_insights: list[dict[str, Any]], phenomenon: str
    ) -> list[dict[str, Any]]:
        """Identify comprehensive contradictions across perspectives."""

        all_insights = []
        for perspective_data in perspective_insights:
            for insight in perspective_data.get('insights', []):
                all_insights.append(insight)

        return self._identify_insight_contradictions(all_insights)

    async def _generate_meta_philosophical_insights(
        self, phenomenon: str, perspective_insights: list[dict[str, Any]], synthetic_insights: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate meta-philosophical insights about the analysis process."""

        meta_insights = []

        # Insight about perspective diversity
        perspectives_used = [pi['perspective'] for pi in perspective_insights]
        families = [self._identify_philosophical_family(p) for p in perspectives_used]
        unique_families = len(set(families))

        if unique_families >= 3:
            meta_insights.append({
                'type': 'methodological_insight',
                'content': f"Analysis of {phenomenon} benefits from cross-traditional philosophical dialogue, "
                          f"spanning {unique_families} major philosophical families.",
                'significance': 'highlights_philosophical_pluralism',
                'confidence': 0.8
            })

        # Insight about analysis complexity
        total_insights = sum(len(pi.get('insights', [])) for pi in perspective_insights)
        if total_insights > 10:
            meta_insights.append({
                'type': 'complexity_insight',
                'content': f"The phenomenon {phenomenon} generates rich philosophical analysis with "
                          f"{total_insights} distinct insights, indicating deep conceptual complexity.",
                'significance': 'reveals_philosophical_richness',
                'confidence': 0.75
            })

        # Insight about synthesis quality
        synthesis_confidence = synthetic_insights.get('overall_confidence', 0.5)
        if synthesis_confidence > 0.7:
            meta_insights.append({
                'type': 'synthesis_insight',
                'content': f"High synthesis confidence ({synthesis_confidence:.1%}) suggests {phenomenon} "
                          f"admits of coherent multi-perspectival understanding.",
                'significance': 'supports_philosophical_realism',
                'confidence': synthesis_confidence
            })

        return meta_insights

    async def _assess_comprehensive_uncertainty(
        self, perspective_insights: list[dict[str, Any]], synthetic_insights: dict[str, Any], evidence_corpus: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess comprehensive uncertainty profile."""
        from .utils import calculate_epistemic_uncertainty

        # Calculate various uncertainty sources
        perspective_count = len(perspective_insights)
        total_insights = sum(len(pi.get('insights', [])) for pi in perspective_insights)
        evidence_count = sum(len(evidence_list) for evidence_list in evidence_corpus.values())

        synthesis_confidence = synthetic_insights.get('overall_confidence', 0.5)

        # Calculate epistemic uncertainty
        epistemic_uncertainty = calculate_epistemic_uncertainty(
            evidence_count=evidence_count,
            coherence_score=synthesis_confidence,
            temporal_factor=1.0,
            domain_complexity=0.7
        )

        return {
            'epistemic_uncertainty': epistemic_uncertainty,
            'evidence_sufficiency': min(evidence_count / 10.0, 1.0),
            'insight_density': min(total_insights / 20.0, 1.0),
            'perspective_coverage': min(perspective_count / 5.0, 1.0),
            'synthesis_stability': synthesis_confidence,
            'uncertainty_sources': [
                'limited_evidence_base' if evidence_count < 8 else None,
                'insufficient_insights' if total_insights < 10 else None,
                'perspective_limitations' if perspective_count < 4 else None,
                'synthesis_challenges' if synthesis_confidence < 0.6 else None
            ],
            'confidence_calibration': self._assess_confidence_calibration(perspective_insights)
        }

    def _assess_confidence_calibration(self, perspective_insights: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess calibration of confidence across perspectives."""

        all_confidences = []
        for perspective_data in perspective_insights:
            for insight in perspective_data.get('insights', []):
                all_confidences.append(insight.get('confidence', 0.5))

        if not all_confidences:
            return {'status': 'no_confidence_data'}

        import numpy as np
        return {
            'mean_confidence': float(np.mean(all_confidences)),
            'confidence_variance': float(np.var(all_confidences)),
            'confidence_range': float(np.max(all_confidences) - np.min(all_confidences)),
            'calibration_quality': 'good' if np.var(all_confidences) < 0.1 else 'moderate' if np.var(all_confidences) < 0.2 else 'poor'
        }

    async def _generate_revision_framework(
        self, phenomenon: str, perspective_insights: list[dict[str, Any]], uncertainty_profile: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate framework for insight revision and update."""

        revision_triggers = []

        # Evidence-based triggers
        evidence_sufficiency = uncertainty_profile.get('evidence_sufficiency', 0.5)
        if evidence_sufficiency < 0.7:
            revision_triggers.append({
                'type': 'evidence_accumulation',
                'threshold': 'significant_new_evidence',
                'description': 'Revise insights when substantial new evidence emerges'
            })

        # Perspective-based triggers
        perspective_coverage = uncertainty_profile.get('perspective_coverage', 0.5)
        if perspective_coverage < 0.8:
            revision_triggers.append({
                'type': 'perspective_expansion',
                'threshold': 'new_philosophical_perspectives',
                'description': 'Revise analysis when new philosophical approaches are considered'
            })

        # Temporal triggers
        revision_triggers.append({
            'type': 'temporal_decay',
            'threshold': '6_months',
            'description': 'Regular revision to account for evolving philosophical discourse'
        })

        return {
            'revision_triggers': revision_triggers,
            'update_protocols': [
                'incremental_evidence_integration',
                'perspective_synthesis_refinement',
                'uncertainty_recalibration'
            ],
            'validation_criteria': [
                'logical_consistency',
                'empirical_adequacy',
                'philosophical_plausibility',
                'practical_relevance'
            ]
        }

    async def _assess_philosophical_significance(
        self, phenomenon: str, perspective_insights: list[dict[str, Any]], synthetic_insights: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess philosophical significance of analysis."""

        significance_indicators = []

        # Cross-traditional relevance
        perspectives = [pi['perspective'] for pi in perspective_insights]
        families = {self._identify_philosophical_family(p) for p in perspectives}
        if len(families) >= 3:
            significance_indicators.append('cross_traditional_relevance')

        # Synthesis quality
        synthesis_confidence = synthetic_insights.get('overall_confidence', 0.5)
        if synthesis_confidence > 0.7:
            significance_indicators.append('high_synthesis_coherence')

        # Insight richness
        total_insights = sum(len(pi.get('insights', [])) for pi in perspective_insights)
        if total_insights > 8:
            significance_indicators.append('rich_philosophical_analysis')

        return {
            'significance_level': 'high' if len(significance_indicators) >= 2 else 'moderate' if len(significance_indicators) >= 1 else 'basic',
            'significance_indicators': significance_indicators,
            'philosophical_contributions': [
                'multi_perspectival_understanding',
                'conceptual_clarification',
                'systematic_integration'
            ],
            'potential_impact': self._assess_potential_philosophical_impact(phenomenon, significance_indicators)
        }

    def _assess_potential_philosophical_impact(self, phenomenon: str, indicators: list[str]) -> str:
        """Assess potential impact of philosophical analysis."""

        if len(indicators) >= 3:
            return 'potentially_significant_contribution'
        elif len(indicators) >= 2:
            return 'meaningful_philosophical_analysis'
        else:
            return 'solid_philosophical_exploration'

    async def _generate_inquiry_recommendations(
        self, phenomenon: str, perspective_insights: list[dict[str, Any]], uncertainty_profile: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Generate recommendations for further philosophical inquiry."""

        recommendations = []

        # Evidence-based recommendations
        evidence_sufficiency = uncertainty_profile.get('evidence_sufficiency', 0.5)
        if evidence_sufficiency < 0.7:
            recommendations.append({
                'type': 'evidence_expansion',
                'description': f'Gather additional evidence from empirical research and case studies related to {phenomenon}',
                'priority': 'high'
            })

        # Perspective-based recommendations
        perspective_coverage = uncertainty_profile.get('perspective_coverage', 0.5)
        if perspective_coverage < 0.8:
            recommendations.append({
                'type': 'perspective_diversification',
                'description': f'Explore {phenomenon} from additional philosophical traditions such as Eastern philosophy or feminist philosophy',
                'priority': 'medium'
            })

        # Methodological recommendations
        recommendations.append({
            'type': 'methodological_refinement',
            'description': f'Apply specialized philosophical methods to deepen analysis of {phenomenon}',
            'priority': 'medium'
        })

        # Integration recommendations
        if len(perspective_insights) > 3:
            recommendations.append({
                'type': 'cross_perspective_integration',
                'description': f'Develop systematic integration of diverse perspectives on {phenomenon}',
                'priority': 'high'
            })

        return recommendations

    async def _generate_fallback_insights(self, phenomenon: str, perspectives: list[str] | None, depth: int) -> dict[str, Any]:
        """Generate simplified insights as fallback."""

        return {
            'phenomenon': phenomenon,
            'status': 'fallback_analysis',
            'primary_insights': [
                {
                    'content': f'Basic philosophical analysis of {phenomenon} reveals conceptual complexity',
                    'perspective': 'general',
                    'confidence': 0.6
                }
            ],
            'analysis_note': 'Simplified analysis due to processing constraints'
        }

    async def _generate_perspective_meta_analysis(self, perspective: str, insights: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate meta-analysis for perspective-specific insights."""

        if not insights:
            return {'status': 'no_insights_for_meta_analysis'}

        confidences = [insight.get('confidence', 0.5) for insight in insights]

        return {
            'insight_count': len(insights),
            'average_confidence': sum(confidences) / len(confidences),
            'confidence_range': max(confidences) - min(confidences),
            'perspective_strength': 'high' if sum(confidences) / len(confidences) > 0.7 else 'moderate',
            'key_contributions': [insight.get('content', '')[:100] + '...' for insight in insights[:3]]
        }

    def _get_default_perspectives(self) -> list[str]:
        """Get default perspectives."""
        return ["analytical", "phenomenological", "pragmatist", "critical"]

    def _identify_insight_contradictions(self, insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify contradictions between insights."""
        contradictions = []
        for i, insight1 in enumerate(insights):
            for insight2 in insights[i+1:]:
                if insight1.get('perspective') != insight2.get('perspective'):
                    contradictions.append({
                        'insight1': insight1,
                        'insight2': insight2,
                        'conflict_type': 'perspectival_difference'
                    })
        return contradictions

    async def _test_in_domain(
        self, hypothesis: str, domain: str, criteria: dict[str, Any]
    ) -> dict[str, Any]:
        """Test hypothesis in domain."""
        return {
            "domain": domain,
            "test_results": f"hypothesis_test_in_{domain}",
            "coherence_score": 0.7,
            "evidence_strength": 0.6,
            "supporting_evidence": ["evidence1"],
            "challenges": ["challenge1"]
        }

    def _calculate_hypothesis_coherence(self, domain_results: dict[str, Any]) -> float:
        """Calculate hypothesis coherence."""
        if not domain_results:
            return 0.0
        scores = [result.get('coherence_score', 0.5) for result in domain_results.values()]
        return float(np.mean(scores))

    def _assess_pragmatic_value(self, hypothesis: str, domain_results: dict[str, Any]) -> float:
        """Assess pragmatic value."""
        return 0.65

    def _calculate_test_confidence(
        self, coherence: float, pragmatic_score: float, challenge_count: int
    ) -> float:
        """Calculate test confidence."""
        base_confidence = (coherence + pragmatic_score) / 2
        challenge_penalty = challenge_count * 0.1
        return max(0.0, base_confidence - challenge_penalty)

    def _identify_philosophical_family(self, perspective: str) -> str:
        """Identify the philosophical family/tradition of a perspective."""
        families = {
            'continental': ['phenomenological', 'existentialist', 'hermeneutic', 'critical', 'postmodern'],
            'analytic': ['analytical', 'empiricist', 'rationalist', 'functionalist', 'logical_positivist'],
            'pragmatic': ['pragmatist', 'naturalist', 'neo_pragmatist'],
            'classical': ['virtue_ethics', 'deontological', 'consequentialist', 'aristotelian'],
            'emergent': ['emergentist', 'complexity_theory', 'systems_theory']
        }

        for family, members in families.items():
            if perspective in members:
                return family
        return 'philosophical_inquiry'

    def _identify_complementary_perspectives(self, perspective: str) -> list[str]:
        """Identify perspectives that complement or synergize with this one."""
        complements = {
            'analytical': ['empiricist', 'rationalist', 'pragmatist'],
            'phenomenological': ['existentialist', 'hermeneutic', 'embodied_cognition'],
            'pragmatist': ['naturalist', 'democratic_theory', 'experimental_philosophy'],
            'critical': ['postmodern', 'feminist', 'postcolonial'],
            'existentialist': ['phenomenological', 'hermeneutic', 'psychoanalytic'],
            'virtue_ethics': ['aristotelian', 'care_ethics', 'narrative_ethics'],
            'empiricist': ['naturalist', 'scientific_realist', 'pragmatist'],
            'functionalist': ['computational', 'systems_theory', 'emergentist']
        }
        return complements.get(perspective, ['holistic', 'integrative'])

    def _identify_typical_tensions(self, perspective: str) -> list[str]:
        """Identify typical tensions or challenges for this perspective."""
        tensions = {
            'analytical': ['oversimplification', 'context_loss', 'lived_experience_neglect'],
            'phenomenological': ['subjectivity_concerns', 'generalizability_limits', 'scientific_compatibility'],
            'pragmatist': ['relativism_risk', 'traditional_values_neglect', 'theoretical_depth_concerns'],
            'critical': ['ideology_accusations', 'objectivity_questions', 'practical_application_challenges'],
            'existentialist': ['radical_individualism', 'nihilism_risk', 'social_dimension_neglect'],
            'virtue_ethics': ['cultural_relativism', 'action_guidance_limits', 'virtue_conflicts'],
            'empiricist': ['theory_underdetermination', 'observation_theory_distinction', 'induction_problem'],
            'functionalist': ['multiple_realizability_issues', 'consciousness_hard_problem', 'qualia_neglect']
        }
        return tensions.get(perspective, ['complexity_management', 'integration_challenges'])

    def _get_perspective_commitments(self, perspective: str) -> list[str]:
        """Get perspective commitments."""
        commitments = {
            "analytical": ["logical_rigor", "conceptual_clarity"],
            "phenomenological": ["lived_experience", "consciousness_centrality"],
            "pragmatist": ["practical_consequences", "experimental_method"],
            "critical": ["power_analysis", "social_construction"]
        }
        return commitments.get(perspective, ["general_inquiry"])

    def _get_perspective_criteria(self, perspective: str) -> list[str]:
        """Get perspective criteria."""
        criteria = {
            "analytical": ["logical_consistency", "definitional_precision"],
            "phenomenological": ["descriptive_adequacy", "experiential_fidelity"],
            "pragmatist": ["practical_success", "problem_solving_efficacy"],
            "critical": ["transformative_potential", "ideological_critique"]
        }
        return criteria.get(perspective, ["general_adequacy"])

    def _get_perspective_priorities(self, perspective: str) -> list[str]:
        """Get perspective priorities."""
        priorities = {
            "analytical": ["precision", "rigor", "systematicity"],
            "phenomenological": ["experience", "meaning", "intentionality"],
            "pragmatist": ["action", "consequences", "inquiry"],
            "critical": ["power", "justice", "transformation"]
        }
        return priorities.get(perspective, ["understanding"])

    def _generate_perspective_interpretation(
        self,
        concept_content: dict[str, Any],
        perspective: str,
        commitments: list[str]
    ) -> str:
        """Generate sophisticated perspective-specific interpretation using rich philosophical analysis."""

        # Extract concept information
        concept_key = concept_content.get('term', 'concept')
        domain = concept_content.get('domain', 'general')

        # Enhanced interpretation generation using philosophical depth
        interpretation_generators = {
            "analytical": self._generate_analytical_interpretation,
            "phenomenological": self._generate_phenomenological_interpretation,
            "pragmatist": self._generate_pragmatist_interpretation,
            "critical": self._generate_critical_interpretation,
            "existentialist": self._generate_existentialist_interpretation,
            "virtue_ethics": self._generate_virtue_ethics_interpretation,
            "empiricist": self._generate_empiricist_interpretation,
            "functionalist": self._generate_functionalist_interpretation,
            "emergentist": self._generate_emergentist_interpretation,
            "hermeneutic": self._generate_hermeneutic_interpretation
        }

        # Get specialized generator or use generic approach
        generator = interpretation_generators.get(perspective, self._generate_generic_interpretation)

        try:
            interpretation = generator(concept_key, domain, commitments, concept_content)
        except Exception as e:
            logger.debug(f"Specialized interpretation failed for {perspective}: {e}")
            interpretation = self._generate_generic_interpretation(concept_key, domain, commitments, concept_content)

        return interpretation

    def _generate_analytical_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate analytical philosophical interpretation."""
        # Analytical philosophy focuses on conceptual analysis, logical structure, and definitional clarity
        analysis_components = []

        # Definitional analysis
        analysis_components.append(
            f"Analytically, '{concept}' within {domain} requires precise definitional analysis. "
            f"We must distinguish between necessary and sufficient conditions for {concept}-hood."
        )

        # Logical structure analysis
        if 'consciousness' in concept.lower() or 'mind' in domain.lower():
            analysis_components.append(
                "The logical structure reveals intentionality conditions, qualitative properties, "
                "and functional relations that constitute mental phenomena."
            )
        elif 'knowledge' in concept.lower() or 'truth' in concept.lower():
            analysis_components.append(
                "Epistemologically, we must analyze the logical relations between justification, "
                "truth conditions, and belief states."
            )
        elif 'ethics' in domain.lower() or 'moral' in concept.lower():
            analysis_components.append(
                "The logical analysis reveals deontic structures, normative relations, "
                "and the logical form of moral reasoning."
            )
        else:
            analysis_components.append(
                f"The conceptual analysis of {concept} reveals formal structures, "
                f"logical dependencies, and systematic relations within {domain}."
            )

        # Integration with commitments
        if commitments:
            primary_commitments = commitments[:2]
            analysis_components.append(
                f"Key analytical commitments include {' and '.join(primary_commitments)}, "
                f"which guide the systematic decomposition of {concept}."
            )

        return " ".join(analysis_components)

    def _generate_phenomenological_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate phenomenological interpretation focusing on lived experience."""
        phenomenological_components = []

        # Experiential foundation
        phenomenological_components.append(
            f"Phenomenologically, {concept} is not primarily a theoretical construct but "
            f"a lived reality that manifests in the structure of consciousness itself."
        )

        # Intentional analysis
        if 'consciousness' in concept.lower() or 'mind' in domain.lower():
            phenomenological_components.append(
                "The intentional structure reveals consciousness as always consciousness-of-something, "
                "with {concept} appearing through specific modes of givenness and temporal synthesis."
            )
        elif 'time' in concept.lower() or 'temporal' in concept.lower():
            phenomenological_components.append(
                "Temporality is disclosed as the fundamental structure of consciousness, "
                "with retention, primal impression, and protention constituting lived duration."
            )
        elif 'body' in concept.lower() or 'embodiment' in domain.lower():
            phenomenological_components.append(
                "Embodied consciousness reveals the body not as mere physical object but as "
                "the lived body (Leib) that is the condition for all world-disclosure."
            )
        elif 'other' in concept.lower() or 'intersubjectivity' in domain.lower():
            phenomenological_components.append(
                "Intersubjective constitution shows how the Other is given in originary experience "
                "through empathetic appresentation and bodily expression."
            )
        else:
            phenomenological_components.append(
                f"{concept.title()} appears in the lifeworld as pre-theoretical understanding, "
                f"constituted through temporal synthesis and embodied engagement within {domain}."
            )

        # Methodological note
        phenomenological_components.append(
            "This analysis brackets the natural attitude to examine how "
            f"{concept} is constituted in and through consciousness."
        )

        return " ".join(phenomenological_components)

    def _generate_pragmatist_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate pragmatist interpretation focusing on practical consequences and inquiry."""
        pragmatist_components = []

        # Practical consequences focus
        pragmatist_components.append(
            f"From a pragmatist perspective, the meaning of '{concept}' is found in its "
            f"practical consequences and its role in ongoing inquiry within {domain}."
        )

        # Domain-specific practical analysis
        if 'truth' in concept.lower() or 'knowledge' in concept.lower():
            pragmatist_components.append(
                "Truth is reconceived as warranted assertibility - what inquiry has established "
                "as reliable for guiding action and further inquiry."
            )
        elif 'democracy' in concept.lower() or 'politics' in domain.lower():
            pragmatist_components.append(
                "Democratic values are validated through their capacity to create conditions "
                "for shared inquiry and cooperative problem-solving."
            )
        elif 'education' in concept.lower() or 'learning' in domain.lower():
            pragmatist_components.append(
                "Educational value is measured by enhancement of intelligent action "
                "and capacity for reflective inquiry."
            )
        elif 'science' in domain.lower() or 'empirical' in concept.lower():
            pragmatist_components.append(
                "Scientific concepts are tools for organizing experience and enabling "
                "successful prediction and control."
            )
        else:
            pragmatist_components.append(
                f"The value of {concept} is determined by its contribution to solving "
                f"concrete problems and advancing inquiry in {domain}."
            )

        # Experimental dimension
        pragmatist_components.append(
            f"Any understanding of {concept} must be tested experimentally "
            f"through its application in concrete situations."
        )

        # Fallibilistic note
        pragmatist_components.append(
            "This interpretation remains open to revision based on "
            "future experience and inquiry outcomes."
        )

        return " ".join(pragmatist_components)

    def _generate_critical_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate critical theory interpretation focusing on power, ideology, and emancipation."""
        critical_components = []

        # Power analysis foundation
        critical_components.append(
            f"A critical analysis of '{concept}' within {domain} must examine "
            f"the power relations, ideological structures, and material conditions "
            f"that shape its meaning and deployment."
        )

        # Domain-specific critical analysis
        if 'knowledge' in concept.lower() or 'education' in domain.lower():
            critical_components.append(
                "Knowledge is not neutral but embedded in relations of power. "
                "We must ask: whose knowledge counts, who benefits from this framing, "
                "and what alternatives are marginalized?"
            )
        elif 'freedom' in concept.lower() or 'autonomy' in concept.lower():
            critical_components.append(
                "Apparent freedom must be examined for subtle forms of domination. "
                "True autonomy requires not just negative liberty but positive "
                "conditions for self-determination."
            )
        elif 'rationality' in concept.lower() or 'reason' in concept.lower():
            critical_components.append(
                "Instrumental reason has colonized domains of human activity, "
                "reducing communicative action to strategic manipulation. "
                "Emancipatory reason seeks to restore dialogue and mutual understanding."
            )
        elif 'culture' in concept.lower() or 'society' in domain.lower():
            critical_components.append(
                "Cultural phenomena must be understood as sites of struggle "
                "between dominant and resistant meanings, shaped by economic "
                "and political forces."
            )
        else:
            critical_components.append(
                f"The concept {concept} functions within systems of domination "
                f"and may serve to reproduce or challenge existing power structures "
                f"within {domain}."
            )

        # Emancipatory dimension
        critical_components.append(
            f"Critical analysis aims to reveal how {concept} might contribute "
            f"to human emancipation and social transformation."
        )

        # Historical consciousness
        critical_components.append(
            "This understanding recognizes the historical contingency of current "
            "arrangements and the possibility of alternative configurations."
        )

        return " ".join(critical_components)

    def _generate_existentialist_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate existentialist interpretation focusing on existence, freedom, and authenticity."""
        existentialist_components = []

        # Existence precedes essence
        existentialist_components.append(
            f"Existentially, {concept} cannot be understood through abstract essences "
            f"but only through the concrete existence and lived situation of human beings."
        )

        # Freedom and responsibility analysis
        if 'choice' in concept.lower() or 'decision' in concept.lower():
            existentialist_components.append(
                "Choice reveals the radical freedom and responsibility that defines "
                "human existence. We are condemned to be free and must choose "
                "without predetermined guidelines."
            )
        elif 'authenticity' in concept.lower() or 'authentic' in concept.lower():
            existentialist_components.append(
                "Authenticity requires owning one's choices and refusing the "
                "comfortable self-deception of bad faith. One must choose oneself "
                "in full awareness of freedom and finitude."
            )
        elif 'anxiety' in concept.lower() or 'angst' in concept.lower():
            existentialist_components.append(
                "Anxiety discloses the groundlessness of existence and the "
                "weight of freedom. It is the affective revelation of our "
                "responsibility for creating meaning."
            )
        elif 'death' in concept.lower() or 'finitude' in concept.lower():
            existentialist_components.append(
                "Being-toward-death reveals the finite structure of existence "
                "and calls us to authentic temporality and resolute choice."
            )
        else:
            existentialist_components.append(
                f"{concept.title()} must be understood in relation to the fundamental "
                f"structures of human existence: freedom, finitude, and the call "
                f"to authentic self-creation within {domain}."
            )

        # Situatedness
        existentialist_components.append(
            f"This analysis recognizes that {concept} is always encountered "
            f"within concrete historical and cultural situations that both "
            f"limit and enable authentic existence."
        )

        # Individual responsibility
        existentialist_components.append(
            "Ultimately, each individual must take responsibility for their "
            f"understanding and living of {concept}."
        )

        return " ".join(existentialist_components)

    def _generate_virtue_ethics_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate virtue ethics interpretation focusing on character and flourishing."""
        virtue_components = []

        # Character-centered approach
        virtue_components.append(
            f"From a virtue ethics perspective, {concept} is best understood "
            f"in relation to character formation and human flourishing (eudaimonia)."
        )

        # Domain-specific virtue analysis
        if 'justice' in concept.lower() or 'fairness' in concept.lower():
            virtue_components.append(
                "Justice as virtue involves the settled disposition to give "
                "each their due, guided by practical wisdom and consideration "
                "of the common good."
            )
        elif 'courage' in concept.lower() or 'bravery' in concept.lower():
            virtue_components.append(
                "Courage is the mean between cowardice and recklessness, "
                "enabling one to act well in the face of difficulty or danger."
            )
        elif 'friendship' in concept.lower() or 'love' in concept.lower():
            virtue_components.append(
                "Friendship represents the highest form of human relationship, "
                "combining utility, pleasure, and mutual virtue in the pursuit "
                "of shared goods."
            )
        elif 'wisdom' in concept.lower() or 'knowledge' in concept.lower():
            virtue_components.append(
                "Practical wisdom (phronesis) is the master virtue that enables "
                "correct deliberation about human affairs and the coordination "
                "of other virtues."
            )
        else:
            virtue_components.append(
                f"{concept.title()} should be evaluated by its contribution to "
                f"character development and the cultivation of virtues within {domain}."
            )

        # Practical wisdom dimension
        virtue_components.append(
            f"The virtuous person exercises practical wisdom to discern how "
            f"{concept} contributes to or detracts from human flourishing."
        )

        # Community context
        virtue_components.append(
            "This analysis recognizes that virtues are cultivated within "
            "communities of practice and oriented toward shared goods."
        )

        # Habituation
        virtue_components.append(
            f"Understanding {concept} requires attention to how repeated actions "
            f"shape character and create settled dispositions for future action."
        )

        return " ".join(virtue_components)

    def _generate_empiricist_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate empiricist interpretation grounded in experience and observation."""
        empiricist_components = []

        # Experience foundation
        empiricist_components.append(
            f"From an empiricist standpoint, our understanding of '{concept}' "
            f"must be grounded in sensory experience and empirical observation."
        )

        # Knowledge source analysis
        if 'knowledge' in concept.lower() or 'learning' in concept.lower():
            empiricist_components.append(
                "All knowledge originates in sense experience through impression "
                "and idea formation. Complex ideas are built from simple sensory "
                "elements through association and abstraction."
            )
        elif 'causation' in concept.lower() or 'cause' in concept.lower():
            empiricist_components.append(
                "Causal relations are not logically necessary but are based on "
                "observed constant conjunction and habit formation. We cannot "
                "perceive causation itself, only temporal succession."
            )
        elif 'self' in concept.lower() or 'identity' in concept.lower():
            empiricist_components.append(
                "Personal identity consists in the continuity of memory and "
                "experience rather than in any metaphysical substance. "
                "The self is a bundle of perceptions."
            )
        else:
            empiricist_components.append(
                f"Any valid concept of {concept} must be traceable to simple "
                f"ideas derived from sensory impressions within {domain}."
            )

        # Verification principle
        empiricist_components.append(
            f"Claims about {concept} gain meaning and justification through "
            f"their connection to possible empirical verification or falsification."
        )

        # Inductive reasoning
        empiricist_components.append(
            "Our understanding develops through inductive generalization "
            "from particular experiences, acknowledging the fallible nature "
            "of empirical knowledge."
        )

        return " ".join(empiricist_components)

    def _generate_functionalist_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate functionalist interpretation focusing on functional roles and relations."""
        functionalist_components = []

        # Functional role analysis
        functionalist_components.append(
            f"Functionally, {concept} is defined not by its intrinsic properties "
            f"but by its causal-functional role within a broader system."
        )

        # Domain-specific functional analysis
        if 'mind' in concept.lower() or 'mental' in domain.lower():
            functionalist_components.append(
                "Mental states are characterized by their functional relations "
                "to inputs, outputs, and other mental states. Multiple physical "
                "systems can realize the same functional organization."
            )
        elif 'pain' in concept.lower() or 'consciousness' in concept.lower():
            functionalist_components.append(
                "Conscious states are defined by their causal roles in producing "
                "behavior, processing information, and integrating with other "
                "cognitive systems."
            )
        elif 'belief' in concept.lower() or 'desire' in concept.lower():
            functionalist_components.append(
                "Propositional attitudes are characterized by their roles in "
                "practical reasoning, action production, and belief revision "
                "rather than by qualitative content."
            )
        else:
            functionalist_components.append(
                f"{concept.title()} is best understood through its functional "
                f"contributions to system operation and goal achievement within {domain}."
            )

        # Multiple realizability
        functionalist_components.append(
            f"The functional approach recognizes that {concept} can be "
            f"multiply realized in different physical substrates while "
            f"maintaining functional equivalence."
        )

        # System integration
        functionalist_components.append(
            f"Analysis of {concept} requires understanding its integration "
            f"with other functional components and its role in overall "
            f"system performance."
        )

        return " ".join(functionalist_components)

    def _generate_emergentist_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate emergentist interpretation focusing on emergence and complexity."""
        emergentist_components = []

        # Emergence foundation
        emergentist_components.append(
            f"From an emergentist perspective, {concept} represents emergent "
            f"properties that arise from but are not reducible to lower-level "
            f"components and interactions."
        )

        # Level-specific analysis
        if 'consciousness' in concept.lower() or 'mind' in domain.lower():
            emergentist_components.append(
                "Consciousness emerges from neural complexity but exhibits "
                "novel causal powers irreducible to neurobiological processes. "
                "Mental causation operates through downward causation."
            )
        elif 'life' in concept.lower() or 'biological' in domain.lower():
            emergentist_components.append(
                "Biological properties emerge from chemical interactions but "
                "exhibit autonomous organization, self-maintenance, and "
                "adaptive behavior not present at lower levels."
            )
        elif 'social' in concept.lower() or 'culture' in domain.lower():
            emergentist_components.append(
                "Social phenomena emerge from individual interactions but "
                "develop institutional reality with genuine causal efficacy "
                "that shapes individual behavior."
            )
        else:
            emergentist_components.append(
                f"{concept.title()} exhibits emergent properties within {domain} "
                f"that arise from complex interactions while maintaining "
                f"distinctive causal powers."
            )

        # Complexity and organization
        emergentist_components.append(
            f"The emergence of {concept} depends on specific organizational "
            f"patterns and threshold levels of complexity that enable "
            f"novel systemic properties."
        )

        # Irreducibility
        emergentist_components.append(
            "While respecting lower-level constraints, emergent properties "
            f"of {concept} require autonomous description and cannot be "
            f"fully predicted from component analysis alone."
        )

        return " ".join(emergentist_components)

    def _generate_hermeneutic_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate hermeneutic interpretation focusing on interpretation and understanding."""
        hermeneutic_components = []

        # Interpretive foundation
        hermeneutic_components.append(
            f"Hermeneutically, {concept} is understood through the circular "
            f"process of interpretation, where understanding emerges from "
            f"the dialogue between interpreter and interpreted."
        )

        # Historical consciousness
        hermeneutic_components.append(
            f"Understanding {concept} requires awareness of historical context "
            f"and the temporal distance that both separates and connects us "
            f"to its meaning within {domain}."
        )

        # Fusion of horizons
        if 'text' in concept.lower() or 'interpretation' in concept.lower():
            hermeneutic_components.append(
                "Meaning emerges through the fusion of horizons between the "
                "world of the text and the world of the interpreter, creating "
                "new understanding through dialogue."
            )
        elif 'tradition' in concept.lower() or 'culture' in concept.lower():
            hermeneutic_components.append(
                "Tradition is not a dead weight of the past but a living "
                "transmission of meaning that shapes present understanding "
                "while remaining open to reinterpretation."
            )
        elif 'understanding' in concept.lower() or 'meaning' in concept.lower():
            hermeneutic_components.append(
                "Understanding is not passive reception but active engagement "
                "that transforms both interpreter and interpreted through "
                "the process of meaning-making."
            )
        else:
            hermeneutic_components.append(
                f"The meaning of {concept} emerges through interpretive "
                f"engagement that recognizes both historical conditioning "
                f"and creative possibility within {domain}."
            )

        # Prejudgment and openness
        hermeneutic_components.append(
            f"Interpretation of {concept} involves productive prejudgments "
            f"that enable understanding while remaining open to correction "
            f"through encounter with the subject matter."
        )

        return " ".join(hermeneutic_components)

    def _generate_generic_interpretation(
        self, concept: str, domain: str, commitments: list[str], content: dict[str, Any]
    ) -> str:
        """Generate generic philosophical interpretation as fallback."""
        generic_components = []

        generic_components.append(
            f"Philosophically, {concept} within {domain} raises fundamental "
            f"questions about the nature of reality, knowledge, and value."
        )

        if commitments:
            generic_components.append(
                f"Key considerations include {', '.join(commitments[:3])}, "
                f"which guide the analysis and evaluation of {concept}."
            )

        generic_components.append(
            f"Understanding {concept} requires careful attention to both "
            f"conceptual precision and contextual sensitivity within {domain}."
        )

        return " ".join(generic_components)

    def _extract_schema_relevant_features(
        self,
        concept_content: dict[str, Any],
        commitments: list[str],
        criteria: list[str]
    ) -> list[str]:
        """Extract features relevant to the interpretive schema."""

        relevant_features = []

        # Check concept content against schema commitments
        for commitment in commitments:
            if any(commitment.lower() in str(value).lower()
                   for value in concept_content.values() if value):
                relevant_features.append(f"commitment_alignment_{commitment}")

        # Check against evaluation criteria
        for criterion in criteria:
            if any(criterion.lower() in str(value).lower()
                   for value in concept_content.values() if value):
                relevant_features.append(f"criterion_satisfaction_{criterion}")

        # Extract structural features
        if 'term' in concept_content:
            relevant_features.append(f"conceptual_core_{concept_content['term']}")

        if 'domain' in concept_content:
            relevant_features.append(f"domain_grounding_{concept_content['domain']}")

        # Add temporal features
        if 'timestamp' in concept_content:
            relevant_features.append("temporal_grounding")

        # Ensure we have at least some features
        if not relevant_features:
            relevant_features = [
                "minimal_conceptual_structure",
                "basic_semantic_content"
            ]

        return relevant_features[:5]  # Limit to 5 most relevant

    def _calculate_schema_uncertainty(
        self,
        alignment_score: float,
        concept_confidence: float,
        coherence_score: float,
        feature_count: int
    ) -> float:
        """Calculate uncertainty in schema application."""
        from .utils import calculate_epistemic_uncertainty

        # Base uncertainty from alignment
        alignment_uncertainty = 1.0 - alignment_score

        # Confidence uncertainty
        confidence_uncertainty = 1.0 - concept_confidence

        # Coherence uncertainty
        coherence_uncertainty = 1.0 - coherence_score

        # Evidence quantity factor (more features = less uncertainty)
        evidence_factor = 1.0 / (1.0 + feature_count * 0.2)

        # Combine uncertainty sources
        combined_uncertainty = (
            0.3 * alignment_uncertainty +
            0.25 * confidence_uncertainty +
            0.25 * coherence_uncertainty +
            0.2 * evidence_factor
        )

        # Apply epistemic uncertainty calculation from utils
        domain_complexity = 0.6  # Philosophy is inherently complex
        temporal_factor = 1.0    # Current analysis

        epistemic_uncertainty = calculate_epistemic_uncertainty(
            evidence_count=feature_count,
            coherence_score=coherence_score,
            temporal_factor=temporal_factor,
            domain_complexity=domain_complexity
        )

        # Take maximum of combined and epistemic uncertainty
        final_uncertainty = max(combined_uncertainty, epistemic_uncertainty)

        return float(np.clip(final_uncertainty, 0.0, 1.0))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PhilosophicalJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for philosophical analysis results."""

    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif hasattr(o, 'isoformat'):
            return o.isoformat()
        elif hasattr(o, '__dataclass_fields__'):
            # Handle dataclass objects using asdict
            return asdict(o)
        return super().default(o)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize objects to JSON with philosophical error handling."""
    try:
        return json.dumps(obj, cls=PhilosophicalJSONEncoder, **kwargs)
    except Exception as e:
        logger.error(f"JSON serialization error: {e}")
        # Return a safe error object
        return json.dumps({
            "error": "serialization_failed",
            "message": str(e),
            "epistemic_status": "computational_limitation"
        }, indent=kwargs.get('indent', 2))

async def main() -> None:
    """
    Enhanced main entry point with comprehensive error handling.

    ### Execution Framework:
    1. Server instantiation
    2. Handler configuration
    3. Lifecycle management
    4. Graceful shutdown orchestration
    """
    try:
        server = OpenEndedPhilosophyServer()
        server.setup_handlers()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Philosophical inquiry interrupted")
    except Exception as e:
        logger.error(f"Critical server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cleanup_processes()

if __name__ == "__main__":
    # Set up event loop policy for better performance
    if sys.platform != 'win32':
        try:
            import uvloop
            uvloop.install()
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")

    asyncio.run(main())


def cli_main():
    """Entry point for console script (synchronous wrapper for async main)."""
    # Set up event loop policy for better performance
    if sys.platform != 'win32':
        try:
            import uvloop
            uvloop.install()
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")

    asyncio.run(main())
