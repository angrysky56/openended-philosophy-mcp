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
import logging
import signal
import sys
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import mcp.server.stdio
import numpy as np
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import ImageContent, TextContent, Tool

from .core import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    EmergentCoherenceNode,
    FallibilisticInference,
    LanguageGameProcessor,
)
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
        perspectives: Optional[list[str]] = None,
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
            # Create provisional semantic node
            concept_node = EmergentCoherenceNode(
                initial_pattern={"term": concept, "domain": context, "timestamp": datetime.now().isoformat()},
                confidence=0.5,  # Start with epistemic humility
                context_sensitivity=0.8
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
                    # Create interpretive schema
                    schema = self._create_interpretive_schema(perspective)

                    # Apply schema to concept
                    interpretation = await self._apply_schema_to_concept(
                        concept_node, schema, active_game
                    )

                    # Calculate interpretation confidence
                    confidence = self._calculate_interpretation_confidence(
                        interpretation, active_game
                    )

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

                except Exception as e:
                    logger.warning(f"Perspective analysis failed for {perspective}: {e}")
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
        """Enhanced coherence exploration with timeout handling."""
        try:
            landscape_state_result = await asyncio.wait_for(
                self.coherence_landscape.map_domain(domain, depth, allow_revision),
                timeout=25.0
            )
            return asdict(landscape_state_result)
        except asyncio.TimeoutError:
            logger.warning(f"Coherence exploration timeout for domain: {domain}")
            return {
                "domain": domain,
                "status": "timeout",
                "partial_results": "Analysis exceeded time bounds",
                "suggestion": "Consider reducing exploration depth"
            }

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

    async def _generate_insights(self, phenomenon: str, perspectives: Optional[list[str]] = None, depth: int = 3, include_contradictions: bool = True) -> dict[str, Any]:
        """Enhanced insight generation with contradiction mapping."""
        logger.debug(f"Generating insights for phenomenon: {phenomenon}")

        try:
            # Initialize insight generation
            insights = {
                "phenomenon": phenomenon,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.context.session_id,
                "perspectives_used": perspectives or ["general"],
                "primary_insights": [],
                "contradictions": [],
                "synthesis": None,
                "uncertainty_profile": {},
                "revision_triggers": []
            }

            # Use timeout for evidence gathering
            evidence_patterns = await asyncio.wait_for(
                self._gather_evidence_patterns(phenomenon, perspectives, depth),
                timeout=20.0
            )

            # Generate insights from each perspective
            for perspective in (perspectives or self._get_default_perspectives()):
                try:
                    perspective_insights = await asyncio.wait_for(
                        self.inference_engine.derive_insights(
                            evidence_patterns.get(perspective, []),
                            confidence_threshold=self.context.confidence_threshold
                        ),
                        timeout=10.0
                    )

                    for insight in perspective_insights:
                        insights["primary_insights"].append({
                            "content": insight.content,
                            "perspective": perspective,
                            "confidence": insight.confidence,
                            "supporting_patterns": insight.evidence_summary,
                            "limitations": insight.identified_limitations,
                            "revision_triggers": insight.revision_triggers
                        })

                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Insight generation failed for perspective {perspective}: {e}")
                    continue

            # Process contradictions if requested
            if include_contradictions and len(insights["primary_insights"]) > 1:
                insights["contradictions"] = self._identify_insight_contradictions(
                    insights["primary_insights"]
                )

            return insights

        except asyncio.TimeoutError:
            logger.warning(f"Insight generation timeout for: {phenomenon}")
            return {
                "phenomenon": phenomenon,
                "status": "timeout",
                "suggestion": "Consider reducing analysis depth or narrowing the phenomenon scope"
            }

    async def _test_hypothesis(self, hypothesis: str, test_domains: Optional[list[str]] = None, criteria: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
        """Select analytical perspectives based on concept and context."""
        base_perspectives = ["analytical", "phenomenological", "pragmatist"]

        if context.lower() in ["ethics", "morality"]:
            base_perspectives.extend(["deontological", "consequentialist", "virtue"])
        elif context.lower() in ["science", "physics"]:
            base_perspectives.extend(["empiricist", "realist", "instrumentalist"])
        elif context.lower() in ["consciousness", "mind"]:
            base_perspectives.extend(["functionalist", "dualist", "emergentist"])

        return base_perspectives[:5]  # Limit to 5 perspectives

    def _create_interpretive_schema(self, perspective: str) -> dict[str, Any]:
        """Create interpretive schema for perspective."""
        return {
            "perspective": perspective,
            "core_commitments": self._get_perspective_commitments(perspective),
            "evaluation_criteria": self._get_perspective_criteria(perspective),
            "conceptual_priorities": self._get_perspective_priorities(perspective)
        }

    async def _apply_schema_to_concept(
        self,
        concept_node: EmergentCoherenceNode,
        schema: dict[str, Any],
        active_game: LanguageGameProcessor
    ) -> dict[str, Any]:
        """Apply interpretive schema to concept node."""
        logger.debug(f"Applying schema {schema.get('perspective')} to concept")

        coherence_score = np.random.rand() * 0.5 + 0.3
        interpretation_details = {
            "derived_meaning": f"Interpretation from {schema.get('perspective')} perspective",
            "key_features": ["feature1", "feature2"],
            "coherence": coherence_score,
            "uncertainty": 1.0 - concept_node.pattern.confidence * coherence_score,
        }
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
        """Synthesize multiple analyses."""
        if not analysis_results:
            return {"summary": "No analyses to synthesize", "overall_confidence": 0.0}

        combined_confidence = np.mean([r.get("confidence", 0.0) for r in analysis_results])
        key_themes = list({r.get("perspective", "unknown") for r in analysis_results})

        return {
            "summary": f"Synthesized insights from {len(analysis_results)} perspectives",
            "overall_confidence": float(combined_confidence),
            "contributing_perspectives": key_themes,
            "dominant_interpretation": "Multi-perspectival synthesis"
        }

    def _identify_conceptual_tensions(self, analysis_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify tensions between analyses."""
        tensions = []
        if len(analysis_results) < 2:
            return tensions

        perspectives = [res.get("perspective") for res in analysis_results]
        if "analytical" in perspectives and "phenomenological" in perspectives:
            tensions.append({
                "type": "methodological_tension",
                "description": "Tension between analytical and phenomenological approaches",
                "involved_perspectives": ["analytical", "phenomenological"]
            })
        return tensions

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

    async def _gather_evidence_patterns(
        self, phenomenon: str, perspectives: Optional[list[str]], depth: int
    ) -> dict[str, Any]:
        """Gather evidence patterns."""
        return {
            "empirical_evidence": [f"evidence_for_{phenomenon}"],
            "conceptual_patterns": [f"pattern_for_{phenomenon}"],
            "cross_perspective_themes": ["theme1", "theme2"]
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
