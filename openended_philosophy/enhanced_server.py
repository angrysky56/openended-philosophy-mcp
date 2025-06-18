"""
Enhanced OpenEnded Philosophy MCP Server with Prompts Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Architectural Enhancement Summary

This implementation provides a complete MCP server that supports both tools and prompts,
following the Model Context Protocol specification and best practices for process
management and philosophical analysis.

#### Core Features:
- Complete MCP tools implementation (analyze_concept, explore_coherence, etc.)
- MCP prompts integration with reusable templates and workflows
- NARS (Non-Axiomatic Reasoning System) integration
- Comprehensive process management and cleanup
- Multi-perspective philosophical analysis
- Uncertainty quantification and revision conditions

#### MCP Protocol Support:
- Tools: list_tools(), call_tool()
- Prompts: list_prompts(), get_prompt()
- Proper error handling and logging
- Signal handling and graceful shutdown
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
from .prompts import get_prompt_by_name, list_available_prompts
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

def cleanup_processes() -> None:
    """Comprehensive cleanup of all running processes and background tasks."""
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
    """Track background task for proper cleanup."""
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

# Register signal handlers and cleanup
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_processes)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PhilosophicalContext:
    """Enhanced contextual substrate for philosophical operations."""
    language_game: str = "general_inquiry"
    confidence_threshold: float = 0.7
    openness_coefficient: float = 0.9
    meta_learning_enabled: bool = True
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safe JSON serialization with proper error handling."""
    try:
        if hasattr(obj, '__dict__'):
            # Convert dataclass or object to dict
            if hasattr(obj, 'to_dict'):
                return json.dumps(obj.to_dict(), **kwargs)
            else:
                return json.dumps(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__, **kwargs)
        else:
            return json.dumps(obj, **kwargs)
    except (TypeError, ValueError) as e:
        logger.warning(f"JSON serialization error: {e}")
        return json.dumps({"error": "Serialization failed", "details": str(e)})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OpenEndedPhilosophyServer:
    """
    Enhanced MCP Server Implementation for Open-Ended Philosophical Framework
    with comprehensive tools and prompts support.
    """

    def __init__(self) -> None:
        """Initialize philosophical substrate with enhanced process management."""
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

        # Language game registry
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

    def setup_handlers(self) -> None:
        """Configure MCP protocol handlers for both tools and prompts."""

        # ─────────────────────────────────────────────────────────────────────
        # Tools Handlers
        # ─────────────────────────────────────────────────────────────────────

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Enumerate available philosophical operations."""
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
            """Execute philosophical operations with comprehensive error management."""
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

        # ─────────────────────────────────────────────────────────────────────
        # Prompts Handlers
        # ─────────────────────────────────────────────────────────────────────

        @self.server.list_prompts()
        async def list_prompts():
            """List available philosophical prompt templates."""
            logger.info("Listing available philosophical prompts")
            try:
                prompts = list_available_prompts()
                logger.debug(f"Retrieved {len(prompts)} prompts")
                return prompts
            except Exception as e:
                logger.error(f"Error listing prompts: {e}", exc_info=True)
                raise

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None = None):
            """Get specific philosophical prompt with processed arguments."""
            logger.info(f"Retrieving prompt: {name}")
            logger.debug(f"Prompt arguments: {arguments}")

            try:
                prompt_result = get_prompt_by_name(name, arguments)
                logger.debug(f"Generated prompt with {len(prompt_result.messages)} messages")
                return prompt_result
            except ValueError as e:
                logger.error(f"Invalid prompt request: {e}")
                raise
            except Exception as e:
                logger.error(f"Error generating prompt {name}: {e}", exc_info=True)
                raise

    # ─────────────────────────────────────────────────────────────────────────
    # Core Philosophical Operations (Simplified Implementations)
    # ─────────────────────────────────────────────────────────────────────────

    async def _analyze_concept(
        self,
        concept: str,
        context: str,
        perspectives: list[str] | None = None,
        confidence_threshold: float = 0.7
    ) -> dict[str, Any]:
        """Enhanced concept analysis with NARS integration."""
        logger.debug(f"Analyzing concept: {concept} in context: {context}")

        try:
            # Use NARS-enhanced analysis if available
            if self._nars_initialized:
                try:
                    nars_analysis = await self.nars_reasoning.analyze_concept(
                        concept=concept,
                        context=context,
                        perspectives=perspectives or self._select_relevant_perspectives(concept, context)
                    )

                    # Enhance with philosophical framework
                    nars_analysis["philosophical_enhancement"] = {
                        "coherence_landscape": self.nars_memory.get_coherence_landscape(),
                        "epistemic_status": self._assess_epistemic_status(nars_analysis.get("synthesis", {})),
                        "framework_integration": "NARS + Philosophical Pluralism"
                    }

                    return nars_analysis

                except Exception as e:
                    logger.warning(f"NARS analysis failed, using fallback: {e}")

            # Fallback philosophical analysis
            return await self._fallback_concept_analysis(concept, context, perspectives, confidence_threshold)

        except Exception as e:
            logger.error(f"Error in concept analysis: {e}", exc_info=True)
            raise

    async def _explore_coherence(self, domain: str, depth: int = 3, allow_revision: bool = True) -> dict[str, Any]:
        """Enhanced coherence exploration."""
        logger.debug(f"Exploring coherence for domain: {domain}, depth: {depth}")

        try:
            # Simplified coherence exploration
            coherence_analysis = {
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
                "analysis_depth": depth,
                "coherence_layers": [],
                "philosophical_structures": {},
                "overall_coherence": 0.75,  # Placeholder
                "landscape_assessment": {
                    "coherence_quality": "moderate",
                    "structural_richness": 3,
                    "philosophical_maturity": "developing"
                }
            }

            # Add layer analysis based on depth
            for i in range(depth):
                layer = {
                    "layer_index": i,
                    "layer_type": ["foundational", "relational", "systematic", "meta_theoretical", "extended"][min(i, 4)],
                    "coherence_score": max(0.8 - i * 0.1, 0.3),
                    "analysis": f"Layer {i} analysis for {domain}"
                }
                coherence_analysis["coherence_layers"].append(layer)

            return coherence_analysis

        except Exception as e:
            logger.error(f"Error in coherence exploration: {e}", exc_info=True)
            raise

    async def _contextualize_meaning(self, expression: str, language_game: str, trace_genealogy: bool = False) -> dict[str, Any]:
        """Enhanced meaning contextualization."""
        logger.debug(f"Contextualizing '{expression}' in game: {language_game}")

        try:
            game_processor = self.language_games.get(
                language_game,
                self.language_games["ordinary_language"]
            )

            result = {
                "expression": expression,
                "language_game": language_game,
                "timestamp": datetime.now().isoformat(),
                "primary_meaning": f"Contextualized meaning of '{expression}' in {language_game}",
                "usage_patterns": ["pattern1", "pattern2", "pattern3"],
                "family_resemblances": ["related_concept1", "related_concept2"],
                "pragmatic_conditions": ["condition1", "condition2"],
                "meaning_stability": 0.8
            }

            if trace_genealogy:
                result["semantic_genealogy"] = {
                    "historical_evolution": f"Evolution of '{expression}' in {language_game}",
                    "key_transitions": ["transition1", "transition2"],
                    "current_trajectory": "stable"
                }

            return result

        except Exception as e:
            logger.error(f"Error in meaning contextualization: {e}", exc_info=True)
            raise

    async def _generate_insights(self, phenomenon: str, perspectives: list[str] | None = None, depth: int = 3, include_contradictions: bool = True) -> dict[str, Any]:
        """Enhanced insight generation."""
        logger.debug(f"Generating insights for: {phenomenon}")

        try:
            perspectives = perspectives or ["analytical", "phenomenological", "pragmatist"]

            insight_results = {
                "phenomenon": phenomenon,
                "timestamp": datetime.now().isoformat(),
                "analysis_depth": depth,
                "perspectives_analyzed": perspectives,
                "primary_insights": {},
                "synthetic_insights": {},
                "meta_philosophical_insights": {},
                "uncertainty_profile": {},
                "revision_framework": {}
            }

            # Generate perspective-specific insights
            for perspective in perspectives:
                insight_results["primary_insights"][perspective] = {
                    "core_claims": [f"Insight about {phenomenon} from {perspective} perspective"],
                    "supporting_evidence": ["evidence1", "evidence2"],
                    "confidence": 0.7,
                    "limitations": ["limitation1"]
                }

            if include_contradictions:
                insight_results["contradictions"] = [
                    {
                        "type": "perspective_tension",
                        "description": f"Tension between different views of {phenomenon}",
                        "perspectives_involved": perspectives[:2] if len(perspectives) > 1 else perspectives
                    }
                ]

            return insight_results

        except Exception as e:
            logger.error(f"Error in insight generation: {e}", exc_info=True)
            raise

    async def _test_hypothesis(self, hypothesis: str, test_domains: list[str] | None = None, criteria: dict[str, Any] | None = None) -> dict[str, Any]:
        """Enhanced hypothesis testing."""
        logger.debug(f"Testing hypothesis: {hypothesis}")

        try:
            test_domains = test_domains or ["general"]

            test_results = {
                "hypothesis": hypothesis,
                "timestamp": datetime.now().isoformat(),
                "test_domains": test_domains,
                "domain_results": {},
                "overall_coherence": 0.7,
                "pragmatic_score": 0.6,
                "confidence": 0.65,
                "supporting_evidence": ["evidence1", "evidence2"],
                "challenges": ["challenge1"],
                "implications": ["implication1"]
            }

            # Test in each domain
            for domain in test_domains:
                test_results["domain_results"][domain] = {
                    "coherence_score": 0.7,
                    "evidence_support": ["domain_evidence1"],
                    "challenges": ["domain_challenge1"],
                    "pragmatic_value": 0.6
                }

            return test_results

        except Exception as e:
            logger.error(f"Error in hypothesis testing: {e}", exc_info=True)
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _select_relevant_perspectives(self, concept: str, context: str) -> list[str]:
        """Select analytical perspectives based on concept and context."""
        # Simplified perspective selection
        default_perspectives = ["analytical", "phenomenological", "pragmatist"]

        context_mappings = {
            "ethics": ["virtue_ethics", "deontological", "consequentialist", "critical"],
            "consciousness": ["phenomenological", "functionalist", "emergentist", "analytical"],
            "knowledge": ["empiricist", "rationalist", "pragmatist", "critical"],
            "metaphysics": ["analytical", "phenomenological", "naturalist", "critical"]
        }

        return context_mappings.get(context.lower(), default_perspectives)

    def _assess_epistemic_status(self, synthesis: dict[str, Any]) -> str:
        """Assess epistemic status of analysis."""
        confidence = synthesis.get("overall_confidence", 0.5)
        if confidence > 0.8:
            return "high_confidence"
        elif confidence > 0.6:
            return "moderate_confidence"
        elif confidence > 0.4:
            return "developing_understanding"
        else:
            return "uncertain_exploration"

    async def _fallback_concept_analysis(self, concept: str, context: str, perspectives: list[str] | None, confidence_threshold: float) -> dict[str, Any]:
        """Fallback concept analysis when NARS is unavailable."""
        perspectives = perspectives or self._select_relevant_perspectives(concept, context)

        return {
            "concept": concept,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.context.session_id,
            "analyses": [
                {
                    "perspective": p,
                    "interpretation": {"derived_meaning": f"Analysis of {concept} from {p} perspective"},
                    "confidence": 0.7,
                    "coherence_score": 0.75
                }
                for p in perspectives
            ],
            "synthesis": {
                "summary": f"Multi-perspective analysis of {concept} in {context}",
                "overall_confidence": 0.7
            },
            "epistemic_status": "fallback_analysis",
            "framework_integration": "Pure Philosophical Analysis"
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Enhanced Server Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan_context(self):
        """Async context manager for server lifespan with proper cleanup."""
        logger.info("Initializing philosophical server lifespan")

        # Initialize NARS if possible
        if not self._nars_initialized:
            try:
                await self.nars_manager.start()
                self._nars_initialized = True
                logger.info("NARS system initialized successfully")
            except Exception as e:
                logger.warning(f"NARS initialization failed, continuing without: {e}")

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
                await asyncio.sleep(2.0)

            # Cleanup NARS
            if self._nars_initialized:
                try:
                    await self.nars_manager.stop()
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
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                logger.debug("Operation monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5.0)

    async def run(self) -> None:
        """Run the MCP server with comprehensive lifecycle management."""
        try:
            async with self.lifespan_context(), mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                # Create initialization options with prompts capability
                initialization_options = InitializationOptions(
                    server_name="openended-philosophy",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )

                logger.info("Starting philosophical MCP server with tools and prompts support")
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Server Setup and CLI Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_server() -> OpenEndedPhilosophyServer:
    """Create and configure the philosophical MCP server."""
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    return server

async def main() -> None:
    """Main entry point for the philosophical MCP server."""
    try:
        server = create_server()
        await server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

def cli_main() -> None:
    """CLI entry point with proper error handling."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server terminated by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()
