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
import dataclasses
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp.server.stdio
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from .core import (
    CoherenceLandscape,
    DynamicPluralismFramework,
    FallibilisticInference,
    LanguageGameProcessor,
)
from .lv_nars_integration import LVNARSIntegrationManager
from .nars import NARSManager, NARSMemory, NARSReasoning
from .operations import PhilosophicalOperations
from .prompts import get_prompt_by_name, list_available_prompts
from .utils import safe_json_dumps, setup_logging

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

        # Initialize philosophical operations using the separated operations module
        # NOTE: LV-NARS Integration is automatically included for ecosystem intelligence
        self.operations = PhilosophicalOperations(
            pluralism_framework=self.pluralism_framework,
            coherence_landscape=self.coherence_landscape,
            inference_engine=self.inference_engine,
            language_games=self.language_games,
            nars_manager=self.nars_manager,
            nars_memory=self.nars_memory,
            nars_reasoning=self.nars_reasoning
        )

        logger.info(f"OpenEnded Philosophy Server initialized with LV-NARS integration, session: {self.context.session_id}")
        logger.debug(f"Full philosophical context: {safe_json_dumps(dataclasses.asdict(self.context))}")

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
                ),
                Tool(
                    name="recursive_self_analysis",
                    description=(
                        "Apply the system's own analytical tools recursively to examine "
                        "its reasoning processes and generate meta-philosophical insights "
                        "about its own operations and effectiveness."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_result": {
                                "type": "object",
                                "description": "Previous analysis result to examine recursively"
                            },
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis performed",
                                "enum": ["concept_analysis", "coherence_exploration", "insight_generation", "hypothesis_testing"]
                            },
                            "meta_depth": {
                                "type": "integer",
                                "description": "Depth of recursive analysis (1-3)",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 3
                            }
                        },
                        "required": ["analysis_result", "analysis_type"]
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
                        return await self.operations.analyze_concept(**arguments)
                    elif name == "explore_coherence":
                        return await self.operations.explore_coherence(**arguments)
                    elif name == "contextualize_meaning":
                        return await self.operations.contextualize_meaning(**arguments)
                    elif name == "generate_insights":
                        return await self.operations.generate_insights(**arguments)
                    elif name == "test_philosophical_hypothesis":
                        return await self.operations.test_philosophical_hypothesis(**arguments)
                    elif name == "recursive_self_analysis":
                        return await self.operations.analyze_own_reasoning_process(**arguments)
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
    # Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan_context(self):
        """
        Manage server lifecycle with proper initialization and cleanup.

        ### Lifecycle Protocol:
        1. Initialize NARS system
        2. Start background monitoring
        3. Yield control to server
        4. Cleanup on shutdown
        """
        logger.info("Initializing philosophical server lifecycle")
        memory_file = Path("philosophy_nars_memory.json")

        try:
            # Initialize NARS system
            if await self.nars_manager.start():
                self._nars_initialized = True
                logger.info("NARS system initialized successfully")
            else:
                logger.warning("NARS initialization failed - continuing without NARS")

            # Load NARS memory if available
            if memory_file.exists():
                self.nars_memory.load(memory_file)
                logger.info(f"Loaded NARS memory from {memory_file}")

            # Start background monitoring
            monitor_task = asyncio.create_task(self._monitor_operations())
            track_background_task(monitor_task)

            yield

        finally:
            logger.info("Beginning philosophical server shutdown")

            # Wait for active operations to complete (with timeout)
            if self._active_operations:
                logger.info(f"Waiting for {len(self._active_operations)} active operations to complete")
                await asyncio.sleep(2.0)  # Grace period

            # Cleanup NARS
            if self._nars_initialized:
                try:
                    await self.nars_manager.stop()
                    # Save NARS memory
                    self.nars_memory.save(memory_file)
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
            @asynccontextmanager
            async def stdio_server_context():
                try:
                    yield mcp.server.stdio.stdio_server()
                finally:
                    pass

            async with (
                self.lifespan_context(),
                stdio_server_context() as streams,
                streams as (read_stream, write_stream)
            ):
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
        finally:
            cleanup_processes()


async def main() -> None:
    """
    Main entry point for the OpenEndedPhilosophyServer.
    Initializes and runs the server instance.
    """
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    await server.run()


def cli_main() -> None:
    """
    Command-line entry point for the openended-philosophy-server.

    This function serves as the synchronous entry point that can be called
    from the command line via the pyproject.toml script configuration.
    """
    # Set up event loop policy for better performance
    if sys.platform != 'win32':
        try:
            import uvloop
            uvloop.install()
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")

    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
