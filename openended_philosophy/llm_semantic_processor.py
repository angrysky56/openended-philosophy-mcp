import logging
from typing import Any, Literal
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class LLMSemanticProcessor:
    """
    Structures prompts and responses for semantic analysis within MCP architecture.

    In the MCP (Model Context Protocol) architecture, the LLM capabilities are provided
    by the client (like Claude Desktop) rather than direct API calls. This processor
    formats requests and structures responses for the MCP client to process.
    """
    def __init__(self, llm_model_name: str = "mcp-client", api_key: str | None = None):
        """
        Initializes the LLM Semantic Processor for MCP architecture.

        Args:
            llm_model_name: Model identifier (not used in MCP - client handles model selection)
            api_key: Not used in MCP architecture
        """
        self.llm_model_name = llm_model_name
        # In MCP, the client handles LLM interactions
        logger.info("LLMSemanticProcessor initialized for MCP client processing")

        # Initialize Jinja2 environment to load templates from the 'prompts' directory
        self.env = Environment(loader=FileSystemLoader('prompts'))

    def _craft_llm_prompt(self, task_type: Literal["interpret_concept_in_game", "generate_insight", "generate_usage_patterns", "assess_alignment", "evaluate_criteria", "assess_resonance", "synthesize_analysis", "identify_contradictions", "generate_meta_insights"], **kwargs: Any) -> str:
        """
        Crafts a detailed, philosophically-informed prompt for the LLM based on the task type.
        """
        template_name = f"{task_type}.md"
        try:
            template = self.env.get_template(template_name)
            prompt = template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error loading or rendering template {template_name}: {e}")
            prompt = f"Perform a general philosophical analysis based on: {kwargs}"
        return prompt

    def _get_expected_response_format(self, task_type: str) -> dict[str, Any]:
        """
        Defines the expected JSON response format for each task type.

        Args:
            task_type: The type of philosophical analysis task

        Returns:
            Expected response format specification
        """
        formats = {
            "interpret_concept_in_game": {
                "type": "object",
                "required_keys": ["primary_sense", "connotations", "semantic_stability", "confidence"],
                "description": "Analysis of concept meaning within a specific language game context"
            },
            "generate_insight": {
                "type": "object",
                "required_keys": ["insight_content", "confidence"],
                "description": "Philosophical insight from a specific perspective"
            },
            "generate_usage_patterns": {
                "type": "array",
                "item_format": {"pattern": "str", "frequency": "float", "contexts": "array"},
                "description": "Usage patterns for expression within language game"
            },
            "assess_alignment": {
                "type": "object",
                "format": "commitment_name -> alignment_score (0.0-1.0)",
                "description": "Alignment scores for philosophical commitments"
            },
            "evaluate_criteria": {
                "type": "object",
                "format": "criterion_name -> evaluation_score (0.0-1.0)",
                "description": "Evaluation scores against philosophical criteria"
            },
            "assess_resonance": {
                "type": "object",
                "format": "priority_name -> resonance_score (0.0-1.0)",
                "description": "Resonance scores with conceptual priorities"
            },
            "synthesize_analysis": {
                "type": "object",
                "required_keys": ["summary", "common_themes", "emergent_insights", "overall_coherence_score", "confidence"],
                "description": "Synthesis of multiple philosophical analyses"
            },
            "identify_contradictions": {
                "type": "array",
                "item_format": {"description": "str", "involved_insights": "array"},
                "description": "Identified contradictions between insights"
            },
            "generate_meta_insights": {
                "type": "array",
                "item_type": "string",
                "description": "Meta-level insights about the analysis process"
            }
        }
        return formats.get(task_type, {"type": "object", "description": "General philosophical analysis response"})

    async def perform_semantic_analysis_with_llm(
        self,
        task_type: Literal["interpret_concept_in_game", "generate_insight", "generate_usage_patterns", "assess_alignment", "evaluate_criteria", "assess_resonance", "synthesize_analysis", "identify_contradictions", "generate_meta_insights"],
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Structures a semantic analysis request for the MCP client to process.

        In MCP architecture, this method prepares the prompt and provides structure
        for the response, but the actual LLM processing is handled by the client.

        Args:
            task_type: The type of philosophical analysis task to perform.
            **kwargs: Arguments specific to the task type.

        Returns:
            A structured prompt and response format for the MCP client.
        """
        prompt = self._craft_llm_prompt(task_type, **kwargs)
        logger.debug(f"Prepared prompt for MCP client - task: '{task_type}'")

        # In MCP architecture, return the structured prompt for client processing
        # The MCP client will handle the actual LLM interaction
        return {
            "prompt": prompt,
            "task_type": task_type,
            "expected_format": self._get_expected_response_format(task_type),
            "processing_instructions": "This prompt should be processed by the MCP client's LLM capabilities",
            "kwargs": kwargs
        }

    async def analyze_statement(
        self,
        statement: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Analyze a philosophical statement within given context.

        Args:
            statement: The statement to analyze
            context: Optional contextual information

        Returns:
            Structured analysis for MCP client processing
        """
        return await self.perform_semantic_analysis_with_llm(
            "interpret_concept_in_game",
            concept=statement,
            language_game=context.get("language_game", "philosophical_inquiry") if context else "philosophical_inquiry",
            context=context or {}
        )

    async def generate_philosophical_insight(
        self,
        phenomenon: str,
        perspective: str,
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate a philosophical insight about a phenomenon from a specific perspective.

        Args:
            phenomenon: The phenomenon to analyze
            perspective: The philosophical perspective to apply
            context: Optional contextual information

        Returns:
            Structured insight for MCP client processing
        """
        return await self.perform_semantic_analysis_with_llm(
            "generate_insight",
            phenomenon=phenomenon,
            perspective=perspective,
            context=context or {}
        )

    async def synthesize_perspectives(
        self,
        analyses: list[dict[str, Any]],
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Synthesize multiple philosophical analyses into a coherent view.

        Args:
            analyses: List of philosophical analyses to synthesize
            context: Optional contextual information

        Returns:
            Structured synthesis for MCP client processing
        """
        return await self.perform_semantic_analysis_with_llm(
            "synthesize_analysis",
            analyses=analyses,
            context=context or {}
        )

