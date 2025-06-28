"""
Enhanced Semantic Types for Philosophical Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Conceptual Framework Deconstruction

This module defines the type system for enhanced semantic processing, embodying:

#### Core Theoretical Foundations:
- **Structured Semantic Representation**: Precise typing for philosophical concepts
- **Context-Dependent Analysis**: Language game and domain awareness
- **Uncertainty Quantification**: Explicit epistemic limitations modeling
- **Multi-Perspectival Architecture**: Support for diverse interpretive frameworks

#### Epistemological Assumptions:
- Concepts exist within interpretive contexts rather than independently
- Semantic relationships are probabilistic and subject to revision
- Multiple valid categorizations can coexist without contradiction
- Understanding emerges through systematic analysis rather than intuitive grasp

### Methodological Approach

The type system employs:
1. **Hierarchical Concept Organization**: Domain-specific ontological structures
2. **Relational Semantic Modeling**: Explicit relationship representation
3. **Context-Aware Processing**: Domain and language game sensitivity
4. **Uncertainty Integration**: Built-in epistemic limitation acknowledgment
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4


class PhilosophicalDomain(Enum):
    """Core philosophical domains for categorization."""
    METAPHYSICS = "metaphysics"
    EPISTEMOLOGY = "epistemology"
    ETHICS = "ethics"
    AESTHETICS = "aesthetics"
    LOGIC = "logic"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    PHILOSOPHY_OF_SCIENCE = "philosophy_of_science"
    POLITICAL_PHILOSOPHY = "political_philosophy"
    PHILOSOPHY_OF_LANGUAGE = "philosophy_of_language"
    PHENOMENOLOGY = "phenomenology"
    EXISTENTIALISM = "existentialism"
    PRAGMATISM = "pragmatism"
    ANALYTIC_PHILOSOPHY = "analytic_philosophy"
    CONTINENTAL_PHILOSOPHY = "continental_philosophy"
    APPLIED_PHILOSOPHY = "applied_philosophy"


class SemanticRelationType(Enum):
    """Types of semantic relationships between concepts."""
    CAUSAL = "causal"
    LOGICAL_IMPLICATION = "logical_implication"
    PART_WHOLE = "part_whole"
    SIMILARITY = "similarity"
    OPPOSITION = "opposition"
    DEPENDENCY = "dependency"
    INSTANTIATION = "instantiation"
    CATEGORY_MEMBERSHIP = "category_membership"
    TEMPORAL_SUCCESSION = "temporal_succession"
    PRAGMATIC_CONSEQUENCE = "pragmatic_consequence"
    HERMENEUTIC_CIRCLE = "hermeneutic_circle"
    DIALECTICAL_TENSION = "dialectical_tension"


class LanguageGame(Enum):
    """Wittgensteinian language games for contextual analysis."""
    SCIENTIFIC_DISCOURSE = "scientific_discourse"
    ETHICAL_DELIBERATION = "ethical_deliberation"
    AESTHETIC_JUDGMENT = "aesthetic_judgment"
    ORDINARY_LANGUAGE = "ordinary_language"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    RELIGIOUS_DISCOURSE = "religious_discourse"
    LEGAL_REASONING = "legal_reasoning"
    THERAPEUTIC_DIALOGUE = "therapeutic_dialogue"
    CRITICAL_ANALYSIS = "critical_analysis"
    HERMENEUTIC_INTERPRETATION = "hermeneutic_interpretation"


@dataclass
class PhilosophicalContext:
    """
    Comprehensive contextual framework for philosophical analysis.

    ### Architectural Components:
    - Domain specification and interdisciplinary connections
    - Language game identification and methodological constraints
    - Temporal and cultural situatedness
    - Epistemic uncertainty acknowledgment
    """
    domain: PhilosophicalDomain
    language_game: LanguageGame = LanguageGame.ORDINARY_LANGUAGE
    inquiry_type: str = "conceptual_analysis"
    depth_requirements: int = 3
    perspective_constraints: list[str] | None = None
    temporal_context: str | None = None
    cultural_context: str | None = None
    interdisciplinary_connections: list[str] = field(default_factory=list)
    methodological_preferences: dict[str, Any] = field(default_factory=dict)
    epistemic_constraints: dict[str, float] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PhilosophicalConcept:
    """
    Enhanced representation of philosophical concepts with systematic categorization.

    ### Conceptual Framework:
    - Multi-dimensional concept representation
    - Domain-specific attribute mapping
    - Relational structure encoding
    - Uncertainty quantification integration
    """
    term: str
    domain: PhilosophicalDomain
    definition: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    relations: list['SemanticRelation'] = field(default_factory=list)
    context_dependencies: list[str] = field(default_factory=list)
    alternative_formulations: list[str] = field(default_factory=list)
    philosophical_tradition: str | None = None
    historical_development: list[str] = field(default_factory=list)
    contemporary_debates: list[str] = field(default_factory=list)
    epistemic_status: str = "provisional"
    confidence_level: float = 0.7
    revision_conditions: list[str] = field(default_factory=list)
    concept_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class SemanticRelation:
    """
    Sophisticated semantic relationship representation.

    ### Methodological Approach:
    - Explicit relationship typing and strength quantification
    - Bidirectional relationship awareness
    - Context-dependent relationship expression
    - Temporal dynamics and revision tracking
    """
    source_concept: str
    target_concept: str
    relation_type: SemanticRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context_dependent: bool = True
    bidirectional: bool = False
    temporal_stability: float = 0.8
    supporting_evidence: list[str] = field(default_factory=list)
    counterevidence: list[str] = field(default_factory=list)
    philosophical_justification: str | None = None
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    relation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PhilosophicalCategory:
    """
    Comprehensive philosophical categorization with hierarchical structure.

    ### Critical Perspective Integration:
    - Multi-level categorization hierarchy
    - Cross-cutting thematic identification
    - Complexity and interdisciplinary assessment
    - Alternative categorization acknowledgment
    """
    primary: PhilosophicalDomain
    secondary: list[PhilosophicalDomain] = field(default_factory=list)
    cross_cutting: list[str] = field(default_factory=list)
    complexity_level: int = 3  # 1-5 scale
    interdisciplinary_connections: list[str] = field(default_factory=list)
    alternative_categorizations: list[dict[str, Any]] = field(default_factory=list)
    categorical_confidence: float = 0.8
    justification: str | None = None
    categorical_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary": self.primary.value,
            "secondary": [d.value for d in self.secondary],
            "cross_cutting": self.cross_cutting,
            "complexity_level": self.complexity_level,
            "interdisciplinary_connections": self.interdisciplinary_connections,
            "alternative_categorizations": self.alternative_categorizations,
            "categorical_confidence": self.categorical_confidence,
            "justification": self.justification,
            "categorical_limitations": self.categorical_limitations
        }


@dataclass
class SemanticAnalysis:
    """
    Comprehensive semantic analysis results with philosophical sophistication.

    ### Argumentative Integrity Analysis:
    - Multi-dimensional concept extraction and relationship mapping
    - Pragmatic implication identification and assessment
    - Epistemic uncertainty systematic quantification
    - Context dependency explicit acknowledgment
    """
    primary_concepts: list[PhilosophicalConcept]
    semantic_relations: list[SemanticRelation]
    pragmatic_implications: list[str]
    epistemic_uncertainty: dict[str, float]
    context_dependencies: list[str]
    revision_triggers: list[str]
    philosophical_presuppositions: list[str] = field(default_factory=list)
    methodological_assumptions: list[str] = field(default_factory=list)
    interpretive_alternatives: list[str] = field(default_factory=list)
    analytical_limitations: list[str] = field(default_factory=list)
    confidence_intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "primary_concepts": [
                {
                    "term": c.term,
                    "domain": c.domain.value,
                    "definition": c.definition,
                    "attributes": c.attributes,
                    "philosophical_tradition": c.philosophical_tradition,
                    "epistemic_status": c.epistemic_status,
                    "confidence_level": c.confidence_level
                }
                for c in self.primary_concepts
            ],
            "semantic_relations": [
                {
                    "source": r.source_concept,
                    "target": r.target_concept,
                    "type": r.relation_type.value,
                    "strength": r.strength,
                    "confidence": r.confidence,
                    "philosophical_justification": r.philosophical_justification
                }
                for r in self.semantic_relations
            ],
            "pragmatic_implications": self.pragmatic_implications,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "context_dependencies": self.context_dependencies,
            "revision_triggers": self.revision_triggers,
            "philosophical_presuppositions": self.philosophical_presuppositions,
            "methodological_assumptions": self.methodological_assumptions,
            "interpretive_alternatives": self.interpretive_alternatives,
            "analytical_limitations": self.analytical_limitations,
            "confidence_intervals": self.confidence_intervals,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "analysis_id": self.analysis_id
        }


class ConceptExtractor(Protocol):
    """
    Protocol for philosophical concept extraction implementations.

    ### Methodological Critique:
    - Domain-specific extraction strategies
    - Context-aware concept identification
    - Uncertainty quantification integration
    - Multi-perspectival validation support
    """

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """
        Extract philosophical concepts from statement within context.

        Args:
            statement: Text to analyze for concept extraction
            context: Philosophical context for interpretation

        Returns:
            List of identified philosophical concepts with categorization
        """
        ...

    def get_domain_focus(self) -> PhilosophicalDomain:
        """Return the primary philosophical domain focus."""
        ...

    def get_extraction_confidence(self) -> float:
        """Return confidence level for extraction capabilities."""
        ...


@dataclass
class ExtractedConcept:
    """Lightweight concept representation for extraction results."""
    term: str
    domain: PhilosophicalDomain
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    context_indicators: list[str] = field(default_factory=list)


@dataclass
class ConceptExtractionResult:
    """Results from concept extraction with analytical metadata."""
    extracted_concepts: list[ExtractedConcept]
    extraction_method: str
    context_applied: PhilosophicalContext
    confidence_distribution: dict[str, float]
    methodological_notes: list[str] = field(default_factory=list)
    extraction_limitations: list[str] = field(default_factory=list)
    alternative_extractions: list[dict[str, Any]] = field(default_factory=list)


# Type aliases for complex composite types
ConceptNetwork = dict[str, list[SemanticRelation]]
SemanticEmbedding = list[float]
ConceptSimilarityMatrix = dict[str, dict[str, float]]
PhilosophicalAnalysisResult = dict[str, Any]

# Utility types for enhanced processing
@dataclass
class AnalysisQualityMetrics:
    """Quality metrics for philosophical analysis assessment."""
    conceptual_coverage: float
    relational_coherence: float
    epistemic_appropriateness: float
    methodological_rigor: float
    interpretive_depth: float
    overall_quality: float
    quality_dimensions: dict[str, float] = field(default_factory=dict)
    improvement_recommendations: list[str] = field(default_factory=list)
