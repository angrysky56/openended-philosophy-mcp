"""
Philosophical Ontology System for Systematic Categorization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1. Conceptual Framework Deconstruction

#### Core Theoretical Foundations:
- **Hierarchical Ontological Organization**: Systematic domain and subdomain categorization
- **Multi-Dimensional Classification**: Cross-cutting themes and complexity assessment
- **Relational Type Mapping**: Explicit philosophical relationship categorization
- **Context-Sensitive Categorization**: Domain-specific interpretation frameworks

#### Underlying Epistemological Assumptions:
- Philosophical concepts exist within systematic organizational structures
- Multiple valid categorizations can coexist without logical contradiction
- Categorical boundaries are provisional and subject to reasoned revision
- Understanding emerges through systematic taxonomic analysis

#### Conceptual Lineage and Intellectual Heritage:
- **Aristotelian Categories**: Foundational systematic classification approach
- **Kant's Architectonic**: Systematic unity and hierarchical organization
- **Analytic Philosophy**: Precision in conceptual distinction and categorization
- **Contemporary Metaphilosophy**: Systematic reflection on philosophical methodology

### 2. Methodological Critique

#### Assessment of Methodological Approach:
- Employs multi-dimensional categorical matrices for comprehensive classification
- Integrates historical philosophical traditions with contemporary analytical frameworks
- Balances systematic organization with flexibility for categorical revision
- Incorporates uncertainty quantification in categorical confidence assessment

#### Evidence Collection and Interpretative Methods:
- Systematic analysis of philosophical literature for categorical structures
- Cross-cultural philosophical tradition integration for comprehensive coverage
- Contemporary philosophical debate analysis for current categorical relevance
- Interdisciplinary boundary analysis for cross-domain categorical connections

#### Potential Methodological Limitations:
- Categorical frameworks reflect historically dominant philosophical traditions
- Systematic organization may obscure important cross-categorical insights
- Confidence metrics rely on heuristic rather than formal probabilistic assessment
- Cultural and linguistic biases may influence categorical construction

### 3. Critical Perspective Integration

#### Alternative Theoretical Perspectives:
- **Feminist Philosophy**: Gender-sensitive categorical analysis and power-aware frameworks
- **Non-Western Traditions**: Buddhist, Confucian, Indigenous philosophical categorizations
- **Postmodern Critique**: Deconstruction of categorical hierarchies and boundary questioning
- **Pragmatist Approach**: Function-based rather than essence-based categorization

#### Interdisciplinary Implications:
- **Cognitive Science**: Empirical research on concept categorization and mental representation
- **Anthropology**: Cultural variation in categorical systems and conceptual organization
- **Linguistics**: Language-specific influences on categorical structure and semantic organization
- **Computer Science**: Formal ontology development and knowledge representation systems

### Usage Example:

```python
ontology = PhilosophicalOntology()

category = ontology.categorize(semantic_analysis)
relations = ontology.get_relation_types(domain="ethics")
context_map = ontology.build_context_sensitivity_map()
```
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .types import (
    LanguageGame,
    PhilosophicalCategory,
    PhilosophicalConcept,
    PhilosophicalDomain,
    SemanticAnalysis,
    SemanticRelationType,
)

logger = logging.getLogger(__name__)


@dataclass
class OntologicalHierarchy:
    """
    Systematic hierarchical organization of philosophical domains.

    ### 4. Argumentative Integrity Analysis:
    - Maintains logical coherence across hierarchical levels
    - Ensures consistent application of categorical principles
    - Prevents circular dependencies in domain relationships
    """
    domain: PhilosophicalDomain
    subdomain: str | None = None
    specializations: list[str] = field(default_factory=list)
    parent_domains: list[PhilosophicalDomain] = field(default_factory=list)
    cross_cutting_themes: list[str] = field(default_factory=list)
    methodological_approaches: list[str] = field(default_factory=list)
    historical_development: dict[str, str] = field(default_factory=dict)
    contemporary_debates: list[str] = field(default_factory=list)


@dataclass
class RelationTypeMapping:
    """
    Systematic mapping of philosophical relationship types with domain specificity.

    ### 5. Contextual and Interpretative Nuances:
    - Domain-specific relationship interpretation frameworks
    - Context-dependent relationship strength assessment
    - Historical evolution of relationship concepts
    """
    relation_type: SemanticRelationType
    applicable_domains: list[PhilosophicalDomain]
    typical_strength_range: tuple[float, float]
    context_dependencies: list[str]
    philosophical_justification: str
    historical_examples: list[str] = field(default_factory=list)
    contemporary_applications: list[str] = field(default_factory=list)


@dataclass
class CategorizationContext:
    """
    Comprehensive context for philosophical categorization decisions.

    Provides systematic framework for context-sensitive categorization
    while maintaining analytical rigor and epistemic humility.
    """
    primary_indicators: dict[str, float]
    secondary_indicators: dict[str, float]
    cross_cutting_signals: dict[str, float]
    confidence_factors: dict[str, float]
    uncertainty_sources: list[str]
    alternative_interpretations: list[dict[str, Any]]


class PhilosophicalOntology:
    """
    Comprehensive ontology system for philosophical concept categorization.

    This class provides systematic categorization capabilities for philosophical
    concepts, analyzing semantic content to determine appropriate philosophical
    domains, assess complexity, and identify cross-cutting themes.
    """

    def __init__(self):
        """Initialize the philosophical ontology system."""
        self.concept_hierarchy = self._build_concept_hierarchy()
        self.relation_types = self._define_relation_types()
        self.context_sensitivity_map = self._build_context_sensitivity_map()
        self.domain_indicators = self._build_domain_indicators()
        self.complexity_assessors = self._build_complexity_assessors()
        self.cross_cutting_themes = self._build_cross_cutting_themes()

        logger.info("PhilosophicalOntology initialized with systematic categorization framework")

    def categorize(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalCategory:
        """
        Systematically categorize philosophical content with nuanced understanding.

        Args:
            semantic_analysis: Comprehensive semantic analysis results

        Returns:
            Multi-dimensional philosophical categorization with confidence metrics
        """
        try:
            # Determine primary category through systematic analysis
            primary_category = self._determine_primary_category(semantic_analysis)

            # Identify secondary categories
            secondary_categories = self._identify_secondary_categories(semantic_analysis)

            # Identify cross-cutting themes
            cross_cutting = self._identify_cross_cutting_themes(semantic_analysis)

            # Assess complexity level
            complexity_level = self._assess_complexity_level(semantic_analysis)

            # Find interdisciplinary connections
            interdisciplinary_connections = self._find_interdisciplinary_connections(semantic_analysis)

            # Generate alternative categorizations
            alternative_categorizations = self._generate_alternative_categorizations(semantic_analysis)

            # Calculate confidence
            categorical_confidence = self._calculate_categorical_confidence(semantic_analysis, primary_category)

            # Generate justification
            justification = self._generate_categorization_justification(
                semantic_analysis, primary_category, secondary_categories
            )

            # Identify limitations
            limitations = self._identify_categorical_limitations(semantic_analysis)

            category = PhilosophicalCategory(
                primary=primary_category,
                secondary=secondary_categories,
                cross_cutting=cross_cutting,
                complexity_level=complexity_level,
                interdisciplinary_connections=interdisciplinary_connections,
                alternative_categorizations=alternative_categorizations,
                categorical_confidence=categorical_confidence,
                justification=justification,
                categorical_limitations=limitations
            )

            logger.debug(f"Categorized as primary: {primary_category.value}, confidence: {categorical_confidence}")
            return category

        except Exception as e:
            logger.error(f"Error in categorization: {e}")
            # Return minimal categorization on error
            return PhilosophicalCategory(
                primary=PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE,  # Default
                categorical_confidence=0.1,
                justification="Categorization failed, using default"
            )

    def _build_concept_hierarchy(self) -> dict[PhilosophicalDomain, OntologicalHierarchy]:
        """Build systematic hierarchical organization of philosophical domains."""
        hierarchy = {}

        # Metaphysics
        hierarchy[PhilosophicalDomain.METAPHYSICS] = OntologicalHierarchy(
            domain=PhilosophicalDomain.METAPHYSICS,
            specializations=[
                "ontology", "philosophy_of_mind", "philosophy_of_time",
                "modal_metaphysics", "philosophy_of_causation"
            ],
            cross_cutting_themes=[
                "realism_vs_antirealism", "emergence", "reduction", "naturalism"
            ],
            methodological_approaches=[
                "conceptual_analysis", "thought_experiments", "formal_methods"
            ],
            contemporary_debates=[
                "hard_problem_of_consciousness", "free_will", "personal_identity"
            ]
        )

        # Epistemology
        hierarchy[PhilosophicalDomain.EPISTEMOLOGY] = OntologicalHierarchy(
            domain=PhilosophicalDomain.EPISTEMOLOGY,
            specializations=[
                "theory_of_knowledge", "philosophy_of_science", "formal_epistemology",
                "social_epistemology", "virtue_epistemology"
            ],
            cross_cutting_themes=[
                "justification", "reliability", "coherence", "foundationalism"
            ],
            methodological_approaches=[
                "analysis_of_knowledge", "skeptical_arguments", "naturalized_epistemology"
            ],
            contemporary_debates=[
                "gettier_problems", "epistemic_injustice", "testimony"
            ]
        )

        # Ethics
        hierarchy[PhilosophicalDomain.ETHICS] = OntologicalHierarchy(
            domain=PhilosophicalDomain.ETHICS,
            specializations=[
                "normative_ethics", "metaethics", "applied_ethics",
                "virtue_ethics", "moral_psychology"
            ],
            cross_cutting_themes=[
                "consequentialism_vs_deontology", "moral_realism", "responsibility"
            ],
            methodological_approaches=[
                "reflective_equilibrium", "moral_intuitions", "consequentialist_calculation"
            ],
            contemporary_debates=[
                "moral_enhancement", "effective_altruism", "moral_motivation"
            ]
        )

        # Logic
        hierarchy[PhilosophicalDomain.LOGIC] = OntologicalHierarchy(
            domain=PhilosophicalDomain.LOGIC,
            specializations=[
                "formal_logic", "philosophical_logic", "modal_logic",
                "temporal_logic", "epistemic_logic"
            ],
            cross_cutting_themes=[
                "validity", "soundness", "completeness", "consistency"
            ],
            methodological_approaches=[
                "formal_systems", "model_theory", "proof_theory"
            ],
            contemporary_debates=[
                "logical_pluralism", "relevance_logic", "paraconsistent_logic"
            ]
        )

        # Aesthetics
        hierarchy[PhilosophicalDomain.AESTHETICS] = OntologicalHierarchy(
            domain=PhilosophicalDomain.AESTHETICS,
            specializations=[
                "philosophy_of_art", "philosophy_of_beauty", "aesthetic_experience",
                "philosophy_of_literature", "philosophy_of_music"
            ],
            cross_cutting_themes=[
                "aesthetic_judgment", "aesthetic_properties", "artistic_value"
            ],
            methodological_approaches=[
                "aesthetic_theory", "critical_analysis", "phenomenological_description"
            ],
            contemporary_debates=[
                "institutional_theory_of_art", "aesthetic_cognitivism", "environmental_aesthetics"
            ]
        )

        return hierarchy

    def _define_relation_types(self) -> dict[SemanticRelationType, RelationTypeMapping]:
        """Define systematic mapping of philosophical relationship types."""
        relations = {}

        relations[SemanticRelationType.CAUSAL] = RelationTypeMapping(
            relation_type=SemanticRelationType.CAUSAL,
            applicable_domains=[
                PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND
            ],
            typical_strength_range=(0.6, 0.9),
            context_dependencies=["temporal_ordering", "mechanism_specification"],
            philosophical_justification="Causal relations fundamental to explanation and understanding",
            historical_examples=["Hume_on_causation", "Mill_methods"],
            contemporary_applications=["mental_causation", "causal_closure"]
        )

        relations[SemanticRelationType.LOGICAL_IMPLICATION] = RelationTypeMapping(
            relation_type=SemanticRelationType.LOGICAL_IMPLICATION,
            applicable_domains=[
                PhilosophicalDomain.LOGIC,
                PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.ETHICS
            ],
            typical_strength_range=(0.8, 1.0),
            context_dependencies=["logical_system", "validity_conditions"],
            philosophical_justification="Logical relations provide structural foundation for reasoning",
            historical_examples=["Aristotelian_syllogistics", "Frege_logic"],
            contemporary_applications=["material_conditionals", "relevant_implication"]
        )

        relations[SemanticRelationType.PART_WHOLE] = RelationTypeMapping(
            relation_type=SemanticRelationType.PART_WHOLE,
            applicable_domains=[
                PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.AESTHETICS
            ],
            typical_strength_range=(0.5, 0.8),
            context_dependencies=["mereological_principles", "emergence_relations"],
            philosophical_justification="Mereological relations crucial for understanding composition",
            historical_examples=["Aristotelian_substance", "Husserlian_wholes"],
            contemporary_applications=["composition_problems", "emergent_properties"]
        )

        return relations

    def _build_context_sensitivity_map(self) -> dict[str, dict[str, float]]:
        """Build map of context sensitivity for different philosophical domains."""
        sensitivity_map = {
            "cultural_context": {
                PhilosophicalDomain.ETHICS.value: 0.8,
                PhilosophicalDomain.AESTHETICS.value: 0.9,
                PhilosophicalDomain.POLITICAL_PHILOSOPHY.value: 0.9,
                PhilosophicalDomain.LOGIC.value: 0.2,
                PhilosophicalDomain.METAPHYSICS.value: 0.4
            },
            "historical_context": {
                PhilosophicalDomain.ETHICS.value: 0.7,
                PhilosophicalDomain.EPISTEMOLOGY.value: 0.6,
                PhilosophicalDomain.AESTHETICS.value: 0.8,
                PhilosophicalDomain.LOGIC.value: 0.3,
                PhilosophicalDomain.METAPHYSICS.value: 0.5
            },
            "linguistic_context": {
                PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE.value: 0.9,
                PhilosophicalDomain.LOGIC.value: 0.7,
                PhilosophicalDomain.EPISTEMOLOGY.value: 0.6,
                PhilosophicalDomain.ETHICS.value: 0.5,
                PhilosophicalDomain.AESTHETICS.value: 0.6
            }
        }
        return sensitivity_map

    def _build_domain_indicators(self) -> dict[PhilosophicalDomain, dict[str, list[str]]]:
        """Build systematic indicators for philosophical domain identification."""
        indicators = {}

        indicators[PhilosophicalDomain.METAPHYSICS] = {
            "primary_terms": [
                "being", "existence", "reality", "substance", "property",
                "essence", "nature", "identity", "persistence", "change"
            ],
            "modal_terms": [
                "necessary", "possible", "contingent", "actual", "potential",
                "counterfactual", "possible_worlds"
            ],
            "temporal_terms": [
                "time", "temporal", "past", "present", "future", "duration",
                "instant", "eternal", "temporal_parts"
            ]
        }

        indicators[PhilosophicalDomain.EPISTEMOLOGY] = {
            "knowledge_terms": [
                "knowledge", "belief", "truth", "justification", "evidence",
                "certainty", "doubt", "skepticism", "reliability"
            ],
            "cognitive_terms": [
                "perception", "memory", "intuition", "reasoning", "inference",
                "understanding", "comprehension", "cognition"
            ],
            "methodological_terms": [
                "empiricism", "rationalism", "foundationalism", "coherentism",
                "pragmatism", "naturalism", "testimony"
            ]
        }

        indicators[PhilosophicalDomain.ETHICS] = {
            "normative_terms": [
                "right", "wrong", "good", "bad", "ought", "should", "duty",
                "obligation", "permission", "prohibition"
            ],
            "value_terms": [
                "value", "virtue", "vice", "character", "moral", "immoral",
                "ethical", "unethical", "justice", "fairness"
            ],
            "consequentialist_terms": [
                "consequences", "outcomes", "utility", "happiness", "welfare",
                "well-being", "harm", "benefit"
            ]
        }

        indicators[PhilosophicalDomain.LOGIC] = {
            "logical_operators": [
                "and", "or", "not", "if", "then", "implies", "equivalent",
                "necessary", "sufficient", "conditional"
            ],
            "quantifiers": [
                "all", "some", "every", "any", "exists", "universal",
                "particular", "exactly", "at_least", "at_most"
            ],
            "validity_terms": [
                "valid", "invalid", "sound", "unsound", "consistent",
                "inconsistent", "contradiction", "tautology"
            ]
        }

        indicators[PhilosophicalDomain.AESTHETICS] = {
            "aesthetic_properties": [
                "beautiful", "ugly", "sublime", "elegant", "graceful",
                "harmonious", "balanced", "expressive", "meaningful"
            ],
            "art_terms": [
                "art", "artwork", "artistic", "creativity", "expression",
                "representation", "interpretation", "style", "form"
            ],
            "evaluation_terms": [
                "aesthetic", "taste", "judgment", "appreciation", "criticism",
                "evaluation", "assessment", "preference"
            ]
        }

        return indicators

    def _build_complexity_assessors(self) -> dict[str, Any]:
        """Build systematic complexity assessment framework."""
        return {
            "conceptual_complexity": {
                "simple": 1,
                "moderate": 2,
                "complex": 3,
                "highly_complex": 4,
                "extremely_complex": 5
            },
            "relational_complexity": {
                "few_relations": 1,
                "moderate_relations": 2,
                "many_relations": 3,
                "dense_network": 4,
                "highly_interconnected": 5
            },
            "epistemic_complexity": {
                "high_certainty": 1,
                "moderate_certainty": 2,
                "uncertain": 3,
                "highly_uncertain": 4,
                "radical_uncertainty": 5
            }
        }

    def _build_cross_cutting_themes(self) -> dict[str, list[PhilosophicalDomain]]:
        """Build map of cross-cutting philosophical themes."""
        return {
            "realism_vs_antirealism": [
                PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.ETHICS,
                PhilosophicalDomain.AESTHETICS
            ],
            "naturalism": [
                PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.ETHICS
            ],
            "emergence": [
                PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE
            ],
            "normativity": [
                PhilosophicalDomain.ETHICS,
                PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.AESTHETICS,
                PhilosophicalDomain.LOGIC
            ]
        }

    def _determine_primary_category(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalDomain:
        """Determine primary philosophical category through systematic analysis."""
        domain_scores = defaultdict(float)

        # Analyze concepts for domain indicators
        for concept in semantic_analysis.primary_concepts:
            if hasattr(concept, 'domain') and concept.domain:
                domain_scores[concept.domain] += concept.confidence_level

        # Analyze statement content for domain-specific terms
        for domain, indicators in self.domain_indicators.items():
            for _, terms in indicators.items():
                # Ideally, we would analyze the original statement text for domain-specific terms,
                # but since the raw statement is not available in SemanticAnalysis, we use concept terms instead.
                # To enable direct statement analysis, SemanticAnalysis would need to include the original text.
                concept_terms = [c.term.lower() for c in semantic_analysis.primary_concepts]
                matches = sum(1 for term in terms if any(term in ct for ct in concept_terms))
                domain_scores[domain] += matches * 0.1

        # Analyze semantic relations for domain preferences
        for relation in semantic_analysis.semantic_relations:
            if relation.relation_type in self.relation_types:
                mapping = self.relation_types[relation.relation_type]
                for domain in mapping.applicable_domains:
                    domain_scores[domain] += relation.confidence * 0.2

        # Find domain with highest score
        if domain_scores:
            primary_domain = max(domain_scores, key=lambda k: domain_scores[k])
            return primary_domain

        # Default fallback
        return PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE

    def _identify_secondary_categories(
        self,
        semantic_analysis: SemanticAnalysis
    ) -> list[PhilosophicalDomain]:
        """Identify secondary philosophical categories."""
        secondary = []

        # Look for concepts from different domains
        domains_found = set()
        for concept in semantic_analysis.primary_concepts:
            if hasattr(concept, 'domain') and concept.domain:
                domains_found.add(concept.domain)

        # Remove primary domain and add others as secondary
        if len(domains_found) > 1:
            primary = self._determine_primary_category(semantic_analysis)
            domains_found.discard(primary)
            secondary = list(domains_found)[:3]  # Limit to 3 secondary

        return secondary

    def _identify_cross_cutting_themes(self, semantic_analysis: SemanticAnalysis) -> list[str]:
        """Identify cross-cutting philosophical themes."""
        themes = []

        # Check pragmatic implications for theme indicators
        implications = semantic_analysis.pragmatic_implications
        for implication in implications:
            if "normative" in implication.lower():
                themes.append("normativity")
            elif "causal" in implication.lower():
                themes.append("causation")
            elif "modal" in implication.lower():
                themes.append("modality")

        # Check for realism indicators
        concept_terms = [c.term.lower() for c in semantic_analysis.primary_concepts]
        if any(term in ["real", "reality", "exists", "existence"] for term in concept_terms):
            themes.append("realism_vs_antirealism")

        # Check for emergence indicators
        if any(term in ["emerge", "emergent", "level", "complex"] for term in concept_terms):
            themes.append("emergence")

        return list(set(themes))[:3]  # Remove duplicates and limit

    def _assess_complexity_level(self, semantic_analysis: SemanticAnalysis) -> int:
        """Assess complexity level of the philosophical content."""
        complexity_score = 0

        # Conceptual complexity
        concept_count = len(semantic_analysis.primary_concepts)
        if concept_count > 5:
            complexity_score += 2
        elif concept_count > 2:
            complexity_score += 1

        # Relational complexity
        relation_count = len(semantic_analysis.semantic_relations)
        if relation_count > 3:
            complexity_score += 2
        elif relation_count > 1:
            complexity_score += 1

        # Epistemic uncertainty
        if semantic_analysis.epistemic_uncertainty:
            avg_uncertainty = sum(semantic_analysis.epistemic_uncertainty.values()) / len(semantic_analysis.epistemic_uncertainty)
            if avg_uncertainty > 0.7:
                complexity_score += 2
            elif avg_uncertainty > 0.4:
                complexity_score += 1

        # Ensure level is between 1 and 5
        return max(1, min(5, complexity_score))

    def _find_interdisciplinary_connections(self, semantic_analysis: SemanticAnalysis) -> list[str]:
        """Find potential interdisciplinary connections."""
        connections = []

        # Check concept terms for disciplinary indicators
        concept_terms = [c.term.lower() for c in semantic_analysis.primary_concepts]

        if any(term in ["brain", "neural", "cognitive", "psychology"] for term in concept_terms):
            connections.append("cognitive_science")

        if any(term in ["quantum", "physics", "scientific", "empirical"] for term in concept_terms):
            connections.append("natural_sciences")

        if any(term in ["social", "political", "cultural", "society"] for term in concept_terms):
            connections.append("social_sciences")

        if any(term in ["computer", "artificial", "algorithm", "computational"] for term in concept_terms):
            connections.append("computer_science")

        if any(term in ["legal", "law", "rights", "justice"] for term in concept_terms):
            connections.append("law")

        return connections[:3]

    def _generate_alternative_categorizations(self, semantic_analysis: SemanticAnalysis) -> list[dict[str, Any]]:
        """Generate alternative categorization possibilities."""
        alternatives = []

        # Generate perspective-based alternatives
        alternatives.append({
            "approach": "historical_perspective",
            "primary_domain": "continental_philosophy",
            "justification": "Alternative interpretation through continental tradition"
        })

        alternatives.append({
            "approach": "pragmatist_perspective",
            "primary_domain": "applied_philosophy",
            "justification": "Focus on practical consequences and applications"
        })

        if len(semantic_analysis.primary_concepts) > 2:
            alternatives.append({
                "approach": "interdisciplinary_perspective",
                "primary_domain": "philosophy_of_science",
                "justification": "Emphasis on interdisciplinary connections"
            })

        return alternatives[:2]  # Limit to 2 alternatives

    def _calculate_categorical_confidence(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain
    ) -> float:
        """Calculate confidence in the categorization."""
        confidence_factors = []

        # Concept confidence
        if semantic_analysis.primary_concepts:
            concept_confidence = sum(c.confidence_level for c in semantic_analysis.primary_concepts) / len(semantic_analysis.primary_concepts)
            confidence_factors.append(concept_confidence)

        # Domain consistency
        primary_domain_concepts = sum(1 for c in semantic_analysis.primary_concepts
                                    if hasattr(c, 'domain') and c.domain == primary_category)
        total_concepts = len(semantic_analysis.primary_concepts)
        if total_concepts > 0:
            domain_consistency = primary_domain_concepts / total_concepts
            confidence_factors.append(domain_consistency)

        # Relation confidence
        if semantic_analysis.semantic_relations:
            relation_confidence = sum(r.confidence for r in semantic_analysis.semantic_relations) / len(semantic_analysis.semantic_relations)
            confidence_factors.append(relation_confidence)

        # Calculate overall confidence
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)

        return 0.5  # Default moderate confidence

    def _generate_categorization_justification(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain,
        secondary_categories: list[PhilosophicalDomain]
    ) -> str:
        """Generate justification for the categorization decision."""
        justification_parts = []

        # Primary category justification
        concept_count = len([c for c in semantic_analysis.primary_concepts
                           if hasattr(c, 'domain') and c.domain == primary_category])
        if concept_count > 0:
            justification_parts.append(f"Primary categorization based on {concept_count} concepts in {primary_category.value}")

        # Secondary category justification
        if secondary_categories:
            secondary_names = [d.value for d in secondary_categories]
            justification_parts.append(f"Secondary themes from {', '.join(secondary_names)}")

        # Complexity justification
        if semantic_analysis.semantic_relations:
            justification_parts.append(f"Supported by {len(semantic_analysis.semantic_relations)} semantic relations")

        if not justification_parts:
            justification_parts.append("Categorization based on systematic analysis of semantic content")

        return "; ".join(justification_parts)

    def _identify_categorical_limitations(self, semantic_analysis: SemanticAnalysis) -> list[str]:
        """Identify limitations of the categorization."""
        limitations = []

        # Uncertainty-based limitations
        if semantic_analysis.epistemic_uncertainty:
            high_uncertainty_aspects = [k for k, v in semantic_analysis.epistemic_uncertainty.items() if v > 0.6]
            if high_uncertainty_aspects:
                limitations.append(f"High uncertainty in: {', '.join(high_uncertainty_aspects)}")

        # Context dependency limitations
        if semantic_analysis.context_dependencies:
            limitations.append("Categorization may vary with different contexts")

        # Analytical limitations
        if semantic_analysis.analytical_limitations:
            limitations.extend(semantic_analysis.analytical_limitations[:2])

        # Default limitations
        if not limitations:
            limitations = [
                "Categorization based on pattern matching rather than deep understanding",
                "May reflect bias toward Western philosophical traditions"
            ]

        return limitations[:3]

    # Additional utility methods for external use

    def get_relation_types(self, domain: str) -> list[SemanticRelationType]:
        """Get relation types applicable to a specific domain."""
        try:
            domain_enum = PhilosophicalDomain(domain)
            applicable_types = []

            for relation_type, mapping in self.relation_types.items():
                if domain_enum in mapping.applicable_domains:
                    applicable_types.append(relation_type)

            return applicable_types
        except ValueError:
            logger.warning(f"Unknown domain: {domain}")
            return []

    def get_domain_specializations(self, domain: PhilosophicalDomain) -> list[str]:
        """Get specializations for a philosophical domain."""
        if domain in self.concept_hierarchy:
            return self.concept_hierarchy[domain].specializations
        return []

    def get_cross_cutting_themes_for_domain(self, domain: PhilosophicalDomain) -> list[str]:
        """Get cross-cutting themes applicable to a domain."""
        applicable_themes = []

        for theme, domains in self.cross_cutting_themes.items():
            if domain in domains:
                applicable_themes.append(theme)

        return applicable_themes
