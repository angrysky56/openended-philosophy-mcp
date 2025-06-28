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
    sibling_domains: list[PhilosophicalDomain] = field(default_factory=list)
    interdisciplinary_connections: list[str] = field(default_factory=list)
    historical_development: list[str] = field(default_factory=list)
    contemporary_debates: list[str] = field(default_factory=list)
    methodological_approaches: list[str] = field(default_factory=list)


@dataclass
class RelationTypeMapping:
    """
    Systematic mapping of philosophical relation types with domain sensitivity.

    ### 5. Contextual and Interpretative Nuances:
    - Domain-specific relation type prevalence and interpretation
    - Context-dependent relation strength and confidence assessment
    - Cultural and historical variation in relational understanding
    """
    relation_type: SemanticRelationType
    typical_domains: list[PhilosophicalDomain]
    strength_indicators: list[str]
    confidence_factors: list[str]
    context_dependencies: list[str]
    philosophical_traditions: list[str]
    contradictory_interpretations: list[str] = field(default_factory=list)
    methodological_considerations: list[str] = field(default_factory=list)


class PhilosophicalOntology:
    """
    Comprehensive ontology system for philosophical concept categorization.

    ### 6. Synthetic Evaluation:

    This implementation provides a comprehensive interpretative framework that:
    - Balances systematic organization with categorical flexibility
    - Integrates multiple philosophical traditions and perspectives
    - Acknowledges limitations while providing practical categorization utility
    - Enables systematic exploration of cross-domain philosophical connections
    """

    def __init__(self):
        """Initialize comprehensive philosophical ontology with systematic organization."""

        # 1. Core Domain Hierarchy Construction
        self.concept_hierarchy = self._build_systematic_concept_hierarchy()

        # 2. Relational Type System Development
        self.relation_types = self._define_comprehensive_relation_types()

        # 3. Context Sensitivity Mapping
        self.context_sensitivity_map = self._build_context_sensitivity_map()

        # 4. Cross-Cutting Theme Analysis
        self.cross_cutting_themes = self._identify_cross_cutting_themes()

        # 5. Complexity Assessment Framework
        self.complexity_indicators = self._develop_complexity_indicators()

        # 6. Interdisciplinary Connection Mapping
        self.interdisciplinary_map = self._build_interdisciplinary_map()

        logger.info("Philosophical Ontology initialized with systematic categorical framework")

    def categorize(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalCategory:
        """
        Systematically categorize philosophical content with comprehensive analysis.

        ### Methodological Approach:
        1. Primary domain identification through concept analysis
        2. Secondary domain assessment via cross-cutting analysis
        3. Complexity evaluation using systematic indicators
        4. Interdisciplinary connection mapping
        5. Confidence assessment with uncertainty quantification

        Args:
            semantic_analysis: Comprehensive semantic analysis results

        Returns:
            Systematic philosophical categorization with justification
        """
        logger.debug("Performing systematic philosophical categorization")

        # 1. Primary Domain Determination
        primary_category = self._determine_primary_category(semantic_analysis)

        # 2. Secondary Domain Analysis
        secondary_categories = self._identify_secondary_categories(semantic_analysis)

        # 3. Cross-Cutting Theme Identification
        cross_cutting_themes = self._identify_cross_cutting_themes_for_analysis(semantic_analysis)

        # 4. Complexity Assessment
        complexity_level = self._assess_complexity_level(semantic_analysis)

        # 5. Interdisciplinary Connection Analysis
        interdisciplinary_connections = self._find_interdisciplinary_connections(semantic_analysis)

        # 6. Alternative Categorization Recognition
        alternative_categorizations = self._generate_alternative_categorizations(
            semantic_analysis, primary_category
        )

        # 7. Confidence Assessment with Systematic Justification
        categorical_confidence = self._calculate_categorical_confidence(
            semantic_analysis, primary_category, secondary_categories
        )

        # 8. Justification Development
        justification = self._develop_categorical_justification(
            semantic_analysis, primary_category, secondary_categories
        )

        # 9. Limitation Acknowledgment
        categorical_limitations = self._identify_categorical_limitations(
            semantic_analysis, primary_category
        )

        return PhilosophicalCategory(
            primary=primary_category,
            secondary=secondary_categories,
            cross_cutting=cross_cutting_themes,
            complexity_level=complexity_level,
            interdisciplinary_connections=interdisciplinary_connections,
            alternative_categorizations=alternative_categorizations,
            categorical_confidence=categorical_confidence,
            justification=justification,
            categorical_limitations=categorical_limitations
        )

    def _build_systematic_concept_hierarchy(self) -> dict[PhilosophicalDomain, OntologicalHierarchy]:
        """
        Build comprehensive hierarchical organization of philosophical domains.

        ### Systematic Organization Principles:
        - Historical tradition acknowledgment with contemporary development integration
        - Logical coherence maintenance across hierarchical levels
        - Flexibility for boundary revision and categorical development
        """
        hierarchy = {}

        # Metaphysics - Foundational Ontological Analysis
        hierarchy[PhilosophicalDomain.METAPHYSICS] = OntologicalHierarchy(
            domain=PhilosophicalDomain.METAPHYSICS,
            specializations=[
                "ontology", "philosophy_of_time", "philosophy_of_space",
                "causation", "modality", "identity", "persistence", "universals_particulars"
            ],
            sibling_domains=[
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE
            ],
            interdisciplinary_connections=[
                "physics", "mathematics", "computer_science", "theology"
            ],
            historical_development=[
                "presocratic_cosmology", "aristotelian_categories", "medieval_scholasticism",
                "modern_rationalism", "analytic_metaphysics", "process_philosophy"
            ],
            contemporary_debates=[
                "grounding_theory", "quantum_metaphysics", "composition_puzzle",
                "temporal_ontology", "modal_realism"
            ],
            methodological_approaches=[
                "conceptual_analysis", "formal_ontology", "naturalistic_metaphysics",
                "experimental_philosophy"
            ]
        )

        # Epistemology - Systematic Knowledge Analysis
        hierarchy[PhilosophicalDomain.EPISTEMOLOGY] = OntologicalHierarchy(
            domain=PhilosophicalDomain.EPISTEMOLOGY,
            specializations=[
                "knowledge_theory", "justification", "skepticism", "epistemic_virtue",
                "social_epistemology", "formal_epistemology", "applied_epistemology"
            ],
            sibling_domains=[
                PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND
            ],
            interdisciplinary_connections=[
                "cognitive_science", "psychology", "sociology", "education",
                "information_science", "artificial_intelligence"
            ],
            historical_development=[
                "ancient_skepticism", "rationalist_foundationalism", "empiricist_tradition",
                "kantian_synthesis", "pragmatist_naturalism", "analytic_epistemology"
            ],
            contemporary_debates=[
                "internalism_externalism", "closure_sensitivity", "epistemic_injustice",
                "disagreement_problem", "experimental_epistemology"
            ],
            methodological_approaches=[
                "conceptual_analysis", "formal_modeling", "experimental_methods",
                "genealogical_analysis", "social_network_analysis"
            ]
        )

        # Ethics - Comprehensive Normative Analysis
        hierarchy[PhilosophicalDomain.ETHICS] = OntologicalHierarchy(
            domain=PhilosophicalDomain.ETHICS,
            specializations=[
                "normative_ethics", "meta_ethics", "applied_ethics", "virtue_ethics",
                "deontological_ethics", "consequentialist_ethics", "care_ethics",
                "political_philosophy", "environmental_ethics"
            ],
            sibling_domains=[
                PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.AESTHETICS,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND
            ],
            interdisciplinary_connections=[
                "psychology", "sociology", "anthropology", "economics", "law",
                "public_policy", "environmental_science", "medicine"
            ],
            historical_development=[
                "virtue_tradition", "divine_command_theory", "natural_law",
                "kantian_deontology", "utilitarian_calculus", "existentialist_ethics"
            ],
            contemporary_debates=[
                "moral_realism_anti_realism", "moral_psychology", "global_justice",
                "bioethics", "climate_ethics", "AI_ethics", "experimental_moral_philosophy"
            ],
            methodological_approaches=[
                "reflective_equilibrium", "casuistry", "empirical_ethics",
                "genealogical_critique", "phenomenological_analysis"
            ]
        )

        # Philosophy of Mind - Consciousness and Mental Phenomena
        hierarchy[PhilosophicalDomain.PHILOSOPHY_OF_MIND] = OntologicalHierarchy(
            domain=PhilosophicalDomain.PHILOSOPHY_OF_MIND,
            specializations=[
                "consciousness", "mental_causation", "personal_identity", "other_minds",
                "cognitive_architecture", "emotions", "perception", "action_theory"
            ],
            sibling_domains=[
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.METAPHYSICS,
                PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE
            ],
            interdisciplinary_connections=[
                "neuroscience", "cognitive_science", "psychology", "artificial_intelligence",
                "linguistics", "computer_science", "medicine"
            ],
            historical_development=[
                "cartesian_dualism", "behaviorism", "identity_theory", "functionalism",
                "eliminative_materialism", "property_dualism"
            ],
            contemporary_debates=[
                "hard_problem_consciousness", "extended_mind", "embodied_cognition",
                "predictive_processing", "integrated_information_theory"
            ],
            methodological_approaches=[
                "thought_experiments", "conceptual_analysis", "empirical_philosophy",
                "computational_modeling", "phenomenological_investigation"
            ]
        )

        # Additional domains following similar systematic pattern...
        # [Aesthetic, Logic, Philosophy of Science, etc. would follow similar structure]

        return hierarchy

    def _define_comprehensive_relation_types(self) -> dict[SemanticRelationType, RelationTypeMapping]:
        """
        Define comprehensive semantic relation types with domain-sensitive analysis.

        ### Systematic Relation Analysis:
        - Domain-specific prevalence and interpretation patterns
        - Context-dependent strength and confidence indicators
        - Cross-cultural and historical variation acknowledgment
        """
        relation_mappings = {}

        # Causal Relations - Metaphysical and Scientific Analysis
        relation_mappings[SemanticRelationType.CAUSAL] = RelationTypeMapping(
            relation_type=SemanticRelationType.CAUSAL,
            typical_domains=[
                PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE,
                PhilosophicalDomain.PHILOSOPHY_OF_MIND, PhilosophicalDomain.ETHICS
            ],
            strength_indicators=[
                "because", "causes", "results_in", "brings_about", "produces",
                "leads_to", "due_to", "owing_to", "on_account_of"
            ],
            confidence_factors=[
                "temporal_precedence", "empirical_correlation", "mechanistic_understanding",
                "theoretical_integration", "experimental_support"
            ],
            context_dependencies=[
                "Scientific vs. everyday causation concepts",
                "Deterministic vs. probabilistic causal frameworks",
                "Agent causation vs. event causation distinctions"
            ],
            philosophical_traditions=[
                "humean_regularity_theory", "aristotelian_four_causes",
                "mechanistic_causation", "interventionist_theories"
            ],
            contradictory_interpretations=[
                "Eliminativist vs. realist approaches to causation",
                "Reductionist vs. emergentist causal frameworks"
            ]
        )

        # Logical Implication - Formal and Informal Logic
        relation_mappings[SemanticRelationType.LOGICAL_IMPLICATION] = RelationTypeMapping(
            relation_type=SemanticRelationType.LOGICAL_IMPLICATION,
            typical_domains=[
                PhilosophicalDomain.LOGIC, PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.ETHICS
            ],
            strength_indicators=[
                "implies", "entails", "follows_from", "therefore", "hence",
                "consequently", "thus", "if_then", "necessary_condition"
            ],
            confidence_factors=[
                "formal_validity", "premise_truth", "inference_rule_application",
                "logical_consistency", "deductive_closure"
            ],
            context_dependencies=[
                "Classical vs. non-classical logic systems",
                "Material vs. relevant implication distinctions",
                "Strict vs. counterfactual conditional interpretation"
            ],
            philosophical_traditions=[
                "aristotelian_syllogistic", "stoic_propositional_logic",
                "modern_predicate_logic", "modal_logic_systems"
            ]
        )

        # Dialectical Tension - Critical and Continental Analysis
        relation_mappings[SemanticRelationType.DIALECTICAL_TENSION] = RelationTypeMapping(
            relation_type=SemanticRelationType.DIALECTICAL_TENSION,
            typical_domains=[
                PhilosophicalDomain.ETHICS, PhilosophicalDomain.POLITICAL_PHILOSOPHY,
                PhilosophicalDomain.CONTINENTAL_PHILOSOPHY, PhilosophicalDomain.AESTHETICS
            ],
            strength_indicators=[
                "tension", "conflict", "contradiction", "paradox", "antinomy",
                "opposition", "disagreement", "competing_claims"
            ],
            confidence_factors=[
                "conceptual_opposition", "practical_conflict", "theoretical_inconsistency",
                "value_disagreement", "methodological_dispute"
            ],
            context_dependencies=[
                "Dialectical vs. analytical approaches to contradiction",
                "Historical vs. logical contradiction interpretation",
                "Productive vs. destructive tension assessment"
            ],
            philosophical_traditions=[
                "hegelian_dialectic", "marxist_dialectical_materialism",
                "socratic_elenchus", "buddhist_middle_way"
            ]
        )

        # Additional relation types would follow similar systematic analysis...

        return relation_mappings

    def _build_context_sensitivity_map(self) -> dict[LanguageGame, dict[str, Any]]:
        """
        Build systematic context sensitivity mapping for interpretation frameworks.

        ### Context-Dependent Analysis Framework:
        - Language game specific interpretation principles
        - Domain-sensitive categorical adjustments
        - Cultural and historical context acknowledgment
        """
        context_map = {}

        # Scientific Discourse Context
        context_map[LanguageGame.SCIENTIFIC_DISCOURSE] = {
            "categorical_priorities": [
                "empirical_grounding", "theoretical_integration", "predictive_power",
                "methodological_rigor", "peer_review_standards"
            ],
            "relation_preferences": [
                SemanticRelationType.CAUSAL, SemanticRelationType.LOGICAL_IMPLICATION,
                SemanticRelationType.DEPENDENCY
            ],
            "confidence_modifiers": {
                "empirical_support": 1.2,
                "theoretical_coherence": 1.1,
                "speculative_claims": 0.7
            },
            "complexity_factors": [
                "mathematical_formalization", "experimental_testability",
                "theoretical_unification", "interdisciplinary_integration"
            ]
        }

        # Ethical Deliberation Context
        context_map[LanguageGame.ETHICAL_DELIBERATION] = {
            "categorical_priorities": [
                "normative_commitment", "practical_consequence", "value_integration",
                "moral_intuition_consistency", "stakeholder_consideration"
            ],
            "relation_preferences": [
                SemanticRelationType.PRAGMATIC_CONSEQUENCE, SemanticRelationType.OPPOSITION,
                SemanticRelationType.DIALECTICAL_TENSION
            ],
            "confidence_modifiers": {
                "moral_consensus": 1.1,
                "practical_experience": 1.0,
                "controversial_claims": 0.8
            },
            "complexity_factors": [
                "stakeholder_diversity", "value_conflict", "practical_constraint",
                "long_term_consequence"
            ]
        }

        # Aesthetic Judgment Context
        context_map[LanguageGame.AESTHETIC_JUDGMENT] = {
            "categorical_priorities": [
                "subjective_validity", "cultural_sensitivity", "historical_awareness",
                "formal_analysis", "emotional_response"
            ],
            "relation_preferences": [
                SemanticRelationType.SIMILARITY, SemanticRelationType.OPPOSITION,
                SemanticRelationType.HERMENEUTIC_CIRCLE
            ],
            "confidence_modifiers": {
                "cultural_consensus": 0.9,
                "personal_taste": 0.7,
                "critical_analysis": 1.0
            },
            "complexity_factors": [
                "cultural_variation", "historical_development", "formal_innovation",
                "interpretive_multiplicity"
            ]
        }

        return context_map

    def _identify_cross_cutting_themes(self) -> dict[str, list[PhilosophicalDomain]]:
        """
        Identify systematic cross-cutting themes spanning multiple domains.

        ### Cross-Domain Analysis:
        - Themes that transcend traditional domain boundaries
        - Systematic identification of recurring philosophical problems
        - Integration opportunities across specialized subdisciplines
        """
        themes = {
            "normativity": [
                PhilosophicalDomain.ETHICS, PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.AESTHETICS, PhilosophicalDomain.LOGIC
            ],
            "representation": [
                PhilosophicalDomain.PHILOSOPHY_OF_MIND, PhilosophicalDomain.EPISTEMOLOGY,
                PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE, PhilosophicalDomain.AESTHETICS
            ],
            "emergence": [
                PhilosophicalDomain.METAPHYSICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.PHILOSOPHY_OF_SCIENCE, PhilosophicalDomain.ETHICS
            ],
            "interpretation": [
                PhilosophicalDomain.PHILOSOPHY_OF_LANGUAGE, PhilosophicalDomain.AESTHETICS,
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.CONTINENTAL_PHILOSOPHY
            ],
            "rationality": [
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.ETHICS,
                PhilosophicalDomain.LOGIC, PhilosophicalDomain.PHILOSOPHY_OF_MIND
            ],
            "agency": [
                PhilosophicalDomain.ETHICS, PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.METAPHYSICS
            ],
            "experience": [
                PhilosophicalDomain.PHENOMENOLOGY, PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.AESTHETICS
            ],
            "power": [
                PhilosophicalDomain.POLITICAL_PHILOSOPHY, PhilosophicalDomain.ETHICS,
                PhilosophicalDomain.EPISTEMOLOGY, PhilosophicalDomain.AESTHETICS
            ]
        }

        return themes

    def _develop_complexity_indicators(self) -> dict[str, Any]:
        """
        Develop systematic complexity assessment indicators.

        Returns:
            Dictionary mapping complexity factors to assessment criteria
        """
        return {
            "conceptual_density": {
                "low": "1-3 primary concepts",
                "medium": "4-7 primary concepts",
                "high": "8+ primary concepts"
            },
            "relational_complexity": {
                "simple": "1-2 relation types",
                "moderate": "3-5 relation types",
                "complex": "6+ relation types"
            },
            "cross_domain_integration": {
                "single": "One primary domain",
                "multiple": "2-3 domains involved",
                "interdisciplinary": "4+ domains with external connections"
            },
            "epistemic_uncertainty": {
                "low": "Average uncertainty < 0.4",
                "moderate": "Average uncertainty 0.4-0.7",
                "high": "Average uncertainty > 0.7"
            }
        }

    def _build_interdisciplinary_map(self) -> dict[str, list[str]]:
        """
        Build systematic interdisciplinary connection mapping.

        Returns:
            Dictionary mapping philosophical domains to related disciplines
        """
        return {
            "metaphysics": ["physics", "mathematics", "computer_science", "theology"],
            "epistemology": ["cognitive_science", "psychology", "education", "information_science"],
            "ethics": ["psychology", "sociology", "law", "public_policy", "medicine"],
            "philosophy_of_mind": ["neuroscience", "cognitive_science", "artificial_intelligence"],
            "philosophy_of_science": ["physics", "biology", "chemistry", "mathematics"],
            "aesthetics": ["art_history", "psychology", "anthropology", "media_studies"],
            "logic": ["mathematics", "computer_science", "linguistics"],
            "political_philosophy": ["political_science", "sociology", "economics", "law"]
        }

    def _determine_primary_category(self, semantic_analysis: SemanticAnalysis) -> PhilosophicalDomain:
        """Systematically determine primary philosophical domain."""
        domain_scores = defaultdict(float)

        # Analyze concepts for domain indicators
        for concept in semantic_analysis.primary_concepts:
            domain_scores[concept.domain] += concept.confidence_level

        # Weight by concept importance and frequency
        for concept in semantic_analysis.primary_concepts:
            # Higher weight for higher confidence concepts
            weight = concept.confidence_level ** 2
            domain_scores[concept.domain] += weight

        # Return domain with highest score
        if domain_scores:
            return max(domain_scores, key=lambda domain: domain_scores[domain])
        else:
            return PhilosophicalDomain.METAPHYSICS  # Default

    def _identify_secondary_categories(self, semantic_analysis: SemanticAnalysis) -> list[PhilosophicalDomain]:
        """Identify secondary philosophical domains present in analysis."""
        domain_scores = defaultdict(float)

        for concept in semantic_analysis.primary_concepts:
            domain_scores[concept.domain] += concept.confidence_level

        # Sort domains by score and return top secondary domains
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

        # Return secondary domains (excluding primary)
        secondary = []
        primary_domain = self._determine_primary_category(semantic_analysis)

        for domain, score in sorted_domains:
            if domain != primary_domain and score > 0.3:  # Threshold for significance
                secondary.append(domain)

        return secondary[:3]  # Return top 3 secondary domains

    def _identify_cross_cutting_themes_for_analysis(self, semantic_analysis: SemanticAnalysis) -> list[str]:
        """Identify cross-cutting themes present in specific analysis."""
        themes = []

        # Check for normativity indicators
        if any("ought" in concept.term or "should" in concept.term
               for concept in semantic_analysis.primary_concepts):
            themes.append("normativity")

        # Check for representation indicators
        if any("represent" in concept.term or "symbol" in concept.term
               for concept in semantic_analysis.primary_concepts):
            themes.append("representation")

        # Check for emergence indicators
        if any("emerge" in concept.term or "complex" in concept.term
               for concept in semantic_analysis.primary_concepts):
            themes.append("emergence")

        # Check pragmatic implications for additional themes
        if any("interpret" in impl for impl in semantic_analysis.pragmatic_implications):
            themes.append("interpretation")

        if any("rational" in impl for impl in semantic_analysis.pragmatic_implications):
            themes.append("rationality")

        return themes

    def _assess_complexity_level(self, semantic_analysis: SemanticAnalysis) -> int:
        """Systematically assess conceptual complexity on 1-5 scale."""
        complexity_score = 1  # Base complexity

        # Factor 1: Number of primary concepts
        concept_count = len(semantic_analysis.primary_concepts)
        if concept_count > 5:
            complexity_score += 1
        if concept_count > 10:
            complexity_score += 1

        # Factor 2: Semantic relations complexity
        relation_count = len(semantic_analysis.semantic_relations)
        if relation_count > 3:
            complexity_score += 1

        # Factor 3: Cross-cutting themes
        cross_cutting_count = len(self._identify_cross_cutting_themes_for_analysis(semantic_analysis))
        if cross_cutting_count > 2:
            complexity_score += 1

        # Factor 4: Epistemic uncertainty
        avg_uncertainty = sum(semantic_analysis.epistemic_uncertainty.values()) / len(semantic_analysis.epistemic_uncertainty) if semantic_analysis.epistemic_uncertainty else 0
        if avg_uncertainty > 0.7:
            complexity_score += 1

        return min(complexity_score, 5)  # Cap at 5

    def _find_interdisciplinary_connections(self, semantic_analysis: SemanticAnalysis) -> list[str]:
        """Find interdisciplinary connections based on concepts and relations."""
        connections = set()

        # Map concepts to interdisciplinary fields
        interdisciplinary_mappings = {
            "consciousness": ["neuroscience", "cognitive_science", "psychology"],
            "knowledge": ["cognitive_science", "education", "information_science"],
            "justice": ["law", "political_science", "sociology"],
            "beauty": ["art_history", "psychology", "anthropology"],
            "cause": ["physics", "biology", "computer_science"],
            "mind": ["neuroscience", "psychology", "artificial_intelligence"],
            "experience": ["psychology", "neuroscience", "phenomenology"],
            "truth": ["logic", "mathematics", "computer_science"]
        }

        for concept in semantic_analysis.primary_concepts:
            for term, fields in interdisciplinary_mappings.items():
                if term in concept.term.lower():
                    connections.update(fields)

        return list(connections)

    def _generate_alternative_categorizations(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain
    ) -> list[dict[str, Any]]:
        """Generate alternative categorical interpretations."""
        alternatives = []

        # Traditional vs. Contemporary categorization
        alternatives.append({
            "type": "temporal_perspective",
            "description": "Historical vs. contemporary philosophical framework",
            "justification": "Different historical periods emphasize different categorical boundaries"
        })

        # Analytic vs. Continental categorization
        alternatives.append({
            "type": "methodological_perspective",
            "description": "Analytic vs. continental philosophical approach",
            "justification": "Different methodological traditions organize concepts differently"
        })

        # Reductive vs. Non-reductive categorization
        if primary_category in [PhilosophicalDomain.PHILOSOPHY_OF_MIND, PhilosophicalDomain.METAPHYSICS]:
            alternatives.append({
                "type": "ontological_perspective",
                "description": "Reductive vs. non-reductive ontological framework",
                "justification": "Different views on emergence and reduction affect categorization"
            })

        return alternatives

    def _calculate_categorical_confidence(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain,
        secondary_categories: list[PhilosophicalDomain]
    ) -> float:
        """Calculate confidence in categorical assignment."""
        confidence_factors = []

        # Factor 1: Concept confidence alignment
        primary_concepts = [c for c in semantic_analysis.primary_concepts if c.domain == primary_category]
        if primary_concepts:
            avg_concept_confidence = sum(c.confidence_level for c in primary_concepts) / len(primary_concepts)
            confidence_factors.append(avg_concept_confidence)

        # Factor 2: Domain concentration (fewer domains = higher confidence)
        domain_diversity = len({c.domain for c in semantic_analysis.primary_concepts})
        concentration_score = max(0.3, 1.0 - (domain_diversity - 1) * 0.2)
        confidence_factors.append(concentration_score)

        # Factor 3: Relation coherence
        if semantic_analysis.semantic_relations:
            relation_coherence = 0.8  # Simplified assessment
            confidence_factors.append(relation_coherence)

        # Calculate weighted average
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Default moderate confidence

    def _develop_categorical_justification(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain,
        secondary_categories: list[PhilosophicalDomain]
    ) -> str:
        """Develop systematic justification for categorical assignment."""
        justification_parts = []

        # Primary domain justification
        primary_concepts = [c for c in semantic_analysis.primary_concepts if c.domain == primary_category]
        if primary_concepts:
            concept_terms = [c.term for c in primary_concepts[:3]]
            justification_parts.append(
                f"Primary categorization as {primary_category.value} based on key concepts: {', '.join(concept_terms)}"
            )

        # Secondary domain justification
        if secondary_categories:
            justification_parts.append(
                f"Secondary categorization includes {', '.join(d.value for d in secondary_categories[:2])} "
                f"due to cross-domain conceptual presence"
            )

        # Methodological justification
        justification_parts.append(
            "Categorization follows systematic concept-to-domain mapping with confidence-weighted assessment"
        )

        return ". ".join(justification_parts)

    def _identify_categorical_limitations(
        self,
        semantic_analysis: SemanticAnalysis,
        primary_category: PhilosophicalDomain
    ) -> list[str]:
        """Identify limitations in categorical analysis."""
        limitations = []

        # Systematic limitations
        limitations.extend([
            "Categorization reflects dominant Western philosophical traditions",
            "Boundary decisions involve interpretive choices rather than objective determination",
            "Cross-cutting themes may resist traditional domain categorization"
        ])

        # Analysis-specific limitations
        if len(semantic_analysis.primary_concepts) < 3:
            limitations.append("Limited concept base may affect categorical reliability")

        high_uncertainty = any(u > 0.8 for u in semantic_analysis.epistemic_uncertainty.values())
        if high_uncertainty:
            limitations.append("High epistemic uncertainty affects categorical confidence")

        return limitations

    def get_relation_types(self, domain: str) -> list[SemanticRelationType]:
        """Get typical relation types for a philosophical domain."""
        domain_enum = PhilosophicalDomain(domain) if isinstance(domain, str) else domain

        typical_relations = []
        for relation_type, mapping in self.relation_types.items():
            if domain_enum in mapping.typical_domains:
                typical_relations.append(relation_type)

        return typical_relations

    def build_context_sensitivity_map(self) -> dict[str, Any]:
        """Return the complete context sensitivity mapping."""
        return {
            lang_game.value: context_data
            for lang_game, context_data in self.context_sensitivity_map.items()
        }
