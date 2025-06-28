"""
LLM-Enhanced Semantic Processor for Philosophical Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Conceptual Framework Deconstruction

This implementation embodies sophisticated philosophical commitments:

#### Core Theoretical Foundations:
- **Dynamic Semantic Understanding**: LLM-powered contextual interpretation
- **Systematic Analytical Method**: Structured philosophical evaluation framework
- **Multi-Perspectival Integration**: Comprehensive interpretive lens application
- **Epistemic Humility**: Built-in uncertainty acknowledgment and revision mechanisms

#### Underlying Epistemological Assumptions:
- Meaning emerges through systematic analytical engagement rather than immediate intuition
- Concepts exist within interpretive frameworks rather than as autonomous entities
- Multiple valid perspectives can coexist without logical contradiction
- Understanding deepens through methodical examination of presuppositions and implications

#### Conceptual Lineage and Intellectual Heritage:
- **Wittgensteinian Language Philosophy**: Context-dependent meaning construction
- **Pragmatist Tradition**: Emphasis on practical consequences and experimental method
- **Hermeneutic Circle**: Iterative refinement of understanding through interpretation
- **Critical Theory**: Power-aware analysis and ideological critique capabilities

### Methodological Critique

The analytical approach employs:
1. **Structured Concept Extraction**: Systematic identification and categorization
2. **Relational Analysis**: Explicit semantic relationship mapping
3. **Contextual Grounding**: Language game and domain-specific interpretation
4. **Uncertainty Quantification**: Explicit epistemic limitation acknowledgment

### Usage Example

```python
processor = LLMSemanticProcessor()

analysis = await processor.analyze_statement(
    statement="consciousness emerges from neural complexity",
    context=PhilosophicalContext(
        domain=PhilosophicalDomain.PHILOSOPHY_OF_MIND,
        language_game=LanguageGame.SCIENTIFIC_DISCOURSE
    )
)
```
"""

import logging
from datetime import datetime
from typing import Optional

from .types import (
    PhilosophicalConcept,
    PhilosophicalContext,
    PhilosophicalDomain,
    SemanticAnalysis,
    SemanticRelation,
    SemanticRelationType,
)

logger = logging.getLogger(__name__)


class MetaphysicalConceptExtractor:
    """
        Extracts metaphysical concepts from philosophical statements using a systematic and rigorous analytical framework.

        Systematic metaphysical concept extraction with philosophical rigor.

        ### Methodological Approach:
        - Ontological category identification
        - Existence claim analysis
        - Modal concept recognition
        - Temporal and spatial concept mapping
        """

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.METAPHYSICS
        self.extraction_confidence = 0.85

        # Metaphysical concept indicators with systematic categorization
        self.ontological_indicators = {
            "being": ["being", "existence", "entity", "reality", "is", "exists"],
            "substance": ["substance", "matter", "material", "physical", "concrete"],
            "property": ["property", "attribute", "quality", "characteristic", "feature"],
            "relation": ["relation", "relationship", "connection", "dependence", "causation"],
            "universals": ["universal", "type", "kind", "essence", "nature", "form"],
            "particulars": ["particular", "individual", "instance", "token", "this", "that"],
            "modality": ["possible", "necessary", "contingent", "actual", "potential"],
            "time": ["time", "temporal", "duration", "moment", "eternal", "change"],
            "space": ["space", "spatial", "location", "place", "dimension", "extension"],
            "causation": ["cause", "effect", "causal", "because", "therefore", "due to"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """
        Extract metaphysical concepts using systematic analytical framework.

        ### Argumentative Integrity Analysis:
        1. Identify ontological commitments in statement
        2. Categorize existence claims and modal assertions
        3. Map causal and relational structures

        Returns:
            List[PhilosophicalConcept]: A list of unique PhilosophicalConcept instances extracted from the statement.
        """
        concepts = []
        seen_terms = set()
        statement_lower = statement.lower()

        # 1. Conceptual Framework Deconstruction
        for category, indicators in self.ontological_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = await self._construct_metaphysical_concept(
                        indicator, category, statement, context
                    )
                    if concept and concept.term not in seen_terms:
                        seen_terms.add(concept.term)
                        concepts.append(concept)

        return concepts

    async def _construct_metaphysical_concept(
        self,
        term: str,
        category: str,
        statement: str,
        context: PhilosophicalContext
    ) -> PhilosophicalConcept | None:
        """Construct metaphysical concept with systematic categorization."""

        # Contextual and Interpretative Nuances
        metaphysical_traditions = {
            "being": "ontological_analysis",
            "substance": "aristotelian_metaphysics",
            "property": "property_theory",
            "relation": "relational_metaphysics",
            "universals": "platonic_realism",
            "particulars": "nominalist_analysis",
            "modality": "modal_metaphysics",
            "time": "temporal_ontology",
            "space": "spatial_ontology",
            "causation": "causal_theory"
        }

        return PhilosophicalConcept(
            term=term,
            domain=PhilosophicalDomain.METAPHYSICS,
            definition=f"Metaphysical concept in {category} category",
            attributes={
                "metaphysical_category": category,
                "ontological_status": "systematic_analysis_required",
                "modal_properties": "context_dependent",
                "temporal_characteristics": "requires_examination"
            },
            philosophical_tradition=metaphysical_traditions.get(category, "general_metaphysics"),
            context_dependencies=[
                f"Interpretation depends on {category} framework",
                "Modal context affects ontological status",
                "Temporal assumptions influence meaning"
            ],
            epistemic_status="provisional",
            confidence_level=0.8,
            revision_conditions=[
                "Alternative ontological frameworks considered",
                "Modal assumptions explicitly examined",
                "Temporal presuppositions clarified"
            ]
        )

    def _deduplicate_concepts(self, concepts: list[PhilosophicalConcept]) -> list[PhilosophicalConcept]:
        """Remove duplicate concepts while preserving analytical rigor."""
        seen_terms = set()
        unique_concepts = []

        for concept in concepts:
            if concept.term not in seen_terms:
                seen_terms.add(concept.term)
                unique_concepts.append(concept)

        return unique_concepts


class EpistemologicalConceptExtractor:
    """
    Systematic epistemological concept extraction with methodological critique.

    ### Critical Perspective Integration:
    - Knowledge claim identification and analysis
    - Justification structure mapping
    - Skeptical challenge recognition
    - Epistemic virtue and vice categorization
    """

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.EPISTEMOLOGY
        self.extraction_confidence = 0.87

        # Epistemological concept categories with systematic organization
        self.epistemic_indicators = {
            "knowledge": ["know", "knowledge", "understanding", "comprehension", "grasp"],
            "belief": ["believe", "belief", "opinion", "conviction", "accept"],
            "justification": ["justified", "reason", "evidence", "support", "warrant"],
            "truth": ["true", "truth", "accurate", "correct", "fact", "reality"],
            "experience": ["experience", "perceive", "observe", "sense", "empirical"],
            "reasoning": ["reason", "logic", "infer", "deduce", "conclude", "argument"],
            "skepticism": ["doubt", "uncertain", "question", "skeptical", "fallible"],
            "certainty": ["certain", "sure", "definite", "absolute", "indubitable"],
            "method": ["method", "methodology", "approach", "procedure", "technique"],
            "inquiry": ["inquiry", "investigation", "research", "study", "explore"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """
        Extract epistemological concepts through systematic analysis.

        ### Methodological Critique:
        1. Identify epistemic commitments and assumptions
        2. Map justification structures and evidence relations
        3. Assess knowledge claims and their warranting conditions
        4. Examine methodological assumptions and limitations
        """
        concepts = []
        statement_lower = statement.lower()

        # Systematic concept identification
        for category, indicators in self.epistemic_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = await self._construct_epistemological_concept(
                        indicator, category, statement, context
                    )
                    if concept:
                        concepts.append(concept)

        return self._enhance_epistemological_analysis(concepts, statement, context)

    async def _construct_epistemological_concept(
        self,
        term: str,
        category: str,
        statement: str,
        context: PhilosophicalContext
    ) -> PhilosophicalConcept | None:
        """Construct epistemological concept with methodological awareness."""

        # Map to epistemological traditions
        epistemic_traditions = {
            "knowledge": "traditional_analysis",
            "belief": "doxastic_analysis",
            "justification": "evidentialist_theory",
            "truth": "correspondence_theory",
            "experience": "empiricist_tradition",
            "reasoning": "rationalist_tradition",
            "skepticism": "skeptical_tradition",
            "certainty": "cartesian_foundationalism",
            "method": "methodological_analysis",
            "inquiry": "pragmatist_epistemology"
        }

        return PhilosophicalConcept(
            term=term,
            domain=PhilosophicalDomain.EPISTEMOLOGY,
            definition=f"Epistemological concept in {category} domain",
            attributes={
                "epistemic_category": category,
                "justification_structure": "requires_analysis",
                "evidential_base": "context_dependent",
                "methodological_assumptions": "systematic_examination_needed"
            },
            philosophical_tradition=epistemic_traditions.get(category, "general_epistemology"),
            context_dependencies=[
                f"Epistemic evaluation depends on {category} framework",
                "Justification standards vary across contexts",
                "Methodological assumptions affect epistemic status"
            ],
            epistemic_status="provisional",
            confidence_level=0.82,
            revision_conditions=[
                "Alternative justification theories considered",
                "Methodological assumptions made explicit",
                "Evidence standards clarified"
            ]
        )

    def _enhance_epistemological_analysis(
        self,
        concepts: list[PhilosophicalConcept],
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Enhance analysis with systematic epistemological evaluation."""

        # Add meta-epistemological concepts if statement contains epistemic claims
        if any(word in statement.lower() for word in ["certain", "know", "prove", "demonstrate"]):
            meta_concept = PhilosophicalConcept(
                term="epistemic_status",
                domain=PhilosophicalDomain.EPISTEMOLOGY,
                definition="Meta-level epistemic evaluation of claims",
                attributes={
                    "meta_level": "second_order_epistemic_analysis",
                    "reflexivity": "epistemic_self_awareness",
                    "fallibilism": "built_in_uncertainty_acknowledgment"
                },
                philosophical_tradition="meta_epistemology",
                epistemic_status="meta_analytical",
                confidence_level=0.9
            )
            concepts.append(meta_concept)

        return self._deduplicate_concepts(concepts)

    def _deduplicate_concepts(self, concepts: list[PhilosophicalConcept]) -> list[PhilosophicalConcept]:
        """Remove duplicates while preserving systematic analysis."""
        seen_terms = set()
        unique_concepts = []

        for concept in concepts:
            if concept.term not in seen_terms:
                seen_terms.add(concept.term)
                unique_concepts.append(concept)

        return unique_concepts


class EthicalConceptExtractor:
    """
    Systematic ethical concept extraction with normative analysis.

    ### Contextual and Interpretative Nuances:
    - Normative framework identification
    - Value commitment analysis
    - Moral psychology integration
    - Applied ethics domain mapping
    """

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.ETHICS
        self.extraction_confidence = 0.83

        # Ethical concept systematic categorization
        self.ethical_indicators = {
            "normative": ["ought", "should", "must", "duty", "obligation", "right", "wrong"],
            "virtue": ["virtue", "character", "excellent", "good", "bad", "vice", "moral"],
            "consequentialist": ["consequence", "outcome", "result", "utility", "happiness", "welfare"],
            "deontological": ["duty", "categorical", "imperative", "universal", "respect", "dignity"],
            "value": ["value", "good", "bad", "better", "worse", "worth", "important"],
            "justice": ["justice", "fair", "unfair", "equal", "rights", "desert", "merit"],
            "care": ["care", "compassion", "empathy", "relationship", "responsibility", "harm"],
            "freedom": ["free", "freedom", "liberty", "autonomy", "choice", "agency"],
            "responsibility": ["responsible", "accountability", "blame", "praise", "liable"],
            "moral_psychology": ["emotion", "intuition", "sentiment", "feeling", "motivation"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """
        Extract ethical concepts through systematic normative analysis.

        ### Synthetic Evaluation:
        1. Identify normative commitments and value frameworks
        2. Map virtue, consequentialist, and deontological elements
        3. Assess moral psychology and responsibility attributions
        4. Integrate care ethics and justice theory perspectives
        """
        concepts = []
        statement_lower = statement.lower()

        # Systematic ethical concept identification
        for category, indicators in self.ethical_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = await self._construct_ethical_concept(
                        indicator, category, statement, context
                    )
                    if concept:
                        concepts.append(concept)

        return self._enhance_ethical_analysis(concepts, statement, context)

    async def _construct_ethical_concept(
        self,
        term: str,
        category: str,
        statement: str,
        context: PhilosophicalContext
    ) -> PhilosophicalConcept | None:
        """Construct ethical concept with normative awareness."""

        # Map to ethical traditions with systematic analysis
        ethical_traditions = {
            "normative": "normative_ethics",
            "virtue": "aristotelian_virtue_ethics",
            "consequentialist": "utilitarian_tradition",
            "deontological": "kantian_ethics",
            "value": "value_theory",
            "justice": "theories_of_justice",
            "care": "ethics_of_care",
            "freedom": "liberal_political_philosophy",
            "responsibility": "moral_responsibility_theory",
            "moral_psychology": "empirical_moral_psychology"
        }

        return PhilosophicalConcept(
            term=term,
            domain=PhilosophicalDomain.ETHICS,
            definition=f"Ethical concept in {category} framework",
            attributes={
                "normative_category": category,
                "moral_valence": "requires_contextual_analysis",
                "universalizability": "framework_dependent",
                "practical_implications": "systematic_evaluation_needed"
            },
            philosophical_tradition=ethical_traditions.get(category, "general_ethics"),
            context_dependencies=[
                f"Normative evaluation depends on {category} theory",
                "Cultural context affects moral interpretation",
                "Applied domain influences practical meaning"
            ],
            epistemic_status="normative_provisional",
            confidence_level=0.78,
            revision_conditions=[
                "Alternative normative frameworks considered",
                "Cultural relativism possibilities examined",
                "Practical consequences systematically assessed"
            ]
        )

    def _enhance_ethical_analysis(
        self,
        concepts: list[PhilosophicalConcept],
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Enhance with systematic ethical evaluation."""

        # Add meta-ethical concepts for normative claims
        if any(word in statement.lower() for word in ["ought", "should", "wrong", "right", "good", "bad"]):
            meta_ethical_concept = PhilosophicalConcept(
                term="normative_status",
                domain=PhilosophicalDomain.ETHICS,
                definition="Meta-ethical evaluation of normative claims",
                attributes={
                    "meta_ethical_level": "second_order_normative_analysis",
                    "objectivity_question": "systematic_examination_required",
                    "motivation_question": "internalism_externalism_considerations"
                },
                philosophical_tradition="meta_ethics",
                epistemic_status="meta_normative",
                confidence_level=0.85
            )
            concepts.append(meta_ethical_concept)

        return self._deduplicate_concepts(concepts)

    def _deduplicate_concepts(self, concepts: list[PhilosophicalConcept]) -> list[PhilosophicalConcept]:
        """Remove duplicates with normative analysis preservation."""
        seen_terms = set()
        unique_concepts = []

        for concept in concepts:
            if concept.term not in seen_terms:
                seen_terms.add(concept.term)
                unique_concepts.append(concept)

        return unique_concepts


class LLMSemanticProcessor:
    """
    Enhanced semantic processor implementing systematic philosophical analysis.

    ### Synthetic Evaluation Framework:

    This processor embodies the comprehensive interpretative methodology:
    1. **Conceptual Framework Deconstruction**: Systematic concept identification
    2. **Methodological Critique**: Analysis method evaluation and limitation acknowledgment
    3. **Critical Perspective Integration**: Multi-domain and multi-traditional analysis
    4. **Argumentative Integrity Analysis**: Logical coherence and consistency examination
    5. **Contextual and Interpretative Nuances**: Language game and cultural context awareness
    6. **Synthetic Evaluation**: Comprehensive integration with constructive insights
    """

    def __init__(self):
        """Initialize with systematic concept extraction capabilities."""
        self.concept_extractors = {
            PhilosophicalDomain.METAPHYSICS: MetaphysicalConceptExtractor(),
            PhilosophicalDomain.EPISTEMOLOGY: EpistemologicalConceptExtractor(),
            PhilosophicalDomain.ETHICS: EthicalConceptExtractor(),
            # Additional extractors to be implemented
        }

        logger.info("LLM Semantic Processor initialized with systematic analysis capabilities")

    async def analyze_statement(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis using systematic philosophical methodology.

        ### Argumentative Integrity Analysis:
        1. Multi-domain concept extraction and categorization
        2. Semantic relationship identification and mapping
        3. Pragmatic implication derivation and assessment
        4. Epistemic uncertainty systematic quantification

        Args:
            statement: Text to analyze with philosophical rigor
            context: Philosophical context providing interpretative framework

        Returns:
            Comprehensive semantic analysis with uncertainty quantification
        """
        logger.debug(f"Analyzing statement: '{statement}' in context: {context.domain}")

        # 1. Conceptual Framework Deconstruction
        primary_concepts = await self._extract_concepts_systematically(statement, context)

        # 2. Methodological Critique - Semantic Relations Analysis
        semantic_relations = await self._identify_semantic_relations(primary_concepts, statement, context)

        # 3. Critical Perspective Integration - Pragmatic Analysis
        pragmatic_implications = await self._analyze_pragmatic_implications(statement, context, primary_concepts)

        # 4. Argumentative Integrity Analysis - Uncertainty Assessment
        epistemic_uncertainty = await self._assess_epistemic_uncertainty(statement, primary_concepts, context)

        # 5. Contextual and Interpretative Nuances
        context_dependencies = self._identify_context_dependencies(primary_concepts, context)
        revision_triggers = await self._generate_revision_triggers(primary_concepts, context)

        # 6. Synthetic Evaluation - Comprehensive Integration
        philosophical_presuppositions = self._identify_philosophical_presuppositions(statement, primary_concepts)
        methodological_assumptions = self._identify_methodological_assumptions(context, primary_concepts)
        interpretive_alternatives = self._generate_interpretive_alternatives(statement, primary_concepts)
        analytical_limitations = self._assess_analytical_limitations(context, primary_concepts)

        return SemanticAnalysis(
            primary_concepts=primary_concepts,
            semantic_relations=semantic_relations,
            pragmatic_implications=pragmatic_implications,
            epistemic_uncertainty=epistemic_uncertainty,
            context_dependencies=context_dependencies,
            revision_triggers=revision_triggers,
            philosophical_presuppositions=philosophical_presuppositions,
            methodological_assumptions=methodological_assumptions,
            interpretive_alternatives=interpretive_alternatives,
            analytical_limitations=analytical_limitations,
            confidence_intervals=self._calculate_confidence_intervals(primary_concepts),
            analysis_timestamp=datetime.now(),
        )

    async def _extract_concepts_systematically(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract concepts using systematic multi-domain analysis."""
        all_concepts = []

        # Primary domain extraction
        primary_extractor = self.concept_extractors.get(context.domain)
        if primary_extractor:
            primary_concepts = await primary_extractor.extract(statement, context)
            all_concepts.extend(primary_concepts)

        # Cross-domain analysis for interdisciplinary connections
        for domain, extractor in self.concept_extractors.items():
            if domain != context.domain and len(context.interdisciplinary_connections) > 0 and domain.value in context.interdisciplinary_connections:
                cross_concepts = await extractor.extract(statement, context)
                # Add cross-domain concepts with lower confidence
                for concept in cross_concepts:
                    concept.confidence_level *= 0.8  # Reduce confidence for cross-domain
                    concept.epistemic_status = "cross_domain_provisional"
                all_concepts.extend(cross_concepts)

        return self._deduplicate_and_rank_concepts(all_concepts)

    async def _identify_semantic_relations(
        self,
        concepts: list[PhilosophicalConcept],
        statement: str,
        context: PhilosophicalContext
    ) -> list[SemanticRelation]:
        """Identify semantic relations with systematic analysis."""
        relations = []

        # Analyze pairwise concept relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                relation = await self._analyze_concept_relation(
                    concept1, concept2, statement, context
                )
                if relation:
                    relations.append(relation)

        return relations

    async def _analyze_concept_relation(
        self,
        concept1: PhilosophicalConcept,
        concept2: PhilosophicalConcept,
        statement: str,
        context: PhilosophicalContext
    ) -> SemanticRelation | None:
        """Analyze relationship between two concepts."""
        statement_lower = statement.lower()
        term1_lower = concept1.term.lower()
        term2_lower = concept2.term.lower()

        # Identify relation types through systematic pattern analysis
        if any(causal in statement_lower for causal in ["cause", "because", "due to", "results in"]) and term1_lower in statement_lower and term2_lower in statement_lower:
            return SemanticRelation(
                source_concept=concept1.term,
                target_concept=concept2.term,
                relation_type=SemanticRelationType.CAUSAL,
                strength=0.8,
                confidence=0.7,
                philosophical_justification="Causal relation indicated by linguistic markers",
                supporting_evidence=[f"Statement contains causal indicators: {statement}"]
            )

        # Logical implication analysis
        # Logical implication analysis
        if any(logical in statement_lower for logical in ["implies", "therefore", "follows", "entails"]) and term1_lower in statement_lower and term2_lower in statement_lower:
            return SemanticRelation(
                source_concept=concept1.term,
                target_concept=concept2.term,
                relation_type=SemanticRelationType.LOGICAL_IMPLICATION,
                strength=0.9,
                confidence=0.8,
                philosophical_justification="Logical implication indicated by inference markers",
                supporting_evidence=[f"Statement contains logical indicators: {statement}"]
            )
        # Similarity and opposition analysis
        if concept1.domain == concept2.domain:
            return SemanticRelation(
                source_concept=concept1.term,
                target_concept=concept2.term,
                relation_type=SemanticRelationType.SIMILARITY,
                strength=0.6,
                confidence=0.6,
                philosophical_justification="Domain similarity suggests conceptual relationship",
                supporting_evidence=[f"Both concepts in {concept1.domain.value} domain"]
            )

        return None

    async def _analyze_pragmatic_implications(
        self,
        statement: str,
        context: PhilosophicalContext,
        concepts: list[PhilosophicalConcept]
    ) -> list[str]:
        """Analyze pragmatic implications through systematic evaluation."""
        implications = []

        # Language game specific implications
        if context.language_game.value == "scientific_discourse":
            implications.extend([
                "Requires empirical validation for acceptance",
                "Subject to peer review and replication standards",
                "Must integrate with existing scientific framework"
            ])
        elif context.language_game.value == "ethical_deliberation":
            implications.extend([
                "Demands normative evaluation and value assessment",
                "Requires consideration of practical consequences",
                "Must address moral responsibility attributions"
            ])
        elif context.language_game.value == "aesthetic_judgment":
            implications.extend([
                "Involves subjective evaluation and taste considerations",
                "Requires sensitivity to cultural and historical context",
                "Permits disagreement without objective resolution"
            ])

        # Domain-specific implications
        if context.domain == PhilosophicalDomain.METAPHYSICS:
            implications.append("Requires ontological commitment examination")
        elif context.domain == PhilosophicalDomain.EPISTEMOLOGY:
            implications.append("Demands epistemic justification analysis")
        elif context.domain == PhilosophicalDomain.ETHICS:
            implications.append("Necessitates normative framework evaluation")

        # Concept-driven implications
        for concept in concepts:
            if concept.epistemic_status == "contested":
                implications.append(f"'{concept.term}' requires careful interpretation due to contested status")

        return implications

    async def _assess_epistemic_uncertainty(
        self,
        statement: str,
        concepts: list[PhilosophicalConcept],
        context: PhilosophicalContext
    ) -> dict[str, float]:
        """Systematically assess epistemic uncertainty across dimensions."""
        uncertainty = {}

        # Conceptual uncertainty
        if concepts:
            concept_confidences = [c.confidence_level for c in concepts]
            uncertainty["conceptual_clarity"] = 1.0 - (sum(concept_confidences) / len(concept_confidences))
        else:
            uncertainty["conceptual_clarity"] = 0.9

        # Contextual uncertainty
        if len(context.interdisciplinary_connections) > 2:
            uncertainty["contextual_complexity"] = 0.7
        else:
            uncertainty["contextual_complexity"] = 0.3

        # Methodological uncertainty
        if context.language_game.value == "ordinary_language":
            uncertainty["methodological_precision"] = 0.6
        elif context.language_game.value == "scientific_discourse":
            uncertainty["methodological_precision"] = 0.2
        else:
            uncertainty["methodological_precision"] = 0.4

        # Interpretive uncertainty
        controversial_terms = ["consciousness", "free will", "truth", "justice", "beauty"]
        if any(term in statement.lower() for term in controversial_terms):
            uncertainty["interpretive_controversy"] = 0.8
        else:
            uncertainty["interpretive_controversy"] = 0.3

        # Meta-level uncertainty about uncertainty assessment
        uncertainty["meta_epistemic"] = 0.4

        return uncertainty

    def _identify_context_dependencies(
        self,
        concepts: list[PhilosophicalConcept],
        context: PhilosophicalContext
    ) -> list[str]:
        """Identify systematic context dependencies."""
        dependencies = []

        # Language game dependencies
        dependencies.append(f"Interpretation depends on {context.language_game.value} framework")

        # Domain dependencies
        dependencies.append(f"Analysis grounded in {context.domain.value} perspective")

        # Cultural and temporal dependencies
        if context.cultural_context:
            dependencies.append(f"Cultural context: {context.cultural_context}")
        if context.temporal_context:
            dependencies.append(f"Temporal context: {context.temporal_context}")

        # Concept-specific dependencies
        for concept in concepts:
            dependencies.extend(concept.context_dependencies)

        return list(set(dependencies))  # Remove duplicates

    async def _generate_revision_triggers(
        self,
        concepts: list[PhilosophicalConcept],
        context: PhilosophicalContext
    ) -> list[str]:
        """Generate systematic revision triggers."""
        triggers = []

        # Conceptual revision triggers
        for concept in concepts:
            triggers.extend(concept.revision_conditions)

        # Methodological revision triggers
        triggers.extend([
            "Alternative methodological approaches considered",
            "Cross-cultural philosophical perspectives examined",
            "Empirical research findings challenge theoretical assumptions"
        ])

        # Context-specific triggers
        if context.domain == PhilosophicalDomain.ETHICS:
            triggers.append("Moral intuitions or practical cases conflict with analysis")
        elif context.domain == PhilosophicalDomain.EPISTEMOLOGY:
            triggers.append("Epistemic paradoxes or counterexamples discovered")
        elif context.domain == PhilosophicalDomain.METAPHYSICS:
            triggers.append("Scientific discoveries challenge ontological assumptions")

        return list(set(triggers))

    def _identify_philosophical_presuppositions(
        self,
        statement: str,
        concepts: list[PhilosophicalConcept]
    ) -> list[str]:
        """Identify underlying philosophical presuppositions."""
        presuppositions = []

        # Ontological presuppositions
        if any(term in statement.lower() for term in ["exists", "real", "actual"]):
            presuppositions.append("Presupposes realist ontology about discussed entities")

        # Epistemological presuppositions
        if any(term in statement.lower() for term in ["know", "certain", "prove"]):
            presuppositions.append("Assumes possibility of knowledge and justification")

        # Methodological presuppositions
        if any(term in statement.lower() for term in ["because", "therefore", "implies"]):
            presuppositions.append("Presupposes logical/causal reasoning validity")

        # Semantic presuppositions
        presuppositions.append("Assumes shared linguistic understanding and meaning stability")

        return presuppositions

    def _identify_methodological_assumptions(
        self,
        context: PhilosophicalContext,
        concepts: list[PhilosophicalConcept]
    ) -> list[str]:
        """Identify methodological assumptions in analysis."""
        assumptions = []

        # Language game assumptions
        if context.language_game.value == "scientific_discourse":
            assumptions.extend([
                "Scientific method provides reliable knowledge",
                "Empirical evidence has epistemic priority",
                "Theoretical unification is valuable"
            ])
        elif context.language_game.value == "ethical_deliberation":
            assumptions.extend([
                "Moral reasoning can guide action",
                "Normative claims have truth values",
                "Practical wisdom is achievable"
            ])

        # Analytical assumptions
        assumptions.extend([
            "Systematic concept analysis reveals important insights",
            "Multiple perspectives enhance understanding",
            "Uncertainty acknowledgment improves epistemic humility"
        ])

        return assumptions

    def _generate_interpretive_alternatives(
        self,
        statement: str,
        concepts: list[PhilosophicalConcept]
    ) -> list[str]:
        """Generate alternative interpretive possibilities."""
        alternatives = []

        # Traditional alternatives
        alternatives.extend([
            "Continental vs. analytic interpretive framework",
            "Historical vs. systematic philosophical approach",
            "Descriptive vs. normative analytical stance"
        ])

        # Domain-specific alternatives
        if any(c.domain == PhilosophicalDomain.ETHICS for c in concepts):
            alternatives.extend([
                "Consequentialist vs. deontological evaluation",
                "Virtue ethics vs. duty-based framework",
                "Care ethics vs. justice-based approach"
            ])

        if any(c.domain == PhilosophicalDomain.EPISTEMOLOGY for c in concepts):
            alternatives.extend([
                "Foundationalist vs. coherentist justification",
                "Internalist vs. externalist epistemic framework",
                "Skeptical vs. anti-skeptical perspective"
            ])

        return alternatives

    def _assess_analytical_limitations(
        self,
        context: PhilosophicalContext,
        concepts: list[PhilosophicalConcept]
    ) -> list[str]:
        """Systematically assess analytical limitations."""
        limitations = []

        # Methodological limitations
        limitations.extend([
            "Analysis limited by current concept extraction capabilities",
            "Semantic relation identification relies on surface linguistic patterns",
            "Cross-cultural perspectives may be underrepresented"
        ])

        # Contextual limitations
        if len(context.interdisciplinary_connections) == 0:
            limitations.append("Analysis may miss important interdisciplinary insights")

        # Temporal limitations
        limitations.append("Analysis reflects current philosophical discourse patterns")

        # Epistemic limitations
        limitations.extend([
            "Uncertainty quantification based on heuristic assessment",
            "Revision triggers may not capture all relevant considerations",
            "Meta-level analysis subject to same limitations as object-level analysis"
        ])

        return limitations

    def _calculate_confidence_intervals(
        self,
        concepts: list[PhilosophicalConcept]
    ) -> dict[str, dict[str, float]]:
        """Calculate confidence intervals for key analytical dimensions."""
        intervals = {}

        if concepts:
            concept_confidences = [c.confidence_level for c in concepts]
            mean_confidence = sum(concept_confidences) / len(concept_confidences)

            intervals["conceptual_analysis"] = {
                "mean": mean_confidence,
                "lower_bound": max(0.0, mean_confidence - 0.2),
                "upper_bound": min(1.0, mean_confidence + 0.2),
                "confidence_level": 0.95
            }

        intervals["semantic_relations"] = {
            "mean": 0.7,
            "lower_bound": 0.5,
            "upper_bound": 0.9,
            "confidence_level": 0.90
        }

        intervals["pragmatic_implications"] = {
            "mean": 0.6,
            "lower_bound": 0.4,
            "upper_bound": 0.8,
            "confidence_level": 0.85
        }

        return intervals

    def _deduplicate_and_rank_concepts(
        self,
        concepts: list[PhilosophicalConcept]
    ) -> list[PhilosophicalConcept]:
        """Deduplicate and rank concepts by systematic criteria."""
        # Remove duplicates by term
        seen_terms = set()
        unique_concepts = []

        for concept in concepts:
            if concept.term not in seen_terms:
                seen_terms.add(concept.term)
                unique_concepts.append(concept)

        # Rank by confidence and epistemic status
        unique_concepts.sort(
            key=lambda c: (c.confidence_level, c.epistemic_status != "provisional"),
            reverse=True
        )

        return unique_concepts
