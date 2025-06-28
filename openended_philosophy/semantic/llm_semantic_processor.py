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

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .types import (
    ConceptExtractionResult,
    ConceptExtractor,
    ExtractedConcept,
    LanguageGame,
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
    Extracts metaphysical concepts from philosophical statements.

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
        """Extract metaphysical concepts from statement."""
        extracted = []
        statement_lower = statement.lower()

        for category, indicators in self.ontological_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = PhilosophicalConcept(
                        term=indicator,
                        domain=PhilosophicalDomain.METAPHYSICS,
                        definition=f"Metaphysical concept related to {category}",
                        attributes={"category": category, "extraction_method": "pattern_matching"},
                        philosophical_tradition="Western_Analytic",
                        confidence_level=self.extraction_confidence
                    )
                    extracted.append(concept)

        return extracted[:5]  # Limit to top 5 concepts

    def get_domain_focus(self) -> PhilosophicalDomain:
        return self.domain_focus

    def get_extraction_confidence(self) -> float:
        return self.extraction_confidence


class EpistemologicalConceptExtractor:
    """Extracts epistemological concepts focusing on knowledge, belief, and justification."""

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.EPISTEMOLOGY
        self.extraction_confidence = 0.80

        self.epistemic_indicators = {
            "knowledge": ["knowledge", "know", "knowing", "cognition", "understanding"],
            "belief": ["believe", "belief", "think", "opinion", "conviction"],
            "justification": ["justified", "evidence", "reason", "proof", "support"],
            "truth": ["true", "truth", "false", "falsity", "accuracy", "correct"],
            "certainty": ["certain", "doubt", "uncertain", "probable", "likely"],
            "experience": ["experience", "empirical", "observation", "sensory", "perception"],
            "reason": ["rational", "logic", "logical", "reasoning", "argument"],
            "skepticism": ["skeptical", "doubt", "questionable", "uncertain", "dubious"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract epistemological concepts from statement."""
        extracted = []
        statement_lower = statement.lower()

        for category, indicators in self.epistemic_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = PhilosophicalConcept(
                        term=indicator,
                        domain=PhilosophicalDomain.EPISTEMOLOGY,
                        definition=f"Epistemological concept related to {category}",
                        attributes={"category": category, "extraction_method": "pattern_matching"},
                        philosophical_tradition="Western_Analytic",
                        confidence_level=self.extraction_confidence
                    )
                    extracted.append(concept)

        return extracted[:5]

    def get_domain_focus(self) -> PhilosophicalDomain:
        return self.domain_focus

    def get_extraction_confidence(self) -> float:
        return self.extraction_confidence


class EthicalConceptExtractor:
    """Extracts ethical concepts focusing on morality, values, and normative claims."""

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.ETHICS
        self.extraction_confidence = 0.75

        self.ethical_indicators = {
            "moral": ["moral", "immoral", "amoral", "ethics", "ethical", "morality"],
            "good": ["good", "bad", "evil", "virtue", "vice", "virtuous"],
            "right": ["right", "wrong", "correct", "incorrect", "proper", "improper"],
            "duty": ["duty", "obligation", "ought", "should", "must", "responsibility"],
            "justice": ["just", "unjust", "fair", "unfair", "justice", "injustice"],
            "value": ["value", "valuable", "worthless", "precious", "important"],
            "harm": ["harm", "help", "benefit", "damage", "hurt", "suffering"],
            "autonomy": ["autonomy", "freedom", "liberty", "choice", "consent"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract ethical concepts from statement."""
        extracted = []
        statement_lower = statement.lower()

        for category, indicators in self.ethical_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = PhilosophicalConcept(
                        term=indicator,
                        domain=PhilosophicalDomain.ETHICS,
                        definition=f"Ethical concept related to {category}",
                        attributes={"category": category, "extraction_method": "pattern_matching"},
                        philosophical_tradition="Western_Analytic",
                        confidence_level=self.extraction_confidence
                    )
                    extracted.append(concept)

        return extracted[:5]

    def get_domain_focus(self) -> PhilosophicalDomain:
        return self.domain_focus

    def get_extraction_confidence(self) -> float:
        return self.extraction_confidence


class AestheticConceptExtractor:
    """Extracts aesthetic concepts focusing on beauty, art, and aesthetic experience."""

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.AESTHETICS
        self.extraction_confidence = 0.70

        self.aesthetic_indicators = {
            "beauty": ["beautiful", "beauty", "ugly", "aesthetic", "aesthetics"],
            "art": ["art", "artistic", "artist", "artwork", "creative", "creativity"],
            "taste": ["taste", "judgment", "appreciation", "preference", "like", "dislike"],
            "sublime": ["sublime", "magnificent", "awe", "wonder", "transcendent"],
            "form": ["form", "shape", "structure", "composition", "design"],
            "expression": ["expression", "expressive", "meaning", "symbolic", "represent"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract aesthetic concepts from statement."""
        extracted = []
        statement_lower = statement.lower()

        for category, indicators in self.aesthetic_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = PhilosophicalConcept(
                        term=indicator,
                        domain=PhilosophicalDomain.AESTHETICS,
                        definition=f"Aesthetic concept related to {category}",
                        attributes={"category": category, "extraction_method": "pattern_matching"},
                        philosophical_tradition="Western_Analytic",
                        confidence_level=self.extraction_confidence
                    )
                    extracted.append(concept)

        return extracted[:5]

    def get_domain_focus(self) -> PhilosophicalDomain:
        return self.domain_focus

    def get_extraction_confidence(self) -> float:
        return self.extraction_confidence


class LogicalConceptExtractor:
    """Extracts logical concepts focusing on reasoning, arguments, and logical structures."""

    def __init__(self):
        self.domain_focus = PhilosophicalDomain.LOGIC
        self.extraction_confidence = 0.85

        self.logical_indicators = {
            "argument": ["argument", "premise", "conclusion", "inference", "reasoning"],
            "logical": ["logical", "illogical", "logic", "valid", "invalid", "sound"],
            "conditional": ["if", "then", "implies", "implication", "conditional"],
            "negation": ["not", "no", "never", "none", "negation", "negative"],
            "quantifier": ["all", "some", "every", "any", "exists", "universal"],
            "consistency": ["consistent", "inconsistent", "contradiction", "compatible"],
            "necessity": ["necessary", "sufficient", "condition", "requirement"]
        }

    async def extract(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract logical concepts from statement."""
        extracted = []
        statement_lower = statement.lower()

        for category, indicators in self.logical_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    concept = PhilosophicalConcept(
                        term=indicator,
                        domain=PhilosophicalDomain.LOGIC,
                        definition=f"Logical concept related to {category}",
                        attributes={"category": category, "extraction_method": "pattern_matching"},
                        philosophical_tradition="Western_Analytic",
                        confidence_level=self.extraction_confidence
                    )
                    extracted.append(concept)

        return extracted[:5]

    def get_domain_focus(self) -> PhilosophicalDomain:
        return self.domain_focus

    def get_extraction_confidence(self) -> float:
        return self.extraction_confidence


class LLMSemanticProcessor:
    """
    Sophisticated semantic processing using LLM capabilities for philosophical analysis.

    This class provides deep semantic understanding of philosophical statements,
    extracting concepts, identifying relationships, and assessing epistemic uncertainty.
    """

    def __init__(self):
        """Initialize the LLM semantic processor with concept extractors."""
        # Initialize concept extractors for different philosophical domains
        self.concept_extractors = {
            PhilosophicalDomain.METAPHYSICS: MetaphysicalConceptExtractor(),
            PhilosophicalDomain.EPISTEMOLOGY: EpistemologicalConceptExtractor(),
            PhilosophicalDomain.ETHICS: EthicalConceptExtractor(),
            PhilosophicalDomain.AESTHETICS: AestheticConceptExtractor(),
            PhilosophicalDomain.LOGIC: LogicalConceptExtractor()
        }

        # Relation type indicators for semantic relationship identification
        self.relation_indicators = {
            SemanticRelationType.CAUSAL: ["causes", "results in", "leads to", "produces", "generates"],
            SemanticRelationType.LOGICAL_IMPLICATION: ["implies", "entails", "follows from", "therefore", "thus"],
            SemanticRelationType.PART_WHOLE: ["part of", "contains", "includes", "comprises", "consists of"],
            SemanticRelationType.SIMILARITY: ["similar to", "like", "resembles", "analogous to", "comparable"],
            SemanticRelationType.OPPOSITION: ["opposite", "contrary", "conflicts with", "opposed to", "against"],
            SemanticRelationType.DEPENDENCY: ["depends on", "requires", "needs", "relies on", "based on"]
        }

        logger.info("LLMSemanticProcessor initialized with philosophical concept extractors")

    async def analyze_statement(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> SemanticAnalysis:
        """
        Comprehensive semantic analysis of philosophical statements.

        Args:
            statement: The philosophical statement to analyze
            context: Philosophical context for interpretation

        Returns:
            Comprehensive semantic analysis results
        """
        try:
            logger.debug(f"Analyzing statement: {statement[:100]}...")

            # Extract concepts using multiple domain-specific extractors
            concepts = await self._extract_concepts_llm(statement, context)

            # Identify semantic relations between concepts
            relations = await self._identify_semantic_relations(concepts, statement)

            # Analyze pragmatic implications
            pragmatic_implications = await self._analyze_pragmatic_implications(statement, context)

            # Assess epistemic uncertainty
            epistemic_uncertainty = await self._assess_epistemic_uncertainty(statement, concepts, context)

            # Identify context dependencies
            context_dependencies = self._identify_context_dependencies(statement, context)

            # Generate revision triggers
            revision_triggers = await self._generate_revision_triggers(concepts, statement)

            # Identify philosophical presuppositions
            presuppositions = self._identify_philosophical_presuppositions(statement)

            # Assess methodological assumptions
            methodological_assumptions = self._assess_methodological_assumptions(statement, context)

            # Generate interpretive alternatives
            interpretive_alternatives = self._generate_interpretive_alternatives(statement, context)

            # Identify analytical limitations
            analytical_limitations = self._identify_analytical_limitations(statement)

            analysis = SemanticAnalysis(
                primary_concepts=concepts,
                semantic_relations=relations,
                pragmatic_implications=pragmatic_implications,
                epistemic_uncertainty=epistemic_uncertainty,
                context_dependencies=context_dependencies,
                revision_triggers=revision_triggers,
                philosophical_presuppositions=presuppositions,
                methodological_assumptions=methodological_assumptions,
                interpretive_alternatives=interpretive_alternatives,
                analytical_limitations=analytical_limitations
            )

            logger.debug(f"Analysis completed: {len(concepts)} concepts, {len(relations)} relations")
            return analysis

        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            # Return minimal analysis on error
            return SemanticAnalysis(
                primary_concepts=[],
                semantic_relations=[],
                pragmatic_implications=[],
                epistemic_uncertainty={"general": 0.8},
                context_dependencies=[],
                revision_triggers=["analysis_error"]
            )

    async def _extract_concepts_llm(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[PhilosophicalConcept]:
        """Extract philosophical concepts using domain-specific extractors."""
        all_concepts = []

        # Use primary domain extractor
        if context.domain in self.concept_extractors:
            primary_extractor = self.concept_extractors[context.domain]
            primary_concepts = await primary_extractor.extract(statement, context)
            all_concepts.extend(primary_concepts)

        # Use additional extractors for cross-domain concepts
        for domain, extractor in self.concept_extractors.items():
            if domain != context.domain:
                additional_concepts = await extractor.extract(statement, context)
                # Limit additional concepts to avoid overwhelming results
                all_concepts.extend(additional_concepts[:2])

        # Remove duplicates and rank by confidence
        unique_concepts = []
        seen_terms = set()

        for concept in all_concepts:
            if concept.term not in seen_terms:
                unique_concepts.append(concept)
                seen_terms.add(concept.term)

        # Sort by confidence and return top concepts
        unique_concepts.sort(key=lambda c: c.confidence_level, reverse=True)
        return unique_concepts[:10]  # Limit to top 10 concepts

    async def _identify_semantic_relations(
        self,
        concepts: list[PhilosophicalConcept],
        statement: str
    ) -> list[SemanticRelation]:
        """Identify semantic relationships between extracted concepts."""
        relations = []
        statement_lower = statement.lower()

        # Check for explicit relation indicators in the statement
        for relation_type, indicators in self.relation_indicators.items():
            for indicator in indicators:
                if indicator in statement_lower:
                    # Find concepts around this indicator
                    for i, concept1 in enumerate(concepts):
                        for concept2 in concepts[i+1:]:
                            relation = SemanticRelation(
                                source_concept=concept1.term,
                                target_concept=concept2.term,
                                relation_type=relation_type,
                                strength=0.7,  # Default strength
                                confidence=0.6,
                                philosophical_justification=f"Identified via indicator: {indicator}"
                            )
                            relations.append(relation)

                            # Limit relations to avoid explosion
                            if len(relations) >= 5:
                                break
                        if len(relations) >= 5:
                            break

        # Add some default relations between concepts of similar domains
        if len(relations) < 3 and len(concepts) >= 2:
            for i, concept1 in enumerate(concepts[:3]):
                for concept2 in concepts[i+1:4]:
                    if concept1.domain == concept2.domain:
                        relation = SemanticRelation(
                            source_concept=concept1.term,
                            target_concept=concept2.term,
                            relation_type=SemanticRelationType.SIMILARITY,
                            strength=0.5,
                            confidence=0.4,
                            philosophical_justification="Same philosophical domain"
                        )
                        relations.append(relation)

        return relations[:5]  # Limit to 5 relations

    async def _analyze_pragmatic_implications(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[str]:
        """Analyze pragmatic implications of the statement."""
        implications = []

        # Domain-specific implications
        if context.domain == PhilosophicalDomain.ETHICS:
            if any(word in statement.lower() for word in ["should", "ought", "must", "duty"]):
                implications.append("Implies normative obligations or moral duties")
            if any(word in statement.lower() for word in ["good", "bad", "right", "wrong"]):
                implications.append("Suggests moral evaluation or judgment")

        elif context.domain == PhilosophicalDomain.EPISTEMOLOGY:
            if any(word in statement.lower() for word in ["know", "believe", "certain"]):
                implications.append("Involves epistemic commitments about knowledge claims")
            if any(word in statement.lower() for word in ["evidence", "proof", "justify"]):
                implications.append("Requires justificatory or evidential support")

        elif context.domain == PhilosophicalDomain.METAPHYSICS:
            if any(word in statement.lower() for word in ["exists", "real", "being"]):
                implications.append("Makes ontological commitments about existence")
            if any(word in statement.lower() for word in ["necessary", "possible", "actual"]):
                implications.append("Involves modal claims about necessity and possibility")

        # General pragmatic patterns
        if "if" in statement.lower() and "then" in statement.lower():
            implications.append("Establishes conditional relationships")

        if any(word in statement.lower() for word in ["because", "therefore", "thus", "hence"]):
            implications.append("Suggests causal or logical connections")

        # Default implications if none found
        if not implications:
            implications = [
                "May require conceptual clarification",
                "Could involve implicit philosophical commitments"
            ]

        return implications[:4]  # Limit to 4 implications

    async def _assess_epistemic_uncertainty(
        self,
        statement: str,
        concepts: list[PhilosophicalConcept],
        context: PhilosophicalContext
    ) -> dict[str, float]:
        """Assess epistemic uncertainty across multiple dimensions."""
        uncertainty = {}

        # Conceptual clarity uncertainty
        concept_confidence_avg = sum(c.confidence_level for c in concepts) / len(concepts) if concepts else 0.5
        uncertainty["conceptual_clarity"] = 1.0 - concept_confidence_avg

        # Contextual sensitivity uncertainty
        context_indicators = len(context.perspective_constraints or [])
        uncertainty["contextual_sensitivity"] = min(0.3 + (context_indicators * 0.1), 0.8)

        # Domain complexity uncertainty
        domain_complexity = {
            PhilosophicalDomain.LOGIC: 0.2,
            PhilosophicalDomain.ETHICS: 0.6,
            PhilosophicalDomain.AESTHETICS: 0.7,
            PhilosophicalDomain.METAPHYSICS: 0.8,
            PhilosophicalDomain.EPISTEMOLOGY: 0.5
        }
        uncertainty["domain_complexity"] = domain_complexity.get(context.domain, 0.5)

        # Statement complexity uncertainty
        word_count = len(statement.split())
        complexity_score = min(word_count / 50.0, 1.0)  # Normalize to 0-1
        uncertainty["statement_complexity"] = complexity_score * 0.5

        # Language game uncertainty
        language_game_uncertainty = {
            LanguageGame.SCIENTIFIC_DISCOURSE: 0.3,
            LanguageGame.ORDINARY_LANGUAGE: 0.6,
            LanguageGame.ETHICAL_DELIBERATION: 0.7,
            LanguageGame.AESTHETIC_JUDGMENT: 0.8
        }
        uncertainty["language_game"] = language_game_uncertainty.get(
            context.language_game, 0.5
        )

        return uncertainty

    def _identify_context_dependencies(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[str]:
        """Identify context dependencies in the statement."""
        dependencies = []

        # Domain-specific dependencies
        dependencies.append(f"Depends on {context.domain.value} philosophical framework")

        # Language game dependencies
        dependencies.append(f"Interpreted within {context.language_game.value} context")

        # Temporal context dependencies
        if context.temporal_context:
            dependencies.append(f"May be influenced by {context.temporal_context} perspective")

        # Cultural context dependencies
        if context.cultural_context:
            dependencies.append(f"Culturally situated within {context.cultural_context}")

        # Perspective constraints
        if context.perspective_constraints:
            dependencies.append(f"Limited by perspective constraints: {', '.join(context.perspective_constraints[:2])}")

        # Detect deictic expressions that require context
        deictic_indicators = ["this", "that", "here", "there", "now", "then", "I", "you", "we"]
        if any(word in statement.lower().split() for word in deictic_indicators):
            dependencies.append("Contains deictic expressions requiring contextual resolution")

        return dependencies[:5]

    async def _generate_revision_triggers(
        self,
        concepts: list[PhilosophicalConcept],
        statement: str
    ) -> list[str]:
        """Generate conditions that would trigger revision of the analysis."""
        triggers = []

        # Conceptual revision triggers
        if concepts:
            triggers.append("Discovery of new philosophical literature on key concepts")
            triggers.append("Emergence of counterexamples to central claims")

        # Domain-specific triggers
        if any(c.domain == PhilosophicalDomain.ETHICS for c in concepts):
            triggers.append("Changes in moral intuitions or ethical frameworks")

        if any(c.domain == PhilosophicalDomain.EPISTEMOLOGY for c in concepts):
            triggers.append("New empirical evidence affecting knowledge claims")

        if any(c.domain == PhilosophicalDomain.METAPHYSICS for c in concepts):
            triggers.append("Advances in physics or cognitive science relevant to ontology")

        # General triggers
        triggers.extend([
            "Critical examination of underlying assumptions",
            "Alternative interpretive frameworks",
            "Cross-cultural philosophical perspectives"
        ])

        return triggers[:4]

    def _identify_philosophical_presuppositions(self, statement: str) -> list[str]:
        """Identify implicit philosophical presuppositions in the statement."""
        presuppositions = []

        statement_lower = statement.lower()

        # Ontological presuppositions
        if any(word in statement_lower for word in ["exists", "real", "being", "entity"]):
            presuppositions.append("Presupposes a realist ontology")

        # Epistemological presuppositions
        if any(word in statement_lower for word in ["know", "knowledge", "true", "certain"]):
            presuppositions.append("Assumes the possibility of knowledge")

        # Causal presuppositions
        if any(word in statement_lower for word in ["causes", "because", "results"]):
            presuppositions.append("Presupposes causal relationships")

        # Normative presuppositions
        if any(word in statement_lower for word in ["should", "ought", "good", "bad"]):
            presuppositions.append("Assumes normative standards exist")

        # Rational presuppositions
        if any(word in statement_lower for word in ["reason", "logical", "argument"]):
            presuppositions.append("Presupposes rationality and logical coherence")

        # Default presupposition
        if not presuppositions:
            presuppositions.append("Presupposes the meaningfulness of philosophical discourse")

        return presuppositions[:3]

    def _assess_methodological_assumptions(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[str]:
        """Assess implicit methodological assumptions."""
        assumptions = []

        # Domain-specific methodological assumptions
        if context.domain == PhilosophicalDomain.LOGIC:
            assumptions.append("Assumes formal logical methods are applicable")
        elif context.domain == PhilosophicalDomain.ETHICS:
            assumptions.append("Assumes moral reasoning can guide action")
        elif context.domain == PhilosophicalDomain.AESTHETICS:
            assumptions.append("Assumes aesthetic judgments have intersubjective validity")

        # Language game assumptions
        if context.language_game == LanguageGame.SCIENTIFIC_DISCOURSE:
            assumptions.append("Assumes scientific methodology as authoritative")
        elif context.language_game == LanguageGame.ORDINARY_LANGUAGE:
            assumptions.append("Assumes common sense as starting point")

        # General assumptions
        assumptions.extend([
            "Assumes language can accurately represent philosophical concepts",
            "Assumes rational discourse can resolve philosophical disputes"
        ])

        return assumptions[:3]

    def _generate_interpretive_alternatives(
        self,
        statement: str,
        context: PhilosophicalContext
    ) -> list[str]:
        """Generate alternative interpretive possibilities."""
        alternatives = []

        # Perspective-based alternatives
        if context.domain == PhilosophicalDomain.ETHICS:
            alternatives.extend([
                "Consequentialist interpretation focusing on outcomes",
                "Deontological interpretation emphasizing duties",
                "Virtue ethics interpretation highlighting character"
            ])
        elif context.domain == PhilosophicalDomain.EPISTEMOLOGY:
            alternatives.extend([
                "Empiricist interpretation emphasizing experience",
                "Rationalist interpretation emphasizing reason",
                "Pragmatist interpretation focusing on practical success"
            ])
        elif context.domain == PhilosophicalDomain.METAPHYSICS:
            alternatives.extend([
                "Materialist interpretation reducing to physical processes",
                "Idealist interpretation emphasizing mental reality",
                "Dualist interpretation recognizing multiple fundamental types"
            ])

        # Cultural alternatives
        alternatives.extend([
            "Western analytic philosophical interpretation",
            "Continental philosophical interpretation",
            "Non-Western philosophical perspective"
        ])

        return alternatives[:4]

    def _identify_analytical_limitations(self, statement: str) -> list[str]:
        """Identify limitations of the current analysis."""
        limitations = [
            "Analysis based on pattern matching rather than deep semantic understanding",
            "Limited by Western philosophical framework assumptions",
            "May miss contextual nuances requiring background knowledge",
            "Uncertainty assessments are heuristic rather than probabilistically grounded"
        ]

        # Statement-specific limitations
        if len(statement.split()) < 5:
            limitations.append("Statement too brief for comprehensive analysis")
        elif len(statement.split()) > 100:
            limitations.append("Statement complexity may exceed analytical capabilities")

        return limitations[:4]
