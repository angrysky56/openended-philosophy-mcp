"""
Enhanced Philosophical Insight Synthesis Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module provides sophisticated synthesis algorithms for generating substantive
philosophical conclusions through multi-perspectival integration and dialectical
resolution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine

from .enhanced_nars_integration import EnhancedNARSMemory, PhilosophicalBelief
from .llm_semantic_processor import (
    LLMSemanticProcessor,
    PhilosophicalConcept,
    SemanticAnalysis,
)

logger = logging.getLogger(__name__)


@dataclass
class PhilosophicalInsight:
    """Represents a substantive philosophical insight."""
    content: str
    confidence: float
    supporting_perspectives: list[str]
    evidence_base: list[str]
    conceptual_foundations: list[PhilosophicalConcept]
    dialectical_tensions: list[str]
    practical_implications: list[str]
    revision_conditions: list[str]
    insight_type: str  # "synthetic", "analytic", "dialectical", "emergent"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerspectivalAnalysis:
    """Analysis from a specific philosophical perspective."""
    perspective: str
    interpretation: dict[str, Any]
    confidence: float
    supporting_beliefs: list[PhilosophicalBelief]
    methodological_commitments: list[str]
    key_findings: list[str]
    limitations: list[str]


@dataclass
class SynthesisPathway:
    """Pathway for dialectical synthesis."""
    type: str  # "convergence", "complementarity", "transcendence"
    description: str
    viability: float
    requirements: list[str]
    expected_outcome: str


@dataclass
class DialecticalTension:
    """Represents productive tension between perspectives."""
    perspective1: str
    perspective2: str
    interpretive_tension: dict[str, Any]
    methodological_tension: dict[str, Any]
    dialectical_potential: float
    synthesis_pathways: list[SynthesisPathway]


class EnhancedInsightSynthesis:
    """
    Advanced synthesis engine for generating substantive philosophical insights.

    This engine uses sophisticated algorithms to integrate multiple perspectives,
    resolve dialectical tensions, and generate novel philosophical insights.
    """

    def __init__(
        self,
        enhanced_memory: EnhancedNARSMemory,
        llm_processor: LLMSemanticProcessor
    ):
        self.memory = enhanced_memory
        self.llm_processor = llm_processor
        self.synthesis_cache = {}

        # Perspective frameworks
        self.perspective_frameworks = self._initialize_perspective_frameworks()

        # Synthesis strategies
        self.synthesis_strategies = {
            "convergent": self._convergent_synthesis,
            "dialectical": self._dialectical_synthesis,
            "complementary": self._complementary_synthesis,
            "emergent": self._emergent_synthesis,
            "pragmatic": self._pragmatic_synthesis
        }

        logger.info("Enhanced Insight Synthesis initialized")

    def _initialize_perspective_frameworks(self) -> dict[str, dict[str, Any]]:
        """Initialize sophisticated perspective frameworks."""
        return {
            "analytical": {
                "commitments": [
                    "logical clarity", "conceptual precision", "systematic analysis",
                    "truth as correspondence", "reductionist methodology"
                ],
                "evaluation_methods": [
                    "logical consistency check", "definitional adequacy",
                    "argument validity", "empirical grounding"
                ],
                "synthesis_strengths": ["precision", "rigor", "clarity"],
                "synthesis_weaknesses": ["experiential blindness", "context insensitivity"]
            },
            "phenomenological": {
                "commitments": [
                    "primacy of experience", "intentionality", "embodiment",
                    "temporal synthesis", "lifeworld grounding"
                ],
                "evaluation_methods": [
                    "eidetic variation", "phenomenological reduction",
                    "horizonal analysis", "constitutive analysis"
                ],
                "synthesis_strengths": ["experiential richness", "meaning depth"],
                "synthesis_weaknesses": ["intersubjective validation", "generalization"]
            },
            "pragmatist": {
                "commitments": [
                    "consequences matter", "experimental method", "fallibilism",
                    "community of inquiry", "meliorism"
                ],
                "evaluation_methods": [
                    "practical workability", "experimental testing",
                    "social validation", "problem-solving efficacy"
                ],
                "synthesis_strengths": ["practical relevance", "adaptability"],
                "synthesis_weaknesses": ["theoretical depth", "universal validity"]
            },
            "critical": {
                "commitments": [
                    "ideology critique", "power analysis", "emancipatory interest",
                    "historical materialism", "praxis orientation"
                ],
                "evaluation_methods": [
                    "genealogical analysis", "ideology critique",
                    "power mapping", "interest analysis"
                ],
                "synthesis_strengths": ["social awareness", "transformative potential"],
                "synthesis_weaknesses": ["constructive proposals", "individual agency"]
            },
            "hermeneutic": {
                "commitments": [
                    "interpretive understanding", "historical consciousness",
                    "fusion of horizons", "prejudice as condition", "tradition"
                ],
                "evaluation_methods": [
                    "textual interpretation", "historical contextualization",
                    "dialogical validation", "meaning reconstruction"
                ],
                "synthesis_strengths": ["contextual sensitivity", "meaning preservation"],
                "synthesis_weaknesses": ["critical distance", "innovation"]
            },
            "existentialist": {
                "commitments": [
                    "existence precedes essence", "radical freedom", "authenticity",
                    "anxiety as revelatory", "finitude awareness"
                ],
                "evaluation_methods": [
                    "authenticity assessment", "freedom analysis",
                    "existential mood attunement", "choice examination"
                ],
                "synthesis_strengths": ["individual focus", "freedom emphasis"],
                "synthesis_weaknesses": ["social dimension", "rational justification"]
            }
        }

    async def synthesize_insights(
        self,
        inquiry_focus: str,
        available_perspectives: list[str],
        depth_level: int = 3,
        synthesis_strategy: str = "auto"
    ) -> list[PhilosophicalInsight]:
        """
        Generate substantive philosophical insights through multi-perspectival synthesis.

        This method orchestrates the entire synthesis process from perspective
        application through dialectical resolution to insight generation.
        """
        logger.info(f"Synthesizing insights for: {inquiry_focus}")

        # Analyze inquiry focus
        from .llm_semantic_processor import PhilosophicalContext

        phil_context = PhilosophicalContext(
            domain="philosophical_synthesis",
            inquiry_type="synthesis",
            depth_requirements=depth_level
        )
        focus_analysis = await self.llm_processor.analyze_statement(
            inquiry_focus, phil_context
        )

        # Retrieve relevant beliefs from enhanced memory
        relevant_beliefs = await self._retrieve_relevant_beliefs(
            focus_analysis, depth_level
        )

        # Apply multiple perspectives
        perspectival_analyses = await self._apply_multiple_perspectives(
            relevant_beliefs, available_perspectives, focus_analysis
        )

        # Identify dialectical tensions
        dialectical_tensions = await self._identify_dialectical_tensions(
            perspectival_analyses
        )

        # Choose synthesis strategy
        if synthesis_strategy == "auto":
            synthesis_strategy = self._select_synthesis_strategy(
                perspectival_analyses, dialectical_tensions
            )

        # Apply synthesis strategy
        strategy_func = self.synthesis_strategies.get(
            synthesis_strategy,
            self._emergent_synthesis
        )

        raw_insights = await strategy_func(
            perspectival_analyses, dialectical_tensions, focus_analysis
        )

        # Refine and substantiate insights
        substantive_insights = await self._substantiate_insights(
            raw_insights, relevant_beliefs, focus_analysis
        )

        # Add meta-insights about the synthesis process
        meta_insights = self._generate_meta_insights(
            substantive_insights, synthesis_strategy, dialectical_tensions
        )
        substantive_insights.extend(meta_insights)

        return substantive_insights

    async def _retrieve_relevant_beliefs(
        self,
        focus_analysis: SemanticAnalysis,
        depth_level: int
    ) -> list[PhilosophicalBelief]:
        """Retrieve beliefs relevant to the inquiry focus."""
        # Get concepts from focus
        focus_concepts = {c.term for c in focus_analysis.primary_concepts}

        # Calculate relevance radius based on depth
        relevance_radius = 0.7 - (depth_level * 0.1)  # Deeper = broader

        relevant_beliefs = []

        for _belief_id, belief in self.memory.philosophical_beliefs.items():
            # Calculate semantic relevance
            if belief.semantic_embedding is not None:
                # Create focus embedding (simplified)
                focus_embedding = np.mean([
                    self.memory._generate_semantic_embedding(c.term)
                    for c in focus_analysis.primary_concepts
                ], axis=0)

                similarity = 1 - cosine(belief.semantic_embedding, focus_embedding)

                if similarity > relevance_radius:
                    relevant_beliefs.append((similarity, belief))
            else:
                # Fallback to keyword matching
                belief_words = set(belief.statement.lower().split())
                if focus_concepts & belief_words:
                    relevant_beliefs.append((0.5, belief))

        # Sort by relevance
        relevant_beliefs.sort(key=lambda x: x[0], reverse=True)

        # Return top beliefs based on depth
        max_beliefs = depth_level * 10
        return [belief for _, belief in relevant_beliefs[:max_beliefs]]

    async def _apply_multiple_perspectives(
        self,
        beliefs: list[PhilosophicalBelief],
        perspectives: list[str],
        focus_analysis: SemanticAnalysis
    ) -> list[PerspectivalAnalysis]:
        """Apply multiple philosophical perspectives to belief sets."""
        analyses = []

        for perspective in perspectives:
            if perspective not in self.perspective_frameworks:
                logger.warning(f"Unknown perspective: {perspective}")
                continue

            framework = self.perspective_frameworks[perspective]

            # Filter beliefs relevant to this perspective
            relevant_beliefs = self._filter_beliefs_for_perspective(
                beliefs, perspective, framework
            )

            # Apply perspective-specific interpretation
            interpretation = await self._interpret_through_perspective(
                relevant_beliefs, perspective, framework, focus_analysis
            )

            # Extract key findings
            key_findings = self._extract_key_findings(
                interpretation, perspective, framework
            )

            # Assess confidence
            confidence = self._assess_perspective_confidence(
                interpretation, relevant_beliefs, framework
            )

            # Identify limitations
            limitations = self._identify_perspective_limitations(
                perspective, focus_analysis, relevant_beliefs
            )

            analyses.append(PerspectivalAnalysis(
                perspective=perspective,
                interpretation=interpretation,
                confidence=confidence,
                supporting_beliefs=relevant_beliefs[:5],  # Top 5
                methodological_commitments=framework["commitments"],
                key_findings=key_findings,
                limitations=limitations
            ))

        return analyses

    def _filter_beliefs_for_perspective(
        self,
        beliefs: list[PhilosophicalBelief],
        perspective: str,
        framework: dict[str, Any]
    ) -> list[PhilosophicalBelief]:
        """Filter beliefs relevant to a specific perspective."""
        relevant_beliefs = []

        # Perspective-specific filtering
        if perspective == "analytical":
            # Focus on logical and definitional beliefs
            for belief in beliefs:
                if any(term in belief.statement.lower()
                       for term in ["definition", "logic", "necessarily", "implies"]) or belief.truth.confidence > 0.8:
                    relevant_beliefs.append(belief)

        elif perspective == "phenomenological":
            # Focus on experiential beliefs
            for belief in beliefs:
                if any(term in belief.statement.lower()
                       for term in ["experience", "consciousness", "appears", "feels"]) or belief.temporal_scope == "temporal":
                    relevant_beliefs.append(belief)

        elif perspective == "pragmatist":
            # Focus on practical beliefs
            for belief in beliefs:
                if any(term in belief.statement.lower()
                       for term in ["works", "useful", "consequence", "practice"]) or belief.philosophical_context.get("action"):
                    relevant_beliefs.append(belief)

        # Add generic relevance for other perspectives
        if not relevant_beliefs:
            relevant_beliefs = beliefs[:10]  # Take top 10 as fallback

        return relevant_beliefs

    async def _interpret_through_perspective(
        self,
        beliefs: list[PhilosophicalBelief],
        perspective: str,
        framework: dict[str, Any],
        focus_analysis: SemanticAnalysis
    ) -> dict[str, Any]:
        """Interpret beliefs through a specific philosophical perspective."""
        interpretation = {
            "perspective": perspective,
            "central_claims": [],
            "supporting_evidence": [],
            "conceptual_structure": {},
            "evaluative_judgments": [],
            "methodological_applications": []
        }

        # Apply evaluation methods
        for method in framework["evaluation_methods"]:
            method_result = await self._apply_evaluation_method(
                method, beliefs, focus_analysis
            )
            interpretation["methodological_applications"].append({
                "method": method,
                "result": method_result
            })

        # Extract central claims based on perspective
        if perspective == "analytical":
            # Look for definitional and logical claims
            for belief in beliefs[:5]:
                if "-->" in belief.narsese_term or "==>" in belief.narsese_term:
                    interpretation["central_claims"].append({
                        "claim": belief.statement,
                        "logical_form": belief.narsese_term,
                        "confidence": belief.truth.confidence
                    })

        elif perspective == "phenomenological":
            # Look for experiential descriptions
            for belief in beliefs[:5]:
                if belief.temporal_scope == "temporal" or "experience" in belief.statement:
                    interpretation["central_claims"].append({
                        "claim": belief.statement,
                        "experiential_mode": "first-person",
                        "temporality": belief.temporal_scope
                    })

        # Build conceptual structure
        concepts = {}
        for belief in beliefs:
            # Extract concepts (simplified)
            words = belief.statement.split()
            for word in words:
                if len(word) > 5:  # Simple heuristic
                    concepts[word] = concepts.get(word, 0) + belief.truth.confidence

        interpretation["conceptual_structure"] = dict(
            sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Generate evaluative judgments
        interpretation["evaluative_judgments"] = self._generate_evaluative_judgments(
            perspective, beliefs, focus_analysis
        )

        return interpretation

    async def _apply_evaluation_method(
        self,
        method: str,
        beliefs: list[PhilosophicalBelief],
        focus_analysis: SemanticAnalysis
    ) -> dict[str, Any]:
        """Apply specific evaluation method from perspective."""
        result = {"method": method, "findings": [], "confidence": 0.5}

        if method == "logical consistency check":
            # Check for contradictions
            for i, belief1 in enumerate(beliefs):
                for belief2 in beliefs[i+1:]:
                    if (belief1.truth.frequency > 0.7 and
                        belief2.truth.frequency < 0.3 and
                        self._beliefs_about_same_topic(belief1, belief2)):
                        result["findings"].append(
                            f"Potential contradiction: {belief1.statement[:50]}... "
                            f"vs {belief2.statement[:50]}..."
                        )
            result["confidence"] = 0.8 if not result["findings"] else 0.4

        elif method == "eidetic variation":
            # Phenomenological method - vary properties to find essence
            concept = focus_analysis.primary_concepts[0].term if focus_analysis.primary_concepts else "phenomenon"
            variations = []
            for belief in beliefs[:5]:
                if concept in belief.statement:
                    variations.append(belief.statement)

            if variations:
                result["findings"].append(
                    f"Essential structure of {concept} persists across variations"
                )
                result["confidence"] = 0.7

        elif method == "practical workability":
            # Pragmatist method - assess practical consequences
            practical_beliefs = [b for b in beliefs if "consequence" in b.statement or "practice" in b.statement]
            if practical_beliefs:
                avg_confidence = np.mean([b.truth.confidence for b in practical_beliefs])
                result["findings"].append(
                    f"Practical viability: {avg_confidence:.2f}"
                )
                result["confidence"] = avg_confidence

        # Add more evaluation methods as needed

        return result

    def _beliefs_about_same_topic(
        self,
        belief1: PhilosophicalBelief,
        belief2: PhilosophicalBelief
    ) -> bool:
        """Check if two beliefs are about the same topic."""
        # Simple overlap check
        words1 = set(belief1.statement.lower().split())
        words2 = set(belief2.statement.lower().split())

        # Remove common words
        common_words = {"the", "is", "are", "was", "were", "a", "an", "and", "or", "but"}
        words1 -= common_words
        words2 -= common_words

        overlap = words1 & words2
        return len(overlap) >= 2  # At least 2 words in common

    def _extract_key_findings(
        self,
        interpretation: dict[str, Any],
        perspective: str,
        framework: dict[str, Any]
    ) -> list[str]:
        """Extract key findings from perspective interpretation."""
        findings = []

        # Extract from central claims
        for claim in interpretation["central_claims"][:3]:
            if isinstance(claim, dict):
                findings.append(f"{perspective}: {claim.get('claim', '')}")

        # Extract from methodological applications
        for application in interpretation["methodological_applications"]:
            if application["result"]["confidence"] > 0.6:
                for finding in application["result"]["findings"][:1]:
                    findings.append(f"{application['method']}: {finding}")

        # Add perspective-specific insights
        if interpretation["conceptual_structure"]:
            top_concept = list(interpretation["conceptual_structure"].keys())[0]
            findings.append(f"Central concept from {perspective}: {top_concept}")

        return findings[:5]  # Limit to 5 key findings

    def _assess_perspective_confidence(
        self,
        interpretation: dict[str, Any],
        beliefs: list[PhilosophicalBelief],
        framework: dict[str, Any]
    ) -> float:
        """Assess confidence in perspective interpretation."""
        confidence_factors = []

        # Belief support
        if beliefs:
            avg_belief_confidence = np.mean([b.truth.confidence for b in beliefs[:10]])
            confidence_factors.append(avg_belief_confidence)

        # Methodological success
        method_confidences = [
            app["result"]["confidence"]
            for app in interpretation["methodological_applications"]
        ]
        if method_confidences:
            confidence_factors.append(np.mean(method_confidences))

        # Central claims strength
        if interpretation["central_claims"]:
            confidence_factors.append(0.7)  # Has central claims
        else:
            confidence_factors.append(0.3)  # No central claims

        return float(np.mean(confidence_factors)) if confidence_factors else 0.5

    def _identify_perspective_limitations(
        self,
        perspective: str,
        focus_analysis: SemanticAnalysis,
        beliefs: list[PhilosophicalBelief]
    ) -> list[str]:
        """Identify limitations of applying this perspective."""
        limitations = []

        framework = self.perspective_frameworks[perspective]

        # Add known weaknesses
        limitations.extend([
            f"{perspective} limitation: {weakness}"
            for weakness in framework["synthesis_weaknesses"]
        ])

        # Context-specific limitations
        if not beliefs:
            limitations.append(f"Limited evidence base for {perspective} analysis")

        if focus_analysis.epistemic_uncertainty > 0.7:
            limitations.append(f"High uncertainty limits {perspective} conclusions")

        # Check for missing evaluation methods
        successful_methods = sum(
            1 for method in framework["evaluation_methods"]
            if any(b for b in beliefs if method.split()[0] in b.statement.lower())
        )

        if successful_methods < len(framework["evaluation_methods"]) / 2:
            limitations.append(f"Many {perspective} methods not fully applicable")

        return limitations[:3]  # Top 3 limitations

    def _generate_evaluative_judgments(
        self,
        perspective: str,
        beliefs: list[PhilosophicalBelief],
        focus_analysis: SemanticAnalysis
    ) -> list[str]:
        """Generate evaluative judgments from perspective."""
        judgments = []

        if perspective == "analytical":
            # Assess logical structure
            logical_beliefs = [b for b in beliefs if "==>" in b.narsese_term or "-->" in b.narsese_term]
            if logical_beliefs:
                judgments.append(
                    f"Logical structure is {'well-defined' if len(logical_beliefs) > 3 else 'emerging'}"
                )

        elif perspective == "phenomenological":
            # Assess experiential richness
            experiential_beliefs = [b for b in beliefs if "experience" in b.statement.lower()]
            if experiential_beliefs:
                judgments.append(
                    f"Experiential dimension is {'rich' if len(experiential_beliefs) > 2 else 'present'}"
                )

        elif perspective == "pragmatist":
            # Assess practical value
            avg_confidence = np.mean([b.truth.confidence for b in beliefs[:5]]) if beliefs else 0.5
            judgments.append(
                f"Practical reliability: {'high' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'low'}"
            )

        # Add generic judgment
        if beliefs:
            judgments.append(
                f"Overall {perspective} assessment: "
                f"{'strong' if len(beliefs) > 10 else 'adequate' if len(beliefs) > 5 else 'limited'} "
                f"evidence base"
            )

        return judgments

    async def _identify_dialectical_tensions(
        self,
        perspectival_analyses: list[PerspectivalAnalysis]
    ) -> list[DialecticalTension]:
        """Identify productive tensions between perspectives."""
        tensions = []

        for i, analysis1 in enumerate(perspectival_analyses):
            for _j, analysis2 in enumerate(perspectival_analyses[i+1:], i+1):
                # Analyze interpretive differences
                interpretive_tension = self._analyze_interpretive_tension(
                    analysis1, analysis2
                )

                # Assess methodological conflicts
                methodological_tension = self._assess_methodological_conflicts(
                    analysis1.methodological_commitments,
                    analysis2.methodological_commitments
                )

                # Calculate dialectical potential
                dialectical_potential = self._calculate_dialectical_potential(
                    interpretive_tension, methodological_tension
                )

                if dialectical_potential > 0.6:  # Significant tension
                    # Identify synthesis pathways
                    synthesis_pathways = await self._identify_synthesis_pathways(
                        analysis1, analysis2, interpretive_tension
                    )

                    tensions.append(DialecticalTension(
                        perspective1=analysis1.perspective,
                        perspective2=analysis2.perspective,
                        interpretive_tension=interpretive_tension,
                        methodological_tension=methodological_tension,
                        dialectical_potential=dialectical_potential,
                        synthesis_pathways=synthesis_pathways
                    ))

        return tensions

    def _analyze_interpretive_tension(
        self,
        analysis1: PerspectivalAnalysis,
        analysis2: PerspectivalAnalysis
    ) -> dict[str, Any]:
        """Analyze tension in interpretations."""
        tension = {
            "conflicting_claims": [],
            "conceptual_divergence": 0.0,
            "evaluative_disagreement": []
        }

        # Compare central claims
        claims1 = set()
        claims2 = set()

        for claim in analysis1.interpretation.get("central_claims", []):
            if isinstance(claim, dict):
                claims1.add(claim.get("claim", ""))

        for claim in analysis2.interpretation.get("central_claims", []):
            if isinstance(claim, dict):
                claims2.add(claim.get("claim", ""))

        # Look for contradictions (simplified)
        for c1 in claims1:
            for c2 in claims2:
                if c1 and c2 and ("not" in c1 and c2 in c1) or ("not" in c2 and c1 in c2):
                    tension["conflicting_claims"].append((c1[:50], c2[:50]))

        # Compare conceptual structures
        concepts1 = set(analysis1.interpretation.get("conceptual_structure", {}).keys())
        concepts2 = set(analysis2.interpretation.get("conceptual_structure", {}).keys())

        if concepts1 and concepts2:
            overlap = len(concepts1 & concepts2)
            total = len(concepts1 | concepts2)
            tension["conceptual_divergence"] = 1.0 - (overlap / total if total > 0 else 0)

        # Compare evaluative judgments
        judgments1 = analysis1.interpretation.get("evaluative_judgments", [])
        judgments2 = analysis2.interpretation.get("evaluative_judgments", [])

        # Simple disagreement detection
        if ("strong" in str(judgments1) and "weak" in str(judgments2)) or \
           ("high" in str(judgments1) and "low" in str(judgments2)):
            tension["evaluative_disagreement"].append(
                "Significant evaluative disagreement detected"
            )

        return tension

    def _assess_methodological_conflicts(
        self,
        commitments1: list[str],
        commitments2: list[str]
    ) -> dict[str, Any]:
        """Assess conflicts in methodological commitments."""
        conflicts = {
            "conflicting_commitments": [],
            "incompatible_methods": [],
            "conflict_severity": 0.0
        }

        # Known conflicts between commitments
        conflict_pairs = [
            ("reductionist methodology", "holistic understanding"),
            ("logical clarity", "experiential richness"),
            ("universal validity", "contextual sensitivity"),
            ("objective truth", "interpretive understanding"),
            ("individual focus", "social awareness")
        ]

        for c1 in commitments1:
            for c2 in commitments2:
                for pair in conflict_pairs:
                    if (c1 in pair[0] and c2 in pair[1]) or (c1 in pair[1] and c2 in pair[0]):
                        conflicts["conflicting_commitments"].append((c1, c2))

        # Calculate severity
        if conflicts["conflicting_commitments"]:
            conflicts["conflict_severity"] = len(conflicts["conflicting_commitments"]) / \
                                           (len(commitments1) + len(commitments2))

        return conflicts

    def _calculate_dialectical_potential(
        self,
        interpretive_tension: dict[str, Any],
        methodological_tension: dict[str, Any]
    ) -> float:
        """Calculate potential for productive dialectical synthesis."""
        factors = []

        # Interpretive tension contributes positively (up to a point)
        if interpretive_tension["conflicting_claims"]:
            factors.append(0.8)  # Clear opposition enables dialectic

        conceptual_divergence = interpretive_tension["conceptual_divergence"]
        if 0.3 < conceptual_divergence < 0.7:
            factors.append(0.9)  # Moderate divergence is productive
        elif conceptual_divergence > 0.9:
            factors.append(0.3)  # Too divergent to synthesize

        # Methodological tension
        severity = methodological_tension["conflict_severity"]
        if 0.2 < severity < 0.5:
            factors.append(0.7)  # Some conflict is productive
        elif severity > 0.7:
            factors.append(0.4)  # Too much conflict

        return float(np.mean(factors)) if factors else 0.5

    async def _identify_synthesis_pathways(
        self,
        analysis1: PerspectivalAnalysis,
        analysis2: PerspectivalAnalysis,
        tension: dict[str, Any]
    ) -> list[SynthesisPathway]:
        """Identify potential pathways for synthesis."""
        pathways = []

        # Convergence pathway - find common ground
        common_concepts = set(analysis1.interpretation.get("conceptual_structure", {}).keys()) & \
                         set(analysis2.interpretation.get("conceptual_structure", {}).keys())

        if common_concepts:
            pathways.append(SynthesisPathway(
                type="convergence",
                description=f"Build on shared concepts: {', '.join(list(common_concepts)[:3])}",
                viability=0.8,
                requirements=["Focus on shared conceptual ground", "Minimize conflicting claims"],
                expected_outcome="Unified understanding based on common elements"
            ))

        # Complementarity pathway - different aspects of same phenomenon
        if tension["conceptual_divergence"] < 0.7:
            pathways.append(SynthesisPathway(
                type="complementarity",
                description=f"Integrate {analysis1.perspective} and {analysis2.perspective} as complementary",
                viability=0.7,
                requirements=["Recognize different levels of analysis", "Avoid reductionism"],
                expected_outcome="Multi-layered understanding preserving both perspectives"
            ))

        # Transcendence pathway - move to higher level
        if tension["conflicting_claims"]:
            pathways.append(SynthesisPathway(
                type="transcendence",
                description="Transcend opposition through higher-order framework",
                viability=0.6,
                requirements=["Identify underlying assumptions", "Create meta-framework"],
                expected_outcome="Novel perspective that sublates both positions"
            ))

        return pathways

    def _select_synthesis_strategy(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension]
    ) -> str:
        """Select appropriate synthesis strategy based on analyses."""
        # High tension suggests dialectical approach
        if tensions and any(t.dialectical_potential > 0.8 for t in tensions):
            return "dialectical"

        # High agreement suggests convergent approach
        avg_confidence = np.mean([a.confidence for a in analyses])
        if avg_confidence > 0.8 and not tensions:
            return "convergent"

        # Multiple perspectives with moderate agreement suggests complementary
        if len(analyses) > 3 and 0.5 < avg_confidence < 0.8:
            return "complementary"

        # Low confidence or few perspectives suggests emergent
        if avg_confidence < 0.5 or len(analyses) < 3:
            return "emergent"

        # Default to pragmatic
        return "pragmatic"

    async def _convergent_synthesis(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension],
        focus_analysis: SemanticAnalysis
    ) -> list[dict[str, Any]]:
        """Synthesize through convergence on common insights."""
        insights = []

        # Find convergent findings across perspectives
        all_findings = []
        for analysis in analyses:
            all_findings.extend(analysis.key_findings)

        # Count occurrences (simplified similarity)
        finding_counts = {}
        for finding in all_findings:
            key = finding[:30]  # First 30 chars as key
            finding_counts[key] = finding_counts.get(key, [])
            finding_counts[key].append(finding)

        # Create insights from convergent findings
        for key, findings in finding_counts.items():
            if len(findings) >= 2:  # Found in multiple perspectives
                perspectives = []
                for analysis in analyses:
                    if any(key in f for f in analysis.key_findings):
                        perspectives.append(analysis.perspective)

                insights.append({
                    "content": f"Convergent insight: {findings[0]}",
                    "type": "convergent",
                    "supporting_perspectives": perspectives,
                    "confidence": min(0.9, len(perspectives) / len(analyses))
                })

        return insights

    async def _dialectical_synthesis(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension],
        focus_analysis: SemanticAnalysis
    ) -> list[dict[str, Any]]:
        """Synthesize through dialectical resolution."""
        insights = []

        for tension in tensions:
            if tension.synthesis_pathways:
                # Take best pathway
                best_pathway = max(tension.synthesis_pathways, key=lambda p: p.viability)

                # Create dialectical insight
                insight_content = (
                    f"Dialectical synthesis of {tension.perspective1} and {tension.perspective2}: "
                    f"{best_pathway.description}. "
                    f"This {best_pathway.type} approach {best_pathway.expected_outcome.lower()}"
                )

                insights.append({
                    "content": insight_content,
                    "type": "dialectical",
                    "supporting_perspectives": [tension.perspective1, tension.perspective2],
                    "confidence": best_pathway.viability,
                    "synthesis_type": best_pathway.type,
                    "requirements": best_pathway.requirements
                })

        return insights

    async def _complementary_synthesis(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension],
        focus_analysis: SemanticAnalysis
    ) -> list[dict[str, Any]]:
        """Synthesize by showing complementary aspects."""
        insights = []

        # Group perspectives by what they emphasize
        emphasis_groups = {
            "experiential": ["phenomenological", "existentialist"],
            "logical": ["analytical", "formal"],
            "practical": ["pragmatist", "critical"],
            "interpretive": ["hermeneutic", "postmodern"]
        }

        covered_groups = set()
        for analysis in analyses:
            for group, perspectives in emphasis_groups.items():
                if analysis.perspective in perspectives:
                    covered_groups.add(group)

        if len(covered_groups) >= 2:
            insight_content = (
                f"Complementary synthesis reveals multiple dimensions: "
                f"{', '.join(covered_groups)} aspects all contribute to understanding. "
                f"Each perspective illuminates different facets without contradiction."
            )

            insights.append({
                "content": insight_content,
                "type": "complementary",
                "supporting_perspectives": [a.perspective for a in analyses],
                "confidence": 0.75,
                "dimensions_covered": list(covered_groups)
            })

        return insights

    async def _emergent_synthesis(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension],
        focus_analysis: SemanticAnalysis
    ) -> list[dict[str, Any]]:
        """Synthesize by identifying emergent patterns."""
        insights = []

        # Look for unexpected patterns across analyses
        all_concepts = []
        for analysis in analyses:
            all_concepts.extend(analysis.interpretation.get("conceptual_structure", {}).keys())

        # Find concepts that appear in unexpected combinations
        concept_pairs = {}
        for i, c1 in enumerate(all_concepts):
            for c2 in all_concepts[i+1:]:
                if c1 != c2:
                    pair = tuple(sorted([c1, c2]))
                    concept_pairs[pair] = concept_pairs.get(pair, 0) + 1

        # Identify emergent patterns
        emergent_pairs = [pair for pair, count in concept_pairs.items() if count >= 2]

        if emergent_pairs:
            pair = emergent_pairs[0]  # Take most common
            insight_content = (
                f"Emergent pattern: The relationship between '{pair[0]}' and '{pair[1]}' "
                f"appears across multiple perspectives, suggesting a deeper structure "
                f"not fully captured by any single viewpoint."
            )

            insights.append({
                "content": insight_content,
                "type": "emergent",
                "supporting_perspectives": [a.perspective for a in analyses],
                "confidence": 0.6,
                "emergent_structure": pair
            })

        return insights

    async def _pragmatic_synthesis(
        self,
        analyses: list[PerspectivalAnalysis],
        tensions: list[DialecticalTension],
        focus_analysis: SemanticAnalysis
    ) -> list[dict[str, Any]]:
        """Synthesize with focus on practical implications."""
        insights = []

        # Extract practical implications from each analysis
        practical_elements = []

        for analysis in analyses:
            # Look for action-oriented findings
            for finding in analysis.key_findings:
                if any(term in finding.lower()
                       for term in ["practice", "action", "consequence", "useful", "work"]):
                    practical_elements.append((analysis.perspective, finding))

        if practical_elements:
            insight_content = (
                f"Pragmatic synthesis: Despite theoretical differences, "
                f"perspectives converge on practical implications. "
                f"Key actionable insight: {practical_elements[0][1]}"
            )

            insights.append({
                "content": insight_content,
                "type": "pragmatic",
                "supporting_perspectives": [p for p, _ in practical_elements],
                "confidence": 0.7,
                "practical_focus": True
            })

        return insights

    async def _substantiate_insights(
        self,
        raw_insights: list[dict[str, Any]],
        beliefs: list[PhilosophicalBelief],
        focus_analysis: SemanticAnalysis
    ) -> list[PhilosophicalInsight]:
        """Convert raw insights into substantive philosophical insights."""
        substantive_insights = []

        for raw in raw_insights:
            # Find supporting evidence from beliefs
            supporting_evidence = []
            for belief in beliefs[:10]:
                # Simple relevance check
                if any(concept.term in belief.statement
                       for concept in focus_analysis.primary_concepts):
                    supporting_evidence.append(belief.statement[:100] + "...")

            # Generate practical implications
            practical_implications = self._generate_practical_implications(
                raw["content"], raw.get("type", "emergent")
            )

            # Create revision conditions
            revision_conditions = [
                "New empirical evidence contradicting core claims",
                "Theoretical advances in constituent perspectives",
                "Pragmatic failure in application"
            ]

            if raw.get("type") == "dialectical":
                revision_conditions.append("Resolution of underlying tensions")

            # Build substantive insight
            insight = PhilosophicalInsight(
                content=raw["content"],
                confidence=raw.get("confidence", 0.5),
                supporting_perspectives=raw.get("supporting_perspectives", []),
                evidence_base=supporting_evidence[:3],
                conceptual_foundations=focus_analysis.primary_concepts[:3],
                dialectical_tensions=[str(t) for t in raw.get("requirements", [])],
                practical_implications=practical_implications,
                revision_conditions=revision_conditions[:3],
                insight_type=raw.get("type", "emergent")
            )

            substantive_insights.append(insight)

        return substantive_insights

    def _generate_practical_implications(
        self,
        insight_content: str,
        insight_type: str
    ) -> list[str]:
        """Generate practical implications of insight."""
        implications = []

        if insight_type == "convergent":
            implications.append("High confidence enables practical application")
            implications.append("Consensus across perspectives supports action")

        elif insight_type == "dialectical":
            implications.append("Tensions require careful navigation in practice")
            implications.append("Both perspectives should inform action")

        elif insight_type == "complementary":
            implications.append("Multiple dimensions must be considered")
            implications.append("Holistic approach needed for implementation")

        elif insight_type == "emergent":
            implications.append("Novel patterns suggest new possibilities")
            implications.append("Experimental approach warranted")

        elif insight_type == "pragmatic":
            implications.append("Direct application to practice possible")
            implications.append("Focus on workability over theory")

        return implications[:2]

    def _generate_meta_insights(
        self,
        insights: list[PhilosophicalInsight],
        strategy: str,
        tensions: list[DialecticalTension]
    ) -> list[PhilosophicalInsight]:
        """Generate meta-level insights about the synthesis process."""
        meta_insights = []

        # Insight about synthesis strategy effectiveness
        avg_confidence = np.mean([i.confidence for i in insights]) if insights else 0.5

        meta_content = (
            f"Meta-insight: The {strategy} synthesis strategy "
            f"{'successfully integrated' if avg_confidence > 0.7 else 'partially integrated'} "
            f"{len({p for i in insights for p in i.supporting_perspectives})} perspectives. "
        )

        if tensions:
            meta_content += (
                f"The presence of {len(tensions)} dialectical tensions "
                f"{'enriched' if strategy == 'dialectical' else 'complicated'} the synthesis."
            )

        meta_insights.append(PhilosophicalInsight(
            content=meta_content,
            confidence=0.8,
            supporting_perspectives=["meta-philosophical"],
            evidence_base=[f"Synthesis of {len(insights)} insights"],
            conceptual_foundations=[],
            dialectical_tensions=[],
            practical_implications=["Synthesis methodology affects conclusions"],
            revision_conditions=["Alternative synthesis strategies may yield different insights"],
            insight_type="meta"
        ))

        # Insight about philosophical methodology
        if len(insights) > 3 and avg_confidence > 0.6:
            meta_insights.append(PhilosophicalInsight(
                content=(
                    "Meta-insight: Multi-perspectival synthesis demonstrates that "
                    "philosophical understanding benefits from methodological pluralism. "
                    "No single perspective captures the full phenomenon."
                ),
                confidence=0.85,
                supporting_perspectives=["meta-philosophical", "pragmatist"],
                evidence_base=["Multiple successful perspective applications"],
                conceptual_foundations=[],
                dialectical_tensions=[],
                practical_implications=[
                    "Philosophical inquiry should embrace multiple methods",
                    "Synthesis skills are as important as analysis"
                ],
                revision_conditions=["Discovery of uniquely privileged perspective"],
                insight_type="meta"
            ))

        return meta_insights
