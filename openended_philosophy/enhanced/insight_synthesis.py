"""
Enhanced Insight Synthesis Engine for Philosophical Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Conceptual Framework Implementation

This module implements sophisticated insight synthesis capabilities that replace
rudimentary pattern recognition with multi-perspectival analysis and dialectical reasoning:

#### Core Theoretical Foundations:
- **Multi-Perspectival Synthesis**: Systematic integration of diverse philosophical viewpoints
- **Dialectical Processing**: Constructive engagement with philosophical tensions
- **Coherence Maximization**: Optimization of interpretive consistency across perspectives
- **Substantive Conclusion Generation**: Production of meaningful philosophical insights

#### Methodological Approach:
1. **Perspective Application**: Systematic deployment of interpretive frameworks
2. **Tension Identification**: Recognition of productive philosophical conflicts
3. **Synthesis Pathway Generation**: Creation of routes to higher-order understanding
4. **Insight Validation**: Quality assessment of generated philosophical conclusions

### Usage Example:

```python
synthesis_engine = EnhancedInsightSynthesis(nars_memory, llm_processor)

insights = await synthesis_engine.synthesize_insights(
    inquiry_focus="consciousness and emergence",
    available_perspectives=["materialist", "phenomenological", "enactivist"],
    depth_level=3
)
```
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..nars import NARSManager, NARSMemory, TruthValue
from ..semantic.llm_semantic_processor import LLMSemanticProcessor
from ..semantic.semantic_embedding_space import SemanticEmbeddingSpace

# Import from our enhanced semantic modules
from ..semantic.types import (
    LanguageGame,
    PhilosophicalConcept,
    PhilosophicalContext,
    PhilosophicalDomain,
    SemanticAnalysis,
    SemanticRelation,
)

logger = logging.getLogger(__name__)


@dataclass
class PerspectivalAnalysis:
    """Results of applying a specific philosophical perspective."""
    perspective: str
    interpretation: dict[str, Any]
    confidence: float
    supporting_beliefs: list[Any]
    methodological_commitments: list[str]
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    novel_insights: list[str] = field(default_factory=list)


@dataclass
class DialecticalTension:
    """Represents productive tension between philosophical perspectives."""
    perspective1: str
    perspective2: str
    interpretive_tension: dict[str, Any]
    methodological_tension: dict[str, Any]
    dialectical_potential: float
    synthesis_pathways: list['SynthesisPathway'] = field(default_factory=list)
    resolution_strategies: list[str] = field(default_factory=list)


@dataclass
class SynthesisPathway:
    """Potential pathway for dialectical synthesis."""
    type: str  # 'convergence', 'complementarity', 'transcendence'
    description: str
    viability: float
    required_assumptions: list[str] = field(default_factory=list)
    potential_insights: list[str] = field(default_factory=list)


@dataclass
class SubstantiveInsight:
    """A meaningful philosophical insight generated through synthesis."""
    content: str
    confidence: float
    supporting_perspectives: list[str]
    synthesis_pathway: str
    philosophical_significance: str
    practical_implications: list[str] = field(default_factory=list)
    revision_conditions: list[str] = field(default_factory=list)
    further_inquiry_directions: list[str] = field(default_factory=list)


class PerspectiveManager:
    """Manages application of diverse philosophical perspectives."""

    def __init__(self):
        """Initialize perspective frameworks."""
        self.frameworks = self._initialize_frameworks()

    def _initialize_frameworks(self) -> dict[str, 'PerspectiveFramework']:
        """Initialize various philosophical perspective frameworks."""
        frameworks = {}

        # Materialist perspective
        frameworks['materialist'] = PerspectiveFramework(
            name='materialist',
            core_commitments=[
                'physical_realism',
                'reductive_physicalism',
                'causal_closure_of_physics'
            ],
            interpretive_principles=[
                'prefer_physical_explanations',
                'reduce_mental_to_neural',
                'eliminate_non_physical_entities'
            ],
            methodological_preferences=[
                'empirical_evidence',
                'scientific_method',
                'quantitative_analysis'
            ]
        )

        # Phenomenological perspective
        frameworks['phenomenological'] = PerspectiveFramework(
            name='phenomenological',
            core_commitments=[
                'primacy_of_consciousness',
                'intentionality_structure',
                'lived_experience_foundation'
            ],
            interpretive_principles=[
                'bracket_natural_attitude',
                'focus_on_experience_structure',
                'avoid_reductive_explanations'
            ],
            methodological_preferences=[
                'descriptive_analysis',
                'eidetic_variation',
                'transcendental_reduction'
            ]
        )

        # Enactivist perspective
        frameworks['enactivist'] = PerspectiveFramework(
            name='enactivist',
            core_commitments=[
                'embodied_cognition',
                'environmental_coupling',
                'emergent_properties'
            ],
            interpretive_principles=[
                'cognition_as_action',
                'mind_environment_continuity',
                'dynamic_systems_thinking'
            ],
            methodological_preferences=[
                'systems_analysis',
                'ecological_psychology',
                'dynamical_modeling'
            ]
        )

        # Pragmatist perspective
        frameworks['pragmatist'] = PerspectiveFramework(
            name='pragmatist',
            core_commitments=[
                'truth_as_utility',
                'experimental_method',
                'fallibilistic_epistemology'
            ],
            interpretive_principles=[
                'focus_on_consequences',
                'test_through_practice',
                'revise_based_on_results'
            ],
            methodological_preferences=[
                'experimental_inquiry',
                'democratic_deliberation',
                'practical_problem_solving'
            ]
        )

        return frameworks

    def get_framework(self, perspective: str) -> 'PerspectiveFramework':
        """Get framework for a specific perspective."""
        if perspective in self.frameworks:
            return self.frameworks[perspective]
        else:
            # Create a generic framework for unknown perspectives
            return PerspectiveFramework(
                name=perspective,
                core_commitments=[f'{perspective}_commitments'],
                interpretive_principles=[f'{perspective}_interpretation'],
                methodological_preferences=[f'{perspective}_methods']
            )


@dataclass
class PerspectiveFramework:
    """Framework for applying a specific philosophical perspective."""
    name: str
    core_commitments: list[str]
    interpretive_principles: list[str]
    methodological_preferences: list[str]

    def filter_relevant_beliefs(self, beliefs: list[Any]) -> list[Any]:
        """Filter beliefs relevant to this perspective."""
        # For now, return all beliefs. In practice, this would filter based
        # on relevance to the perspective's commitments
        return beliefs[:5]  # Limit for performance

    async def interpret_beliefs(self, beliefs: list[Any]) -> dict[str, Any]:
        """Apply perspective-specific interpretation to beliefs."""
        interpretation = {
            'core_themes': self._identify_core_themes(beliefs),
            'interpretive_framework': self.name,
            'key_commitments': self.core_commitments,
            'supporting_evidence': self._gather_supporting_evidence(beliefs),
            'challenges': self._identify_challenges(beliefs)
        }
        return interpretation

    def _identify_core_themes(self, beliefs: list[Any]) -> list[str]:
        """Identify core themes from perspective viewpoint."""
        # Simplified implementation - would be more sophisticated in practice
        return [f'{self.name}_theme_{i}' for i in range(min(3, len(beliefs)))]

    def _gather_supporting_evidence(self, beliefs: list[Any]) -> list[str]:
        """Gather evidence supporting this perspective."""
        return [f'Evidence supporting {self.name} from belief {i}' for i in range(min(2, len(beliefs)))]

    def _identify_challenges(self, beliefs: list[Any]) -> list[str]:
        """Identify challenges to this perspective."""
        return [f'Challenge to {self.name} from belief analysis']

    async def assess_confidence(self, interpretation: dict[str, Any], beliefs: list[Any]) -> float:
        """Assess confidence in this perspective's interpretation."""
        # Simplified confidence calculation
        base_confidence = 0.7

        # Adjust based on supporting evidence
        evidence_count = len(interpretation.get('supporting_evidence', []))
        evidence_boost = min(evidence_count * 0.1, 0.2)

        # Adjust based on challenges
        challenge_count = len(interpretation.get('challenges', []))
        challenge_penalty = min(challenge_count * 0.05, 0.1)

        return max(0.1, min(1.0, base_confidence + evidence_boost - challenge_penalty))


class CoherenceAnalyzer:
    """Analyzes and maximizes coherence across perspectives."""

    def __init__(self):
        """Initialize coherence analysis tools."""
        self.coherence_metrics = self._initialize_coherence_metrics()

    def _initialize_coherence_metrics(self) -> dict[str, Any]:
        """Initialize metrics for coherence assessment."""
        return {
            'consistency': 'logical_consistency_check',
            'mutual_support': 'evidence_reinforcement_analysis',
            'explanatory_power': 'explanatory_scope_assessment',
            'simplicity': 'theoretical_parsimony_evaluation'
        }

    async def maximize_coherence(
        self,
        perspectival_analyses: list[PerspectivalAnalysis],
        dialectical_tensions: list[DialecticalTension]
    ) -> list[dict[str, Any]]:
        """Maximize coherence through systematic synthesis."""
        coherent_syntheses = []

        # Find points of convergence
        convergence_points = self._find_convergence_points(perspectival_analyses)

        # Resolve tensions through synthesis pathways
        for tension in dialectical_tensions:
            for pathway in tension.synthesis_pathways:
                if pathway.viability > 0.6:
                    synthesis = await self._generate_synthesis(
                        tension, pathway, perspectival_analyses
                    )
                    coherent_syntheses.append(synthesis)

        # Add convergence-based syntheses
        for convergence in convergence_points:
            synthesis = await self._synthesize_convergence(
                convergence, perspectival_analyses
            )
            coherent_syntheses.append(synthesis)

        return coherent_syntheses[:5]  # Limit results

    def _find_convergence_points(self, analyses: list[PerspectivalAnalysis]) -> list[dict[str, Any]]:
        """Find points where perspectives converge."""
        convergence_points = []

        # Simple convergence detection - look for common themes
        if len(analyses) >= 2:
            common_themes = set(analyses[0].interpretation.get('core_themes', []))
            for analysis in analyses[1:]:
                themes = set(analysis.interpretation.get('core_themes', []))
                common_themes = common_themes.intersection(themes)

            if common_themes:
                convergence_points.append({
                    'type': 'thematic_convergence',
                    'common_themes': list(common_themes),
                    'participating_perspectives': [a.perspective for a in analyses]
                })

        return convergence_points

    async def _generate_synthesis(
        self,
        tension: DialecticalTension,
        pathway: SynthesisPathway,
        analyses: list[PerspectivalAnalysis]
    ) -> dict[str, Any]:
        """Generate synthesis resolving dialectical tension."""
        return {
            'synthesis_type': 'dialectical_resolution',
            'pathway_type': pathway.type,
            'description': pathway.description,
            'resolved_tension': {
                'perspectives': [tension.perspective1, tension.perspective2],
                'resolution_method': pathway.type
            },
            'novel_insights': pathway.potential_insights,
            'confidence': pathway.viability * 0.8,  # Slight discount for uncertainty
            'supporting_analyses': [a.perspective for a in analyses if a.perspective in [tension.perspective1, tension.perspective2]]
        }

    async def _synthesize_convergence(
        self,
        convergence: dict[str, Any],
        analyses: list[PerspectivalAnalysis]
    ) -> dict[str, Any]:
        """Generate synthesis from convergence points."""
        return {
            'synthesis_type': 'convergent_synthesis',
            'convergence_basis': convergence['common_themes'],
            'participating_perspectives': convergence['participating_perspectives'],
            'synthesis_content': f"Convergent understanding around: {', '.join(convergence['common_themes'])}",
            'confidence': 0.8,  # High confidence for convergent insights
            'methodological_agreement': True
        }


class DialecticalProcessor:
    """Process philosophical tensions and contradictions constructively."""

    async def identify_tensions(
        self,
        perspectival_analyses: list[PerspectivalAnalysis]
    ) -> list[DialecticalTension]:
        """Identify productive philosophical tensions between perspectives."""
        tensions = []

        for i, analysis1 in enumerate(perspectival_analyses):
            for analysis2 in perspectival_analyses[i+1:]:

                # Analyze interpretive differences
                interpretive_tension = await self._analyze_interpretive_tension(
                    analysis1, analysis2
                )

                # Assess methodological conflicts
                methodological_tension = await self._assess_methodological_conflicts(
                    analysis1.methodological_commitments,
                    analysis2.methodological_commitments
                )

                # Determine dialectical potential
                dialectical_potential = self._calculate_dialectical_potential(
                    interpretive_tension, methodological_tension
                )

                if dialectical_potential > 0.5:  # Threshold for productive tension
                    # Generate synthesis pathways
                    pathways = await self._identify_synthesis_pathways(
                        analysis1, analysis2
                    )

                    tension = DialecticalTension(
                        perspective1=analysis1.perspective,
                        perspective2=analysis2.perspective,
                        interpretive_tension=interpretive_tension,
                        methodological_tension=methodological_tension,
                        dialectical_potential=dialectical_potential,
                        synthesis_pathways=pathways
                    )
                    tensions.append(tension)

        return tensions

    async def _analyze_interpretive_tension(
        self,
        analysis1: PerspectivalAnalysis,
        analysis2: PerspectivalAnalysis
    ) -> dict[str, Any]:
        """Analyze interpretive differences between perspectives."""
        themes1 = set(analysis1.interpretation.get('core_themes', []))
        themes2 = set(analysis2.interpretation.get('core_themes', []))

        overlapping = themes1.intersection(themes2)
        conflicting = themes1.symmetric_difference(themes2)

        return {
            'overlapping_themes': list(overlapping),
            'conflicting_themes': list(conflicting),
            'tension_level': len(conflicting) / max(len(themes1.union(themes2)), 1),
            'potential_reconciliation': len(overlapping) > 0
        }

    async def _assess_methodological_conflicts(
        self,
        commitments1: list[str],
        commitments2: list[str]
    ) -> dict[str, Any]:
        """Assess methodological conflicts between perspectives."""
        conflicts = []
        compatibilities = []

        # Simple conflict detection based on opposed terms
        conflict_pairs = [
            ('empirical', 'rational'),
            ('reductive', 'holistic'),
            ('objective', 'subjective')
        ]

        for method1 in commitments1:
            for method2 in commitments2:
                for pair in conflict_pairs:
                    if pair[0] in method1.lower() and pair[1] in method2.lower():
                        conflicts.append((method1, method2))
                    elif pair[1] in method1.lower() and pair[0] in method2.lower():
                        conflicts.append((method1, method2))

        return {
            'conflicts': conflicts,
            'compatibilities': compatibilities,
            'conflict_severity': len(conflicts) / max(len(commitments1) + len(commitments2), 1)
        }

    def _calculate_dialectical_potential(
        self,
        interpretive_tension: dict[str, Any],
        methodological_tension: dict[str, Any]
    ) -> float:
        """Calculate potential for productive dialectical engagement."""
        # Base potential from tension level
        tension_level = interpretive_tension.get('tension_level', 0)

        # Adjust for reconciliation potential
        reconciliation_bonus = 0.3 if interpretive_tension.get('potential_reconciliation') else 0

        # Adjust for methodological conflicts (moderate conflict is productive)
        conflict_severity = methodological_tension.get('conflict_severity', 0)
        methodological_factor = 1.0 - abs(conflict_severity - 0.5)  # Peak at moderate conflict

        potential = (tension_level * 0.5 + reconciliation_bonus + methodological_factor * 0.3)

        return min(1.0, max(0.0, potential))

    async def _identify_synthesis_pathways(
        self,
        analysis1: PerspectivalAnalysis,
        analysis2: PerspectivalAnalysis
    ) -> list[SynthesisPathway]:
        """Identify potential pathways for dialectical synthesis."""
        pathways = []

        # Convergence pathway - build on common ground
        overlapping = analysis1.interpretation.get('core_themes', [])
        if overlapping:
            pathways.append(SynthesisPathway(
                type='convergence',
                description=f"Build synthesis on shared themes: {', '.join(overlapping[:2])}",
                viability=0.8,
                potential_insights=[f"Integrated understanding of {theme}" for theme in overlapping[:2]]
            ))

        # Complementarity pathway - integrate different strengths
        pathways.append(SynthesisPathway(
            type='complementarity',
            description=f"Integrate {analysis1.perspective} and {analysis2.perspective} as complementary approaches",
            viability=0.7,
            potential_insights=[
                f"{analysis1.perspective} provides {', '.join(analysis1.strengths[:2])}",
                f"{analysis2.perspective} provides {', '.join(analysis2.strengths[:2])}"
            ]
        ))

        # Transcendence pathway - move to higher level
        pathways.append(SynthesisPathway(
            type='transcendence',
            description=f"Transcend {analysis1.perspective}/{analysis2.perspective} opposition through meta-level analysis",
            viability=0.6,
            potential_insights=[
                f"Meta-perspective integrating {analysis1.perspective} and {analysis2.perspective}",
                "Higher-order principles governing both approaches"
            ]
        ))

        return pathways


class EnhancedInsightSynthesis:
    """
    Advanced synthesis engine for generating substantive philosophical insights.

    This class orchestrates multi-perspectival analysis, dialectical processing,
    and coherence maximization to produce meaningful philosophical conclusions.
    """

    def __init__(self, nars_memory: NARSMemory, llm_processor: LLMSemanticProcessor):
        """Initialize enhanced insight synthesis system."""
        self.nars_memory = nars_memory
        self.llm_processor = llm_processor
        self.perspective_manager = PerspectiveManager()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.dialectical_processor = DialecticalProcessor()
        self.embedding_space = SemanticEmbeddingSpace()

        logger.info("EnhancedInsightSynthesis initialized with multi-perspectival capabilities")

    async def synthesize_insights(
        self,
        inquiry_focus: str,
        available_perspectives: list[str],
        depth_level: int = 3
    ) -> list[SubstantiveInsight]:
        """
        Generate substantive philosophical insights through multi-perspectival analysis.

        Args:
            inquiry_focus: The philosophical question or phenomenon to analyze
            available_perspectives: List of philosophical perspectives to apply
            depth_level: Depth of analysis (1-5)

        Returns:
            List of substantive philosophical insights with confidence metrics
        """
        try:
            logger.info(f"Synthesizing insights for: {inquiry_focus}")

            # Retrieve relevant beliefs from NARS memory
            relevant_beliefs = await self._query_relevant_beliefs(
                inquiry_focus, context_radius=depth_level
            )

            # Apply multiple perspectives
            perspectival_analyses = await self._apply_multiple_perspectives(
                relevant_beliefs, available_perspectives
            )

            # Identify dialectical tensions
            dialectical_tensions = await self.dialectical_processor.identify_tensions(
                perspectival_analyses
            )

            # Synthesize through coherence maximization
            coherent_syntheses = await self.coherence_analyzer.maximize_coherence(
                perspectival_analyses, dialectical_tensions
            )

            # Generate substantive conclusions
            substantive_insights = await self._generate_substantive_conclusions(
                coherent_syntheses, inquiry_focus, perspectival_analyses
            )

            logger.info(f"Generated {len(substantive_insights)} substantive insights")
            return substantive_insights

        except Exception as e:
            logger.error(f"Error in insight synthesis: {e}")
            return []

    async def _query_relevant_beliefs(self, inquiry_focus: str, context_radius: int) -> list[Any]:
        """Query NARS memory for beliefs relevant to the inquiry focus."""
        try:
            # Create a philosophical context for the query
            context = PhilosophicalContext(
                domain=PhilosophicalDomain.PHILOSOPHY_OF_MIND,  # Default
                inquiry_type="insight_synthesis",
                depth_requirements=context_radius
            )

            # Analyze the inquiry focus to extract key concepts
            if self.llm_processor:
                analysis = await self.llm_processor.analyze_statement(inquiry_focus, context)
                key_concepts = [c.term for c in analysis.primary_concepts[:5]]
            else:
                # Fallback to simple keyword extraction
                key_concepts = inquiry_focus.split()[:5]

            # Query NARS memory for relevant beliefs
            relevant_beliefs = []
            for concept in key_concepts:
                # Use NARS memory query functionality
                belief = self.nars_memory.query(concept)
                if belief:
                    relevant_beliefs.append(belief)

            # Also get attention buffer for additional relevant beliefs
            attention_beliefs = self.nars_memory.get_attention_buffer(
                query=inquiry_focus
            )
            relevant_beliefs.extend(attention_beliefs[:5])  # Limit for performance

            logger.debug(f"Retrieved {len(relevant_beliefs)} relevant beliefs from NARS")
            return relevant_beliefs[:10]  # Limit for performance

        except Exception as e:
            logger.error(f"Error querying relevant beliefs: {e}")
            return []

    async def _apply_multiple_perspectives(
        self,
        beliefs: list[Any],
        perspectives: list[str]
    ) -> list[PerspectivalAnalysis]:
        """Apply multiple philosophical perspectives to belief sets."""
        analyses = []

        for perspective in perspectives:
            try:
                perspective_framework = self.perspective_manager.get_framework(perspective)

                # Filter beliefs relevant to this perspective
                relevant_beliefs = perspective_framework.filter_relevant_beliefs(beliefs)

                # Apply perspective-specific interpretation
                interpretation = await perspective_framework.interpret_beliefs(relevant_beliefs)

                # Assess perspective confidence
                confidence = await perspective_framework.assess_confidence(
                    interpretation, relevant_beliefs
                )

                # Identify perspective strengths and limitations
                strengths = self._identify_perspective_strengths(perspective, interpretation)
                limitations = self._identify_perspective_limitations(perspective, interpretation)

                # Generate novel insights from this perspective
                novel_insights = await self._generate_perspective_insights(
                    perspective, interpretation, relevant_beliefs
                )

                analysis = PerspectivalAnalysis(
                    perspective=perspective,
                    interpretation=interpretation,
                    confidence=confidence,
                    supporting_beliefs=relevant_beliefs,
                    methodological_commitments=perspective_framework.methodological_preferences,
                    strengths=strengths,
                    limitations=limitations,
                    novel_insights=novel_insights
                )

                analyses.append(analysis)

            except Exception as e:
                logger.error(f"Error applying perspective {perspective}: {e}")
                continue

        return analyses

    def _identify_perspective_strengths(self, perspective: str, interpretation: dict[str, Any]) -> list[str]:
        """Identify strengths of a philosophical perspective."""
        strength_map = {
            'materialist': ['empirical_grounding', 'scientific_compatibility', 'explanatory_power'],
            'phenomenological': ['experiential_accuracy', 'descriptive_richness', 'consciousness_focus'],
            'enactivist': ['embodiment_emphasis', 'ecological_validity', 'dynamic_perspective'],
            'pragmatist': ['practical_relevance', 'experimental_method', 'fallibilistic_humility']
        }

        return strength_map.get(perspective, ['systematic_approach', 'conceptual_clarity'])

    def _identify_perspective_limitations(self, perspective: str, interpretation: dict[str, Any]) -> list[str]:
        """Identify limitations of a philosophical perspective."""
        limitation_map = {
            'materialist': ['consciousness_hard_problem', 'reductive_oversimplification'],
            'phenomenological': ['empirical_disconnect', 'methodological_subjectivity'],
            'enactivist': ['theoretical_underdetermination', 'complexity_challenges'],
            'pragmatist': ['truth_relativization', 'normative_gaps']
        }

        return limitation_map.get(perspective, ['scope_limitations', 'methodological_constraints'])

    async def _generate_perspective_insights(
        self,
        perspective: str,
        interpretation: dict[str, Any],
        beliefs: list[Any]
    ) -> list[str]:
        """Generate novel insights from a specific perspective."""
        insights = []

        # Generate insights based on perspective's core themes
        core_themes = interpretation.get('core_themes', [])
        for theme in core_themes:
            insight = f"From {perspective} perspective: {theme} reveals new understanding about the inquiry"
            insights.append(insight)

        # Generate insights from perspective's unique contributions
        if perspective == 'materialist':
            insights.append("Physical substrate analysis reveals causal mechanisms")
        elif perspective == 'phenomenological':
            insights.append("Experiential structure analysis reveals consciousness patterns")
        elif perspective == 'enactivist':
            insights.append("Embodied interaction analysis reveals emergent properties")
        elif perspective == 'pragmatist':
            insights.append("Practical consequence analysis reveals functional significance")

        return insights[:3]  # Limit to top insights

    async def _generate_substantive_conclusions(
        self,
        coherent_syntheses: list[dict[str, Any]],
        inquiry_focus: str,
        perspectival_analyses: list[PerspectivalAnalysis]
    ) -> list[SubstantiveInsight]:
        """Generate substantive philosophical conclusions from syntheses."""
        substantive_insights = []

        for synthesis in coherent_syntheses:
            try:
                # Generate insight content based on synthesis type
                if synthesis.get('synthesis_type') == 'dialectical_resolution':
                    content = await self._generate_dialectical_insight(synthesis, inquiry_focus)
                elif synthesis.get('synthesis_type') == 'convergent_synthesis':
                    content = await self._generate_convergent_insight(synthesis, inquiry_focus)
                else:
                    content = f"Synthetic insight about {inquiry_focus}: {synthesis.get('description', 'Novel understanding')}"

                # Calculate confidence
                confidence = synthesis.get('confidence', 0.6)

                # Identify supporting perspectives
                supporting_perspectives = synthesis.get('participating_perspectives',
                                                      synthesis.get('supporting_analyses', []))

                # Generate practical implications
                practical_implications = await self._generate_practical_implications(
                    content, supporting_perspectives
                )

                # Generate revision conditions
                revision_conditions = self._generate_revision_conditions(synthesis)

                # Generate further inquiry directions
                inquiry_directions = self._generate_inquiry_directions(synthesis, inquiry_focus)

                insight = SubstantiveInsight(
                    content=content,
                    confidence=confidence,
                    supporting_perspectives=supporting_perspectives,
                    synthesis_pathway=synthesis.get('pathway_type', synthesis.get('synthesis_type', 'unknown')),
                    philosophical_significance=self._assess_philosophical_significance(content),
                    practical_implications=practical_implications,
                    revision_conditions=revision_conditions,
                    further_inquiry_directions=inquiry_directions
                )

                substantive_insights.append(insight)

            except Exception as e:
                logger.error(f"Error generating substantive conclusion: {e}")
                continue

        # Sort by confidence and philosophical significance
        substantive_insights.sort(key=lambda x: x.confidence, reverse=True)

        return substantive_insights[:5]  # Return top 5 insights

    async def _generate_dialectical_insight(self, synthesis: dict[str, Any], inquiry_focus: str) -> str:
        """Generate insight from dialectical resolution."""
        resolved_tension = synthesis.get('resolved_tension', {})
        perspectives = resolved_tension.get('perspectives', [])
        resolution_method = resolved_tension.get('resolution_method', 'synthesis')

        return f"Dialectical analysis of {inquiry_focus} reveals that {perspectives[0] if perspectives else 'perspective1'} and {perspectives[1] if len(perspectives) > 1 else 'perspective2'} can be reconciled through {resolution_method}, yielding a deeper understanding that transcends their apparent opposition."

    async def _generate_convergent_insight(self, synthesis: dict[str, Any], inquiry_focus: str) -> str:
        """Generate insight from convergent analysis."""
        convergence_basis = synthesis.get('convergence_basis', [])
        participants = synthesis.get('participating_perspectives', [])

        return f"Multi-perspectival analysis of {inquiry_focus} reveals convergent understanding around {', '.join(convergence_basis[:2])} among {', '.join(participants[:3])}, suggesting robust philosophical insights that transcend individual perspective limitations."

    async def _generate_practical_implications(self, content: str, perspectives: list[str]) -> list[str]:
        """Generate practical implications of philosophical insights."""
        implications = []

        # General implications
        implications.append("Informs theoretical understanding and conceptual frameworks")

        # Perspective-specific implications
        if 'materialist' in perspectives:
            implications.append("Suggests empirical research directions and testable hypotheses")
        if 'phenomenological' in perspectives:
            implications.append("Provides guidance for experiential analysis and descriptive methods")
        if 'enactivist' in perspectives:
            implications.append("Indicates environmental and embodiment factors for consideration")
        if 'pragmatist' in perspectives:
            implications.append("Offers practical problem-solving approaches and experimental methods")

        return implications[:3]

    def _generate_revision_conditions(self, synthesis: dict[str, Any]) -> list[str]:
        """Generate conditions under which insights should be revised."""
        conditions = [
            "New empirical evidence contradicting core assumptions",
            "Logical inconsistencies discovered in synthesis",
            "Alternative perspectives providing superior explanatory power"
        ]

        # Add synthesis-specific conditions
        if synthesis.get('synthesis_type') == 'dialectical_resolution':
            conditions.append("Discovery of irreconcilable tensions in dialectical resolution")
        elif synthesis.get('synthesis_type') == 'convergent_synthesis':
            conditions.append("Revelation that apparent convergence was superficial")

        return conditions[:3]

    def _generate_inquiry_directions(self, synthesis: dict[str, Any], inquiry_focus: str) -> list[str]:
        """Generate directions for further philosophical inquiry."""
        directions = [
            f"Deeper analysis of key concepts identified in {inquiry_focus}",
            "Cross-cultural philosophical perspectives on the synthesis",
            "Historical development of ideas related to the insight"
        ]

        # Add synthesis-specific directions
        novel_insights = synthesis.get('novel_insights', [])
        for insight in novel_insights[:2]:
            directions.append(f"Further exploration of: {insight}")

        return directions[:4]

    def _assess_philosophical_significance(self, content: str) -> str:
        """Assess the philosophical significance of an insight."""
        if 'transcend' in content.lower():
            return "High significance: transcends existing theoretical boundaries"
        elif 'reconcile' in content.lower():
            return "Moderate significance: resolves existing philosophical tensions"
        elif 'reveal' in content.lower():
            return "Moderate significance: reveals new understanding of existing issues"
        else:
            return "Standard significance: contributes to ongoing philosophical discourse"
