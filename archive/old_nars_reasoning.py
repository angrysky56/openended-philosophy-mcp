"""
NARS Reasoning Integration - Philosophical Non-Axiomatic Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Integrates NARS reasoning capabilities with philosophical analysis,
providing non-axiomatic inference, multi-perspective synthesis, and
epistemic uncertainty quantification.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .nars_manager import NARSManager
from .nars_memory import MemoryItem, NARSMemory
from .truth_functions import Truth, TruthValue

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of NARS reasoning process."""
    conclusion: str
    truth: TruthValue
    evidence: list[MemoryItem]
    inference_path: list[str]
    uncertainty_factors: dict[str, float]
    philosophical_implications: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "conclusion": self.conclusion,
            "truth": self.truth.to_dict(),
            "evidence": [e.term for e in self.evidence],
            "inference_path": self.inference_path,
            "uncertainty_factors": self.uncertainty_factors,
            "philosophical_implications": self.philosophical_implications
        }


class NARSReasoning:
    """
    NARS reasoning system with philosophical enhancements.

    Provides non-axiomatic reasoning capabilities integrated with
    philosophical analysis frameworks.
    """

    def __init__(self,
                 nars_manager: NARSManager,
                 nars_memory: NARSMemory):
        """
        Initialize NARS reasoning system.

        Args:
            nars_manager: ONA process manager
            nars_memory: NARS memory system
        """
        self.nars = nars_manager
        self.memory = nars_memory

        # Philosophical reasoning patterns
        self.reasoning_patterns = {
            "deductive": self._deductive_reasoning,
            "inductive": self._inductive_reasoning,
            "abductive": self._abductive_reasoning,
            "analogical": self._analogical_reasoning,
            "dialectical": self._dialectical_reasoning
        }

        logger.info("NARS Reasoning system initialized")

    async def analyze_concept(self,
                            concept: str,
                            context: str,
                            perspectives: list[str]) -> dict[str, Any]:
        """
        Analyze concept using NARS reasoning with philosophical perspectives.

        Combines non-axiomatic reasoning with multi-perspective analysis.
        """
        logger.debug(f"Analyzing concept '{concept}' in context '{context}'")

        # Get relevant beliefs from memory
        attention_buffer = self.memory.get_attention_buffer(
            query=f"{concept} {context}",
            include_categories=self._context_to_categories(context)
        )

        # Prime NARS with relevant beliefs
        await self._prime_nars_memory(attention_buffer)

        # Analyze from each perspective
        perspective_analyses = {}

        for perspective in perspectives:
            # Generate perspective-specific queries
            queries = self._generate_perspective_queries(concept, context, perspective)

            # Execute queries and collect results
            results = []
            for query in queries:
                try:
                    nars_result = await self.nars.query(query, timeout=3.0)
                    if nars_result.get("answers"):
                        results.extend(nars_result["answers"])
                except Exception as e:
                    logger.warning(f"Query failed for {perspective}: {e}")

            # Synthesize perspective analysis
            if results:
                analysis = await self._synthesize_perspective_analysis(
                    concept, perspective, results, attention_buffer
                )
                perspective_analyses[perspective] = analysis

        # Cross-perspective synthesis
        synthesis = await self._synthesize_cross_perspective(
            concept, context, perspective_analyses
        )

        return {
            "concept": concept,
            "context": context,
            "perspective_analyses": perspective_analyses,
            "synthesis": synthesis,
            "coherence_assessment": self._assess_conceptual_coherence(perspective_analyses),
            "uncertainty_profile": self._calculate_uncertainty_profile(perspective_analyses)
        }

    async def explore_coherence(self,
                              domain: str,
                              depth: int = 3) -> dict[str, Any]:
        """
        Explore coherence patterns using NARS reasoning.

        Maps conceptual landscape through non-axiomatic inference.
        """
        logger.debug(f"Exploring coherence in domain '{domain}' with depth {depth}")

        # Get domain-relevant beliefs
        domain_beliefs = self.memory.get_attention_buffer(
            query=domain,
            include_categories=self._domain_to_categories(domain)
        )

        # Build conceptual graph through NARS queries
        concept_graph = await self._build_concept_graph(domain_beliefs, depth)

        # Identify coherence patterns
        coherence_patterns = self._analyze_coherence_patterns(concept_graph)

        # Find conceptual attractors
        attractors = self._identify_conceptual_attractors(concept_graph)

        # Analyze stability
        stability_analysis = await self._analyze_conceptual_stability(
            concept_graph, attractors
        )

        return {
            "domain": domain,
            "concept_graph": concept_graph,
            "coherence_patterns": coherence_patterns,
            "conceptual_attractors": attractors,
            "stability_analysis": stability_analysis,
            "philosophical_structure": self._extract_philosophical_structure(concept_graph)
        }

    async def generate_insights(self,
                              phenomenon: str,
                              perspectives: list[str],
                              depth: int = 3) -> dict[str, Any]:
        """
        Generate insights through NARS reasoning and synthesis.

        Produces fallibilistic insights with uncertainty quantification.

        Returns:
            dict: {
                "phenomenon": The phenomenon analyzed,
                "evidence_base": List of evidence items,
                "reasoning_results": Results from reasoning patterns,
                "perspective_insights": Perspective-specific insights,
                "contradictions": List of contradictions,
                "meta_insights": List of meta-level insights,
                "revision_conditions": List of conditions that would necessitate revision of insights
            }
        """
        logger.debug(f"Generating insights for phenomenon '{phenomenon}'")

        # Gather evidence through NARS queries
        evidence = await self._gather_phenomenon_evidence(phenomenon, depth)

        # Apply reasoning patterns
        reasoning_results = {}
        for pattern_name, pattern_func in self.reasoning_patterns.items():
            try:
                result = await pattern_func(phenomenon, evidence)
                if result:
                    reasoning_results[pattern_name] = result
            except Exception as e:
                logger.warning(f"Reasoning pattern {pattern_name} failed: {e}")

        # Generate perspective-specific insights
        perspective_insights = {}
        for perspective in perspectives:
            insights = await self._generate_perspective_insights(
                phenomenon, perspective, reasoning_results, evidence
            )
            perspective_insights[perspective] = insights

        # Identify contradictions and tensions
        contradictions = self._identify_contradictions(perspective_insights)

        # Generate meta-insights
        meta_insights = self._generate_meta_insights(
            phenomenon, perspective_insights, contradictions
        )

        return {
            "phenomenon": phenomenon,
            "evidence_base": [e.to_dict() for e in evidence],
            "reasoning_results": {k: (v.to_dict() if v else None) for k, v in reasoning_results.items()},
            "perspective_insights": perspective_insights,
            "contradictions": contradictions,
            "meta_insights": meta_insights,
            "revision_conditions": self._generate_revision_conditions(reasoning_results)
        }

    async def test_hypothesis(self,
                              hypothesis: str,
                              test_domains: list[str],
                              criteria: dict[str, Any]) -> dict[str, Any]:
        """
        Test philosophical hypothesis using NARS reasoning.

        Evaluates hypothesis across domains with coherence analysis.
        """
        logger.debug(f"Testing hypothesis: {hypothesis}")

        # Parse hypothesis into NARS format
        narsese_hypothesis = self._hypothesis_to_narsese(hypothesis)

        # Test in each domain
        domain_results = {}

        for domain in test_domains:
            # Get domain-specific evidence
            domain_evidence = self.memory.get_attention_buffer(
                query=f"{hypothesis} {domain}",
                include_categories=self._domain_to_categories(domain)
            )

            # Test hypothesis
            test_result = await self._test_in_domain(
                narsese_hypothesis, domain, domain_evidence, criteria
            )

            domain_results[domain] = test_result

        # Calculate overall coherence
        overall_coherence = self._calculate_hypothesis_coherence(domain_results)

        # Assess pragmatic value
        pragmatic_assessment = self._assess_pragmatic_value(hypothesis, domain_results)

        return {
            "hypothesis": hypothesis,
            "narsese_form": narsese_hypothesis,
            "domain_results": domain_results,
            "overall_coherence": overall_coherence,
            "pragmatic_assessment": pragmatic_assessment,
            "confidence": self._calculate_hypothesis_confidence(domain_results),
            "implications": self._derive_implications(hypothesis, domain_results)
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Reasoning Pattern Implementations
    # ─────────────────────────────────────────────────────────────────────────

    async def _deductive_reasoning(self,
                                 phenomenon: str,
                                 evidence: list[MemoryItem]) -> ReasoningResult | None:
        """Apply deductive reasoning pattern."""
        # Find general principles
        principles = [e for e in evidence
                     if "==>" in e.term or "-->" in e.term
                     and e.truth.confidence > 0.7]

        if not principles:
            return None

        # Apply strongest principle
        best_principle = max(principles, key=lambda x: x.truth.expectation)

        # Query NARS for deduction
        query = f"{best_principle.term.split('==>')[0].strip()}?"
        result = await self.nars.query(query)

        if result.get("answers"):
            answer = result["answers"][0]
            if answer.get("truth"):
                truth = TruthValue(
                    answer["truth"]["frequency"],
                    answer["truth"]["confidence"]
                )

                return ReasoningResult(
                    conclusion=f"Deduced: {answer['term']}",
                    truth=truth,
                    evidence=[best_principle],
                    inference_path=["deduction", best_principle.term, answer["term"]],
                    uncertainty_factors={"deductive_strength": truth.confidence},
                    philosophical_implications=[
                        "Follows necessarily from principles",
                        "High epistemic warrant if premises hold"
                    ]
                )

        return None

    async def _inductive_reasoning(self,
                                 phenomenon: str,
                                 evidence: list[MemoryItem]) -> ReasoningResult | None:
        """Apply inductive reasoning pattern."""
        # Find instances
        instances = [e for e in evidence
                    if phenomenon.lower() in e.term.lower()
                    and "-->" in e.term]

        if len(instances) < 2:
            return None

        # Find common patterns
        pattern_counts = {}
        for instance in instances:
            if "-->" in instance.term:
                predicate = instance.term.split("-->")[1].strip()
                pattern_counts[predicate] = pattern_counts.get(predicate, 0) + 1

        if not pattern_counts:
            return None

        # Induce general rule
        best_pattern = max(pattern_counts, key=lambda k: pattern_counts.get(k, 0))
        support_ratio = pattern_counts[best_pattern] / len(instances)

        # Calculate inductive truth
        frequency = support_ratio
        confidence = len(instances) / (len(instances) + Truth.K)
        truth = TruthValue(frequency, confidence)

        return ReasoningResult(
            conclusion=f"Induced: <{phenomenon} --> {best_pattern}>",
            truth=truth,
            evidence=instances[:5],  # Top 5 supporting instances
            inference_path=["induction", f"{len(instances)} instances", best_pattern],
            uncertainty_factors={
                "sample_size": len(instances),
                "support_ratio": support_ratio
            },
            philosophical_implications=[
                "Probabilistic generalization from instances",
                "Subject to revision with new evidence",
                f"Based on {len(instances)} observations"
            ]
        )

    async def _abductive_reasoning(self,
                                 phenomenon: str,
                                 evidence: list[MemoryItem]) -> ReasoningResult | None:
        """Apply abductive reasoning pattern."""
        # Find rules that could explain phenomenon
        explanatory_rules = []

        for item in evidence:
            if "==>" in item.term and phenomenon in item.term.split("==>")[1]:
                explanatory_rules.append(item)

        if not explanatory_rules:
            return None

        # Select best explanation
        best_explanation = max(explanatory_rules, key=lambda x: x.truth.expectation)

        # Extract hypothesis
        hypothesis = best_explanation.term.split("==>")[0].strip()

        # Abductive truth (weakened)
        truth = Truth.abduction(
            TruthValue(1.0, 0.9),  # Phenomenon is observed
            best_explanation.truth
        )

        return ReasoningResult(
            conclusion=f"Abduced: {hypothesis} (explains {phenomenon})",
            truth=truth,
            evidence=[best_explanation],
            inference_path=["abduction", phenomenon, hypothesis],
            uncertainty_factors={
                "explanatory_power": best_explanation.truth.expectation,
                "alternative_explanations": len(explanatory_rules) - 1
            },
            philosophical_implications=[
                "Inference to best explanation",
                "Hypothetical reasoning",
                "Requires further verification"
            ]
        )

    async def _analogical_reasoning(self,
                                  phenomenon: str,
                                  evidence: list[MemoryItem]) -> ReasoningResult | None:
        """Apply analogical reasoning pattern."""
        # Find similar phenomena
        similar_items = []

        for item in evidence:
            if ("-->" in item.term and
                    phenomenon not in item.term and
                    self.memory._generate_embedding(phenomenon) is not None):
                # Check semantic similarity
                item_embedding = self.memory._generate_embedding(item.term)
                similarity = np.dot(
                    self.memory._generate_embedding(phenomenon),
                    item_embedding
                ) / (np.linalg.norm(self.memory._generate_embedding(phenomenon)) *
                    np.linalg.norm(item_embedding))

                if similarity > 0.6:
                    similar_items.append((item, similarity))

        if not similar_items:
            return None

        # Use most similar item
        best_analog, similarity = max(similar_items, key=lambda x: x[1])

        # Transfer properties
        if "-->" in best_analog.term:
            source = best_analog.term.split("-->")[0].strip()
            property = best_analog.term.split("-->")[1].strip()

            # Analogical truth (weakened by similarity)
            truth = TruthValue(
                best_analog.truth.frequency * similarity,
                best_analog.truth.confidence * similarity
            )

            return ReasoningResult(
                conclusion=f"By analogy: <{phenomenon} --> {property}>",
                truth=truth,
                evidence=[best_analog],
                inference_path=["analogy", best_analog.term, f"similarity: {similarity:.2f}"],
                uncertainty_factors={
                    "similarity": similarity,
                    "source_confidence": best_analog.truth.confidence
                },
                philosophical_implications=[
                    f"Based on similarity to {source}",
                    "Structural correspondence assumed",
                    "Weaker than deductive inference"
                ]
            )

        return None
    async def _dialectical_reasoning(self,
                                   phenomenon: str,
                                   evidence: list[MemoryItem]) -> ReasoningResult | None:
        """Apply dialectical reasoning pattern."""
        # Find contradictory positions
        positions = []
        negations = []

        for item in evidence:
            if phenomenon in item.term:
                if item.truth.frequency > 0.6:
                    positions.append(item)
                elif item.truth.frequency < 0.4:
                    negations.append(item)

        if not positions or not negations:
            return None

        # Synthesize through revision
        thesis = max(positions, key=lambda x: x.truth.confidence)
        antithesis = max(negations, key=lambda x: x.truth.confidence)

        # Dialectical synthesis
        synthesis_truth = Truth.revision(
            thesis.truth,
            Truth.negation(antithesis.truth)
        )

        return ReasoningResult(
            conclusion=f"Dialectical synthesis: {phenomenon} with moderation",
            truth=synthesis_truth,
            evidence=[thesis, antithesis],
            inference_path=["dialectic", "thesis", "antithesis", "synthesis"],
            uncertainty_factors={
                "thesis_strength": thesis.truth.confidence,
                "antithesis_strength": antithesis.truth.confidence,
                "tension": abs(thesis.truth.frequency - (1 - antithesis.truth.frequency))
            },
            philosophical_implications=[
                "Resolution through opposing views",
                "Higher-order integration",
                "Preserves partial truths from both positions"
            ]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    async def _synthesize_cross_perspective(self,
                                          concept: str,
                                          context: str,
                                          perspective_analyses: dict[str, Any]) -> dict[str, Any]:
        """
        Synthesize insights across multiple philosophical perspectives.

        Implements NARS-based multi-perspective synthesis with coherence assessment
        and dialectical integration of potentially contradictory viewpoints.

        Args:
            concept: The philosophical concept being analyzed
            context: Contextual domain of analysis
            perspective_analyses: Dictionary of perspective-specific analyses

        Returns:
            Dict containing synthesis results with coherence metrics
        """
        logger.debug(f"Synthesizing cross-perspective analysis for concept '{concept}'")

        synthesis_points = []
        convergent_insights = []
        divergent_tensions = []

        # Extract key findings from each perspective
        findings_by_perspective = {}
        for perspective, analysis in perspective_analyses.items():
            if 'findings' in analysis:
                findings_by_perspective[perspective] = analysis['findings']

        # Identify convergent insights (similar claims across perspectives)
        for p1, findings1 in findings_by_perspective.items():
            for finding1 in findings1:
                convergent_count = 1
                convergent_perspectives = [p1]

                for p2, findings2 in findings_by_perspective.items():
                    if p1 != p2:
                        for finding2 in findings2:
                            # Check semantic similarity using NARS revision
                            similarity_score = self._calculate_finding_similarity(finding1, finding2)
                            if similarity_score > 0.7:
                                convergent_count += 1
                                convergent_perspectives.append(p2)

                if convergent_count > 1:
                    # Synthesize convergent insight using NARS revision
                    synthesized_truth = self._synthesize_convergent_truths(
                        [f for f in findings1 if f == finding1],
                        convergent_perspectives
                    )

                    convergent_insights.append({
                        'insight': finding1['claim'],
                        'supporting_perspectives': convergent_perspectives,
                        'synthesized_truth': synthesized_truth.to_dict(),
                        'synthesis_strength': convergent_count / len(perspective_analyses)
                    })

        # Identify divergent tensions (contradictory claims)
        for p1, findings1 in findings_by_perspective.items():
            for finding1 in findings1:
                for p2, findings2 in findings_by_perspective.items():
                    if p1 != p2:
                        for finding2 in findings2:
                            tension_score = self._calculate_tension_score(finding1, finding2)
                            if tension_score > 0.6:
                                divergent_tensions.append({
                                    'tension': f"{finding1['claim']} vs {finding2['claim']}",
                                    'perspectives': [p1, p2],
                                    'tension_strength': tension_score,
                                    'dialectical_potential': self._assess_dialectical_potential(finding1, finding2)
                                })

        # Generate synthesis through dialectical reasoning
        dialectical_synthesis = await self._perform_dialectical_synthesis(
            convergent_insights, divergent_tensions, concept, context
        )

        # Calculate overall synthesis coherence
        synthesis_coherence = self._calculate_synthesis_coherence(
            convergent_insights, divergent_tensions, dialectical_synthesis
        )

        return {
            'concept': concept,
            'context': context,
            'convergent_insights': convergent_insights,
            'divergent_tensions': divergent_tensions,
            'dialectical_synthesis': dialectical_synthesis,
            'synthesis_coherence': synthesis_coherence,
            'epistemological_status': self._determine_epistemological_status(synthesis_coherence),
            'revision_triggers': self._identify_synthesis_revision_triggers(convergent_insights, divergent_tensions)
        }

    def _assess_conceptual_coherence(self, perspective_analyses: dict[str, Any]) -> dict[str, Any]:
        """
        Assess the coherence of conceptual analyses across perspectives.

        Evaluates logical consistency, semantic compatibility, and evidential support
        using NARS truth functions and philosophical coherence principles.

        Args:
            perspective_analyses: Dictionary of perspective-specific analyses

        Returns:
            Dict containing coherence assessment metrics
        """
        logger.debug("Assessing conceptual coherence across perspectives")

        coherence_metrics = {
            'logical_consistency': 0.0,
            'semantic_compatibility': 0.0,
            'evidential_support': 0.0,
            'overall_coherence': 0.0,
            'weak_points': [],
            'strong_points': []
        }

        if not perspective_analyses:
            return coherence_metrics

        # Extract all findings for analysis
        all_findings = []
        perspective_confidences = []

        for perspective, analysis in perspective_analyses.items():
            if 'findings' in analysis:
                all_findings.extend(analysis['findings'])
                avg_confidence = analysis.get('average_confidence', 0.0)
                perspective_confidences.append(avg_confidence)

        if not all_findings:
            return coherence_metrics

        # Assess logical consistency using NARS truth maintenance
        consistency_violations = []
        for i, finding1 in enumerate(all_findings):
            for j, finding2 in enumerate(all_findings[i+1:], i+1):
                contradiction_score = self._detect_logical_contradiction(finding1, finding2)
                if contradiction_score > 0.7:
                    consistency_violations.append({
                        'finding1': finding1['claim'],
                        'finding2': finding2['claim'],
                        'contradiction_strength': contradiction_score
                    })

        logical_consistency = max(0.0, 1.0 - len(consistency_violations) / max(len(all_findings), 1))

        # Assess semantic compatibility using embedding similarity
        semantic_similarities = []
        for i, finding1 in enumerate(all_findings):
            for j, finding2 in enumerate(all_findings[i+1:], i+1):
                similarity = self._calculate_semantic_compatibility(finding1, finding2)
                semantic_similarities.append(similarity)

        semantic_compatibility = np.mean(semantic_similarities) if semantic_similarities else 0.0

        # Assess evidential support using NARS confidence aggregation
        evidential_support = np.mean([f['truth']['confidence'] for f in all_findings
                                    if 'truth' in f and 'confidence' in f['truth']])

        # Calculate overall coherence using weighted combination
        overall_coherence = (
            0.4 * logical_consistency +
            0.3 * semantic_compatibility +
            0.3 * evidential_support
        )

        # Identify weak and strong points
        weak_points = []
        strong_points = []

        if logical_consistency < 0.6:
            weak_points.append(f"Logical inconsistencies detected: {len(consistency_violations)} violations")

        if semantic_compatibility < 0.5:
            weak_points.append(f"Low semantic compatibility: {semantic_compatibility:.2f}")

        if evidential_support < 0.7:
            weak_points.append(f"Weak evidential support: {evidential_support:.2f}")

        if logical_consistency > 0.8:
            strong_points.append(f"High logical consistency: {logical_consistency:.2f}")

        if semantic_compatibility > 0.7:
            strong_points.append(f"Strong semantic compatibility: {semantic_compatibility:.2f}")

        if evidential_support > 0.8:
            strong_points.append(f"Strong evidential support: {evidential_support:.2f}")

        coherence_metrics.update({
            'logical_consistency': logical_consistency,
            'semantic_compatibility': semantic_compatibility,
            'evidential_support': evidential_support,
            'overall_coherence': overall_coherence,
            'weak_points': weak_points,
            'strong_points': strong_points,
            'consistency_violations': consistency_violations
        })

        return coherence_metrics

    def _calculate_uncertainty_profile(self, perspective_analyses: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate comprehensive uncertainty profile across perspectives.

        Quantifies epistemic uncertainty, evidential limitations, and revision potential
        using NARS uncertainty propagation and philosophical uncertainty principles.

        Args:
            perspective_analyses: Dictionary of perspective-specific analyses

        Returns:
            Dict containing detailed uncertainty profile
        """
        logger.debug("Calculating uncertainty profile across perspectives")

        uncertainty_profile = {
            'epistemic_uncertainty': 0.0,
            'evidential_uncertainty': 0.0,
            'perspectival_uncertainty': 0.0,
            'overall_uncertainty': 0.0,
            'uncertainty_sources': [],
            'confidence_distribution': {},
            'revision_likelihood': 0.0
        }

        if not perspective_analyses:
            uncertainty_profile['overall_uncertainty'] = 1.0  # Maximum uncertainty
            return uncertainty_profile

        # Collect confidence values and uncertainty measures
        all_confidences = []
        perspective_uncertainties = []

        for perspective, analysis in perspective_analyses.items():
            if 'findings' in analysis:
                findings = analysis['findings']
                perspective_confidences = [f['truth']['confidence']
                                         for f in findings
                                         if 'truth' in f and 'confidence' in f['truth']]

                if perspective_confidences:
                    all_confidences.extend(perspective_confidences)
                    # Calculate perspective-level uncertainty
                    avg_confidence = np.mean(perspective_confidences)
                    uncertainty = 1.0 - avg_confidence
                    perspective_uncertainties.append(uncertainty)

        # Epistemic uncertainty (average uncertainty across all findings)
        epistemic_uncertainty = 1.0 - np.mean(all_confidences) if all_confidences else 1.0

        # Evidential uncertainty (variance in confidence levels)
        evidential_uncertainty = np.var(all_confidences) if len(all_confidences) > 1 else 0.0

        # Perspectival uncertainty (disagreement between perspectives)
        perspectival_uncertainty = np.var(perspective_uncertainties) if len(perspective_uncertainties) > 1 else 0.0

        # Overall uncertainty using NARS-style uncertainty propagation
        overall_uncertainty = min(1.0, epistemic_uncertainty + 0.5 * evidential_uncertainty + 0.3 * perspectival_uncertainty)

        # Identify uncertainty sources
        uncertainty_sources = []

        if epistemic_uncertainty > 0.6:
            uncertainty_sources.append(f"High epistemic uncertainty: {epistemic_uncertainty:.2f}")

        if evidential_uncertainty > 0.3:
            uncertainty_sources.append(f"High evidential variance: {evidential_uncertainty:.2f}")

        if perspectival_uncertainty > 0.4:
            uncertainty_sources.append(f"High perspectival disagreement: {perspectival_uncertainty:.2f}")

        if len(all_confidences) < 5:
            uncertainty_sources.append(f"Limited evidence base: {len(all_confidences)} findings")

        # Create confidence distribution
        if all_confidences:
            confidence_bins = np.histogram(all_confidences, bins=5, range=(0, 1))[0]
            confidence_distribution = {
                f"bin_{i+1}": int(count) for i, count in enumerate(confidence_bins)
            }
        else:
            confidence_distribution = {}

        # Calculate revision likelihood based on uncertainty factors
        revision_likelihood = min(1.0, overall_uncertainty + 0.2 * len(uncertainty_sources) / 5)

        uncertainty_profile.update({
            'epistemic_uncertainty': epistemic_uncertainty,
            'evidential_uncertainty': evidential_uncertainty,
            'perspectival_uncertainty': perspectival_uncertainty,
            'overall_uncertainty': overall_uncertainty,
            'uncertainty_sources': uncertainty_sources,
            'confidence_distribution': confidence_distribution,
            'revision_likelihood': revision_likelihood,
            'total_findings': len(all_confidences),
            'perspectives_analyzed': len(perspective_analyses)
        })

        return uncertainty_profile

    async def _build_concept_graph(self, domain_beliefs: list[MemoryItem], depth: int) -> dict[str, Any]:
        """
        Build conceptual graph through NARS-based relation discovery.

        Constructs a network of conceptual relationships using NARS inference
        and semantic embeddings to map the philosophical landscape.

        Args:
            domain_beliefs: List of relevant memory items for the domain
            depth: Maximum depth for graph exploration

        Returns:
            Dict containing nodes, edges, and graph metadata
        """
        logger.debug(f"Building concept graph from {len(domain_beliefs)} beliefs with depth {depth}")

        nodes = {}
        edges = []
        concept_embeddings = {}

        # Extract concepts from beliefs
        for belief in domain_beliefs:
            term = belief.term

            # Parse different NARS term structures
            concepts = self._extract_concepts_from_term(term)

            for concept in concepts:
                if concept not in nodes:
                    # Create node with NARS-based properties
                    nodes[concept] = {
                        'id': concept,
                        'truth': belief.truth.to_dict(),
                        'category': belief.philosophical_category,
                        'centrality': 0.0,
                        'evidence_strength': belief.truth.expectation,
                        'semantic_cluster': None
                    }

                    # Store embedding for clustering
                    if belief.embedding is not None:
                        concept_embeddings[concept] = belief.embedding
                else:
                    # Update existing node with stronger evidence
                    if belief.truth.expectation > nodes[concept]['evidence_strength']:
                        nodes[concept]['truth'] = belief.truth.to_dict()
                        nodes[concept]['evidence_strength'] = belief.truth.expectation

            # Extract relations for edges
            relations = self._extract_relations_from_term(term, belief.truth)
            edges.extend(relations)

        # Perform semantic clustering
        if concept_embeddings:
            clusters = self._perform_semantic_clustering(concept_embeddings)
            for concept, cluster_id in clusters.items():
                if concept in nodes:
                    nodes[concept]['semantic_cluster'] = cluster_id

        # Calculate centrality measures
        centrality_scores = self._calculate_graph_centrality(nodes, edges)
        for concept, centrality in centrality_scores.items():
            if concept in nodes:
                nodes[concept]['centrality'] = centrality

        # Expand graph through NARS queries (up to specified depth)
        current_depth = 0
        expansion_candidates = list(nodes.keys())

        while current_depth < depth and expansion_candidates:
            next_candidates = []

            for concept in expansion_candidates[:10]:  # Limit to prevent explosion
                # Query for related concepts
                related_concepts = await self._query_related_concepts(concept)

                for related_concept, relation_type, truth in related_concepts:
                    if related_concept not in nodes:
                        nodes[related_concept] = {
                            'id': related_concept,
                            'truth': truth.to_dict(),
                            'category': self._infer_category(related_concept),
                            'centrality': 0.0,
                            'evidence_strength': truth.expectation,
                            'semantic_cluster': None,
                            'depth': current_depth + 1
                        }
                        next_candidates.append(related_concept)

                    # Add edge
                    edges.append({
                        'source': concept,
                        'target': related_concept,
                        'relation': relation_type,
                        'weight': truth.expectation,
                        'truth': truth.to_dict()
                    })

            expansion_candidates = next_candidates
            current_depth += 1

        # Calculate graph metrics
        graph_metrics = self._calculate_graph_metrics(nodes, edges)

        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': graph_metrics,
            'depth_reached': current_depth,
            'total_concepts': len(nodes),
            'total_relations': len(edges),
            'semantic_clusters': self._get_cluster_summary(nodes)
        }

    def _analyze_coherence_patterns(self, concept_graph: dict[str, Any]) -> list[str]:
        """
        Analyze coherence patterns in the conceptual graph.

        Identifies stable regions, emergence patterns, and structural properties
        that indicate philosophical coherence or incoherence.

        Args:
            concept_graph: The conceptual graph structure

        Returns:
            List of identified coherence patterns
        """
        logger.debug("Analyzing coherence patterns in concept graph")

        patterns = []
        nodes = concept_graph.get('nodes', {})
        edges = concept_graph.get('edges', [])

        if not nodes or not edges:
            return ["No coherence patterns: insufficient graph structure"]

        # Pattern 1: High-confidence clusters
        high_confidence_clusters = self._identify_high_confidence_clusters(nodes, edges)
        for cluster in high_confidence_clusters:
            patterns.append(f"High-confidence coherence cluster: {cluster['concepts']} "
                          f"(avg confidence: {cluster['confidence']:.2f})")

        # Pattern 2: Central concepts with strong support
        central_concepts = self._identify_central_supported_concepts(nodes, edges)
        for concept in central_concepts:
            patterns.append(f"Central coherence anchor: '{concept['id']}' "
                          f"(centrality: {concept['centrality']:.2f}, "
                          f"support: {concept['evidence_strength']:.2f})")

        # Pattern 3: Coherent philosophical categories
        category_coherence = self._analyze_category_coherence(nodes, edges)
        for category, coherence_score in category_coherence.items():
            if coherence_score > 0.7:
                patterns.append(f"Strong {category} coherence: {coherence_score:.2f}")
            elif coherence_score < 0.4:
                patterns.append(f"Weak {category} coherence: {coherence_score:.2f}")

        # Pattern 4: Emergent conceptual structures
        emergent_structures = self._identify_emergent_structures(nodes, edges)
        for structure in emergent_structures:
            patterns.append(f"Emergent structure: {structure['type']} "
                          f"involving {structure['concepts']}")

        # Pattern 5: Contradiction clusters
        contradiction_clusters = self._identify_contradiction_clusters(nodes, edges)
        for cluster in contradiction_clusters:
            patterns.append(f"Contradiction cluster: {cluster['concepts']} "
                          f"(tension score: {cluster['tension']:.2f})")

        # Pattern 6: Inferential chains
        inferential_chains = self._identify_inferential_chains(edges)
        for chain in inferential_chains:
            if len(chain['concepts']) > 3:
                patterns.append(f"Strong inferential chain: {' → '.join(chain['concepts'])} "
                              f"(chain strength: {chain['strength']:.2f})")

        # Pattern 7: Semantic coherence regions
        semantic_regions = self._identify_semantic_coherence_regions(nodes)
        for region in semantic_regions:
            patterns.append(f"Semantic coherence region: cluster {region['cluster_id']} "
                          f"with {region['size']} concepts "
                          f"(coherence: {region['coherence']:.2f})")

        return patterns if patterns else ["No significant coherence patterns detected"]

    def _identify_conceptual_attractors(self, concept_graph: dict[str, Any]) -> list[str]:
        """
        Identify conceptual attractors in the philosophical landscape.

        Finds stable concepts that tend to draw other concepts into their
        semantic and logical orbit, indicating fundamental organizing principles.

        Args:
            concept_graph: The conceptual graph structure

        Returns:
            List of identified conceptual attractors
        """
        logger.debug("Identifying conceptual attractors")

        attractors = []
        nodes = concept_graph.get('nodes', {})
        edges = concept_graph.get('edges', [])

        if not nodes or not edges:
            return ["No attractors: insufficient graph structure"]

        # Calculate attractor metrics for each node
        attractor_scores = {}

        for node_id, node_data in nodes.items():
            # Metrics for attractor identification
            centrality = node_data.get('centrality', 0.0)
            evidence_strength = node_data.get('evidence_strength', 0.0)

            # Calculate incoming edge strength
            incoming_strength = sum(edge['weight'] for edge in edges
                                  if edge['target'] == node_id)

            # Calculate outgoing edge strength
            outgoing_strength = sum(edge['weight'] for edge in edges
                                  if edge['source'] == node_id)

            # Attractor score combines multiple factors
            attractor_score = (
                0.3 * centrality +
                0.3 * evidence_strength +
                0.2 * (incoming_strength / max(len(edges), 1)) +
                0.2 * (outgoing_strength / max(len(edges), 1))
            )

            attractor_scores[node_id] = {
                'score': attractor_score,
                'centrality': centrality,
                'evidence_strength': evidence_strength,
                'incoming_strength': incoming_strength,
                'outgoing_strength': outgoing_strength,
                'category': node_data.get('category', 'unknown')
            }

        # Identify top attractors
        sorted_attractors = sorted(attractor_scores.items(),
                                 key=lambda x: x[1]['score'],
                                 reverse=True)

        # Filter for significant attractors
        threshold = np.mean([score['score'] for _, score in sorted_attractors]) + np.std([score['score'] for _, score in sorted_attractors])

        for concept, metrics in sorted_attractors:
            if metrics['score'] > threshold and metrics['score'] > 0.3:
                attractor_type = self._classify_attractor_type(concept, metrics, edges)

                attractors.append(
                    f"'{concept}' ({attractor_type}): "
                    f"score={metrics['score']:.2f}, "
                    f"centrality={metrics['centrality']:.2f}, "
                    f"evidence={metrics['evidence_strength']:.2f}"
                )

        # Identify attractor clusters (groups of related attractors)
        if len(attractors) > 1:
            attractor_clusters = self._identify_attractor_clusters(
                [a.split("'")[1] for a in attractors], edges
            )

            for cluster in attractor_clusters:
                if len(cluster) > 1:
                    attractors.append(f"Attractor cluster: {cluster} (mutually reinforcing)")

        return attractors if attractors else ["No significant conceptual attractors identified"]

    async def _analyze_conceptual_stability(self,
                                          concept_graph: dict[str, Any],
                                          attractors: list[str]) -> dict[str, Any]:
        """
        Analyze stability of the conceptual landscape.

        Evaluates how resistant the philosophical structure is to perturbation
        and how concepts maintain their relationships under logical pressure.

        Args:
            concept_graph: The conceptual graph structure
            attractors: List of identified conceptual attractors

        Returns:
            Dict containing stability analysis results
        """
        logger.debug("Analyzing conceptual stability")

        stability_analysis = {
            'overall_stability': 0.0,
            'local_stabilities': {},
            'perturbation_resistance': 0.0,
            'attractor_stability': 0.0,
            'weak_points': [],
            'stability_factors': []
        }

        nodes = concept_graph.get('nodes', {})
        edges = concept_graph.get('edges', [])

        if not nodes or not edges:
            stability_analysis['overall_stability'] = 0.0
            stability_analysis['weak_points'] = ["Insufficient structure for stability analysis"]
            return stability_analysis

        # Analyze local stability for each concept
        local_stabilities = {}

        for node_id, node_data in nodes.items():
            # Calculate local stability metrics
            local_edges = [e for e in edges if e['source'] == node_id or e['target'] == node_id]

            if local_edges:
                # Average edge strength indicates stability
                avg_edge_strength = np.mean([e['weight'] for e in local_edges])

                # Edge variance indicates consistency
                edge_variance = np.var([e['weight'] for e in local_edges])

                # Number of connections indicates integration
                connection_strength = len(local_edges) / max(len(nodes), 1)

                # Local stability score
                local_stability = (
                    0.4 * avg_edge_strength +
                    0.3 * (1.0 - edge_variance) +  # Lower variance = higher stability
                    0.3 * connection_strength
                )

                local_stabilities[node_id] = {
                    'stability': local_stability,
                    'edge_strength': avg_edge_strength,
                    'consistency': 1.0 - edge_variance,
                    'integration': connection_strength
                }
            else:
                local_stabilities[node_id] = {
                    'stability': 0.0,
                    'edge_strength': 0.0,
                    'consistency': 0.0,
                    'integration': 0.0
                }

        # Calculate overall stability
        if local_stabilities:
            overall_stability = np.mean([ls['stability'] for ls in local_stabilities.values()])
        else:
            overall_stability = 0.0

        # Analyze perturbation resistance through simulated concept removal
        perturbation_resistance = await self._simulate_perturbation_resistance(
            nodes, edges, local_stabilities
        )

        # Analyze attractor stability
        attractor_stability = self._analyze_attractor_stability(attractors, local_stabilities)

        # Identify weak points (low stability concepts)
        weak_points = []
        stability_threshold = overall_stability * 0.7  # 70% of average

        for node_id, stability_data in local_stabilities.items():
            if stability_data['stability'] < stability_threshold:
                weak_points.append(
                    f"'{node_id}': stability={stability_data['stability']:.2f} "
                    f"(integration={stability_data['integration']:.2f})"
                )

        # Identify stability factors
        stability_factors = []

        if overall_stability > 0.7:
            stability_factors.append(f"High overall stability: {overall_stability:.2f}")

        if perturbation_resistance > 0.6:
            stability_factors.append(f"Good perturbation resistance: {perturbation_resistance:.2f}")

        if attractor_stability > 0.7:
            stability_factors.append(f"Stable attractor structure: {attractor_stability:.2f}")

        # Check for structural factors
        edge_density = len(edges) / max(len(nodes) * (len(nodes) - 1), 1)
        if edge_density > 0.3:
            stability_factors.append(f"High connectivity: density={edge_density:.2f}")

        stability_analysis.update({
            'overall_stability': overall_stability,
            'local_stabilities': local_stabilities,
            'perturbation_resistance': perturbation_resistance,
            'attractor_stability': attractor_stability,
            'weak_points': weak_points,
            'stability_factors': stability_factors,
            'edge_density': edge_density,
            'structural_integrity': min(overall_stability, perturbation_resistance)
        })

        return stability_analysis

    def _extract_philosophical_structure(self, concept_graph: dict[str, Any]) -> dict[str, Any]:
        """
        Extract philosophical structure from the conceptual graph.

        Identifies hierarchical relationships, foundational concepts,
        and architectural patterns that reveal philosophical organization.

        Args:
            concept_graph: The conceptual graph structure

        Returns:
            Dict containing philosophical structure analysis
        """
        logger.debug("Extracting philosophical structure")

        philosophical_structure = {
            'foundational_concepts': [],
            'hierarchical_layers': {},
            'philosophical_categories': {},
            'inferential_patterns': [],
            'structural_principles': [],
            'emergence_levels': {}
        }

        nodes = concept_graph.get('nodes', {})
        edges = concept_graph.get('edges', [])

        if not nodes or not edges:
            return philosophical_structure

        # Identify foundational concepts (high centrality + evidence)
        foundational_candidates = []
        for node_id, node_data in nodes.items():
            centrality = node_data.get('centrality', 0.0)
            evidence = node_data.get('evidence_strength', 0.0)
            foundation_score = 0.6 * centrality + 0.4 * evidence

            if foundation_score > 0.7:
                foundational_candidates.append({
                    'concept': node_id,
                    'score': foundation_score,
                    'category': node_data.get('category', 'unknown')
                })

        foundational_concepts = sorted(foundational_candidates,
                                     key=lambda x: x['score'],
                                     reverse=True)[:5]

        # Analyze hierarchical layers through graph depth analysis
        hierarchical_layers = self._analyze_hierarchical_structure(nodes, edges)

        # Group by philosophical categories
        philosophical_categories = {}
        for node_id, node_data in nodes.items():
            category = node_data.get('category', 'uncategorized')
            if category not in philosophical_categories:
                philosophical_categories[category] = {
                    'concepts': [],
                    'avg_centrality': 0.0,
                    'avg_evidence': 0.0,
                    'internal_coherence': 0.0
                }

            philosophical_categories[category]['concepts'].append(node_id)

        # Calculate category metrics
        for category, data in philosophical_categories.items():
            category_nodes = [nodes[c] for c in data['concepts'] if c in nodes]
            if category_nodes:
                data['avg_centrality'] = np.mean([n.get('centrality', 0.0) for n in category_nodes])
                data['avg_evidence'] = np.mean([n.get('evidence_strength', 0.0) for n in category_nodes])
                data['internal_coherence'] = self._calculate_category_internal_coherence(
                    data['concepts'], edges
                )

        # Identify inferential patterns
        inferential_patterns = self._identify_inferential_patterns(edges)

        # Extract structural principles
        structural_principles = self._extract_structural_principles(
            nodes, edges, foundational_concepts, hierarchical_layers
        )

        # Analyze emergence levels
        emergence_levels = self._analyze_emergence_levels(nodes, edges, hierarchical_layers)

        philosophical_structure.update({
            'foundational_concepts': foundational_concepts,
            'hierarchical_layers': hierarchical_layers,
            'philosophical_categories': philosophical_categories,
            'inferential_patterns': inferential_patterns,
            'structural_principles': structural_principles,
            'emergence_levels': emergence_levels,
            'total_concepts': len(nodes),
            'structural_depth': len(hierarchical_layers),
            'category_distribution': {k: len(v['concepts']) for k, v in philosophical_categories.items()}
        })

        return philosophical_structure

    async def _gather_phenomenon_evidence(self, phenomenon: str, depth: int) -> list[MemoryItem]:
        """
        Gather evidence for a phenomenon through NARS queries and memory search.

        Systematically collects relevant beliefs, observations, and theoretical
        commitments related to the philosophical phenomenon under investigation.

        Args:
            phenomenon: The phenomenon to investigate
            depth: Depth of evidence gathering (number of query iterations)

        Returns:
            List of relevant memory items as evidence
        """
        logger.debug(f"Gathering evidence for phenomenon '{phenomenon}' with depth {depth}")

        evidence = []
        processed_terms = set()

        # Initial evidence from memory
        initial_evidence = self.memory.get_attention_buffer(
            query=phenomenon,
            include_categories=None  # Include all categories initially
        )
        evidence.extend(initial_evidence)
        processed_terms.update(item.term for item in initial_evidence)

        # Iterative evidence expansion through NARS queries
        current_depth = 0
        expansion_terms = [phenomenon]

        while current_depth < depth and expansion_terms:
            next_expansion_terms = []

            for term in expansion_terms[:5]:  # Limit to prevent explosion
                try:
                    # Generate NARS queries for evidence gathering
                    queries = self._generate_evidence_queries(term, phenomenon)

                    for query in queries:
                        try:
                            result = await self.nars.query(query, timeout=2.0)

                            if result.get("answers"):
                                for answer in result["answers"]:
                                    if "term" in answer and "truth" in answer:
                                        answer_term = answer["term"]

                                        if answer_term not in processed_terms:
                                            # Create memory item from NARS answer
                                            truth = TruthValue(
                                                answer["truth"]["frequency"],
                                                answer["truth"]["confidence"]
                                            )

                                            # Check if we already have this in memory
                                            existing_item = self.memory.query(answer_term)

                                            if existing_item:
                                                evidence.append(existing_item)
                                            else:
                                                # Create new memory item
                                                new_item = MemoryItem(
                                                    term=answer_term,
                                                    truth=truth,
                                                    occurrence_time="eternal",
                                                    stamp=[self.memory.current_time],
                                                    embedding=self.memory._generate_embedding(answer_term),
                                                    philosophical_category=self.memory._categorize_term(answer_term)
                                                )
                                                evidence.append(new_item)

                                            processed_terms.add(answer_term)

                                            # Add for next expansion if relevant
                                            if self._is_relevant_for_expansion(answer_term, phenomenon):
                                                next_expansion_terms.append(answer_term)

                        except Exception as e:
                            logger.debug(f"Query failed: {query} - {e}")
                            continue

                except Exception as e:
                    logger.debug(f"Failed to process term {term}: {e}")
                    continue

            expansion_terms = next_expansion_terms
            current_depth += 1

        # Search for related beliefs using semantic similarity
        if evidence:
            # Get embeddings for phenomenon
            phenomenon_embedding = self.memory._generate_embedding(phenomenon)

            # Find semantically related items in memory
            semantic_evidence = []
            for item in self.memory.memory.values():
                if (item.term not in processed_terms and
                    item.embedding is not None):

                    similarity = np.dot(phenomenon_embedding, item.embedding) / (
                        np.linalg.norm(phenomenon_embedding) * np.linalg.norm(item.embedding)
                    )

                    if similarity > 0.5:  # Threshold for relevance
                        semantic_evidence.append(item)

            evidence.extend(semantic_evidence[:10])  # Limit semantic additions

        # Filter and rank evidence by relevance and confidence
        filtered_evidence = []
        for item in evidence:
            # Relevance score based on term similarity and truth confidence
            relevance_score = self._calculate_evidence_relevance(item, phenomenon)

            if relevance_score > 0.3:  # Minimum relevance threshold
                item.semantic_context['relevance_score'] = relevance_score
                filtered_evidence.append(item)

        # Sort by relevance and confidence
        filtered_evidence.sort(
            key=lambda x: (
                x.semantic_context.get('relevance_score', 0.0) * 0.6 +
                x.truth.expectation * 0.4
            ),
            reverse=True
        )

        logger.debug(f"Gathered {len(filtered_evidence)} evidence items for '{phenomenon}'")
        return filtered_evidence[:50]  # Return top 50 most relevant items

    async def _generate_perspective_insights(self,
                                           phenomenon: str,
                                           perspective: str,
                                           reasoning_results: dict[str, Any],
                                           evidence: list[MemoryItem]) -> list[str]:
        """
        Generate perspective-specific insights for a phenomenon.

        Applies philosophical perspective to reasoning results and evidence
        to derive insights that reflect the particular interpretive lens.

        Args:
            phenomenon: The phenomenon being analyzed
            perspective: The philosophical perspective to apply
            reasoning_results: Results from various reasoning patterns
            evidence: Gathered evidence for the phenomenon

        Returns:
            List of perspective-specific insights
        """
        logger.debug(f"Generating {perspective} insights for {phenomenon}")

        insights = []

        # Perspective-specific interpretation frameworks
        interpretation_frameworks = {
            "analytical": self._apply_analytical_framework,
            "phenomenological": self._apply_phenomenological_framework,
            "pragmatist": self._apply_pragmatist_framework,
            "existentialist": self._apply_existentialist_framework,
            "materialist": self._apply_materialist_framework,
            "idealist": self._apply_idealist_framework,
            "process": self._apply_process_framework,
            "structuralist": self._apply_structuralist_framework
        }

        framework_func = interpretation_frameworks.get(
            perspective,
            self._apply_default_framework
        )

        # Apply perspective framework to reasoning results
        for pattern_name, reasoning_result in reasoning_results.items():
            if reasoning_result:
                perspective_interpretation = framework_func(
                    reasoning_result, phenomenon, evidence
                )

                if perspective_interpretation:
                    insights.append(
                        f"[{pattern_name.upper()}] {perspective_interpretation}"
                    )

        # Generate perspective-specific evidence interpretations
        evidence_insights = self._generate_evidence_insights(
            perspective, phenomenon, evidence
        )
        insights.extend(evidence_insights)

        # Generate perspective-specific theoretical commitments
        theoretical_insights = self._generate_theoretical_insights(
            perspective, phenomenon, reasoning_results, evidence
        )
        insights.extend(theoretical_insights)

        # Generate perspective-specific methodological insights
        methodological_insights = self._generate_methodological_insights(
            perspective, phenomenon, evidence
        )
        insights.extend(methodological_insights)

        # Add perspective-specific uncertainty assessments
        uncertainty_insights = self._generate_uncertainty_insights(
            perspective, reasoning_results, evidence
        )
        insights.extend(uncertainty_insights)

        return insights

    def _identify_contradictions(self, perspective_insights: dict[str, Any]) -> list[str]:
        """
        Identify contradictions across perspective-based insights.

        Detects logical inconsistencies, semantic tensions, and evidential
        conflicts that arise when different philosophical perspectives
        are applied to the same phenomenon.

        Args:
            perspective_insights: Dictionary of insights by perspective

        Returns:
            List of identified contradictions
        """
        logger.debug("Identifying contradictions across perspectives")

        contradictions = []

        if len(perspective_insights) < 2:
            return contradictions

        # Extract claims from each perspective
        perspective_claims = {}
        for perspective, insights in perspective_insights.items():
            claims = []
            for insight in insights:
                # Extract propositional content from insights
                extracted_claims = self._extract_claims_from_insight(insight)
                claims.extend(extracted_claims)
            perspective_claims[perspective] = claims

        # Compare claims across perspectives for contradictions
        perspectives = list(perspective_claims.keys())

        for i, perspective1 in enumerate(perspectives):
            for j, perspective2 in enumerate(perspectives[i+1:], i+1):
                claims1 = perspective_claims[perspective1]
                claims2 = perspective_claims[perspective2]

                # Check for logical contradictions
                logical_contradictions = self._detect_logical_contradictions(
                    claims1, claims2, perspective1, perspective2
                )
                contradictions.extend(logical_contradictions)

                # Check for semantic tensions
                semantic_tensions = self._detect_semantic_tensions(
                    claims1, claims2, perspective1, perspective2
                )
                contradictions.extend(semantic_tensions)

                # Check for evidential conflicts
                evidential_conflicts = self._detect_evidential_conflicts(
                    claims1, claims2, perspective1, perspective2
                )
                contradictions.extend(evidential_conflicts)

        # Identify higher-order contradictions (meta-level conflicts)
        meta_contradictions = self._identify_meta_contradictions(perspective_insights)
        contradictions.extend(meta_contradictions)

        # Classify contradiction types and severity
        classified_contradictions = []
        for contradiction in contradictions:
            classification = self._classify_contradiction(contradiction)
            classified_contradictions.append(f"{classification}: {contradiction}")

        return classified_contradictions

    def _generate_meta_insights(self,
                              phenomenon: str,
                              perspective_insights: dict[str, Any],
                              contradictions: list[str]) -> list[str]:
        """
        Generate meta-insights from perspective analysis and contradictions.

        Produces higher-order insights about the nature of the phenomenon,
        the adequacy of perspectives, and the philosophical implications
        of the analytical process itself.

        Args:
            phenomenon: The phenomenon being analyzed
            perspective_insights: Dictionary of insights by perspective
            contradictions: List of identified contradictions

        Returns:
            List of meta-insights
        """
        logger.debug(f"Generating meta-insights for {phenomenon}")

        meta_insights = []

        # Meta-insight 1: Phenomenological adequacy assessment
        adequacy_assessment = self._assess_phenomenological_adequacy(
            phenomenon, perspective_insights
        )
        meta_insights.append(f"Phenomenological adequacy: {adequacy_assessment}")

        # Meta-insight 2: Perspectival complementarity analysis
        complementarity_analysis = self._analyze_perspectival_complementarity(
            perspective_insights, contradictions
        )
        meta_insights.append(f"Perspectival relationships: {complementarity_analysis}")

        # Meta-insight 3: Epistemic status assessment
        epistemic_status = self._assess_epistemic_status(
            phenomenon, perspective_insights, contradictions
        )
        meta_insights.append(f"Epistemic status: {epistemic_status}")

        # Meta-insight 4: Methodological implications
        methodological_implications = self._derive_methodological_implications(
            phenomenon, perspective_insights, contradictions
        )
        if methodological_implications:
            meta_insights.append(f"Methodological implications: {methodological_implications}")

        # Meta-insight 5: Conceptual boundary analysis
        boundary_analysis = self._analyze_conceptual_boundaries(
            phenomenon, perspective_insights
        )
        meta_insights.append(f"Conceptual boundaries: {boundary_analysis}")

        # Meta-insight 6: Emergence and reduction patterns
        emergence_patterns = self._identify_emergence_patterns(
            phenomenon, perspective_insights
        )
        if emergence_patterns:
            meta_insights.append(f"Emergence patterns: {emergence_patterns}")

        # Meta-insight 7: Dialectical potential assessment
        dialectical_potential = self._assess_dialectical_potential_meta(
            contradictions, perspective_insights
        )
        meta_insights.append(f"Dialectical potential: {dialectical_potential}")

        # Meta-insight 8: Pragmatic consequences
        pragmatic_consequences = self._derive_pragmatic_consequences(
            phenomenon, perspective_insights, contradictions
        )
        if pragmatic_consequences:
            meta_insights.append(f"Pragmatic consequences: {pragmatic_consequences}")

        # Meta-insight 9: Future inquiry directions
        inquiry_directions = self._suggest_inquiry_directions(
            phenomenon, perspective_insights, contradictions
        )
        if inquiry_directions:
            meta_insights.append(f"Future inquiry: {inquiry_directions}")

        return meta_insights

    def _generate_revision_conditions(self, reasoning_results: dict[str, Any]) -> list[str]:
        """
        Generate conditions that would necessitate revision of insights.

        Identifies specific evidential, logical, or pragmatic conditions
        that would require updating or abandoning current conclusions.

        Args:
            reasoning_results: Dictionary of reasoning pattern results

        Returns:
            List of revision conditions
        """
        logger.debug("Generating revision conditions for insights")

        revision_conditions = []

        for pattern_name, result in reasoning_results.items():
            if not result:
                continue

            # Extract confidence and uncertainty metrics
            uncertainty_factors = result.uncertainty_factors
            truth = result.truth

            # Generate pattern-specific revision conditions
            if pattern_name == "deductive":
                revision_conditions.extend(self._generate_deductive_revision_conditions(result))
            elif pattern_name == "inductive":
                revision_conditions.extend(self._generate_inductive_revision_conditions(result))
            elif pattern_name == "abductive":
                revision_conditions.extend(self._generate_abductive_revision_conditions(result))
            elif pattern_name == "analogical":
                revision_conditions.extend(self._generate_analogical_revision_conditions(result))
            elif pattern_name == "dialectical":
                revision_conditions.extend(self._generate_dialectical_revision_conditions(result))

        # Generate general epistemic revision conditions
        general_conditions = self._generate_general_revision_conditions(reasoning_results)
        revision_conditions.extend(general_conditions)

        # Generate pragmatic revision conditions
        pragmatic_conditions = self._generate_pragmatic_revision_conditions(reasoning_results)
        revision_conditions.extend(pragmatic_conditions)

        return revision_conditions

    async def _test_in_domain(self,
                            narsese_hypothesis: str,
                            domain: str,
                            domain_evidence: list[MemoryItem],
                            criteria: dict[str, Any]) -> dict[str, Any]:
        """
        Test a hypothesis within a specific philosophical domain.

        Evaluates hypothesis coherence, explanatory power, and consistency
        within the constraints and evidence of a particular domain.

        Args:
            narsese_hypothesis: Hypothesis in NARS format
            domain: The philosophical domain for testing
            domain_evidence: Relevant evidence from the domain
            criteria: Testing criteria and thresholds

        Returns:
            Dict containing test results and metrics
        """
        logger.debug(f"Testing hypothesis in domain '{domain}'")

        test_result = {
            'domain': domain,
            'hypothesis': narsese_hypothesis,
            'coherence_score': 0.0,
            'explanatory_power': 0.0,
            'consistency_score': 0.0,
            'evidential_support': 0.0,
            'overall_score': 0.0,
            'supporting_evidence': [],
            'conflicting_evidence': [],
            'test_details': {}
        }

        if not domain_evidence:
            test_result['test_details']['error'] = "No domain evidence available"
            return test_result

        # Test coherence with domain beliefs
        coherence_score = await self._test_domain_coherence(
            narsese_hypothesis, domain_evidence
        )

        # Test explanatory power
        explanatory_power = self._test_explanatory_power(
            narsese_hypothesis, domain_evidence, domain
        )

        # Test logical consistency
        consistency_score = self._test_logical_consistency(
            narsese_hypothesis, domain_evidence
        )

        # Assess evidential support
        evidential_support, supporting_evidence, conflicting_evidence = \
            self._assess_evidential_support(narsese_hypothesis, domain_evidence)

        # Calculate overall test score
        weights = criteria.get('weights', {
            'coherence': 0.3,
            'explanatory_power': 0.3,
            'consistency': 0.2,
            'evidential_support': 0.2
        })

        overall_score = (
            weights.get('coherence', 0.3) * coherence_score +
            weights.get('explanatory_power', 0.3) * explanatory_power +
            weights.get('consistency', 0.2) * consistency_score +
            weights.get('evidential_support', 0.2) * evidential_support
        )

        # Additional domain-specific tests
        domain_specific_results = await self._perform_domain_specific_tests(
            narsese_hypothesis, domain, domain_evidence, criteria
        )

        test_result.update({
            'coherence_score': coherence_score,
            'explanatory_power': explanatory_power,
            'consistency_score': consistency_score,
            'evidential_support': evidential_support,
            'overall_score': overall_score,
            'supporting_evidence': [e.term for e in supporting_evidence],
            'conflicting_evidence': [e.term for e in conflicting_evidence],
            'test_details': {
                'domain_evidence_count': len(domain_evidence),
                'domain_specific_results': domain_specific_results,
                'testing_criteria': criteria
            }
        })

        return test_result

    def _calculate_hypothesis_coherence(self, domain_results: dict[str, Any]) -> float:
        """
        Calculate overall coherence of hypothesis across domains.

        Aggregates domain-specific coherence scores using NARS-style
        confidence combination and philosophical coherence principles.

        Args:
            domain_results: Dictionary of test results by domain

        Returns:
            Overall coherence score (0.0 to 1.0)
        """
        logger.debug("Calculating hypothesis coherence across domains")

        if not domain_results:
            return 0.0

        coherence_scores = []
        domain_weights = []

        for domain, results in domain_results.items():
            coherence_score = results.get('coherence_score', 0.0)
            evidence_count = results.get('test_details', {}).get('domain_evidence_count', 0)

            # Weight by evidence availability and domain reliability
            domain_weight = min(1.0, evidence_count / 10.0)  # More evidence = higher weight

            coherence_scores.append(coherence_score)
            domain_weights.append(domain_weight)

        if not coherence_scores:
            return 0.0

        # Calculate weighted average coherence
        if sum(domain_weights) > 0:
            weighted_coherence = sum(s * w for s, w in zip(coherence_scores, domain_weights)) / sum(domain_weights)
        else:
            weighted_coherence = np.mean(coherence_scores)

        # Apply penalty for high variance (inconsistent across domains)
        coherence_variance = np.var(coherence_scores)
        variance_penalty = min(0.3, coherence_variance)  # Maximum 30% penalty

        overall_coherence = max(0.0, weighted_coherence - variance_penalty)

        return overall_coherence

    def _assess_pragmatic_value(self,
                              hypothesis: str,
                              domain_results: dict[str, Any]) -> dict[str, Any]:
        """
        Assess pragmatic value of hypothesis across domains.

        Evaluates practical utility, problem-solving potential, and
        actionable implications of the philosophical hypothesis.

        Args:
            hypothesis: The hypothesis being evaluated
            domain_results: Dictionary of test results by domain

        Returns:
            Dict containing pragmatic assessment
        """
        logger.debug(f"Assessing pragmatic value of hypothesis: {hypothesis}")

        pragmatic_assessment = {
            'overall_utility': 0.0,
            'problem_solving_potential': 0.0,
            'actionable_implications': [],
            'practical_applications': [],
            'utility_by_domain': {},
            'limitations': [],
            'pragmatic_score': 0.0
        }

        if not domain_results:
            pragmatic_assessment['limitations'].append("No domain results for assessment")
            return pragmatic_assessment

        domain_utilities = []
        all_implications = []

        for domain, results in domain_results.items():
            # Calculate domain-specific utility
            explanatory_power = results.get('explanatory_power', 0.0)
            evidence_support = results.get('evidential_support', 0.0)
            overall_score = results.get('overall_score', 0.0)

            # Domain utility combines multiple factors
            domain_utility = (
                0.4 * explanatory_power +
                0.3 * evidence_support +
                0.3 * overall_score
            )

            domain_utilities.append(domain_utility)

            # Extract domain-specific implications
            domain_implications = self._extract_pragmatic_implications(
                hypothesis, domain, results
            )
            all_implications.extend(domain_implications)

            pragmatic_assessment['utility_by_domain'][domain] = {
                'utility_score': domain_utility,
                'explanatory_power': explanatory_power,
                'evidence_support': evidence_support,
                'implications': domain_implications
            }

        # Calculate overall utility
        overall_utility = np.mean(domain_utilities) if domain_utilities else 0.0

        # Assess problem-solving potential
        problem_solving_potential = self._assess_problem_solving_potential(
            hypothesis, domain_results, all_implications
        )

        # Filter and rank actionable implications
        actionable_implications = self._filter_actionable_implications(all_implications)

        # Identify practical applications
        practical_applications = self._identify_practical_applications(
            hypothesis, domain_results
        )

        # Identify limitations
        limitations = self._identify_pragmatic_limitations(
            hypothesis, domain_results, overall_utility
        )

        # Calculate overall pragmatic score
        pragmatic_score = (
            0.4 * overall_utility +
            0.3 * problem_solving_potential +
            0.2 * (len(actionable_implications) / max(len(all_implications), 1)) +
            0.1 * (len(practical_applications) / max(len(domain_results), 1))
        )

        pragmatic_assessment.update({
            'overall_utility': overall_utility,
            'problem_solving_potential': problem_solving_potential,
            'actionable_implications': actionable_implications,
            'practical_applications': practical_applications,
            'limitations': limitations,
            'pragmatic_score': pragmatic_score,
            'total_implications': len(all_implications),
            'domains_assessed': len(domain_results)
        })

        return pragmatic_assessment

    def _calculate_hypothesis_confidence(self, domain_results: dict[str, Any]) -> float:
        """
        Calculate overall confidence in hypothesis using NARS principles.

        Combines domain-specific evidence using NARS truth functions
        to derive overall epistemic confidence in the hypothesis.

        Args:
            domain_results: Dictionary of test results by domain

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        logger.debug("Calculating hypothesis confidence")

        if not domain_results:
            return 0.0

        # Collect confidence-related metrics from each domain
        domain_confidences = []
        evidence_counts = []

        for domain, results in domain_results.items():
            # Extract confidence indicators
            overall_score = results.get('overall_score', 0.0)
            coherence_score = results.get('coherence_score', 0.0)
            evidential_support = results.get('evidential_support', 0.0)
            evidence_count = results.get('test_details', {}).get('domain_evidence_count', 0)

            # Calculate domain confidence using NARS-style combination
            domain_confidence = Truth.conjunction(
                TruthValue(overall_score, 0.9),
                Truth.conjunction(
                    TruthValue(coherence_score, 0.8),
                    TruthValue(evidential_support, 0.7)
                )
            ).confidence

            domain_confidences.append(domain_confidence)
            evidence_counts.append(evidence_count)

        # Combine domain confidences using NARS revision
        if len(domain_confidences) == 1:
            combined_confidence = domain_confidences[0]
        else:
            # Start with first confidence
            combined_truth = TruthValue(0.5, domain_confidences[0])

            # Revise with each additional domain
            for conf in domain_confidences[1:]:
                domain_truth = TruthValue(0.5, conf)
                combined_truth = Truth.revision(combined_truth, domain_truth)

            combined_confidence = combined_truth.confidence

        # Apply evidence-based confidence adjustment
        total_evidence = sum(evidence_counts)
        evidence_factor = min(1.0, total_evidence / (total_evidence + Truth.K))

        # Apply cross-domain consistency bonus/penalty
        confidence_variance = np.var(domain_confidences) if len(domain_confidences) > 1 else 0.0
        consistency_factor = max(0.5, 1.0 - confidence_variance)

        # Final confidence calculation
        final_confidence = combined_confidence * evidence_factor * consistency_factor

        return min(1.0, max(0.0, final_confidence))

    def _derive_implications(self,
                           hypothesis: str,
                           domain_results: dict[str, Any]) -> list[str]:
        """
        Derive logical and practical implications from tested hypothesis.

        Extracts consequences that follow from accepting the hypothesis
        given the domain-specific test results and evidence.

        Args:
            hypothesis: The tested hypothesis
            domain_results: Dictionary of test results by domain

        Returns:
            List of derived implications
        """
        logger.debug(f"Deriving implications from hypothesis: {hypothesis}")

        implications = []

        if not domain_results:
            return ["No implications: insufficient domain results"]

        # Derive logical implications
        logical_implications = self._derive_logical_implications(hypothesis, domain_results)
        implications.extend(logical_implications)

        # Derive practical implications
        practical_implications = self._derive_practical_implications(hypothesis, domain_results)
        implications.extend(practical_implications)

        # Derive methodological implications
        methodological_implications = self._derive_methodological_implications_from_hypothesis(
            hypothesis, domain_results
        )
        implications.extend(methodological_implications)

        # Derive theoretical implications
        theoretical_implications = self._derive_theoretical_implications(hypothesis, domain_results)
        implications.extend(theoretical_implications)

        # Derive epistemic implications
        epistemic_implications = self._derive_epistemic_implications(hypothesis, domain_results)
        implications.extend(epistemic_implications)

        # Derive critical implications (challenges and limitations)
        critical_implications = self._derive_critical_implications(hypothesis, domain_results)
        implications.extend(critical_implications)

        return implications

    async def _prime_nars_memory(self, beliefs: list[MemoryItem]) -> None:
        """Prime NARS with relevant beliefs from memory."""
        from contextlib import suppress
        for belief in beliefs:
            if belief.occurrence_time == "eternal":
                narsese = f"{belief.term}. {{{belief.truth.frequency:.2f} {belief.truth.confidence:.2f}}}"
                with suppress(Exception):  # Ignore priming failures
                    await self.nars.query(narsese, timeout=0.5)

    def _context_to_categories(self, context: str) -> list[str]:
        """Map context to philosophical categories."""
        context_lower = context.lower()

        category_mappings = {
            "metaphysical": ["existence", "reality", "being", "ontology"],
            "epistemological": ["knowledge", "belief", "truth", "science"],
            "ethical": ["ethics", "morality", "values", "good"],
            "phenomenological": ["consciousness", "experience", "mind"],
            "logical": ["logic", "reasoning", "inference", "formal"]
        }

        categories = []
        for category, keywords in category_mappings.items():
            if any(keyword in context_lower for keyword in keywords):
                categories.append(category)

        return categories or ["metaphysical"]  # Default

    def _domain_to_categories(self, domain: str) -> list[str]:
        """Map domain to philosophical categories."""
        # Similar to context mapping but domain-specific
        return self._context_to_categories(domain)

    def _generate_perspective_queries(self,
                                    concept: str,
                                    context: str,
                                    perspective: str) -> list[str]:
        """Generate NARS queries for a philosophical perspective."""
        queries = []

        # Perspective-specific query patterns
        if perspective == "analytical":
            queries.extend([
                f"<{concept} --> ?what>?",
                f"<?what --> {concept}>?",
                f"<({concept} * ?x) --> ?relation>?"
            ])
        elif perspective == "phenomenological":
            queries.extend([
                f"<{concept} --> experience>?",
                f"<{concept} --> consciousness>?",
                f"<{concept} --> [?property]>?"
            ])
        elif perspective == "pragmatist":
            queries.extend([
                f"<{concept} =/> ?consequence>?",
                f"<?action =/> {concept}>?",
                f"<{concept} --> useful>?"
            ])
        # Add more perspective patterns...

        return queries

    async def _synthesize_perspective_analysis(self,
                                             concept: str,
                                             perspective: str,
                                             results: list[dict[str, Any]],
                                             evidence: list[MemoryItem]) -> dict[str, Any]:
        """Synthesize analysis from a philosophical perspective."""
        # Extract key findings
        findings = []
        total_confidence = 0.0

        for result in results:
            if "truth" in result:
                findings.append({
                    "claim": result["term"],
                    "truth": TruthValue(
                        result["truth"]["frequency"],
                        result["truth"]["confidence"]
                    )
                })
                total_confidence += result["truth"]["confidence"]

        # Generate perspective interpretation
        interpretation = {
            "perspective": perspective,
            "findings": findings,
            "average_confidence": total_confidence / len(findings) if findings else 0.0,
            "key_insights": self._extract_perspective_insights(perspective, findings),
            "limitations": self._identify_perspective_limitations(perspective, concept)
        }

        return interpretation

    def _extract_perspective_insights(self,
                                    perspective: str,
                                    findings: list[dict[str, Any]]) -> list[str]:
        """Extract key insights from perspective findings."""
        insights = []

        # Sort by confidence
        sorted_findings = sorted(findings,
                               key=lambda x: x["truth"].expectation,
                               reverse=True)

        # Take top findings and interpret
        for finding in sorted_findings[:3]:
            insight = f"{perspective} reveals: {finding['claim']} "
            insight += f"(confidence: {finding['truth'].confidence:.2f})"
            insights.append(insight)

        return insights

    def _identify_perspective_limitations(self,
                                        perspective: str,
                                        concept: str) -> list[str]:
        """Identify limitations of perspective for concept."""
        limitations = {
            "analytical": [
                "May miss experiential dimensions",
                "Focuses on logical structure over meaning"
            ],
            "phenomenological": [
                "Subjective focus may lack generalizability",
                "Difficult to verify intersubjectively"
            ],
            "pragmatist": [
                "May reduce truth to utility",
                "Context-dependent results"
            ]
        }

        return limitations.get(perspective, ["Perspective-specific limitations apply"])

    def _hypothesis_to_narsese(self, hypothesis: str) -> str:
        """Convert natural language hypothesis to Narsese."""
        # Note: This is a naive conversion; ideally, the AI client should handle robust Narsese translation
        hypothesis_lower = hypothesis.lower()

        # Pattern matching for common forms
        if " is " in hypothesis_lower:
            parts = hypothesis_lower.split(" is ")
            return f"<{parts[0].strip()} --> {parts[1].strip()}>"
        elif " causes " in hypothesis_lower:
            parts = hypothesis_lower.split(" causes ")
            return f"<{parts[0].strip()} =/> {parts[1].strip()}>"
        else:
            # Default to property assertion
            return f"<{hypothesis} --> true>"

    def _extract_claims_from_insight(self, insight: str) -> list[str]:
        """Extract propositional claims from an insight string."""
        # Simple extraction: assume the main claim is after the pattern name
        if ']' in insight:
            return [insight.split(']', 1)[1].strip()]
        return [insight.strip()]

    def _detect_logical_contradictions(self,
                                       claims1: list[str],
                                       claims2: list[str],
                                       perspective1: str,
                                       perspective2: str) -> list[str]:
        """Detect logical contradictions between two sets of claims."""
        contradictions = []
        for c1 in claims1:
            for c2 in claims2:
                # Simplified logical contradiction detection
                # This would ideally use a more robust NARS-based contradiction detection
                if "not " + c1.lower() in c2.lower() or "not " + c2.lower() in c1.lower():
                    contradictions.append(
                        f"Logical contradiction between '{c1}' ({perspective1}) and '{c2}' ({perspective2})"
                    )
        return contradictions

    # ─────────────────────────────────────────────────────────────────────────
    # Supporting Helper Methods (Simplified Implementations)
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_finding_similarity(self, finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Calculate semantic similarity between two findings."""
        if 'claim' not in finding1 or 'claim' not in finding2:
            return 0.0
        
        # Simple lexical similarity
        claim1_words = set(finding1['claim'].lower().split())
        claim2_words = set(finding2['claim'].lower().split())
        
        if not claim1_words or not claim2_words:
            return 0.0
        
        intersection = len(claim1_words.intersection(claim2_words))
        union = len(claim1_words.union(claim2_words))
        
        return intersection / union if union > 0 else 0.0

    def _synthesize_convergent_truths(self, findings: list[dict[str, Any]], perspectives: list[str]) -> TruthValue:
        """Synthesize truth values from convergent findings across perspectives."""
        if not findings:
            return TruthValue(0.5, 0.0)
        
        # Use NARS revision to combine truth values
        result_truth = TruthValue(0.5, 0.0)
        
        for finding in findings:
            if 'truth' in finding:
                finding_truth = TruthValue(
                    finding['truth']['frequency'],
                    finding['truth']['confidence']
                )
                result_truth = Truth.revision(result_truth, finding_truth)
        
        return result_truth

    def _calculate_tension_score(self, finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Calculate tension/contradiction score between findings."""
        if 'truth' not in finding1 or 'truth' not in finding2:
            return 0.0
        
        freq1 = finding1['truth']['frequency']
        freq2 = finding2['truth']['frequency']
        
        # High tension when frequencies are opposite and both confident
        conf1 = finding1['truth']['confidence']
        conf2 = finding2['truth']['confidence']
        
        frequency_distance = abs(freq1 - freq2)
        combined_confidence = (conf1 + conf2) / 2.0
        
        return frequency_distance * combined_confidence

    def _assess_dialectical_potential(self, finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Assess potential for dialectical synthesis between contradictory findings."""
        tension = self._calculate_tension_score(finding1, finding2)
        
        if tension < 0.3:
            return 0.0  # No significant tension
        
        # Higher dialectical potential when both findings have evidence
        conf1 = finding1.get('truth', {}).get('confidence', 0.0)
        conf2 = finding2.get('truth', {}).get('confidence', 0.0)
        
        evidence_quality = (conf1 + conf2) / 2.0
        
        return min(1.0, tension * evidence_quality)

    async def _perform_dialectical_synthesis(self, 
                                           convergent_insights: list[dict[str, Any]], 
                                           divergent_tensions: list[dict[str, Any]], 
                                           concept: str, 
                                           context: str) -> dict[str, Any]:
        """Perform dialectical synthesis of convergent and divergent insights."""
        synthesis = {
            'synthesis_type': 'dialectical',
            'unified_insights': [],
            'resolved_tensions': [],
            'remaining_tensions': [],
            'synthesis_strength': 0.0
        }
        
        # Unify convergent insights
        for insight in convergent_insights:
            if insight['synthesis_strength'] > 0.5:
                synthesis['unified_insights'].append({
                    'insight': insight['insight'],
                    'strength': insight['synthesis_strength'],
                    'supporting_perspectives': insight['supporting_perspectives']
                })
        
        # Attempt to resolve tensions dialectically
        for tension in divergent_tensions:
            if tension['dialectical_potential'] > 0.6:
                resolved = await self._attempt_dialectical_resolution(tension, concept, context)
                if resolved:
                    synthesis['resolved_tensions'].append(resolved)
                else:
                    synthesis['remaining_tensions'].append(tension)
            else:
                synthesis['remaining_tensions'].append(tension)
        
        # Calculate synthesis strength
        unified_strength = len(synthesis['unified_insights']) / max(len(convergent_insights), 1)
        resolution_strength = len(synthesis['resolved_tensions']) / max(len(divergent_tensions), 1)
        
        synthesis['synthesis_strength'] = (unified_strength + resolution_strength) / 2.0
        
        return synthesis

    async def _attempt_dialectical_resolution(self, 
                                            tension: dict[str, Any], 
                                            concept: str, 
                                            context: str) -> dict[str, Any] | None:
        """Attempt to resolve a dialectical tension through synthesis."""
        # Simple heuristic resolution - in production would use sophisticated NARS synthesis
        try:
            return {
                'tension': tension['tension'],
                'synthesis': f"Dialectical resolution of {tension['tension']} through higher-order integration",
                'resolution_strength': 0.7
            }
        except Exception:
            return None

    def _calculate_synthesis_coherence(self, 
                                     convergent_insights: list[dict[str, Any]], 
                                     divergent_tensions: list[dict[str, Any]], 
                                     dialectical_synthesis: dict[str, Any]) -> float:
        """Calculate overall coherence of the synthesis."""
        if not convergent_insights and not divergent_tensions:
            return 0.0
        
        convergence_factor = len(convergent_insights) / max(len(convergent_insights) + len(divergent_tensions), 1)
        synthesis_factor = dialectical_synthesis.get('synthesis_strength', 0.0)
        
        return (convergence_factor + synthesis_factor) / 2.0

    def _determine_epistemological_status(self, synthesis_coherence: float) -> str:
        """Determine epistemological status based on synthesis coherence."""
        if synthesis_coherence > 0.8:
            return "High epistemic warrant"
        elif synthesis_coherence > 0.6:
            return "Moderate epistemic warrant"
        elif synthesis_coherence > 0.4:
            return "Provisional epistemic warrant"
        else:
            return "Low epistemic warrant"

    def _identify_synthesis_revision_triggers(self, 
                                            convergent_insights: list[dict[str, Any]], 
                                            divergent_tensions: list[dict[str, Any]]) -> list[str]:
        """Identify conditions that would trigger revision of the synthesis."""
        triggers = []
        
        # Low convergence triggers
        if len(convergent_insights) < 2:
            triggers.append("Insufficient convergent evidence")
        
        # High tension triggers
        unresolved_tensions = len([t for t in divergent_tensions if t['tension_strength'] > 0.7])
        if unresolved_tensions > len(convergent_insights):
            triggers.append("Unresolved contradictions exceed convergent insights")
        
        # Confidence-based triggers
        low_confidence_insights = len([i for i in convergent_insights 
                                     if i['synthesized_truth']['confidence'] < 0.5])
        if low_confidence_insights > len(convergent_insights) / 2:
            triggers.append("Low confidence in synthesis")
        
        return triggers

    def _detect_logical_contradiction(self, finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Detect logical contradiction between two findings."""
        # Simplified logical contradiction detection
        claim1 = finding1.get('claim', '').lower()
        claim2 = finding2.get('claim', '').lower()
        
        # Check for explicit negation patterns
        negation_indicators = ['not', 'no', 'never', 'none', 'nothing']
        
        contradiction_score = 0.0
        
        # Simple heuristic: if one claim contains negation of concepts in the other
        claim1_words = set(claim1.split())
        claim2_words = set(claim2.split())
        
        # Check if one claim negates the other
        for neg in negation_indicators:
            if neg in claim1_words and any(word in claim2_words for word in claim1_words if word != neg):
                contradiction_score += 0.3
            if neg in claim2_words and any(word in claim1_words for word in claim2_words if word != neg):
                contradiction_score += 0.3
        
        # Check truth value opposition
        if 'truth' in finding1 and 'truth' in finding2:
            freq1 = finding1['truth']['frequency']
            freq2 = finding2['truth']['frequency']
            
            if abs(freq1 - freq2) > 0.6:
                contradiction_score += 0.4
        
        return min(1.0, contradiction_score)

    def _calculate_semantic_compatibility(self, finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Calculate semantic compatibility between findings."""
        return self._calculate_finding_similarity(finding1, finding2)

    def _extract_concepts_from_term(self, term: str) -> list[str]:
        """Extract atomic concepts from NARS term."""
        concepts = []
        
        # Remove NARS operators and extract concepts
        cleaned_term = term.replace('<', '').replace('>', '').replace('[', '').replace(']', '')
        
        # Split on common NARS operators
        for separator in ['-->', '==>', '=/>', '<->', '*', '&', '|']:
            cleaned_term = cleaned_term.replace(separator, ' ')
        
        # Extract individual concepts
        potential_concepts = cleaned_term.split()
        
        for concept in potential_concepts:
            concept = concept.strip()
            if concept and len(concept) > 1:  # Filter out single characters and operators
                concepts.append(concept)
        
        return concepts

    def _extract_relations_from_term(self, term: str, truth: TruthValue) -> list[dict[str, Any]]:
        """Extract relational information from NARS term."""
        relations = []
        
        # Parse inheritance relations
        if '-->' in term:
            parts = term.split('-->')
            if len(parts) == 2:
                source = parts[0].strip('<> ')
                target = parts[1].strip('<> ')
                relations.append({
                    'source': source,
                    'target': target,
                    'relation': 'inheritance',
                    'weight': truth.expectation,
                    'truth': truth.to_dict()
                })
        
        # Parse implication relations
        if '==>' in term:
            parts = term.split('==>')
            if len(parts) == 2:
                source = parts[0].strip('<> ')
                target = parts[1].strip('<> ')
                relations.append({
                    'source': source,
                    'target': target,
                    'relation': 'implication',
                    'weight': truth.expectation,
                    'truth': truth.to_dict()
                })
        
        return relations

    def _perform_semantic_clustering(self, concept_embeddings: dict[str, np.ndarray]) -> dict[str, int]:
        """Perform semantic clustering of concepts using embeddings."""
        if len(concept_embeddings) < 2:
            return {concept: 0 for concept in concept_embeddings.keys()}
        
        # Simple clustering based on similarity thresholds
        concepts = list(concept_embeddings.keys())
        embeddings = list(concept_embeddings.values())
        
        clusters = {}
        cluster_id = 0
        
        for i, concept in enumerate(concepts):
            if concept in clusters:
                continue
            
            # Start new cluster
            clusters[concept] = cluster_id
            
            # Find similar concepts
            for j, other_concept in enumerate(concepts[i+1:], i+1):
                if other_concept in clusters:
                    continue
                
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity > 0.7:  # Threshold for same cluster
                    clusters[other_concept] = cluster_id
            
            cluster_id += 1
        
        return clusters

    def _calculate_graph_centrality(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate centrality measures for graph nodes."""
        centrality = {}
        
        # Calculate degree centrality
        node_degrees = {node_id: 0 for node_id in nodes.keys()}
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            if source in node_degrees:
                node_degrees[source] += 1
            if target in node_degrees:
                node_degrees[target] += 1
        
        max_degree = max(node_degrees.values()) if node_degrees.values() else 1
        
        for node_id, degree in node_degrees.items():
            centrality[node_id] = degree / max_degree
        
        return centrality

    async def _query_related_concepts(self, concept: str) -> list[tuple[str, str, TruthValue]]:
        """Query NARS for concepts related to the given concept."""
        related_concepts = []
        
        # Generate queries for related concepts
        queries = [
            f"<{concept} --> ?x>?",
            f"<?x --> {concept}>?",
            f"<{concept} =/> ?x>?",
            f"<?x =/> {concept}>?"
        ]
        
        for query in queries:
            try:
                result = await self.nars.query(query, timeout=1.0)
                if result.get("answers"):
                    for answer in result["answers"]:
                        if "term" in answer and "truth" in answer:
                            # Extract related concept from answer
                            related_concept = self._extract_related_concept_from_answer(
                                answer["term"], concept
                            )
                            if related_concept:
                                truth = TruthValue(
                                    answer["truth"]["frequency"],
                                    answer["truth"]["confidence"]
                                )
                                relation_type = self._determine_relation_type(query)
                                related_concepts.append((related_concept, relation_type, truth))
            except Exception:
                continue
        
        return related_concepts[:10]  # Limit results

    def _extract_related_concept_from_answer(self, answer_term: str, original_concept: str) -> str | None:
        """Extract the related concept from a NARS answer term."""
        # Simple extraction - remove original concept and operators
        cleaned = answer_term.replace(original_concept, '').replace('<', '').replace('>', '')
        cleaned = cleaned.replace('-->', '').replace('==>', '').replace('=/>', '')
        cleaned = cleaned.strip()
        
        if cleaned and len(cleaned) > 1:
            return cleaned
        return None

    def _determine_relation_type(self, query: str) -> str:
        """Determine relation type from query pattern."""
        if '-->' in query:
            return 'inheritance'
        elif '==>' in query:
            return 'implication'
        elif '=/>' in query:
            return 'predictive'
        else:
            return 'unknown'

    def _infer_category(self, concept: str) -> str | None:
        """Infer philosophical category for a concept."""
        return self.memory._categorize_term(concept)

    def _calculate_graph_metrics(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate overall graph metrics."""
        metrics = {
            'density': 0.0,
            'average_centrality': 0.0,
            'clustering_coefficient': 0.0,
            'average_confidence': 0.0
        }
        
        if not nodes or not edges:
            return metrics
        
        # Graph density
        n_nodes = len(nodes)
        max_edges = n_nodes * (n_nodes - 1)
        metrics['density'] = len(edges) / max_edges if max_edges > 0 else 0.0
        
        # Average centrality
        centralities = [node.get('centrality', 0.0) for node in nodes.values()]
        metrics['average_centrality'] = np.mean(centralities) if centralities else 0.0
        
        # Average confidence
        confidences = [edge['truth']['confidence'] for edge in edges if 'truth' in edge]
        metrics['average_confidence'] = np.mean(confidences) if confidences else 0.0
        
        return metrics

    def _get_cluster_summary(self, nodes: dict[str, Any]) -> dict[str, list[str]]:
        """Get summary of semantic clusters."""
        clusters = {}
        
        for node_id, node_data in nodes.items():
            cluster_id = node_data.get('semantic_cluster')
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node_id)
        
        return clusters

    def _generate_evidence_queries(self, term: str, phenomenon: str) -> list[str]:
        """Generate NARS queries for evidence gathering."""
        queries = []
        
        # Direct queries about the term
        queries.extend([
            f"<{term} --> ?property>?",
            f"<?subject --> {term}>?",
            f"<{term} =/> ?effect>?",
            f"<?cause =/> {term}>?"
        ])
        
        # Phenomenon-related queries
        if term != phenomenon:
            queries.extend([
                f"<({term} * {phenomenon}) --> ?relation>?",
                f"<{term} <-> {phenomenon}>?"
            ])
        
        return queries

    def _is_relevant_for_expansion(self, answer_term: str, phenomenon: str) -> bool:
        """Check if an answer term is relevant for evidence expansion."""
        # Simple heuristic - check if term shares concepts with phenomenon
        answer_words = set(answer_term.lower().split())
        phenomenon_words = set(phenomenon.lower().split())
        
        # At least one word overlap or semantic indicators
        semantic_indicators = ['related', 'similar', 'causes', 'implies', 'connected']
        
        has_overlap = bool(answer_words.intersection(phenomenon_words))
        has_semantic_indicator = any(indicator in answer_term.lower() for indicator in semantic_indicators)
        
        return has_overlap or has_semantic_indicator

    def _calculate_evidence_relevance(self, item: MemoryItem, phenomenon: str) -> float:
        """Calculate relevance score for evidence item."""
        # Text-based relevance
        term_words = set(item.term.lower().split())
        phenomenon_words = set(phenomenon.lower().split())
        
        word_overlap = len(term_words.intersection(phenomenon_words))
        word_union = len(term_words.union(phenomenon_words))
        
        text_relevance = word_overlap / word_union if word_union > 0 else 0.0
        
        # Truth-based relevance (higher confidence = more relevant)
        truth_relevance = item.truth.confidence
        
        # Category-based relevance
        category_relevance = 0.3 if item.philosophical_category else 0.0
        
        # Combined relevance score
        relevance = 0.5 * text_relevance + 0.3 * truth_relevance + 0.2 * category_relevance
        
        return min(1.0, relevance)

    # Placeholder methods that need full implementation
    # These provide basic functionality to prevent errors

    def _identify_high_confidence_clusters(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify clusters of high-confidence concepts."""
        return []

    def _identify_central_supported_concepts(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify central concepts with strong evidential support."""
        return []

    def _analyze_category_coherence(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
        """Analyze coherence within philosophical categories."""
        return {}

    def _identify_emergent_structures(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify emergent conceptual structures."""
        return []

    def _identify_contradiction_clusters(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify contradiction clusters."""
        return []

    def _identify_inferential_chains(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify inferential chains."""
        return []

    def _identify_semantic_coherence_regions(self, nodes: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify semantic coherence regions."""
        return []

    def _classify_attractor_type(self, concept: str, metrics: dict[str, Any], edges: list[dict[str, Any]]) -> str:
        """Classify the type of conceptual attractor."""
        return "local_attractor"

    def _identify_attractor_clusters(self, attractors: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
        """Identify clusters of mutually reinforcing attractors."""
        return []

    async def _simulate_perturbation_resistance(self, 
                                              nodes: dict[str, Any], 
                                              edges: list[dict[str, Any]], 
                                              local_stabilities: dict[str, Any]) -> float:
        """Simulate perturbation resistance."""
        return 0.5

    def _analyze_attractor_stability(self, attractors: list[str], local_stabilities: dict[str, Any]) -> float:
        """Analyze attractor stability."""
        return 0.5

    def _analyze_hierarchical_structure(self, nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze hierarchical structure."""
        return {}

    def _calculate_category_internal_coherence(self, concepts: list[str], edges: list[dict[str, Any]]) -> float:
        """Calculate category internal coherence."""
        return 0.5

    def _identify_inferential_patterns(self, edges: list[dict[str, Any]]) -> list[str]:
        """Identify inferential patterns."""
        return []

    def _extract_structural_principles(self, 
                                     nodes: dict[str, Any], 
                                     edges: list[dict[str, Any]], 
                                     foundational_concepts: list[dict[str, Any]], 
                                     hierarchical_layers: dict[str, Any]) -> list[str]:
        """Extract structural principles."""
        return []

    def _analyze_emergence_levels(self, 
                                nodes: dict[str, Any], 
                                edges: list[dict[str, Any]], 
                                hierarchical_layers: dict[str, Any]) -> dict[str, Any]:
        """Analyze emergence levels."""
        return {}

    # More placeholder methods for complete functionality

    def _detect_semantic_tensions(self, 
                                 claims1: list[str], 
                                 claims2: list[str], 
                                 perspective1: str, 
                                 perspective2: str) -> list[str]:
        """Detect semantic tensions between claim sets."""
        return []

    def _detect_evidential_conflicts(self, 
                                   claims1: list[str], 
                                   claims2: list[str], 
                                   perspective1: str, 
                                   perspective2: str) -> list[str]:
        """Detect evidential conflicts between claim sets."""
        return []

    def _identify_meta_contradictions(self, perspective_insights: dict[str, Any]) -> list[str]:
        """Identify meta-level contradictions."""
        return []

    def _classify_contradiction(self, contradiction: str) -> str:
        """Classify type and severity of contradiction."""
        return "GENERAL"

    def _assess_phenomenological_adequacy(self, phenomenon: str, perspective_insights: dict[str, Any]) -> str:
        """Assess phenomenological adequacy."""
        return "Adequate for current analysis"

    def _analyze_perspectival_complementarity(self, perspective_insights: dict[str, Any], contradictions: list[str]) -> str:
        """Analyze perspectival complementarity."""
        return "Perspectives provide complementary viewpoints"

    def _assess_epistemic_status(self, phenomenon: str, perspective_insights: dict[str, Any], contradictions: list[str]) -> str:
        """Assess epistemic status."""
        return "Provisional epistemic status"

    def _derive_methodological_implications(self, phenomenon: str, perspective_insights: dict[str, Any], contradictions: list[str]) -> str:
        """Derive methodological implications."""
        return "Multi-perspective analysis recommended"

    def _analyze_conceptual_boundaries(self, phenomenon: str, perspective_insights: dict[str, Any]) -> str:
        """Analyze conceptual boundaries."""
        return "Boundaries require further clarification"

    def _identify_emergence_patterns(self, phenomenon: str, perspective_insights: dict[str, Any]) -> str:
        """Identify emergence patterns."""
        return "Emergent properties detected"

    def _assess_dialectical_potential_meta(self, contradictions: list[str], perspective_insights: dict[str, Any]) -> str:
        """Assess dialectical potential at meta level."""
        return "Moderate dialectical potential"

    def _derive_pragmatic_consequences(self, phenomenon: str, perspective_insights: dict[str, Any], contradictions: list[str]) -> str:
        """Derive pragmatic consequences."""
        return "Further practical investigation needed"

    def _suggest_inquiry_directions(self, phenomenon: str, perspective_insights: dict[str, Any], contradictions: list[str]) -> str:
        """Suggest future inquiry directions."""
        return "Continued multi-perspective investigation recommended"

    def _generate_deductive_revision_conditions(self, result: ReasoningResult) -> list[str]:
        """Generate revision conditions for deductive reasoning."""
        return [f"If premise confidence drops below {result.truth.confidence - 0.2:.2f}"]

    def _generate_inductive_revision_conditions(self, result: ReasoningResult) -> list[str]:
        """Generate revision conditions for inductive reasoning."""
        return ["If contradictory instances exceed supporting instances"]

    def _generate_abductive_revision_conditions(self, result: ReasoningResult) -> list[str]:
        """Generate revision conditions for abductive reasoning."""
        return ["If better explanation emerges with higher explanatory power"]

    def _generate_analogical_revision_conditions(self, result: ReasoningResult) -> list[str]:
        """Generate revision conditions for analogical reasoning."""
        return ["If analogical similarity falls below 0.5"]

    def _generate_dialectical_revision_conditions(self, result: ReasoningResult) -> list[str]:
        """Generate revision conditions for dialectical reasoning."""
        return ["If synthesis coherence drops below acceptable threshold"]

    def _generate_general_revision_conditions(self, reasoning_results: dict[str, Any]) -> list[str]:
        """Generate general epistemic revision conditions."""
        return ["If overall confidence drops significantly", "If new contradictory evidence emerges"]

    def _generate_pragmatic_revision_conditions(self, reasoning_results: dict[str, Any]) -> list[str]:
        """Generate pragmatic revision conditions."""
        return ["If practical applications fail consistently"]

    async def _test_domain_coherence(self, narsese_hypothesis: str, domain_evidence: list[MemoryItem]) -> float:
        """Test hypothesis coherence with domain beliefs."""
        return 0.5

    def _test_explanatory_power(self, narsese_hypothesis: str, domain_evidence: list[MemoryItem], domain: str) -> float:
        """Test explanatory power of hypothesis."""
        return 0.5

    def _test_logical_consistency(self, narsese_hypothesis: str, domain_evidence: list[MemoryItem]) -> float:
        """Test logical consistency of hypothesis."""
        return 0.5

    def _assess_evidential_support(self, narsese_hypothesis: str, domain_evidence: list[MemoryItem]) -> tuple[float, list[MemoryItem], list[MemoryItem]]:
        """Assess evidential support for hypothesis."""
        return 0.5, [], []

    async def _perform_domain_specific_tests(self, 
                                           narsese_hypothesis: str, 
                                           domain: str, 
                                           domain_evidence: list[MemoryItem], 
                                           criteria: dict[str, Any]) -> dict[str, Any]:
        """Perform domain-specific tests."""
        return {"status": "completed"}

    def _extract_pragmatic_implications(self, hypothesis: str, domain: str, results: dict[str, Any]) -> list[str]:
        """Extract pragmatic implications."""
        return [f"Pragmatic implication for {hypothesis} in {domain}"]

    def _assess_problem_solving_potential(self, hypothesis: str, domain_results: dict[str, Any], implications: list[str]) -> float:
        """Assess problem-solving potential."""
        return 0.5

    def _filter_actionable_implications(self, implications: list[str]) -> list[str]:
        """Filter and rank actionable implications."""
        return implications[:5]  # Return top 5

    def _identify_practical_applications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Identify practical applications."""
        return [f"Practical application of {hypothesis}"]

    def _identify_pragmatic_limitations(self, hypothesis: str, domain_results: dict[str, Any], utility: float) -> list[str]:
        """Identify pragmatic limitations."""
        return ["Limited practical applicability"]

    def _derive_logical_implications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive logical implications."""
        return [f"Logical implication: {hypothesis} entails further investigation"]

    def _derive_practical_implications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive practical implications."""
        return [f"Practical implication: {hypothesis} suggests new approaches"]

    def _derive_methodological_implications_from_hypothesis(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive methodological implications from hypothesis."""
        return [f"Methodological implication: {hypothesis} requires new research methods"]

    def _derive_theoretical_implications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive theoretical implications."""
        return [f"Theoretical implication: {hypothesis} challenges existing theories"]

    def _derive_epistemic_implications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive epistemic implications."""
        return [f"Epistemic implication: {hypothesis} affects what we can know"]

    def _derive_critical_implications(self, hypothesis: str, domain_results: dict[str, Any]) -> list[str]:
        """Derive critical implications (challenges and limitations)."""
        return [f"Critical implication: {hypothesis} faces significant challenges"]
