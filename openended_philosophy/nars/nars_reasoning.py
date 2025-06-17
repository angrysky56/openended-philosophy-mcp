"""
NARS Reasoning Integration - Philosophical Non-Axiomatic Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Integrates NARS reasoning capabilities with philosophical analysis,
providing non-axiomatic inference, multi-perspective synthesis, and
epistemic uncertainty quantification.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .nars_manager import NARSManager
from .nars_memory import NARSMemory, MemoryItem
from .truth_functions import Truth, TruthValue

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of NARS reasoning process."""
    conclusion: str
    truth: TruthValue
    evidence: List[MemoryItem]
    inference_path: List[str]
    uncertainty_factors: Dict[str, float]
    philosophical_implications: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
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
                            perspectives: List[str]) -> Dict[str, Any]:
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
                              depth: int = 3) -> Dict[str, Any]:
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
                              perspectives: List[str],
                              depth: int = 3) -> Dict[str, Any]:
        """
        Generate insights through NARS reasoning and synthesis.
        
        Produces fallibilistic insights with uncertainty quantification.
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
            "reasoning_results": {k: v.to_dict() for k, v in reasoning_results.items()},
            "perspective_insights": perspective_insights,
            "contradictions": contradictions,
            "meta_insights": meta_insights,
            "revision_conditions": self._generate_revision_conditions(reasoning_results)
        }
        
    async def test_hypothesis(self,
                            hypothesis: str,
                            test_domains: List[str],
                            criteria: Dict[str, Any]) -> Dict[str, Any]:
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
                                 evidence: List[MemoryItem]) -> Optional[ReasoningResult]:
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
                                 evidence: List[MemoryItem]) -> Optional[ReasoningResult]:
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
        best_pattern = max(pattern_counts, key=pattern_counts.get)
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
                                 evidence: List[MemoryItem]) -> Optional[ReasoningResult]:
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
                                  evidence: List[MemoryItem]) -> Optional[ReasoningResult]:
        """Apply analogical reasoning pattern."""
        # Find similar phenomena
        similar_items = []
        
        for item in evidence:
            if "-->" in item.term and phenomenon not in item.term:
                # Check semantic similarity
                if self.memory._generate_embedding(phenomenon) is not None:
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
                                   evidence: List[MemoryItem]) -> Optional[ReasoningResult]:
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
    
    async def _prime_nars_memory(self, beliefs: List[MemoryItem]) -> None:
        """Prime NARS with relevant beliefs from memory."""
        for belief in beliefs:
            if belief.occurrence_time == "eternal":
                narsese = f"{belief.term}. {{{belief.truth.frequency:.2f} {belief.truth.confidence:.2f}}}"
                try:
                    await self.nars.query(narsese, timeout=0.5)
                except:
                    pass  # Ignore priming failures
                    
    def _context_to_categories(self, context: str) -> List[str]:
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
        
    def _domain_to_categories(self, domain: str) -> List[str]:
        """Map domain to philosophical categories."""
        # Similar to context mapping but domain-specific
        return self._context_to_categories(domain)
        
    def _generate_perspective_queries(self,
                                    concept: str,
                                    context: str,
                                    perspective: str) -> List[str]:
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
                                             results: List[Dict[str, Any]],
                                             evidence: List[MemoryItem]) -> Dict[str, Any]:
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
                                    findings: List[Dict[str, Any]]) -> List[str]:
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
                                        concept: str) -> List[str]:
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
        # Simple conversion - in production use NLP
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
