"""
Enhanced NARS Integration for Deep Philosophical Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module provides deep integration between NARS non-axiomatic reasoning
and philosophical analysis, enabling sophisticated belief revision, temporal
reasoning, and multi-perspective synthesis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..nars import NARSManager, NARSMemory, TruthValue
from ..nars.truth_functions import Truth
from .llm_semantic_processor import (
    LLMSemanticProcessor,
    PhilosophicalConcept,
    SemanticAnalysis,
)

logger = logging.getLogger(__name__)


@dataclass
class PhilosophicalBelief:
    """Enhanced belief structure for philosophical reasoning in NARS."""
    statement: str
    narsese_term: str
    truth: TruthValue
    philosophical_context: dict[str, Any]
    supporting_evidence: list[str]
    challenging_evidence: list[str]
    perspective_source: str
    temporal_scope: str  # "eternal", "temporal", "historical"
    revision_count: int = 0
    last_revised: datetime = field(default_factory=datetime.now)
    semantic_embedding: np.ndarray | None = None


@dataclass
class BeliefRevisionEvent:
    """Record of belief revision for tracking epistemic changes."""
    original_belief: PhilosophicalBelief
    revised_belief: PhilosophicalBelief
    revision_reason: str
    evidence_delta: dict[str, Any]
    confidence_change: float
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedNARSMemory:
    """
    NARS memory with deep semantic understanding and philosophical categorization.

    This enhanced memory system integrates semantic embeddings, philosophical
    categorization, and sophisticated belief revision mechanisms.
    """

    def __init__(
        self,
        nars_memory: NARSMemory,
        llm_processor: LLMSemanticProcessor
    ):
        self.base_memory = nars_memory
        self.llm_processor = llm_processor
        self.philosophical_beliefs: dict[str, PhilosophicalBelief] = {}
        self.belief_networks: dict[str, set[str]] = {}  # belief_id -> related_beliefs
        self.revision_history: list[BeliefRevisionEvent] = []
        self.perspective_indices: dict[str, set[str]] = {}  # perspective -> belief_ids
        self.temporal_index: dict[str, list[tuple[datetime, str]]] = {}  # concept -> [(time, belief_id)]

        logger.info("Enhanced NARS Memory initialized")
    async def process_philosophical_statement(
        self,
        statement: str,
        context: dict[str, Any],
        perspective: str = "general"
    ) -> PhilosophicalBelief:
        """
        Process a philosophical statement into NARS memory with semantic understanding.
        """
        # Convert dict context to PhilosophicalContext if needed
        from .llm_semantic_processor import PhilosophicalContext

        if isinstance(context, dict):
            phil_context = PhilosophicalContext(
                domain=context.get("domain", "general"),
                inquiry_type=context.get("inquiry_type", "analysis"),
                depth_requirements=context.get("depth", 3)
            )
        else:
            phil_context = context

        # Analyze statement semantically
        semantic_analysis = await self.llm_processor.analyze_statement(
            statement,
            phil_context
        )

        # Convert to NARS format with philosophical grounding
        narsese_term = self._to_philosophically_grounded_narsese(
            statement,
            semantic_analysis
        )

        # Calculate initial truth value based on evidence and uncertainty
        truth_value = self._calculate_philosophical_truth(
            semantic_analysis,
            context
        )

        # Create philosophical belief
        belief = PhilosophicalBelief(
            statement=statement,
            narsese_term=narsese_term,
            truth=truth_value,
            philosophical_context=context,
            supporting_evidence=[],
            challenging_evidence=[],
            perspective_source=perspective,
            temporal_scope=self._determine_temporal_scope(statement, semantic_analysis),
            semantic_embedding=self._generate_semantic_embedding(statement)
        )

        # Store in enhanced memory
        belief_id = self._generate_belief_id(belief)
        self.philosophical_beliefs[belief_id] = belief

        # Update indices
        self._update_indices(belief_id, belief, semantic_analysis)

        # Store in base NARS memory
        self.base_memory.add_belief(
            term=narsese_term,
            truth=truth_value,
            occurrence_time=belief.temporal_scope
        )

        return belief

    def _to_philosophically_grounded_narsese(
        self,
        statement: str,
        semantic_analysis: SemanticAnalysis
    ) -> str:
        """
        Convert natural language to NARS format with philosophical grounding.

        This method creates sophisticated NARS representations that preserve
        philosophical nuance and conceptual relationships.
        """
        primary_concepts = semantic_analysis.primary_concepts

        if not primary_concepts:
            # Fallback to simple property assertion
            return f"<{statement} --> philosophical_claim>"

        # Identify main subject and predicate
        if len(primary_concepts) == 1:
            # Single concept - create property assertion
            concept = primary_concepts[0]
            return f"<{concept.term} --> [{concept.domain}_concept]>"

        elif len(primary_concepts) == 2:
            # Two concepts - look for relations
            concept1, concept2 = primary_concepts[0], primary_concepts[1]

            # Check semantic relations
            for rel_type, relations in semantic_analysis.semantic_relations.items():
                for (c1, c2, strength) in relations:
                    if c1 == concept1.term and c2 == concept2.term:
                        return self._create_relation_term(
                            concept1.term, concept2.term, rel_type, strength
                        )

            # Default to inheritance
            return f"<{concept1.term} --> {concept2.term}>"

        else:
            # Multiple concepts - create compound term
            return self._create_compound_philosophical_term(
                primary_concepts,
                semantic_analysis
            )

    def _create_relation_term(
        self,
        concept1: str,
        concept2: str,
        relation_type: str,
        strength: float
    ) -> str:
        """Create NARS term for philosophical relations."""
        relation_mapping = {
            "implication": "==>",
            "similarity": "<->",
            "inheritance": "-->",
            "causation": "=/>",
            "property": "-->",
            "instance": "{--",
            "part": "--[",
            "temporal": "=/>"
        }

        connector = relation_mapping.get(relation_type, "-->")

        # Add strength modifier for weak relations
        if strength < 0.5:
            return f"<{concept1} {connector} [{concept2}]>"  # Intensional
        else:
            return f"<{concept1} {connector} {concept2}>"

    def _create_compound_philosophical_term(
        self,
        concepts: list[PhilosophicalConcept],
        analysis: SemanticAnalysis
    ) -> str:
        """Create compound NARS term for complex philosophical statements."""
        # Group by philosophical category
        category_groups = {}
        for concept in concepts:
            if concept.domain not in category_groups:
                category_groups[concept.domain] = []
            category_groups[concept.domain].append(concept.term)

        if len(category_groups) == 1:
            # Single category - create conjunction
            terms = list(category_groups.values())[0]
            if len(terms) == 2:
                return f"<({terms[0]} & {terms[1]}) --> philosophical_position>"
            else:
                # Multiple terms - create set
                term_set = " * ".join(terms)
                return f"<{{{term_set}}} --> philosophical_configuration>"
        else:
            # Multiple categories - create complex relation
            components = []
            for category, terms in category_groups.items():
                if len(terms) == 1:
                    components.append(f"[{terms[0]}:{category}]")
                else:
                    components.append(f"[({' & '.join(terms)}):{category}]")

            return f"<{' * '.join(components)} --> philosophical_synthesis>"

    def _calculate_philosophical_truth(
        self,
        semantic_analysis: SemanticAnalysis,
        context: dict[str, Any]
    ) -> TruthValue:
        """
        Calculate truth value incorporating philosophical uncertainty.

        This method considers epistemic uncertainty, conceptual clarity,
        and contextual factors in determining truth values.
        """
        # Base frequency on conceptual coherence
        base_frequency = 0.5  # Start neutral

        # Adjust based on concept confidence
        if semantic_analysis.primary_concepts:
            avg_confidence = np.mean([
                c.confidence for c in semantic_analysis.primary_concepts
            ])
            base_frequency = 0.5 + (avg_confidence - 0.5) * 0.5

        # Incorporate epistemic uncertainty
        epistemic_factor = 1.0 - semantic_analysis.epistemic_uncertainty
        frequency = base_frequency * epistemic_factor

        # Calculate confidence based on evidence and context
        evidence_weight = context.get("evidence_strength", 0.5)
        perspective_diversity = len(context.get("perspectives", [])) / 10.0  # Normalize

        confidence = min(0.9, evidence_weight * 0.6 + perspective_diversity * 0.4)

        # Apply philosophical modesty - avoid extreme confidence
        confidence = confidence * 0.9 + 0.05  # Keep between 0.05 and 0.95

        return TruthValue(
            frequency=float(frequency),
            confidence=float(confidence)
        )

    def _determine_temporal_scope(
        self,
        statement: str,
        semantic_analysis: SemanticAnalysis
    ) -> str:
        """Determine temporal scope of philosophical belief."""
        # Check for temporal indicators
        temporal_indicators = {
            "eternal": ["always", "necessarily", "universally", "timelessly"],
            "temporal": ["now", "currently", "today", "recently"],
            "historical": ["was", "used to", "historically", "once"]
        }

        statement_lower = statement.lower()

        for scope, indicators in temporal_indicators.items():
            if any(indicator in statement_lower for indicator in indicators):
                return scope

        # Check concept types
        concept_domains = [c.domain for c in semantic_analysis.primary_concepts]
        if "logical" in concept_domains or "metaphysical" in concept_domains:
            return "eternal"
        elif "phenomenological" in concept_domains:
            return "temporal"

        return "eternal"  # Default for philosophical claims

    def _generate_semantic_embedding(self, statement: str) -> np.ndarray:
        """Generate semantic embedding for similarity calculations."""
        # Simplified embedding generation
        # In reality, would use sentence transformers or similar
        words = statement.lower().split()
        embedding = np.zeros(384)  # Standard embedding size

        for i, word in enumerate(words[:100]):  # Limit to 100 words
            # Simple hash-based embedding
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0 / (i + 1)  # Position-weighted

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _generate_belief_id(self, belief: PhilosophicalBelief) -> str:
        """Generate unique ID for philosophical belief."""
        # Use hash of key components
        components = f"{belief.narsese_term}_{belief.perspective_source}_{belief.temporal_scope}"
        return f"belief_{abs(hash(components)) % 1000000}"

    def _update_indices(
        self,
        belief_id: str,
        belief: PhilosophicalBelief,
        semantic_analysis: SemanticAnalysis
    ):
        """Update various indices for efficient retrieval."""
        # Update perspective index
        if belief.perspective_source not in self.perspective_indices:
            self.perspective_indices[belief.perspective_source] = set()
        self.perspective_indices[belief.perspective_source].add(belief_id)

        # Update temporal index for concepts
        for concept in semantic_analysis.primary_concepts:
            if concept.term not in self.temporal_index:
                self.temporal_index[concept.term] = []
            self.temporal_index[concept.term].append(
                (datetime.now(), belief_id)
            )

        # Update belief network based on relations
        self._update_belief_network(belief_id, semantic_analysis)

    def _update_belief_network(
        self,
        belief_id: str,
        semantic_analysis: SemanticAnalysis
    ):
        """Update belief network with semantic relations."""
        if belief_id not in self.belief_networks:
            self.belief_networks[belief_id] = set()

        # Find related beliefs based on shared concepts
        for concept in semantic_analysis.primary_concepts:
            # Search existing beliefs with this concept
            for other_id, other_belief in self.philosophical_beliefs.items():
                if other_id != belief_id and concept.term in other_belief.statement:
                    self.belief_networks[belief_id].add(other_id)
                    if other_id not in self.belief_networks:
                        self.belief_networks[other_id] = set()
                    self.belief_networks[other_id].add(belief_id)

    async def revise_belief(
        self,
        belief_id: str,
        new_evidence: dict[str, Any],
        revision_type: str = "evidence_based"
    ) -> BeliefRevisionEvent | None:
        """
        Revise philosophical belief based on new evidence or reasoning.

        This implements sophisticated belief revision that preserves
        philosophical insights while updating based on new information.
        """
        if belief_id not in self.philosophical_beliefs:
            logger.warning(f"Belief {belief_id} not found for revision")
            return None

        original_belief = self.philosophical_beliefs[belief_id]

        # Calculate revision based on type
        if revision_type == "evidence_based":
            revised_belief = await self._evidence_based_revision(
                original_belief, new_evidence
            )
        elif revision_type == "coherence_based":
            revised_belief = await self._coherence_based_revision(
                original_belief, new_evidence
            )
        elif revision_type == "dialectical":
            revised_belief = await self._dialectical_revision(
                original_belief, new_evidence
            )
        else:
            logger.warning(f"Unknown revision type: {revision_type}")
            return None

        # Create revision event
        revision_event = BeliefRevisionEvent(
            original_belief=original_belief,
            revised_belief=revised_belief,
            revision_reason=revision_type,
            evidence_delta=new_evidence,
            confidence_change=revised_belief.truth.confidence - original_belief.truth.confidence
        )

        # Update memory
        self.philosophical_beliefs[belief_id] = revised_belief
        self.revision_history.append(revision_event)

        # Update base NARS memory
        self.base_memory.add_belief(
            term=revised_belief.narsese_term,
            truth=revised_belief.truth,
            occurrence_time=revised_belief.temporal_scope
        )

        return revision_event

    async def _evidence_based_revision(
        self,
        belief: PhilosophicalBelief,
        new_evidence: dict[str, Any]
    ) -> PhilosophicalBelief:
        """Revise belief based on new evidence."""
        # Extract evidence type and strength
        evidence_type = new_evidence.get("type", "general")
        evidence_strength = new_evidence.get("strength", 0.5)
        evidence_content = new_evidence.get("content", "")

        # Deep copy belief
        revised = PhilosophicalBelief(
            statement=belief.statement,
            narsese_term=belief.narsese_term,
            truth=TruthValue(belief.truth.frequency, belief.truth.confidence),
            philosophical_context=belief.philosophical_context.copy(),
            supporting_evidence=belief.supporting_evidence.copy(),
            challenging_evidence=belief.challenging_evidence.copy(),
            perspective_source=belief.perspective_source,
            temporal_scope=belief.temporal_scope,
            revision_count=belief.revision_count + 1,
            semantic_embedding=belief.semantic_embedding
        )

        # Update evidence lists
        if evidence_type == "supporting":
            revised.supporting_evidence.append(evidence_content)
            # Strengthen truth
            revised.truth = Truth.revision(
                belief.truth,
                TruthValue(0.9, evidence_strength)
            )
        elif evidence_type == "challenging":
            revised.challenging_evidence.append(evidence_content)
            # Weaken truth
            revised.truth = Truth.revision(
                belief.truth,
                TruthValue(0.1, evidence_strength)
            )

        return revised

    async def _coherence_based_revision(
        self,
        belief: PhilosophicalBelief,
        context: dict[str, Any]
    ) -> PhilosophicalBelief:
        """Revise belief based on coherence with belief network."""
        # Get related beliefs
        belief_id = self._find_belief_id(belief)
        related_ids = self.belief_networks.get(belief_id, set()) if belief_id else set()

        if not related_ids:
            return belief  # No revision without context

        # Calculate coherence pressure
        coherence_sum = 0.0
        coherence_weights = 0.0

        for related_id in related_ids:
            related_belief = self.philosophical_beliefs[related_id]

            # Calculate similarity
            similarity = self._calculate_belief_similarity(belief, related_belief)

            # Weight by similarity and confidence
            weight = similarity * related_belief.truth.confidence
            coherence_sum += related_belief.truth.frequency * weight
            coherence_weights += weight

        if coherence_weights > 0:
            # Revise toward coherent position
            target_frequency = coherence_sum / coherence_weights

            # Create revised belief
            revised = PhilosophicalBelief(
                statement=belief.statement,
                narsese_term=belief.narsese_term,
                truth=Truth.revision(
                    belief.truth,
                    TruthValue(target_frequency, 0.7)  # Moderate confidence in coherence
                ),
                philosophical_context=belief.philosophical_context.copy(),
                supporting_evidence=belief.supporting_evidence.copy(),
                challenging_evidence=belief.challenging_evidence.copy(),
                perspective_source=belief.perspective_source,
                temporal_scope=belief.temporal_scope,
                revision_count=belief.revision_count + 1,
                semantic_embedding=belief.semantic_embedding
            )

            return revised

        return belief

    async def _dialectical_revision(
        self,
        belief: PhilosophicalBelief,
        antithesis: dict[str, Any]
    ) -> PhilosophicalBelief:
        """Revise belief through dialectical synthesis."""
        # Extract antithesis components
        antithesis_statement = antithesis.get("statement", "")
        antithesis_truth = antithesis.get("truth", TruthValue(0.5, 0.5))

        # Create synthesis statement
        synthesis_statement = f"Synthesis: {belief.statement} considering {antithesis_statement}"

        # Dialectical truth combination
        synthesis_truth = Truth.revision(
            belief.truth,
            Truth.negation(antithesis_truth)
        )

        # Create synthesized belief
        revised = PhilosophicalBelief(
            statement=synthesis_statement,
            narsese_term=f"<({belief.narsese_term} | dialectical_synthesis) --> resolved>",
            truth=synthesis_truth,
            philosophical_context={
                **belief.philosophical_context,
                "dialectical_synthesis": True,
                "thesis": belief.statement,
                "antithesis": antithesis_statement
            },
            supporting_evidence=belief.supporting_evidence + ["dialectical_resolution"],
            challenging_evidence=[],
            perspective_source="dialectical",
            temporal_scope=belief.temporal_scope,
            revision_count=belief.revision_count + 1,
            semantic_embedding=self._generate_semantic_embedding(synthesis_statement)
        )

        return revised

    def _find_belief_id(self, belief: PhilosophicalBelief) -> str | None:
        """Find belief ID for a given belief."""
        for belief_id, stored_belief in self.philosophical_beliefs.items():
            if stored_belief.narsese_term == belief.narsese_term:
                return belief_id
        return None

    def _calculate_belief_similarity(
        self,
        belief1: PhilosophicalBelief,
        belief2: PhilosophicalBelief
    ) -> float:
        """Calculate semantic similarity between beliefs."""
        if (belief1.semantic_embedding is not None and
            belief2.semantic_embedding is not None):
            # Cosine similarity of embeddings
            similarity = np.dot(
                belief1.semantic_embedding,
                belief2.semantic_embedding
            )
            return float(similarity)

        # Fallback to simple text similarity
        words1 = set(belief1.statement.lower().split())
        words2 = set(belief2.statement.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def get_coherence_landscape(self) -> dict[str, Any]:
        """
        Generate coherence landscape from belief network.

        This creates a topological view of the belief space showing
        clusters, tensions, and conceptual relationships.
        """
        # Identify belief clusters
        clusters = self._identify_belief_clusters()

        # Calculate cluster coherence
        cluster_coherence = {}
        for cluster_id, belief_ids in clusters.items():
            coherence = self._calculate_cluster_coherence(belief_ids)
            cluster_coherence[cluster_id] = coherence

        # Identify conceptual tensions
        tensions = self._identify_conceptual_tensions()

        # Find bridge concepts
        bridges = self._identify_bridge_concepts()

        return {
            "clusters": clusters,
            "cluster_coherence": cluster_coherence,
            "tensions": tensions,
            "bridges": bridges,
            "total_beliefs": len(self.philosophical_beliefs),
            "revision_count": len(self.revision_history),
            "perspective_distribution": {
                p: len(beliefs)
                for p, beliefs in self.perspective_indices.items()
            }
        }

    def _identify_belief_clusters(self) -> dict[str, set[str]]:
        """Identify clusters of related beliefs."""
        visited = set()
        clusters = {}
        cluster_id = 0

        for belief_id in self.philosophical_beliefs:
            if belief_id not in visited:
                # DFS to find connected component
                cluster = set()
                stack = [belief_id]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)

                        # Add connected beliefs
                        connected = self.belief_networks.get(current, set())
                        stack.extend(connected - visited)

                if len(cluster) > 1:  # Only keep multi-belief clusters
                    clusters[f"cluster_{cluster_id}"] = cluster
                    cluster_id += 1

        return clusters

    def _calculate_cluster_coherence(self, belief_ids: set[str]) -> float:
        """Calculate internal coherence of a belief cluster."""
        if len(belief_ids) < 2:
            return 1.0

        coherence_scores = []
        belief_list = list(belief_ids)

        for i in range(len(belief_list)):
            for j in range(i + 1, len(belief_list)):
                belief1 = self.philosophical_beliefs[belief_list[i]]
                belief2 = self.philosophical_beliefs[belief_list[j]]

                # Check truth value alignment
                freq_diff = abs(belief1.truth.frequency - belief2.truth.frequency)
                coherence = 1.0 - freq_diff

                # Weight by confidence
                weight = (belief1.truth.confidence + belief2.truth.confidence) / 2
                coherence_scores.append(coherence * weight)

        return float(np.mean(coherence_scores)) if coherence_scores else 0.5

    def _identify_conceptual_tensions(self) -> list[dict[str, Any]]:
        """Identify tensions between beliefs."""
        tensions = []

        for belief_id, connected_ids in self.belief_networks.items():
            belief = self.philosophical_beliefs[belief_id]

            for connected_id in connected_ids:
                connected_belief = self.philosophical_beliefs[connected_id]

                # Check for opposing truth values
                freq_diff = abs(belief.truth.frequency - connected_belief.truth.frequency)

                if freq_diff > 0.6:  # Significant opposition
                    tensions.append({
                        "belief1": belief.statement,
                        "belief2": connected_belief.statement,
                        "tension_strength": freq_diff,
                        "perspectives": [
                            belief.perspective_source,
                            connected_belief.perspective_source
                        ]
                    })

        return tensions

    def _identify_bridge_concepts(self) -> list[dict[str, Any]]:
        """Identify concepts that bridge different clusters."""
        # Track concept occurrence across clusters
        clusters = self._identify_belief_clusters()
        concept_cluster_map = {}

        for cluster_id, belief_ids in clusters.items():
            cluster_concepts = set()

            for belief_id in belief_ids:
                belief = self.philosophical_beliefs[belief_id]
                # Extract concepts from statement (simplified)
                words = belief.statement.lower().split()
                concepts = [w for w in words if len(w) > 4]  # Simple heuristic
                cluster_concepts.update(concepts)

            for concept in cluster_concepts:
                if concept not in concept_cluster_map:
                    concept_cluster_map[concept] = set()
                concept_cluster_map[concept].add(cluster_id)

        # Find concepts in multiple clusters
        bridges = []
        for concept, cluster_set in concept_cluster_map.items():
            if len(cluster_set) > 1:
                bridges.append({
                    "concept": concept,
                    "clusters_connected": list(cluster_set),
                    "bridge_strength": len(cluster_set) / len(clusters)
                })

        return sorted(bridges, key=lambda x: x["bridge_strength"], reverse=True)[:10]


class PhilosophicalNARSReasoning:
    """
    Enhanced NARS reasoning system for philosophical analysis.

    Integrates deep semantic understanding with NARS inference mechanisms
    for sophisticated philosophical reasoning.
    """

    def __init__(
        self,
        nars_manager: NARSManager,
        enhanced_memory: EnhancedNARSMemory,
        llm_processor: 'LLMSemanticProcessor'
    ):
        self.nars = nars_manager
        self.memory = enhanced_memory
        self.llm_processor = llm_processor
        self.reasoning_cache = {}

        # Enhanced reasoning patterns
        self.reasoning_patterns = {
            "transcendental": self._transcendental_reasoning,
            "phenomenological": self._phenomenological_reasoning,
            "pragmatic": self._pragmatic_reasoning,
            "critical": self._critical_reasoning,
            "systematic": self._systematic_reasoning
        }

        logger.info("Philosophical NARS Reasoning initialized")

    async def philosophical_inference(
        self,
        query: str,
        context: dict[str, Any],
        reasoning_type: str = "mixed"
    ) -> dict[str, Any]:
        """
        Perform philosophical inference using NARS with semantic understanding.
        """
        # Convert dict context to PhilosophicalContext if needed
        from .llm_semantic_processor import PhilosophicalContext

        if isinstance(context, dict):
            phil_context = PhilosophicalContext(
                domain=context.get("domain", "general"),
                inquiry_type=context.get("inquiry_type", "inference"),
                depth_requirements=context.get("depth", 3)
            )
        else:
            phil_context = context

        # Analyze query semantically
        query_analysis = await self.llm_processor.analyze_statement(query, phil_context)

        # Convert to NARS query format
        narsese_query = self.memory._to_philosophically_grounded_narsese(
            query, query_analysis
        )

        # Retrieve relevant beliefs from memory
        relevant_beliefs = await self._retrieve_relevant_beliefs(
            query_analysis, context
        )

        # Prime NARS with relevant beliefs
        await self._prime_nars_with_beliefs(relevant_beliefs)

        # Execute NARS query
        nars_result = await self.nars.query(narsese_query + "?", timeout=5.0)

        # Apply philosophical reasoning patterns
        if reasoning_type == "mixed":
            # Apply multiple patterns and synthesize
            pattern_results = {}
            for pattern_name, pattern_func in self.reasoning_patterns.items():
                try:
                    result = await pattern_func(
                        query, query_analysis, relevant_beliefs, nars_result
                    )
                    if result:
                        pattern_results[pattern_name] = result
                except Exception as e:
                    logger.warning(f"Pattern {pattern_name} failed: {e}")

            # Synthesize results
            inference_result = await self._synthesize_pattern_results(
                pattern_results, query_analysis
            )
        else:
            # Apply specific reasoning pattern
            pattern_func = self.reasoning_patterns.get(
                reasoning_type,
                self._systematic_reasoning
            )
            inference_result = await pattern_func(
                query, query_analysis, relevant_beliefs, nars_result
            )

        return {
            "query": query,
            "narsese_query": narsese_query,
            "inference_result": inference_result,
            "relevant_beliefs": [
                {
                    "statement": b.statement,
                    "truth": {"frequency": b.truth.frequency, "confidence": b.truth.confidence},
                    "perspective": b.perspective_source
                }
                for b in relevant_beliefs[:5]  # Top 5
            ],
            "reasoning_type": reasoning_type,
            "coherence_impact": await self._assess_coherence_impact(inference_result)
        }

    async def _retrieve_relevant_beliefs(
        self,
        query_analysis: SemanticAnalysis,
        context: dict[str, Any]
    ) -> list[PhilosophicalBelief]:
        """Retrieve beliefs relevant to the query."""
        relevant_beliefs = []

        # Get concepts from query
        query_concepts = {c.term for c in query_analysis.primary_concepts}

        # Search through philosophical beliefs
        for _belief_id, belief in self.memory.philosophical_beliefs.items():
            # Calculate relevance score
            relevance = 0.0

            # Concept overlap
            belief_words = set(belief.statement.lower().split())
            concept_overlap = len(query_concepts & belief_words)
            if concept_overlap > 0:
                relevance += concept_overlap * 0.3

            # Semantic similarity (if embeddings available)
            if belief.semantic_embedding is not None and query_concepts:
                # Simplified - would use proper query embedding
                relevance += 0.3

            # Context match
            if belief.philosophical_context.get("domain") == context.get("domain"):
                relevance += 0.2

            # Perspective relevance
            if context.get("perspectives") and belief.perspective_source in context["perspectives"]:
                relevance += 0.2

            if relevance > 0.3:  # Threshold
                relevant_beliefs.append((relevance, belief))

        # Sort by relevance and return top beliefs
        relevant_beliefs.sort(key=lambda x: x[0], reverse=True)
        return [belief for _, belief in relevant_beliefs[:20]]  # Top 20

    async def _prime_nars_with_beliefs(self, beliefs: list[PhilosophicalBelief]):
        """Prime NARS with relevant beliefs for inference."""
        for belief in beliefs[:10]:  # Limit to prevent overload
            try:
                # Input belief to NARS
                input_str = f"{belief.narsese_term}. " \
                          f"{{{belief.truth.frequency:.2f} {belief.truth.confidence:.2f}}}"
                await self.nars.query(input_str, timeout=0.5)
            except Exception as e:
                logger.debug(f"Failed to prime belief: {e}")

    async def _transcendental_reasoning(
        self,
        query: str,
        query_analysis: SemanticAnalysis,
        relevant_beliefs: list[PhilosophicalBelief],
        nars_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply transcendental reasoning pattern (conditions of possibility)."""
        # Identify what must be true for the query to make sense
        necessary_conditions = []

        # Analyze presuppositions in query
        for concept in query_analysis.primary_concepts:
            # What must exist/be true for this concept to be meaningful?
            if concept.domain == "epistemological":
                necessary_conditions.append(
                    f"Cognitive subjects capable of {concept.term} must exist"
                )
            elif concept.domain == "ethical":
                necessary_conditions.append(
                    f"Agents capable of moral agency presupposed by {concept.term}"
                )

        # Look for foundational beliefs
        foundational = [
            b for b in relevant_beliefs
            if b.temporal_scope == "eternal" and b.truth.confidence > 0.7
        ]

        return {
            "reasoning_type": "transcendental",
            "necessary_conditions": necessary_conditions,
            "foundational_beliefs": [b.statement for b in foundational[:3]],
            "transcendental_argument": (
                f"For '{query}' to be meaningful, the following conditions must hold: "
                f"{'; '.join(necessary_conditions[:2])}"
            ),
            "confidence": 0.7 if necessary_conditions else 0.3
        }

    async def _phenomenological_reasoning(
        self,
        query: str,
        query_analysis: SemanticAnalysis,
        relevant_beliefs: list[PhilosophicalBelief],
        nars_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply phenomenological reasoning (experiential description)."""
        # Focus on experiential and first-person aspects
        experiential_beliefs = [
            b for b in relevant_beliefs
            if any(term in b.statement.lower()
                   for term in ["experience", "consciousness", "feeling", "perception"])
        ]

        # Extract phenomenological insights
        insights = []
        if query_analysis.primary_concepts:
            for concept in query_analysis.primary_concepts:
                if concept.domain in ["phenomenological", "aesthetic"]:
                    insights.append(
                        f"The lived experience of {concept.term} involves "
                        f"{concept.contextual_variations.get('phenomenology', 'direct givenness')}"
                    )

        return {
            "reasoning_type": "phenomenological",
            "experiential_descriptions": [b.statement for b in experiential_beliefs[:3]],
            "phenomenological_insights": insights,
            "intentional_structure": f"Consciousness directed toward {query}",
            "confidence": 0.6 if experiential_beliefs else 0.4
        }

    async def _pragmatic_reasoning(
        self,
        query: str,
        query_analysis: SemanticAnalysis,
        relevant_beliefs: list[PhilosophicalBelief],
        nars_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply pragmatic reasoning (practical consequences)."""
        # Focus on practical implications and consequences
        practical_implications = query_analysis.pragmatic_implications

        # Find action-oriented beliefs
        action_beliefs = [
            b for b in relevant_beliefs
            if any(term in b.statement.lower()
                   for term in ["action", "practice", "consequence", "result", "effect"])
        ]

        # Generate pragmatic evaluation
        if nars_result.get("answers"):
            # Use NARS results for practical assessment
            best_answer = nars_result["answers"][0]
            practical_value = best_answer.get("truth", {}).get("expectation", 0.5)
        else:
            practical_value = 0.5

        return {
            "reasoning_type": "pragmatic",
            "practical_implications": practical_implications,
            "action_recommendations": [
                f"Based on {b.statement}, consider {b.philosophical_context.get('action', 'reflection')}"
                for b in action_beliefs[:2]
            ],
            "pragmatic_value": practical_value,
            "workability_assessment": f"The practical workability of '{query}' is "
                                    f"{'high' if practical_value > 0.7 else 'moderate' if practical_value > 0.4 else 'low'}",
            "confidence": practical_value
        }

    async def _critical_reasoning(
        self,
        query: str,
        query_analysis: SemanticAnalysis,
        relevant_beliefs: list[PhilosophicalBelief],
        nars_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply critical reasoning (power/ideology critique)."""
        # Identify potential ideological assumptions
        assumptions = []

        # Check for value-laden terms
        value_terms = ["good", "right", "just", "fair", "natural", "normal", "rational"]
        for term in value_terms:
            if term in query.lower():
                assumptions.append(
                    f"The query assumes '{term}' has a fixed, uncontested meaning"
                )

        # Look for challenging beliefs
        critical_beliefs = [
            b for b in relevant_beliefs
            if b.perspective_source == "critical" or
            any(term in b.statement.lower()
                for term in ["power", "ideology", "critique", "oppression"])
        ]

        return {
            "reasoning_type": "critical",
            "ideological_assumptions": assumptions,
            "critical_perspectives": [b.statement for b in critical_beliefs[:3]],
            "power_analysis": f"The concept '{query}' may serve interests of "
                            f"{query_analysis.philosophical_categorization.get('ethical', 'established order')}",
            "alternative_framings": [
                f"Consider '{query}' from marginalized perspectives",
                "Question who benefits from this conceptualization"
            ],
            "confidence": 0.6
        }

    async def _systematic_reasoning(
        self,
        query: str,
        query_analysis: SemanticAnalysis,
        relevant_beliefs: list[PhilosophicalBelief],
        nars_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply systematic reasoning (comprehensive integration)."""
        # Default comprehensive reasoning pattern

        # Process NARS results
        if nars_result.get("answers"):
            nars_conclusions = []
            for answer in nars_result["answers"][:3]:
                truth = answer.get("truth", {})
                nars_conclusions.append({
                    "conclusion": answer.get("term", ""),
                    "confidence": truth.get("confidence", 0),
                    "frequency": truth.get("frequency", 0.5)
                })
        else:
            nars_conclusions = [{"conclusion": "No direct NARS inference available", "confidence": 0}]

        # Systematic integration
        integrated_view = {
            "conceptual_analysis": {
                "primary_concepts": [c.term for c in query_analysis.primary_concepts],
                "domains": list(query_analysis.philosophical_categorization.keys()),
                "uncertainty": query_analysis.epistemic_uncertainty
            },
            "belief_integration": {
                "supporting_beliefs": len([b for b in relevant_beliefs if b.truth.frequency > 0.6]),
                "challenging_beliefs": len([b for b in relevant_beliefs if b.truth.frequency < 0.4]),
                "total_relevant": len(relevant_beliefs)
            },
            "nars_inference": nars_conclusions[0] if nars_conclusions else None
        }

        # Generate systematic conclusion
        conclusion = self._generate_systematic_conclusion(
            query, integrated_view, query_analysis
        )

        return {
            "reasoning_type": "systematic",
            "integrated_analysis": integrated_view,
            "systematic_conclusion": conclusion,
            "confidence": self._calculate_systematic_confidence(integrated_view),
            "further_investigation": query_analysis.revision_triggers[:3]
        }

    def _generate_systematic_conclusion(
        self,
        query: str,
        integrated_view: dict[str, Any],
        query_analysis: SemanticAnalysis
    ) -> str:
        """Generate systematic conclusion from integrated analysis."""
        # Extract key information
        concepts = integrated_view["conceptual_analysis"]["primary_concepts"]
        uncertainty = integrated_view["conceptual_analysis"]["uncertainty"]
        belief_stats = integrated_view["belief_integration"]

        # Build conclusion
        if belief_stats["supporting_beliefs"] > belief_stats["challenging_beliefs"]:
            stance = "provisionally supported"
        elif belief_stats["challenging_beliefs"] > belief_stats["supporting_beliefs"]:
            stance = "faces significant challenges"
        else:
            stance = "remains contested"

        conclusion = f"The philosophical position '{query}' {stance} "

        if concepts:
            conclusion += f"when understood through the concepts of {', '.join(concepts[:2])}. "

        if uncertainty > 0.7:
            conclusion += "High epistemic uncertainty suggests further investigation needed. "
        elif uncertainty < 0.3:
            conclusion += "Low uncertainty indicates relatively stable understanding. "

        if integrated_view.get("nars_inference") and integrated_view["nars_inference"]["confidence"] > 0.5:
            conclusion += f"NARS inference suggests: {integrated_view['nars_inference']['conclusion']}."

        return conclusion

    def _calculate_systematic_confidence(self, integrated_view: dict[str, Any]) -> float:
        """Calculate confidence for systematic reasoning."""
        factors = []

        # Belief support ratio
        belief_stats = integrated_view["belief_integration"]
        total_beliefs = belief_stats["total_relevant"]
        if total_beliefs > 0:
            support_ratio = belief_stats["supporting_beliefs"] / total_beliefs
            factors.append(support_ratio)

        # NARS confidence
        if integrated_view.get("nars_inference"):
            factors.append(integrated_view["nars_inference"]["confidence"])

        # Inverse uncertainty
        uncertainty = integrated_view["conceptual_analysis"]["uncertainty"]
        factors.append(1.0 - uncertainty)

        return float(np.mean(factors)) if factors else 0.5

    async def _synthesize_pattern_results(
        self,
        pattern_results: dict[str, dict[str, Any]],
        query_analysis: SemanticAnalysis
    ) -> dict[str, Any]:
        """Synthesize results from multiple reasoning patterns."""
        if not pattern_results:
            return {"error": "No reasoning patterns succeeded"}

        # Extract key insights from each pattern
        synthesis = {
            "multi_pattern_analysis": True,
            "patterns_applied": list(pattern_results.keys()),
            "convergent_insights": [],
            "divergent_insights": [],
            "integrated_confidence": 0.0
        }

        # Collect all insights
        all_insights = []
        confidence_values = []

        for pattern, result in pattern_results.items():
            confidence_values.append(result.get("confidence", 0.5))

            # Extract pattern-specific insights
            if pattern == "transcendental":
                all_insights.extend(result.get("necessary_conditions", []))
            elif pattern == "phenomenological":
                all_insights.extend(result.get("phenomenological_insights", []))
            elif pattern == "pragmatic":
                all_insights.extend(result.get("practical_implications", []))
            elif pattern == "critical":
                all_insights.extend(result.get("alternative_framings", []))
            elif pattern == "systematic" and result.get("systematic_conclusion"):
                    all_insights.append(result["systematic_conclusion"])

        # Identify convergent insights (appear in multiple patterns)
        insight_counts = {}
        for insight in all_insights:
            # Simple similarity check
            key = insight[:30]  # First 30 chars as key
            insight_counts[key] = insight_counts.get(key, 0) + 1

        synthesis["convergent_insights"] = [
            insight for insight in all_insights
            if insight_counts.get(insight[:30], 0) > 1
        ][:3]

        synthesis["divergent_insights"] = [
            insight for insight in all_insights
            if insight_counts.get(insight[:30], 0) == 1
        ][:3]

        # Calculate integrated confidence
        synthesis["integrated_confidence"] = float(np.mean(confidence_values))

        # Generate meta-synthesis
        synthesis["meta_synthesis"] = (
            f"Multi-pattern analysis reveals "
            f"{len(synthesis['convergent_insights'])} convergent and "
            f"{len(synthesis['divergent_insights'])} divergent insights. "
            f"Overall epistemic confidence: {synthesis['integrated_confidence']:.2f}"
        )

        return synthesis

    async def _assess_coherence_impact(
        self,
        inference_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess how the inference impacts overall coherence landscape."""
        # Assess potential impact
        impact = {
            "strengthens_coherence": False,
            "creates_tension": False,
            "bridges_clusters": False,
            "impact_description": ""
        }

        # Check if inference strengthens existing clusters
        if inference_result.get("confidence", 0) > 0.7:
            impact["strengthens_coherence"] = True
            impact["impact_description"] = "High-confidence inference reinforces existing beliefs"

        # Check for tensions
        if "divergent_insights" in inference_result and inference_result["divergent_insights"]:
            impact["creates_tension"] = True
            impact["impact_description"] += ". Introduces conceptual tensions requiring resolution"

        # Check for bridging potential
        if inference_result.get("patterns_applied") and len(inference_result["patterns_applied"]) > 3:
            impact["bridges_clusters"] = True
            impact["impact_description"] += ". Multi-pattern synthesis may bridge belief clusters"

        return impact
