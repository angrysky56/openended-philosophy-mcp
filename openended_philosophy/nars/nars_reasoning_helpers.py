"""
NARS Reasoning Helper Methods - Supporting Functions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Supporting helper methods for NARS philosophical reasoning system.
Provides specialized functions for philosophical analysis, pattern recognition,
and dialectical synthesis operations.
"""

import logging
from typing import Any

import numpy as np

from .truth_functions import Truth
from .types import MemoryItem, TruthValue

logger = logging.getLogger(__name__)


class NARSReasoningHelpers:
    """Collection of helper methods for NARS philosophical reasoning."""

    @staticmethod
    def calculate_finding_similarity(finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Calculate semantic similarity between two findings."""
        if 'claim' not in finding1 or 'claim' not in finding2:
            return 0.0

        # Simple lexical similarity for now - in production use embeddings
        claim1_words = set(finding1['claim'].lower().split())
        claim2_words = set(finding2['claim'].lower().split())

        if not claim1_words or not claim2_words:
            return 0.0

        intersection = len(claim1_words.intersection(claim2_words))
        union = len(claim1_words.union(claim2_words))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def synthesize_convergent_truths(findings: list[dict[str, Any]], perspectives: list[str]) -> TruthValue:
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

    @staticmethod
    def calculate_tension_score(finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
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

    @staticmethod
    def assess_dialectical_potential(finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Assess potential for dialectical synthesis between contradictory findings."""
        tension = NARSReasoningHelpers.calculate_tension_score(finding1, finding2)

        if tension < 0.3:
            return 0.0  # No significant tension

        # Higher dialectical potential when both findings have evidence
        conf1 = finding1.get('truth', {}).get('confidence', 0.0)
        conf2 = finding2.get('truth', {}).get('confidence', 0.0)

        evidence_quality = (conf1 + conf2) / 2.0

        return min(1.0, tension * evidence_quality)

    @staticmethod
    def detect_logical_contradiction(finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
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

    @staticmethod
    def calculate_semantic_compatibility(finding1: dict[str, Any], finding2: dict[str, Any]) -> float:
        """Calculate semantic compatibility between findings."""
        return NARSReasoningHelpers.calculate_finding_similarity(finding1, finding2)

    @staticmethod
    def extract_concepts_from_term(term: str) -> list[str]:
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

    @staticmethod
    def extract_relations_from_term(term: str, truth: TruthValue) -> list[dict[str, Any]]:
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

    @staticmethod
    def perform_semantic_clustering(concept_embeddings: dict[str, np.ndarray]) -> dict[str, int]:
        """Perform semantic clustering of concepts using embeddings."""
        if len(concept_embeddings) < 2:
            return dict.fromkeys(concept_embeddings, 0)

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

    @staticmethod
    def calculate_graph_centrality(nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate centrality measures for graph nodes."""
        centrality = {}

        # Calculate degree centrality
        node_degrees = dict.fromkeys(nodes, 0)

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

    @staticmethod
    def calculate_graph_metrics(nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
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
        metrics['average_centrality'] = float(np.mean(centralities)) if centralities else 0.0

        # Average confidence
        confidences = [edge['truth']['confidence'] for edge in edges if 'truth' in edge]
        metrics['average_confidence'] = float(np.mean(confidences)) if confidences else 0.0

        return metrics

    @staticmethod
    def get_cluster_summary(nodes: dict[str, Any]) -> dict[str, list[str]]:
        """Get summary of semantic clusters."""
        clusters = {}

        for node_id, node_data in nodes.items():
            cluster_id = node_data.get('semantic_cluster')
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(node_id)

        return clusters

    @staticmethod
    def extract_related_concept_from_answer(answer_term: str, original_concept: str) -> str | None:
        """Extract the related concept from a NARS answer term."""
        # Simple extraction - remove original concept and operators
        cleaned = answer_term.replace(original_concept, '').replace('<', '').replace('>', '')
        cleaned = cleaned.replace('-->', '').replace('==>', '').replace('=/>', '')
        cleaned = cleaned.strip()

        if cleaned and len(cleaned) > 1:
            return cleaned
        return None

    @staticmethod
    def determine_relation_type(query: str) -> str:
        """Determine relation type from query pattern."""
        if '-->' in query:
            return 'inheritance'
        elif '==>' in query:
            return 'implication'
        elif '=/>' in query:
            return 'predictive'
        else:
            return 'unknown'

    @staticmethod
    def generate_evidence_queries(term: str, phenomenon: str) -> list[str]:
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

    @staticmethod
    def is_relevant_for_expansion(answer_term: str, phenomenon: str) -> bool:
        """Check if an answer term is relevant for evidence expansion."""
        # Simple heuristic - check if term shares concepts with phenomenon
        answer_words = set(answer_term.lower().split())
        phenomenon_words = set(phenomenon.lower().split())

        # At least one word overlap or semantic indicators
        semantic_indicators = ['related', 'similar', 'causes', 'implies', 'connected']

        has_overlap = bool(answer_words.intersection(phenomenon_words))
        has_semantic_indicator = any(indicator in answer_term.lower() for indicator in semantic_indicators)

        return has_overlap or has_semantic_indicator

    @staticmethod
    def calculate_evidence_relevance(item: MemoryItem, phenomenon: str) -> float:
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

    @staticmethod
    def identify_high_confidence_clusters(nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify clusters of high-confidence concepts."""
        clusters = []

        # Group nodes by semantic cluster
        cluster_groups = {}
        for node_id, node_data in nodes.items():
            cluster_id = node_data.get('semantic_cluster', 'default')
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append((node_id, node_data))

        # Analyze each cluster
        for cluster_id, cluster_nodes in cluster_groups.items():
            if len(cluster_nodes) < 2:
                continue

            # Calculate average confidence
            confidences = [node['truth']['confidence'] for _, node in cluster_nodes]
            avg_confidence = float(np.mean(confidences))

            if avg_confidence > 0.7:  # High confidence threshold
                clusters.append({
                    'cluster_id': cluster_id,
                    'concepts': [node_id for node_id, _ in cluster_nodes],
                    'confidence': avg_confidence,
                    'size': len(cluster_nodes)
                })

        return clusters

    @staticmethod
    def identify_central_supported_concepts(nodes: dict[str, Any], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify central concepts with strong evidential support."""
        central_concepts = []

        for node_id, node_data in nodes.items():
            centrality = node_data.get('centrality', 0.0)
            evidence_strength = node_data.get('evidence_strength', 0.0)

            # Combined score for centrality and evidence
            combined_score = 0.6 * centrality + 0.4 * evidence_strength

            if combined_score > 0.7:  # High combined score threshold
                central_concepts.append({
                    'id': node_id,
                    'centrality': centrality,
                    'evidence_strength': evidence_strength,
                    'combined_score': combined_score,
                    'category': node_data.get('category', 'unknown')
                })

        # Sort by combined score
        central_concepts.sort(key=lambda x: x['combined_score'], reverse=True)

        return central_concepts[:5]  # Top 5 central concepts

    @staticmethod
    def analyze_category_coherence(nodes: dict[str, Any], edges: list[dict[str, Any]]) -> dict[str, float]:
        """Analyze coherence within philosophical categories."""
        category_coherence = {}

        # Group nodes by category
        category_groups = {}
        for node_id, node_data in nodes.items():
            category = node_data.get('category', 'uncategorized')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(node_id)

        # Calculate coherence for each category
        for category, category_nodes in category_groups.items():
            if len(category_nodes) < 2:
                category_coherence[category] = 1.0  # Single node = perfect coherence
                continue

            # Find edges within category
            internal_edges = []
            for edge in edges:
                if (edge['source'] in category_nodes and
                    edge['target'] in category_nodes):
                    internal_edges.append(edge)

            # Calculate coherence based on internal connectivity
            max_internal_edges = len(category_nodes) * (len(category_nodes) - 1)
            connectivity = len(internal_edges) / max_internal_edges if max_internal_edges > 0 else 0.0

            # Factor in average edge strength
            if internal_edges:
                avg_edge_strength = float(np.mean([edge['weight'] for edge in internal_edges]))
                coherence = 0.7 * connectivity + 0.3 * avg_edge_strength
            else:
                coherence = 0.0

            category_coherence[category] = coherence

        return category_coherence

    @staticmethod
    def classify_attractor_type(concept: str, metrics: dict[str, Any], edges: list[dict[str, Any]]) -> str:
        """Classify the type of conceptual attractor."""
        centrality = metrics['centrality']
        incoming_strength = metrics['incoming_strength']
        outgoing_strength = metrics['outgoing_strength']
        category = metrics['category']

        # Determine attractor type based on patterns
        if incoming_strength > outgoing_strength * 1.5:
            return "convergence_attractor"
        elif outgoing_strength > incoming_strength * 1.5:
            return "generative_attractor"
        elif centrality > 0.8:
            return "hub_attractor"
        elif category in ['metaphysical', 'epistemological']:
            return "foundational_attractor"
        else:
            return "local_attractor"

    @staticmethod
    def identify_attractor_clusters(attractors: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
        """Identify clusters of mutually reinforcing attractors."""
        clusters = []
        processed = set()

        for attractor in attractors:
            if attractor in processed:
                continue

            # Find connected attractors
            cluster = [attractor]
            processed.add(attractor)

            # Look for edges connecting this attractor to others
            for edge in edges:
                if (edge['source'] == attractor and edge['target'] in attractors and
                        edge['target'] not in processed and edge['weight'] > 0.6):
                    cluster.append(edge['target'])
                    processed.add(edge['target'])
                elif (edge['target'] == attractor and edge['source'] in attractors and
                      edge['source'] not in processed and edge['weight'] > 0.6):
                    cluster.append(edge['source'])
                    processed.add(edge['source'])

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    @staticmethod
    def generate_placeholder_method_implementations() -> dict[str, str]:
        """Generate placeholder implementations for missing methods."""
        # This method provides basic implementations for methods that need to be defined
        # but are complex enough to require domain-specific implementation

        placeholder_methods = {
            "_perform_dialectical_synthesis": "return {'synthesis_type': 'basic', 'synthesis_strength': 0.5}",
            "_attempt_dialectical_resolution": "return None",
            "_calculate_synthesis_coherence": "return 0.5",
            "_determine_epistemological_status": "return 'provisional'",
            "_identify_synthesis_revision_triggers": "return ['insufficient_evidence']",
            "_identify_emergent_structures": "return []",
            "_identify_contradiction_clusters": "return []",
            "_identify_inferential_chains": "return []",
            "_identify_semantic_coherence_regions": "return []",
            "_simulate_perturbation_resistance": "return 0.5",
            "_analyze_attractor_stability": "return 0.5",
            "_analyze_hierarchical_structure": "return {}",
            "_calculate_category_internal_coherence": "return 0.5",
            "_identify_inferential_patterns": "return []",
            "_extract_structural_principles": "return []",
            "_analyze_emergence_levels": "return {}"
        }

        return placeholder_methods
