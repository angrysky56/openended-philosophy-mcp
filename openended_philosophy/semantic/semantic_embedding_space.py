"""
Semantic Embedding Space for Philosophical Concept Representation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Conceptual Framework Deconstruction

This module implements sophisticated semantic embedding capabilities that replace
traditional static similarity measures with dynamic, context-aware semantic spaces:

#### Core Theoretical Foundations:
- **Vector Space Semantics**: Philosophical concepts represented in high-dimensional spaces
- **Context-Dependent Embedding**: Wittgensteinian language game awareness
- **Dynamic Similarity Calculation**: Multi-dimensional relationship assessment
- **Hierarchical Semantic Organization**: Clustered conceptual neighborhoods

#### Epistemological Assumptions:
- Conceptual similarity emerges through relational patterns rather than fixed properties
- Semantic spaces are revisable and context-dependent
- Multiple valid similarity metrics can coexist without contradiction
- Understanding deepens through systematic exploration of semantic neighborhoods

### Methodological Approach

The embedding system employs:
1. **Multi-Modal Embedding Generation**: Text, philosophical context, and relational features
2. **Dynamic Similarity Metrics**: Context-aware distance calculations
3. **Semantic Clustering**: Automatic discovery of conceptual neighborhoods
4. **Temporal Evolution**: Embedding space adaptation over time
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .types import (
    PhilosophicalConcept,
    PhilosophicalContext,
    PhilosophicalDomain,
    SemanticAnalysis,
    SemanticEmbedding,
    SemanticRelation,
)

logger = logging.getLogger(__name__)


@dataclass
class SemanticCluster:
    """Represents a cluster of semantically related concepts."""
    cluster_id: str
    center_concept: str
    member_concepts: list[str]
    coherence_score: float
    philosophical_theme: str | None = None
    temporal_stability: float = 0.8
    cluster_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingMetrics:
    """Quality metrics for semantic embeddings."""
    dimensional_density: float
    semantic_coherence: float
    contextual_sensitivity: float
    temporal_consistency: float
    overall_quality: float


class SemanticEmbeddingSpace:
    """
    Sophisticated semantic embedding space for philosophical concepts.

    This class manages high-dimensional representations of philosophical concepts,
    providing context-aware similarity calculations and dynamic clustering capabilities.
    """

    def __init__(self, embedding_dimension: int = 768):
        """
        Initialize the semantic embedding space.

        Args:
            embedding_dimension: Dimensionality of the embedding vectors
        """
        self.embedding_dimension = embedding_dimension
        self.concept_embeddings: dict[str, np.ndarray] = {}
        self.context_embeddings: dict[str, np.ndarray] = {}
        self.embedding_metadata: dict[str, dict[str, Any]] = {}
        self.semantic_clusters: list[SemanticCluster] = []
        self.similarity_cache: dict[tuple[str, str], float] = {}
        self.evolution_history: list[dict[str, Any]] = []

        # Initialize clustering algorithms
        self.dbscan_clusterer = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        self.kmeans_clusterer = None  # Initialized dynamically based on data

        logger.info(f"Initialized SemanticEmbeddingSpace with {embedding_dimension} dimensions")

    async def generate_philosophical_embedding(
        self,
        statement: str,
        semantic_analysis: SemanticAnalysis,
        philosophical_category: Any,
        context: PhilosophicalContext | None = None
    ) -> np.ndarray:
        """
        Generate comprehensive philosophical embedding for a statement.

        Args:
            statement: The philosophical statement to embed
            semantic_analysis: Semantic analysis results
            philosophical_category: Categorical information
            context: Philosophical context for interpretation

        Returns:
            High-dimensional embedding vector
        """
        try:
            # Base semantic embedding (would typically use a transformer model)
            base_embedding = await self._generate_base_semantic_embedding(statement)

            # Philosophical feature enhancement
            philosophical_features = self._extract_philosophical_features(
                semantic_analysis, philosophical_category
            )

            # Context-dependent adjustments
            if context:
                context_features = self._extract_context_features(context)
                philosophical_features = np.concatenate([philosophical_features, context_features])

            # Combine embeddings with appropriate weighting
            enhanced_embedding = self._combine_embeddings(
                base_embedding, philosophical_features
            )

            # Store embedding with metadata
            concept_key = self._generate_concept_key(statement, context)
            self.concept_embeddings[concept_key] = enhanced_embedding
            self.embedding_metadata[concept_key] = {
                'statement': statement,
                'context': context.domain.value if context else None,
                'timestamp': datetime.now().isoformat(),
                'semantic_analysis_id': semantic_analysis.analysis_id,
                'embedding_quality': await self._assess_embedding_quality(enhanced_embedding)
            }

            logger.debug(f"Generated embedding for concept: {concept_key}")
            return enhanced_embedding

        except Exception as e:
            logger.error(f"Error generating philosophical embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dimension)

    async def _generate_base_semantic_embedding(self, statement: str) -> np.ndarray:
        """
        Generate base semantic embedding using simulated transformer model.

        In a real implementation, this would use a pre-trained language model
        like BERT, RoBERTa, or a specialized philosophical model.
        """
        # Simulated embedding generation (replace with actual model)
        # This creates a deterministic but varied embedding based on statement content
        hash_value = hash(statement) % (2**31)
        np.random.seed(hash_value)

        # Generate embedding with some semantic structure
        base_vector = np.random.normal(0, 0.1, self.embedding_dimension)

        # Add statement-specific features
        statement_length_feature = len(statement) / 1000.0
        complexity_feature = len(statement.split()) / 100.0

        # Incorporate basic linguistic features
        base_vector[0] = statement_length_feature
        base_vector[1] = complexity_feature

        # Normalize to unit vector
        base_vector = base_vector / np.linalg.norm(base_vector)

        return base_vector

    def _extract_philosophical_features(
        self,
        semantic_analysis: SemanticAnalysis,
        philosophical_category: Any
    ) -> np.ndarray:
        """Extract philosophical features for embedding enhancement."""
        features = []

        # Domain-specific features
        if hasattr(philosophical_category, 'primary'):
            domain_encoding = self._encode_philosophical_domain(
                philosophical_category.primary
            )
            features.extend(domain_encoding)

        # Complexity and uncertainty features
        if semantic_analysis.epistemic_uncertainty:
            uncertainty_avg = np.mean(list(semantic_analysis.epistemic_uncertainty.values()))
            features.append(uncertainty_avg)
        else:
            features.append(0.5)  # Default uncertainty

        # Relational complexity
        relation_count = len(semantic_analysis.semantic_relations)
        features.append(min(relation_count / 10.0, 1.0))  # Normalized

        # Concept count
        concept_count = len(semantic_analysis.primary_concepts)
        features.append(min(concept_count / 5.0, 1.0))  # Normalized

        # Pad or truncate to ensure consistent size
        target_size = 50  # Philosophical feature vector size
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        return np.array(features)

    def _extract_context_features(self, context: PhilosophicalContext) -> np.ndarray:
        """Extract context-specific features."""
        features = []

        # Domain encoding
        domain_encoding = self._encode_philosophical_domain(context.domain)
        features.extend(domain_encoding)

        # Language game encoding
        language_game_encoding = self._encode_language_game(context.language_game)
        features.extend(language_game_encoding)

        # Depth and complexity
        features.append(context.depth_requirements / 5.0)  # Normalized

        # Constraint features
        constraint_count = len(context.perspective_constraints or [])
        features.append(min(constraint_count / 5.0, 1.0))

        # Pad to consistent size
        target_size = 30
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        return np.array(features)

    def _encode_philosophical_domain(self, domain: PhilosophicalDomain) -> list[float]:
        """Encode philosophical domain as feature vector."""
        domain_map = {
            PhilosophicalDomain.METAPHYSICS: [1.0, 0.0, 0.0, 0.0, 0.0],
            PhilosophicalDomain.EPISTEMOLOGY: [0.0, 1.0, 0.0, 0.0, 0.0],
            PhilosophicalDomain.ETHICS: [0.0, 0.0, 1.0, 0.0, 0.0],
            PhilosophicalDomain.AESTHETICS: [0.0, 0.0, 0.0, 1.0, 0.0],
            PhilosophicalDomain.LOGIC: [0.0, 0.0, 0.0, 0.0, 1.0],
        }
        return domain_map.get(domain, [0.2, 0.2, 0.2, 0.2, 0.2])

    def _encode_language_game(self, language_game) -> list[float]:
        """Encode language game as feature vector."""
        # Simplified encoding - in practice would be more sophisticated
        return [hash(str(language_game)) % 100 / 100.0] * 5

    def _combine_embeddings(
        self,
        base_embedding: np.ndarray,
        philosophical_features: np.ndarray
    ) -> np.ndarray:
        """Combine base and philosophical embeddings."""
        # Ensure consistent dimensionality
        if len(base_embedding) + len(philosophical_features) > self.embedding_dimension:
            # Truncate philosophical features if needed
            philosophical_features = philosophical_features[:self.embedding_dimension - len(base_embedding)]
        elif len(base_embedding) + len(philosophical_features) < self.embedding_dimension:
            # Pad with zeros if needed
            padding_size = self.embedding_dimension - len(base_embedding) - len(philosophical_features)
            philosophical_features = np.concatenate([philosophical_features, np.zeros(padding_size)])

        combined = np.concatenate([base_embedding, philosophical_features])

        # Normalize to unit vector
        return combined / np.linalg.norm(combined)

    def _generate_concept_key(self, statement: str, context: PhilosophicalContext | None) -> str:
        """Generate unique key for concept storage."""
        context_str = context.domain.value if context else "general"
        return f"{statement[:50]}_{context_str}_{hash(statement) % 10000}"

    async def _assess_embedding_quality(self, embedding: np.ndarray) -> EmbeddingMetrics:
        """Assess the quality of a generated embedding."""
        # Calculate various quality metrics
        dimensional_density = np.std(embedding)  # Measure of information distribution
        semantic_coherence = 1.0 - np.abs(np.mean(embedding))  # Measure of balance
        contextual_sensitivity = min(np.max(embedding) - np.min(embedding), 1.0)
        temporal_consistency = 0.8  # Would be calculated from historical data

        overall_quality = np.mean(np.array([
            dimensional_density, semantic_coherence,
            contextual_sensitivity, temporal_consistency
        ]))

        return EmbeddingMetrics(
            dimensional_density=float(dimensional_density),
            semantic_coherence=float(semantic_coherence),
            contextual_sensitivity=float(contextual_sensitivity),
            temporal_consistency=float(temporal_consistency),
            overall_quality=float(overall_quality)
        )

    def calculate_semantic_similarity(
        self,
        concept1: str,
        concept2: str,
        context: PhilosophicalContext | None = None
    ) -> float:
        """
        Calculate semantic similarity between two concepts.

        Args:
            concept1: First concept identifier
            concept2: Second concept identifier
            context: Optional context for similarity calculation

        Returns:
            Similarity score between 0 and 1
        """
        # Check cache first
        cache_key = (concept1, concept2)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Get embeddings
        embedding1 = self._get_concept_embedding(concept1, context)
        embedding2 = self._get_concept_embedding(concept2, context)

        if embedding1 is None or embedding2 is None:
            logger.warning(f"Missing embeddings for similarity calculation: {concept1}, {concept2}")
            return 0.0

        # Calculate cosine similarity
        similarity = cosine_similarity(
            np.expand_dims(embedding1, axis=0),
            np.expand_dims(embedding2, axis=0)
        )[0][0]

        # Apply context-dependent adjustments
        if context:
            similarity = self._adjust_similarity_for_context(similarity, context)

        # Cache result
        self.similarity_cache[cache_key] = similarity

        return float(similarity)

    def _get_concept_embedding(
        self,
        concept: str,
        context: PhilosophicalContext | None = None
    ) -> np.ndarray | None:
        """Retrieve embedding for a concept."""
        # Try exact match first
        concept_key = self._generate_concept_key(concept, context)
        if concept_key in self.concept_embeddings:
            return self.concept_embeddings[concept_key]

        # Try partial matches
        for key, embedding in self.concept_embeddings.items():
            if concept in key:
                return embedding

        return None

    def _adjust_similarity_for_context(
        self,
        similarity: float,
        context: PhilosophicalContext
    ) -> float:
        """Adjust similarity score based on philosophical context."""
        # Domain-specific adjustments
        if context.domain == PhilosophicalDomain.LOGIC:
            # Logic requires higher precision
            return similarity * 0.9
        elif context.domain == PhilosophicalDomain.AESTHETICS:
            # Aesthetics allows more interpretive flexibility
            return min(similarity * 1.1, 1.0)

        return similarity

    async def discover_semantic_clusters(
        self,
        min_cluster_size: int = 3,
        max_clusters: int = 10
    ) -> list[SemanticCluster]:
        """
        Discover semantic clusters in the embedding space.

        Args:
            min_cluster_size: Minimum number of concepts per cluster
            max_clusters: Maximum number of clusters to create

        Returns:
            List of discovered semantic clusters
        """
        if len(self.concept_embeddings) < min_cluster_size:
            logger.warning("Not enough concepts for clustering")
            return []

        # Prepare data for clustering
        concept_keys = list(self.concept_embeddings.keys())
        embeddings_matrix = np.array([
            self.concept_embeddings[key] for key in concept_keys
        ])

        # Apply DBSCAN clustering
        dbscan_labels = self.dbscan_clusterer.fit_predict(embeddings_matrix)

        # Create cluster objects
        clusters = []
        unique_labels = set(dbscan_labels)

        for label in unique_labels:
            if label == -1:  # Noise points
                continue

            cluster_indices = [i for i, label_value in enumerate(dbscan_labels) if label_value == label]
            if len(cluster_indices) < min_cluster_size:
                continue

            cluster_concepts = [concept_keys[i] for i in cluster_indices]
            cluster_embeddings = embeddings_matrix[cluster_indices]

            # Calculate cluster center
            center_embedding = np.mean(cluster_embeddings, axis=0)
            center_concept = self._find_most_central_concept(
                cluster_concepts, center_embedding
            )

            # Calculate coherence score
            coherence_score = self._calculate_cluster_coherence(cluster_embeddings)

            # Identify philosophical theme
            philosophical_theme = await self._identify_cluster_theme(cluster_concepts)

            cluster = SemanticCluster(
                cluster_id=f"cluster_{label}_{len(clusters)}",
                center_concept=center_concept,
                member_concepts=cluster_concepts,
                coherence_score=coherence_score,
                philosophical_theme=philosophical_theme
            )

            clusters.append(cluster)

        self.semantic_clusters = clusters
        logger.info(f"Discovered {len(clusters)} semantic clusters")

        return clusters

    def _find_most_central_concept(
        self,
        concepts: list[str],
        center_embedding: np.ndarray
    ) -> str:
        """Find the concept closest to the cluster center."""
        min_distance = float('inf')
        central_concept = concepts[0]

        for concept in concepts:
            embedding = self._get_concept_embedding(concept)
            if embedding is not None:
                distance = np.linalg.norm(embedding - center_embedding)
                if distance < min_distance:
                    min_distance = distance
                    central_concept = concept

        return central_concept

    def _calculate_cluster_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate coherence score for a cluster."""
        if len(embeddings) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Return average similarity (excluding diagonal)
        n = len(embeddings)
        total_similarity = np.sum(similarities) - np.trace(similarities)
        average_similarity = total_similarity / (n * (n - 1))

        return float(average_similarity)

    async def _identify_cluster_theme(self, concepts: list[str]) -> str:
        """Identify the philosophical theme of a cluster."""
        # Simple theme identification based on concept analysis
        # In practice, this would use more sophisticated NLP

        concept_domains = []
        for concept in concepts:
            metadata = self.embedding_metadata.get(concept, {})
            if 'context' in metadata:
                concept_domains.append(metadata['context'])

        if concept_domains:
            # Find most common domain
            domain_counts = {}
            for domain in concept_domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            most_common_domain = max(domain_counts, key=lambda k: domain_counts[k])
            return f"philosophical_theme_{most_common_domain}"

        return f"philosophical_theme_cluster_{len(concepts)}_concepts"

    def get_semantic_neighborhood(
        self,
        concept: str,
        radius: float = 0.7,
        max_neighbors: int = 10
    ) -> list[tuple[str, float]]:
        """
        Get semantic neighborhood of a concept.

        Args:
            concept: Target concept
            radius: Similarity threshold for neighborhood
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of (neighbor_concept, similarity_score) tuples
        """
        concept_embedding = self._get_concept_embedding(concept)
        if concept_embedding is None:
            return []

        neighbors = []

        for other_concept, other_embedding in self.concept_embeddings.items():
            if other_concept == concept:
                continue

            similarity = cosine_similarity(
                np.expand_dims(concept_embedding, axis=0),
                np.expand_dims(other_embedding, axis=0)
            )[0][0]

            if similarity >= radius:
                neighbors.append((other_concept, float(similarity)))

        # Sort by similarity and limit results
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]

    def get_embedding_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the embedding space."""
        if not self.concept_embeddings:
            return {"status": "empty"}

        embeddings_matrix = np.array(list(self.concept_embeddings.values()))

        return {
            "total_concepts": len(self.concept_embeddings),
            "embedding_dimension": self.embedding_dimension,
            "total_clusters": len(self.semantic_clusters),
            "embedding_statistics": {
                "mean_norm": float(np.mean([np.linalg.norm(emb) for emb in embeddings_matrix])),
                "std_norm": float(np.std([np.linalg.norm(emb) for emb in embeddings_matrix])),
                "dimension_variance": float(np.mean(np.var(embeddings_matrix, axis=0))),
                "sparsity": float(np.mean(embeddings_matrix == 0))
            },
            "cache_statistics": {
                "similarity_cache_size": len(self.similarity_cache),
                "cache_hit_rate": getattr(self, '_cache_hit_rate', 0.0)
            },
            "quality_metrics": {
                "average_quality": np.mean([
                    metadata.get('embedding_quality', {}).get('overall_quality', 0.5)
                    for metadata in self.embedding_metadata.values()
                    if 'embedding_quality' in metadata
                ])
            }
        }

    def save_embedding_space(self, filepath: str) -> None:
        """Save the embedding space to disk."""
        data = {
            "concept_embeddings": {
                key: embedding.tolist()
                for key, embedding in self.concept_embeddings.items()
            },
            "embedding_metadata": self.embedding_metadata,
            "semantic_clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "center_concept": cluster.center_concept,
                    "member_concepts": cluster.member_concepts,
                    "coherence_score": cluster.coherence_score,
                    "philosophical_theme": cluster.philosophical_theme
                }
                for cluster in self.semantic_clusters
            ],
            "embedding_dimension": self.embedding_dimension,
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved embedding space to {filepath}")

    def load_embedding_space(self, filepath: str) -> None:
        """Load embedding space from disk."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Restore embeddings
            self.concept_embeddings = {
                key: np.array(embedding)
                for key, embedding in data.get("concept_embeddings", {}).items()
            }

            # Restore metadata
            self.embedding_metadata = data.get("embedding_metadata", {})

            # Restore clusters
            cluster_data = data.get("semantic_clusters", [])
            self.semantic_clusters = [
                SemanticCluster(
                    cluster_id=cluster["cluster_id"],
                    center_concept=cluster["center_concept"],
                    member_concepts=cluster["member_concepts"],
                    coherence_score=cluster["coherence_score"],
                    philosophical_theme=cluster["philosophical_theme"]
                )
                for cluster in cluster_data
            ]

            self.embedding_dimension = data.get("embedding_dimension", 768)

            logger.info(f"Loaded embedding space from {filepath}")

        except Exception as e:
            logger.error(f"Error loading embedding space: {e}")
            raise
