"""
NARS Memory Integration - Philosophical Knowledge Management
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Integrates NARS memory capabilities with philosophical coherence landscape,
providing semantic embeddings, attention buffer, and evidence tracking.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .truth_functions import Truth, TruthValue

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """
    Represents a belief or concept in NARS memory.

    Combines NARS-style truth maintenance with semantic embeddings
    for philosophical coherence analysis.
    """
    term: str
    truth: TruthValue
    occurrence_time: str  # "eternal" or timestamp
    stamp: list[int]  # Evidence base IDs
    embedding: np.ndarray | None = None
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    usefulness: int = 0
    semantic_context: dict[str, Any] = field(default_factory=dict)
    philosophical_category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "term": self.term,
            "truth": self.truth.to_dict(),
            "occurrence_time": self.occurrence_time,
            "stamp": self.stamp,
            "last_used": self.last_used,
            "usefulness": self.usefulness,
            "semantic_context": self.semantic_context,
            "philosophical_category": self.philosophical_category
        }


class NARSMemory:
    """
    NARS-inspired memory system with philosophical enhancements.

    Provides:
    - Semantic embeddings for conceptual similarity
    - Attention buffer for relevant belief retrieval
    - Truth maintenance and evidence tracking
    - Coherence landscape integration
    """

    def __init__(self,
                 embedding_model: TfidfVectorizer | None = None,
                 memory_file: Path | None = None,
                 attention_size: int = 30,
                 recency_size: int = 10,
                 eternalization_distance: int = 3):
        """
        Initialize NARS memory system.

        Args:
            embedding_model: Model for generating semantic embeddings
            memory_file: Path to persist memory
            attention_size: Size of attention buffer for relevance
            recency_size: Size of recency buffer
            eternalization_distance: Time distance for belief eternalization
        """
        self.memory: dict[tuple[str, str], MemoryItem] = {}
        self.atoms: dict[str, np.ndarray] = {}  # Atomic concept embeddings
        self.embedding_model: TfidfVectorizer = embedding_model or TfidfVectorizer(max_features=1000)
        self.memory_file = memory_file
        self.attention_size = attention_size
        self.recency_size = recency_size
        self.eternalization_distance = eternalization_distance
        self.current_time = 1

        # Philosophical categorization
        self.philosophical_categories = {
            "metaphysical": ["existence", "being", "reality", "substance", "essence"],
            "epistemological": ["knowledge", "belief", "truth", "justification", "evidence"],
            "ethical": ["good", "right", "duty", "virtue", "value"],
            "phenomenological": ["experience", "consciousness", "qualia", "intentionality"],
            "logical": ["implies", "and", "or", "not", "necessary", "possible"]
        }

        # Load existing memory if available
        if memory_file and memory_file.exists():
            self.load(memory_file)

        logger.info(f"NARS Memory initialized with {len(self.memory)} items")

    def add_belief(self,
                   term: str,
                   truth: TruthValue,
                   occurrence_time: str = "eternal",
                   stamp: list[int] | None = None,
                   embedding: np.ndarray | None = None) -> MemoryItem:
        """
        Add or update a belief in memory.

        Implements NARS revision if belief already exists.
        """
        key = (term, occurrence_time)
        stamp = stamp or [self.current_time]

        # Categorize philosophically
        category = self._categorize_term(term)

        if key in self.memory:
            # Revise existing belief
            existing = self.memory[key]
            revised_truth = Truth.revision(existing.truth, truth)

            # Update with higher confidence
            if truth.confidence > existing.truth.confidence:
                existing.truth = revised_truth
                existing.stamp.extend(stamp)
                existing.last_used = datetime.now().timestamp()
                existing.usefulness += 1

            logger.debug(f"Revised belief: {term} to {revised_truth}")
            return existing

        else:
            # Create new belief
            if embedding is None:
                embedding = self._generate_embedding(term)

            item = MemoryItem(
                term=term,
                truth=truth,
                occurrence_time=occurrence_time,
                stamp=stamp,
                embedding=embedding,
                philosophical_category=category
            )

            self.memory[key] = item
            logger.debug(f"Added belief: {term} with truth {truth}")
            return item

    def query(self, term: str, occurrence_time: str = "eternal") -> MemoryItem | None:
        """
        Query memory for a specific belief.

        Updates usage statistics for memory management.
        """
        key = (term, occurrence_time)

        if key in self.memory:
            item = self.memory[key]
            item.last_used = datetime.now().timestamp()
            item.usefulness += 1
            return item

        return None

    def get_attention_buffer(self,
                            query: str | None = None,
                            include_categories: list[str] | None = None) -> list[MemoryItem]:
        """
        Get attention buffer of relevant beliefs.

        Combines recency and relevance for philosophical inquiry.
        """
        items = list(self.memory.values())

        # Filter by philosophical categories if specified
        if include_categories:
            items = [item for item in items
                    if item.philosophical_category in include_categories]

        # Sort by recency
        recent_items = sorted(items, key=lambda x: x.last_used, reverse=True)[:self.recency_size]

        # Add relevant items if query provided
        relevant_items = []
        if query:
            query_embedding = self._generate_embedding(query)

            # Calculate relevance scores
            relevance_scores = []
            for item in items:
                if item not in recent_items and item.embedding is not None:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        item.embedding.reshape(1, -1)
                    )[0][0]

                    # Weight by truth expectation
                    weighted_score = similarity * item.truth.expectation
                    relevance_scores.append((item, weighted_score))

            # Sort by relevance and take top items
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_items = [item for item, _ in relevance_scores[:self.attention_size - len(recent_items)]]

        # Combine and return
        attention_buffer = relevant_items + recent_items

        logger.debug(f"Attention buffer: {len(relevant_items)} relevant + {len(recent_items)} recent items")
        return attention_buffer

    def eternalize_beliefs(self) -> None:
        """
        Convert temporal beliefs to eternal through evidence accumulation.

        Implements NARS eternalization for long-term memory.
        """
        # Group beliefs by term
        term_beliefs = defaultdict(list)

        for (term, time), item in list(self.memory.items()):
            if time != "eternal" and \
               isinstance(time, int | float) and \
               (self.current_time - float(time) > self.eternalization_distance):
                term_beliefs[term].append(item)

        # Eternalize each term
        for term, temporal_items in term_beliefs.items():
            if not temporal_items:
                continue

            # Get existing eternal belief if any
            eternal_key = (term, "eternal")
            existing_eternal = self.memory.get(eternal_key)

            # Collect truth values
            truth_values = [item.truth for item in temporal_items]
            if existing_eternal:
                truth_values.append(existing_eternal.truth)

            # Eternalize through revision
            eternal_truth = Truth.eternalization(truth_values)

            # Collect all evidence stamps
            all_stamps = []
            for item in temporal_items:
                all_stamps.extend(item.stamp)
            if existing_eternal:
                all_stamps.extend(existing_eternal.stamp)

            # Use embedding from most confident belief
            best_item = max(temporal_items, key=lambda x: x.truth.confidence)

            # Update or create eternal belief
            eternal_item = MemoryItem(
                term=term,
                truth=eternal_truth,
                occurrence_time="eternal",
                stamp=list(set(all_stamps)),
                embedding=best_item.embedding,
                philosophical_category=best_item.philosophical_category,
                usefulness=sum(item.usefulness for item in temporal_items)
            )

            self.memory[eternal_key] = eternal_item

            # Remove temporal beliefs
            for item in temporal_items:
                del self.memory[(item.term, item.occurrence_time)]

            logger.debug(f"Eternalized {len(temporal_items)} beliefs for term: {term}")

    def _categorize_term(self, term: str) -> str | None:
        """Categorize term philosophically based on content."""
        term_lower = term.lower()

        for category, keywords in self.philosophical_categories.items():
            if any(keyword in term_lower for keyword in keywords):
                return category

        # Analyze term structure for categorization
        if "-->" in term:
            if "[" in term and "]" in term:
                return "metaphysical"  # Property relation
            elif "*" in term:
                return "logical"  # Compound relation
            else:
                return "epistemological"  # Simple relation

        return None

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text."""
        # Simple TF-IDF based embedding
        # In production, use more sophisticated embeddings
        try:
            # Ensure vectorizer is fitted
            if not hasattr(self.embedding_model, 'vocabulary_'):
                # Fit on philosophical corpus
                corpus = [
                    "consciousness experience qualia phenomenology",
                    "truth knowledge belief justification evidence",
                    "good right duty virtue ethics morality",
                    "existence being reality substance essence",
                    "logic implies necessary possible contingent"
                ]
                self.embedding_model.fit(corpus)

            embedding = self.embedding_model.transform([text]).toarray()[0]  # type: ignore
            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            # Return random embedding as fallback
            # Use getattr to be robust against type checker issues and potential absence of the attribute.
            mf = getattr(self.embedding_model, 'max_features', None)
            # Ensure max_features is a positive integer; otherwise, use a default dimension.
            dimension = mf if isinstance(mf, int) and mf > 0 else 100
            return np.random.rand(dimension)

    def get_coherence_landscape(self) -> dict[str, Any]:
        """
        Generate coherence landscape from memory contents.

        Returns philosophical structure analysis of beliefs.
        """
        # Categorize beliefs
        category_beliefs = defaultdict(list)
        for item in self.memory.values():
            if item.philosophical_category:
                category_beliefs[item.philosophical_category].append(item)

        # Analyze coherence by category
        landscape = {}

        for category, items in category_beliefs.items():
            if not items:
                continue

            # Calculate category coherence
            truth_values = [item.truth for item in items]
            avg_confidence = np.mean([tv.confidence for tv in truth_values])
            avg_expectation = np.mean([tv.expectation for tv in truth_values])

            # Calculate semantic coherence
            if len(items) > 1:
                embeddings = [item.embedding for item in items if item.embedding is not None]
                if len(embeddings) > 1:
                    # Average pairwise similarity
                    similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = cosine_similarity(
                                embeddings[i].reshape(1, -1),
                                embeddings[j].reshape(1, -1)
                            )[0][0]
                            similarities.append(sim)
                    semantic_coherence = np.mean(similarities) if similarities else 0.0
                else:
                    semantic_coherence = 1.0
            else:
                semantic_coherence = 1.0

            landscape[category] = {
                "belief_count": len(items),
                "average_confidence": float(avg_confidence),
                "average_expectation": float(avg_expectation),
                "semantic_coherence": float(semantic_coherence),
                "key_beliefs": [item.term for item in sorted(items,
                               key=lambda x: x.truth.expectation,
                               reverse=True)[:5]]
            }

        return landscape

    def save(self, path: Path) -> None:
        """Save memory to file."""
        data = {
            "memory": {str(k): v.to_dict() for k, v in self.memory.items()},
            "current_time": self.current_time,
            "atoms": {k: v.tolist() for k, v in self.atoms.items()}
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.memory)} memory items to {path}")

    def load(self, path: Path) -> None:
        """Load memory from file."""
        with open(path) as f:
            data = json.load(f)

        self.current_time = data.get("current_time", 1)

        # Load atoms
        self.atoms = {k: np.array(v) for k, v in data.get("atoms", {}).items()}

        # Load memory items
        self.memory.clear()
        for key_str, item_data in data.get("memory", {}).items():
            # Parse key
            key_tuple = eval(key_str)  # Safe since we control the format

            # Reconstruct truth value
            truth_data = item_data["truth"]
            truth = TruthValue(truth_data["frequency"], truth_data["confidence"])

            # Create memory item
            item = MemoryItem(
                term=item_data["term"],
                truth=truth,
                occurrence_time=item_data["occurrence_time"],
                stamp=item_data["stamp"],
                last_used=item_data.get("last_used", datetime.now().timestamp()),
                usefulness=item_data.get("usefulness", 0),
                semantic_context=item_data.get("semantic_context", {}),
                philosophical_category=item_data.get("philosophical_category")
            )

            # Generate embedding if needed
            if item.term in self.atoms:
                item.embedding = self.atoms[item.term]
            else:
                item.embedding = self._generate_embedding(item.term)

            self.memory[key_tuple] = item

        logger.info(f"Loaded {len(self.memory)} memory items from {path}")
