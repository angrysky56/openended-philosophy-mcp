"""
types.py - Core Data Structures and Type Definitions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This file defines all shared data models for the NARS Philosophical Reasoning Engine.
By centralizing these definitions, we establish a single source of truth and prevent
circular dependencies between other modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Type alias for semantic embedding vectors
Embedding = np.ndarray


@dataclass(frozen=True)
class TruthValue:
    """
    Represents the two-dimensional NARS truth-value, consisting of frequency
    and confidence. This class is immutable (`frozen=True`) to prevent accidental
    modification of truth values during reasoning.

    In NARS, truth is represented as (frequency, confidence) where:
    - frequency: degree of positive evidence (0-1)
    - confidence: degree of evidential support (0-1)
    """
    frequency: float
    confidence: float

    def __post_init__(self):
        """Validate values after initialization to ensure they are within the [0,1] range."""
        # Use object.__setattr__ because the dataclass is frozen
        freq = float(np.clip(self.frequency, 0.0, 1.0))
        conf = float(np.clip(self.confidence, 0.0, 1.0))

        if freq != self.frequency:
            object.__setattr__(self, 'frequency', freq)
        if conf != self.confidence:
            object.__setattr__(self, 'confidence', conf)

    @property
    def expectation(self) -> float:
        """Calculate expectation value for decision making."""
        return self.confidence * (self.frequency - 0.5) + 0.5

    @property
    def uncertainty(self) -> float:
        """Calculate epistemic uncertainty."""
        return 1.0 - self.confidence

    def to_dict(self) -> dict[str, float]:
        """Serialize the TruthValue to a dictionary."""
        return {
            "frequency": self.frequency,
            "confidence": self.confidence,
            "expectation": self.expectation,
            "uncertainty": self.uncertainty
        }


@dataclass
class MemoryItem:
    """
    Represents a single item of knowledge within the NARS memory, such as a
    concept, belief, or judgment. This is the primary object managed by NARSMemory.

    Combines NARS-style truth maintenance with semantic embeddings
    for philosophical coherence analysis.
    """
    term: str
    truth: TruthValue
    stamp: list[int]  # Evidence base IDs

    # Time this belief is considered valid. "eternal" is default for general principles
    occurrence_time: str | int | float = "eternal"

    # Semantic vector embedding of the term, crucial for similarity checks
    embedding: Embedding | None = None

    # Category tag for organizing philosophical concepts
    philosophical_category: str | None = None

    # Usage tracking
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    usefulness: int = 0

    # Temporary dictionary to hold contextual data during processing
    semantic_context: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the MemoryItem to a dictionary for external use."""
        return {
            "term": self.term,
            "truth": self.truth.to_dict(),
            "stamp": self.stamp,
            "occurrence_time": self.occurrence_time,
            "philosophical_category": self.philosophical_category,
            "last_used": self.last_used,
            "usefulness": self.usefulness,
            "semantic_context": self.semantic_context
        }


@dataclass(frozen=True)
class ReasoningResult:
    """
    A standardized container for the output of any philosophical reasoning pattern.
    Its immutability ensures that once a result is generated, it cannot be altered.
    """
    conclusion: str
    truth: TruthValue
    evidence: list[MemoryItem]
    inference_path: list[str]
    uncertainty_factors: dict[str, float | int]
    philosophical_implications: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ReasoningResult to a dictionary for detailed output."""
        return {
            "conclusion": self.conclusion,
            "truth": self.truth.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "inference_path": self.inference_path,
            "uncertainty_factors": self.uncertainty_factors,
            "philosophical_implications": self.philosophical_implications,
        }
