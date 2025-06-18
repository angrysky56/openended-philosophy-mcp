"""
types.py - Core Data Structures and Type Definitions

This file defines all shared data models for the NARS Philosophical Reasoning Engine.
By centralizing these definitions, we establish a single source of truth and prevent
circular dependencies between other modules.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict
import numpy as np

# A type alias to clearly represent a semantic embedding vector.
Embedding = np.ndarray

@dataclass(frozen=True)
class TruthValue:
    """
    Represents the two-dimensional NARS truth-value, consisting of frequency
    and confidence. This class is immutable (`frozen=True`) to prevent accidental
    modification.
    """
    frequency: float
    confidence: float

    def __post_init__(self):
        """Validate values after initialization."""
        if not 0.0 <= self.frequency <= 1.0:
            raise ValueError("Frequency must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    @property
    def expectation(self) -> float:
        """
        Calculates the expectation value of the truth.
        This provides a single scalar value useful for ranking beliefs, where
        values closer to 1.0 are "more true" and values closer to 0.0 are
        "more false".
        """
        return self.confidence * (self.frequency - 0.5) + 0.5

    def to_dict(self) -> Dict[str, float]:
        """Serializes the TruthValue to a dictionary."""
        return {"frequency": self.frequency, "confidence": self.confidence}


@dataclass
class MemoryItem:
    """
    Represents a single item of knowledge within the NARS memory, such as a
    concept, belief, or judgment.
    """
    term: str
    truth: TruthValue
    stamp: List[int]
    
    # The time this belief is considered valid. "eternal" is the default for
    # general principles.
    occurrence_time: str | None = "eternal"
    
    # The semantic vector embedding of the term, crucial for similarity checks.
    embedding: Embedding | None = None
    
    # A category tag for organizing philosophical concepts.
    philosophical_category: str = "uncategorized"
    
    # A temporary dictionary to hold contextual data during processing,
    # such as a relevance score for a specific query. This is not stored
    # permanently in memory.
    semantic_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the MemoryItem to a dictionary for external use."""
        return {
            "term": self.term,
            "truth": self.truth.to_dict(),
            "stamp": self.stamp,
            "occurrence_time": self.occurrence_time,
            "philosophical_category": self.philosophical_category
        }


@dataclass(frozen=True)
class ReasoningResult:
    """
    A standardized container for the output of any philosophical reasoning pattern.
    This ensures that results from deductive, inductive, abductive, etc.,
    processes can be handled uniformly. Immutable to ensure result integrity.
    """
    conclusion: str
    truth: TruthValue
    evidence: List[MemoryItem]
    inference_path: List[str]
    uncertainty_factors: Dict[str, float | int]
    philosophical_implications: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ReasoningResult to a dictionary."""
        return {
            "conclusion": self.conclusion,
            "truth": self.truth.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "inference_path": self.inference_path,
            "uncertainty_factors": self.uncertainty_factors,
            "philosophical_implications": self.philosophical_implications,
        }