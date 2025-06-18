"""
NARS Truth Functions - Non-Axiomatic Logic Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements NARS truth value operations and uncertainty calculations
based on Non-Axiomatic Logic (NAL) principles.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TruthValue:
    """
    Represents a NARS truth value with frequency and confidence.

    In NARS, truth is represented as (frequency, confidence) where:
    - frequency: degree of positive evidence (0-1)
    - confidence: degree of evidential support (0-1)
    """
    frequency: float
    confidence: float

    def __post_init__(self):
        """Validate truth values are in valid range."""
        self.frequency = float(np.clip(self.frequency, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    @property
    def expectation(self) -> float:
        """Calculate expectation value for decision making."""
        return self.confidence * (self.frequency - 0.5) + 0.5

    @property
    def uncertainty(self) -> float:
        """Calculate epistemic uncertainty."""
        return 1.0 - self.confidence

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "frequency": self.frequency,
            "confidence": self.confidence,
            "expectation": self.expectation,
            "uncertainty": self.uncertainty
        }


class Truth:
    """NARS truth value operations following Non-Axiomatic Logic."""

    # Evidence horizon constant
    K = 1.0

    @staticmethod
    def revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Revise two truth values with same content but different evidence.

        This implements the NARS revision rule for combining evidence
        from different sources about the same statement.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        # Calculate weights based on confidence
        w1 = c1 / (1 - c1)
        w2 = c2 / (1 - c2)
        w = w1 + w2

        # Revised frequency (weighted average)
        f = (w1 * f1 + w2 * f2) / w

        # Revised confidence (approaches 1 as evidence accumulates)
        c = w / (w + Truth.K)

        return TruthValue(f, c)

    @staticmethod
    def deduction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Deductive inference: (A → B), A ⊢ B

        Strong inference preserving truth with confidence reduction.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        f = f1 * f2
        c = f1 * f2 * c1 * c2

        return TruthValue(f, c)

    @staticmethod
    def induction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Inductive inference: B, (A → B) ⊢ A

        Weak inference with significant confidence reduction.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        w = f2 * c1 * c2 / (f2 * c1 * c2 + Truth.K)
        f = f1
        c = w * c1 * c2 / (w * c1 * c2 + Truth.K)

        return TruthValue(f, c)

    @staticmethod
    def abduction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Abductive inference: A, (A → B) ⊢ B

        Inference to best explanation with uncertainty.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        w = f1 * c1 * c2 / (f1 * c1 * c2 + Truth.K)
        f = f2
        c = w * c1 * c2 / (w * c1 * c2 + Truth.K)

        return TruthValue(f, c)

    @staticmethod
    def analogy(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Analogical inference: (A → B), (S → B) ⊢ (S → A)

        Similarity-based inference with high uncertainty.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        f = f1 * f2
        c = c1 * c2 * f2

        return TruthValue(f, c)

    @staticmethod
    def negation(tv: TruthValue) -> TruthValue:
        """
        Negation: ¬A

        Frequency inverts, confidence remains.
        """
        return TruthValue(1.0 - tv.frequency, tv.confidence)

    @staticmethod
    def conjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Conjunction: A ∧ B

        Both must be true with reduced confidence.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        f = f1 * f2
        c = c1 * c2

        return TruthValue(f, c)

    @staticmethod
    def disjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """
        Disjunction: A ∨ B

        At least one must be true.
        """
        f1, c1 = tv1.frequency, tv1.confidence
        f2, c2 = tv2.frequency, tv2.confidence

        f = 1 - (1 - f1) * (1 - f2)
        c = c1 * c2

        return TruthValue(f, c)

    @staticmethod
    def projection(tv: TruthValue, source_time: float, target_time: float,
                   decay_rate: float = 0.9) -> TruthValue:
        """
        Project truth value from one time to another with confidence decay.

        Args:
            tv: Original truth value
            source_time: Time of original observation
            target_time: Time to project to
            decay_rate: Confidence decay factor per time unit

        Returns:
            Projected truth value with decayed confidence
        """
        time_diff = abs(target_time - source_time)

        # Confidence decays exponentially with time
        decayed_confidence = tv.confidence * (decay_rate ** time_diff)

        # Frequency remains but confidence decreases
        return TruthValue(tv.frequency, decayed_confidence)

    @staticmethod
    def eternalization(temporal_truths: list[TruthValue]) -> TruthValue:
        """
        Convert temporal truths to eternal truth through evidence accumulation.

        Args:
            temporal_truths: List of time-stamped truth values

        Returns:
            Eternalized truth value
        """
        if not temporal_truths:
            return TruthValue(0.5, 0.0)  # No evidence

        # Accumulate evidence through revision
        result = temporal_truths[0]
        for tv in temporal_truths[1:]:
            result = Truth.revision(result, tv)

        return result

    @staticmethod
    def expectation_to_truth(expectation: float, confidence: float = 0.9) -> TruthValue:
        """
        Convert expectation value to truth value.

        Useful for integrating with other systems that use single values.
        """
        # Solve for frequency given expectation and confidence
        # E = c * (f - 0.5) + 0.5
        # f = (E - 0.5) / c + 0.5

        if confidence == 0:
            return TruthValue(expectation, 0.0)

        frequency = (expectation - 0.5) / confidence + 0.5
        frequency = np.clip(frequency, 0.0, 1.0)

        return TruthValue(frequency, confidence)

    @staticmethod
    def from_evidence(positive: int, total: int) -> TruthValue:
        """
        Create truth value from evidence counts.

        Args:
            positive: Number of positive observations
            total: Total number of observations

        Returns:
            Truth value derived from evidence
        """
        if total == 0:
            return TruthValue(0.5, 0.0)

        frequency = positive / total
        confidence = total / (total + Truth.K)

        return TruthValue(frequency, confidence)
