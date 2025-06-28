"""
LV-Enhanced NARS Reasoning Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module integrates the Lotka-Volterra Ecosystem Intelligence Framework
with NARS (Non-Axiomatic Reasoning System) to provide diversity-preserving
philosophical reasoning capabilities.

Core Innovation:
- Applies ecological dynamics to NARS reasoning strategy selection
- Maintains diverse truth value populations to prevent mode collapse
- Uses entropy-based adaptation for context-appropriate reasoning depth

Author: AI Memory System MVP + NeoCoder LV Framework Integration
Mathematical Foundation: Lotka-Volterra competition equations
Philosophical Foundation: Non-axiomatic reasoning with epistemic humility
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .nars import (
    MemoryItem,
    NARSManager,
    NARSMemory,
    NARSReasoning,
    ReasoningResult,
    Truth,
    TruthValue,
)

logger = logging.getLogger(__name__)


@dataclass
class LVReasoningCandidate:
    """
    Represents a reasoning strategy candidate in the LV ecosystem.

    Each candidate embodies a different approach to philosophical reasoning,
    competing for selection based on ecological fitness metrics.
    """
    strategy_name: str
    reasoning_pattern: str  # deductive, inductive, abductive, analogical, dialectical
    truth_approach: str     # revision, synthesis, projection, eternalization
    content: str

    # LV ecosystem metrics
    quality_score: float = 0.0
    novelty_score: float = 0.0
    coherence_score: float = 0.0
    diversity_score: float = 0.0
    epistemic_value: float = 0.0

    # NARS-specific metrics
    truth_confidence: float = 0.0
    evidence_strength: float = 0.0
    inference_depth: int = 0

    # Ecosystem dynamics
    population: float = 1.0
    fitness: float = 0.0

    # Supporting data
    reasoning_result: ReasoningResult | None = None
    truth_values: list[TruthValue] = field(default_factory=list)
    evidence_items: list[MemoryItem] = field(default_factory=list)

    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """Calculate ecological fitness based on weighted criteria."""
        self.fitness = (
            weights.get('quality', 0.3) * self.quality_score +
            weights.get('novelty', 0.2) * self.novelty_score +
            weights.get('coherence', 0.2) * self.coherence_score +
            weights.get('diversity', 0.15) * self.diversity_score +
            weights.get('epistemic_value', 0.15) * self.epistemic_value
        )
        return self.fitness


@dataclass
class LVEntropyProfile:
    """
    Entropy-based behavioral adaptation for NARS reasoning.

    Defines how reasoning strategies should be weighted based on
    the contextual uncertainty (entropy) of the philosophical inquiry.
    """
    low_threshold: float = 0.3
    high_threshold: float = 0.6

    # Weight schemes for different entropy levels
    low_entropy_weights: dict[str, float] = field(default_factory=lambda: {
        "quality": 0.8, "novelty": 0.05, "coherence": 0.1,
        "diversity": 0.025, "epistemic_value": 0.025
    })

    medium_entropy_weights: dict[str, float] = field(default_factory=lambda: {
        "quality": 0.5, "novelty": 0.25, "coherence": 0.15,
        "diversity": 0.075, "epistemic_value": 0.025
    })

    high_entropy_weights: dict[str, float] = field(default_factory=lambda: {
        "quality": 0.2, "novelty": 0.4, "coherence": 0.1,
        "diversity": 0.2, "epistemic_value": 0.1
    })


class LVEntropyEstimator:
    """
    Estimates contextual entropy for philosophical inquiries.

    Uses semantic analysis to determine the uncertainty and ambiguity
    inherent in philosophical questions, guiding LV ecosystem behavior.
    """

    def __init__(self, embedder_model: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedder_model)

    def estimate_philosophical_entropy(self,
                                     inquiry: str,
                                     context: dict[str, Any] | None = None) -> float:
        """
        Estimate entropy of philosophical inquiry.

        Args:
            inquiry: The philosophical question or concept to analyze
            context: Additional context including philosophical domain, perspective, etc.

        Returns:
            Entropy value between 0 (deterministic) and 1 (maximum uncertainty)
        """
        try:
            # Method 1: Philosophical complexity indicators
            philosophical_entropy = self._analyze_philosophical_characteristics(inquiry)

            # Method 2: Semantic dispersion analysis
            semantic_entropy = self._calculate_semantic_entropy(inquiry)

            # Method 3: Context-specific entropy (if available)
            context_entropy = self._analyze_contextual_uncertainty(inquiry, context)

            # Weighted combination
            final_entropy = (
                0.4 * philosophical_entropy +
                0.35 * semantic_entropy +
                0.25 * context_entropy
            )

            return np.clip(final_entropy, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Entropy estimation failed: {e}, using default 0.5")
            return 0.5

    def _analyze_philosophical_characteristics(self, inquiry: str) -> float:
        """Analyze inherent philosophical complexity."""
        inquiry_lower = inquiry.lower()

        # High entropy indicators (uncertain, ambiguous, creative)
        high_entropy_terms = [
            'consciousness', 'meaning', 'existence', 'reality', 'truth', 'beauty',
            'justice', 'free will', 'identity', 'time', 'causation', 'mind',
            'qualia', 'phenomenology', 'metaphysics', 'epistemology', 'ethics',
            'what if', 'suppose', 'imagine', 'consider', 'explore', 'multiple',
            'perspective', 'interpretation', 'ambiguous', 'paradox', 'mystery'
        ]

        # Low entropy indicators (definite, factual, analytical)
        low_entropy_terms = [
            'define', 'calculate', 'prove', 'demonstrate', 'factual', 'precise',
            'exactly', 'specifically', 'determine', 'measure', 'logic', 'formal',
            'theorem', 'axiom', 'definition', 'classification', 'category'
        ]

        high_count = sum(1 for term in high_entropy_terms if term in inquiry_lower)
        low_count = sum(1 for term in low_entropy_terms if term in inquiry_lower)

        # Normalize by inquiry length
        word_count = len(inquiry.split())
        high_ratio = high_count / max(word_count, 1)
        low_ratio = low_count / max(word_count, 1)

        return max(0.0, min(1.0, high_ratio - low_ratio + 0.5))

    def _calculate_semantic_entropy(self, inquiry: str) -> float:
        """Calculate semantic entropy using embedding variance."""
        try:
            # Split into conceptual components
            sentences = [s.strip() for s in inquiry.replace('?', '.').split('.') if s.strip()]
            if len(sentences) < 2:
                # Single concept - analyze word-level variance
                words = inquiry.split()
                if len(words) < 3:
                    return 0.3  # Simple, direct inquiry
                sentences = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

            # Generate embeddings
            embeddings = self.embedder.encode(sentences)

            # Calculate variance in semantic space
            mean_embed = np.mean(embeddings, axis=0)
            variances = [np.linalg.norm(emb - mean_embed) for emb in embeddings]
            avg_variance = np.mean(variances)

            # Normalize (heuristic based on embedding space characteristics)
            return min(1.0, float(avg_variance) / 1.5)

        except Exception:
            return 0.5

    def _analyze_contextual_uncertainty(self,
                                      inquiry: str,
                                      context: dict[str, Any] | None) -> float:
        """Analyze context-specific uncertainty factors."""
        if not context:
            return 0.5

        uncertainty = 0.5

        # Domain-specific uncertainty
        domain = context.get('domain', '')
        if domain in ['metaphysics', 'consciousness', 'aesthetics']:
            uncertainty += 0.2
        elif domain in ['logic', 'mathematics', 'formal_systems']:
            uncertainty -= 0.2

        # Perspective multiplicity
        perspectives = context.get('perspectives', [])
        if len(perspectives) > 2:
            uncertainty += 0.1 * (len(perspectives) - 2)

        # Historical context
        if context.get('historical_perspectives'):
            uncertainty += 0.1

        return np.clip(uncertainty, 0.0, 1.0)


class LVNARSEcosystem:
    """
    Core LV-Enhanced NARS Reasoning System.

    Implements ecological dynamics for NARS reasoning strategy selection,
    maintaining diverse populations of reasoning approaches while preventing
    convergence to single-perspective philosophical analyses.
    """

    def __init__(self,
                 nars_manager: NARSManager,
                 nars_memory: NARSMemory,
                 nars_reasoning: NARSReasoning,
                 embedder_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize LV-enhanced NARS ecosystem.

        Args:
            nars_manager: NARS process manager
            nars_memory: NARS memory system
            nars_reasoning: NARS reasoning engine
            embedder_model: Sentence transformer for semantic analysis
        """
        self.nars_manager = nars_manager
        self.nars_memory = nars_memory
        self.nars_reasoning = nars_reasoning

        self.embedder = SentenceTransformer(embedder_model)
        self.entropy_estimator = LVEntropyEstimator(embedder_model)
        self.entropy_profile = LVEntropyProfile()

        # LV ecosystem parameters
        self.max_iterations = 12
        self.damping_factor = 0.1
        self.convergence_threshold = 1e-6
        self.min_population_threshold = 0.1

        # Reasoning strategy registry
        self.reasoning_strategies = {
            'deductive_synthesis': self._deductive_synthesis_strategy,
            'inductive_exploration': self._inductive_exploration_strategy,
            'abductive_hypothesis': self._abductive_hypothesis_strategy,
            'analogical_mapping': self._analogical_mapping_strategy,
            'dialectical_integration': self._dialectical_integration_strategy,
            'truth_value_ecology': self._truth_value_ecology_strategy,
            'epistemic_revision': self._epistemic_revision_strategy
        }

        logger.info("LV-Enhanced NARS Ecosystem initialized")

    async def enhanced_philosophical_analysis(self,
                                            concept: str,
                                            context: dict[str, Any],
                                            perspectives: list[str]) -> dict[str, Any]:
        """
        Perform LV-enhanced philosophical analysis using diverse NARS reasoning strategies.

        Args:
            concept: The philosophical concept to analyze
            context: Analysis context including domain, criteria, etc.
            perspectives: list of philosophical perspectives to consider

        Returns:
            Comprehensive analysis with LV diversity preservation
        """
        try:
            # Estimate contextual entropy
            entropy = self.entropy_estimator.estimate_philosophical_entropy(
                concept, context
            )

            logger.info(f"Philosophical entropy estimated: {entropy:.3f}")

            # Generate reasoning strategy candidates
            candidates = await self._generate_reasoning_candidates(
                concept, context, perspectives, entropy
            )

            # Apply LV ecosystem dynamics
            selected_strategies = await self._apply_lv_dynamics(
                candidates, entropy, context
            )

            # Synthesize results
            synthesis = await self._synthesize_lv_results(
                concept, context, selected_strategies, entropy
            )

            # Calculate diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(selected_strategies)

            return {
                'concept': concept,
                'context': context,
                'entropy': entropy,
                'selected_strategies': [
                    {
                        'strategy_name': s.strategy_name,
                        'reasoning_pattern': s.reasoning_pattern,
                        'content': s.content,
                        'population': s.population,
                        'fitness': s.fitness,
                        'truth_confidence': s.truth_confidence,
                        'evidence_strength': s.evidence_strength,
                        'reasoning_result': s.reasoning_result.to_dict() if s.reasoning_result else None
                    }
                    for s in selected_strategies
                ],
                'synthesis': synthesis,
                'diversity_metrics': diversity_metrics,
                'ecosystem_status': 'stable' if len(selected_strategies) >= 2 else 'converged',
                'epistemic_implications': self._extract_epistemic_implications(selected_strategies)
            }

        except Exception as e:
            logger.error(f"LV-enhanced analysis failed: {e}")
            return {
                'error': str(e),
                'fallback_analysis': await self._fallback_analysis(concept, context, perspectives)
            }

    async def _generate_reasoning_candidates(self,
                                           concept: str,
                                           context: dict[str, Any],
                                           perspectives: list[str],
                                           entropy: float) -> list[LVReasoningCandidate]:
        """Generate diverse reasoning strategy candidates."""
        candidates = []

        # Generate strategy candidates based on available reasoning patterns
        for strategy_name, strategy_func in self.reasoning_strategies.items():
            try:
                candidate = await strategy_func(concept, context, perspectives, entropy)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")

        return candidates

    async def _apply_lv_dynamics(self,
                               candidates: list[LVReasoningCandidate],
                               entropy: float,
                               context: dict[str, Any]) -> list[LVReasoningCandidate]:
        """Apply Lotka-Volterra ecosystem dynamics to select reasoning strategies."""
        if not candidates:
            return []

        # Get entropy-appropriate weights
        weights = self._get_entropy_weights(entropy)

        # Calculate initial fitness values
        for candidate in candidates:
            candidate.calculate_fitness(weights)

        # Build LV interaction matrix
        alpha_matrix = self._build_strategy_interaction_matrix(candidates)

        # Extract growth rates (initial fitness values)
        growth_rates = np.array([c.fitness for c in candidates])

        # Initialize populations
        initial_populations = np.ones(len(candidates))

        # Run LV dynamics simulation
        final_populations, convergence_data = self._simulate_lv_dynamics(
            growth_rates, alpha_matrix, initial_populations
        )

        # Update candidate populations
        for i, candidate in enumerate(candidates):
            candidate.population = final_populations[i]

        # Select candidates above population threshold
        selected = [
            c for c in candidates
            if c.population > self.min_population_threshold
        ]

        # Ensure at least top candidate survives
        if not selected and candidates:
            selected = [max(candidates, key=lambda c: c.population)]

        # Sort by population (ecosystem fitness)
        selected.sort(key=lambda c: c.population, reverse=True)

        logger.info(f"LV dynamics selected {len(selected)}/{len(candidates)} strategies")

        return selected

    def _simulate_lv_dynamics(self,
                             growth_rates: np.ndarray,
                             alpha_matrix: np.ndarray,
                             initial_populations: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Simulate Lotka-Volterra competition dynamics.

        dx_i/dt = r_i * x_i * (1 - Σ(α_ij * x_j))
        """
        x = initial_populations.copy()
        n = len(x)

        convergence_data = {
            'iterations': 0,
            'converged': False,
            'final_variance': 0.0
        }

        for iteration in range(self.max_iterations):
            x_prev = x.copy()

            # Calculate competition terms
            for i in range(n):
                competition = np.sum(alpha_matrix[i] * x)
                dx_dt = growth_rates[i] * x[i] * (1.0 - competition)
                x[i] = max(0.0, x[i] + self.damping_factor * dx_dt)

            # Normalize to prevent explosion
            total_pop = np.sum(x)
            if total_pop > 0:
                x = x / total_pop * n  # Maintain total ecosystem size

            # Check convergence
            change = np.linalg.norm(x - x_prev)
            if change < self.convergence_threshold:
                convergence_data['converged'] = True
                break

            convergence_data['iterations'] = iteration + 1

        convergence_data['final_variance'] = float(np.var(x))

        return x, convergence_data

    def _build_strategy_interaction_matrix(self,
                                         candidates: list[LVReasoningCandidate]) -> np.ndarray:
        """Build interaction matrix for reasoning strategy competition."""
        n = len(candidates)
        alpha = np.zeros((n, n))

        # Calculate semantic similarities between strategies
        embeddings = []
        for candidate in candidates:
            embedding = self.embedder.encode([candidate.content])[0]
            embeddings.append(embedding)

        # Build interaction matrix based on similarity
        for i in range(n):
            for j in range(n):
                if i == j:
                    alpha[i][j] = 1.0  # Self-competition
                else:
                    # Competition strength based on semantic similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )

                    # High similarity = higher competition
                    alpha[i][j] = max(0.1, float(similarity))

        return alpha

    def _get_entropy_weights(self, entropy: float) -> dict[str, float]:
        """Get appropriate weights based on entropy level."""
        if entropy < self.entropy_profile.low_threshold:
            return self.entropy_profile.low_entropy_weights
        elif entropy < self.entropy_profile.high_threshold:
            return self.entropy_profile.medium_entropy_weights
        else:
            return self.entropy_profile.high_entropy_weights

    async def _synthesize_lv_results(self,
                                   concept: str,
                                   context: dict[str, Any],
                                   selected_strategies: list[LVReasoningCandidate],
                                   entropy: float) -> dict[str, Any]:
        """Synthesize results from selected LV strategies."""
        if not selected_strategies:
            return {"synthesis": "No strategies survived ecosystem selection"}

        # Extract key insights from each strategy
        insights = []
        truth_values = []
        evidence_items = []

        for strategy in selected_strategies:
            insights.append({
                'strategy': strategy.strategy_name,
                'pattern': strategy.reasoning_pattern,
                'content': strategy.content,
                'population_weight': strategy.population,
                'confidence': strategy.truth_confidence
            })

            truth_values.extend(strategy.truth_values)
            evidence_items.extend(strategy.evidence_items)

        # Perform ecological truth value synthesis
        synthesized_truth = self._synthesize_truth_values_ecologically(
            truth_values, [s.population for s in selected_strategies]
        )

        # Generate meta-insights about the ecosystem state
        meta_insights = self._generate_ecosystem_meta_insights(
            selected_strategies, entropy
        )

        return {
            'primary_insights': insights,
            'synthesized_truth': synthesized_truth.to_dict() if synthesized_truth else None,
            'meta_insights': meta_insights,
            'ecosystem_diversity': self._calculate_strategy_diversity(selected_strategies),
            'convergence_indicators': self._analyze_convergence_patterns(selected_strategies)
        }

    def _synthesize_truth_values_ecologically(self,
                                            truth_values: list[TruthValue],
                                            populations: list[float]) -> TruthValue | None:
        """Synthesize truth values using ecological weighting."""
        if not truth_values:
            return None

        # Weight truth values by their strategy populations
        total_weight = sum(populations)
        if total_weight == 0:
            return truth_values[0]  # Fallback

        weighted_frequency = 0.0
        weighted_confidence = 0.0

        for i, tv in enumerate(truth_values[:len(populations)]):
            weight = populations[i] / total_weight
            weighted_frequency += weight * tv.frequency
            weighted_confidence += weight * tv.confidence

        # Apply ecological synthesis (preserves diversity)
        synthesized_confidence = min(0.95, weighted_confidence * 0.9)  # Slight reduction for synthesis uncertainty

        return TruthValue(weighted_frequency, synthesized_confidence)

    def _calculate_diversity_metrics(self,
                                   strategies: list[LVReasoningCandidate]) -> dict[str, float]:
        """Calculate ecosystem diversity metrics."""
        if not strategies:
            return {"diversity": 0.0}

        # Population diversity (Shannon entropy)
        populations = [s.population for s in strategies]
        total_pop = sum(populations)

        if total_pop == 0:
            shannon_diversity = 0.0
        else:
            probs = [p / total_pop for p in populations]
            shannon_diversity = -sum(p * np.log(p) if p > 0 else 0 for p in probs)

        # Strategy pattern diversity
        patterns = [s.reasoning_pattern for s in strategies]
        pattern_diversity = len(set(patterns)) / len(patterns) if patterns else 0.0

        # Semantic diversity
        if len(strategies) > 1:
            embeddings = [self.embedder.encode([s.content])[0] for s in strategies]
            pairwise_similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    pairwise_similarities.append(sim)

            avg_similarity = np.mean(pairwise_similarities)
            semantic_diversity = 1.0 - avg_similarity
        else:
            semantic_diversity = 0.0

        return {
            "shannon_diversity": float(shannon_diversity),
            "pattern_diversity": float(pattern_diversity),
            "semantic_diversity": float(semantic_diversity),
            "overall_diversity": float(np.mean([
                float(shannon_diversity / np.log(len(strategies))) if len(strategies) > 1 else 0.0,
                float(pattern_diversity),
                float(semantic_diversity)
            ]))
        }


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Reasoning Strategy Implementations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _deductive_synthesis_strategy(self,
                                          concept: str,
                                          context: dict[str, Any],
                                          perspectives: list[str],
                                          entropy: float) -> LVReasoningCandidate | None:
        """Generate deductive synthesis reasoning strategy."""
        try:
            # Get relevant beliefs for deductive reasoning
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} deduction",
                include_categories=['logical', 'epistemological']
            )

            # Find general principles for deduction
            principles = [item for item in attention_buffer
                         if ("==>" in item.term or "-->" in item.term)
                         and item.truth.confidence > 0.7]

            if not principles:
                return None

            # Apply NARS deductive reasoning
            deductive_result = await self.nars_reasoning._deductive_reasoning(
                concept, attention_buffer
            )

            if not deductive_result:
                return None

            # Create LV candidate
            candidate = LVReasoningCandidate(
                strategy_name="deductive_synthesis",
                reasoning_pattern="deductive",
                truth_approach="synthesis",
                content=f"Deductive analysis of {concept}: {deductive_result.conclusion}",
                reasoning_result=deductive_result,
                truth_values=[deductive_result.truth],
                evidence_items=deductive_result.evidence
            )

            # Calculate strategy-specific scores
            candidate.quality_score = min(0.95, deductive_result.truth.confidence * 1.1)
            candidate.novelty_score = 0.3  # Deduction is conservative
            candidate.coherence_score = 0.9  # High logical coherence
            candidate.epistemic_value = deductive_result.truth.expectation
            candidate.truth_confidence = deductive_result.truth.confidence
            candidate.evidence_strength = len(deductive_result.evidence) / 10.0
            candidate.inference_depth = len(deductive_result.inference_path)

            return candidate

        except Exception as e:
            logger.warning(f"Deductive synthesis strategy failed: {e}")
            return None

    async def _inductive_exploration_strategy(self,
                                            concept: str,
                                            context: dict[str, Any],
                                            perspectives: list[str],
                                            entropy: float) -> LVReasoningCandidate | None:
        """Generate inductive exploration reasoning strategy."""
        try:
            # Get instances for inductive reasoning
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} instances examples",
                include_categories=['empirical', 'experimental']
            )

            # Apply NARS inductive reasoning
            inductive_result = await self.nars_reasoning._inductive_reasoning(
                concept, attention_buffer
            )

            if not inductive_result:
                return None

            candidate = LVReasoningCandidate(
                strategy_name="inductive_exploration",
                reasoning_pattern="inductive",
                truth_approach="generalization",
                content=f"Inductive exploration of {concept}: {inductive_result.conclusion}",
                reasoning_result=inductive_result,
                truth_values=[inductive_result.truth],
                evidence_items=inductive_result.evidence
            )

            # Inductive reasoning favors novelty and evidence strength
            candidate.quality_score = inductive_result.truth.confidence * 0.8
            candidate.novelty_score = min(0.9, 0.6 + entropy * 0.3)  # Higher for uncertain contexts
            candidate.coherence_score = 0.7  # Moderate logical coherence
            candidate.epistemic_value = inductive_result.truth.expectation * 0.9
            candidate.truth_confidence = inductive_result.truth.confidence
            candidate.evidence_strength = len(inductive_result.evidence) / 5.0  # Induction needs more evidence
            candidate.inference_depth = len(inductive_result.inference_path)

            return candidate

        except Exception as e:
            logger.warning(f"Inductive exploration strategy failed: {e}")
            return None

    async def _abductive_hypothesis_strategy(self,
                                           concept: str,
                                           context: dict[str, Any],
                                           perspectives: list[str],
                                           entropy: float) -> LVReasoningCandidate | None:
        """Generate abductive hypothesis reasoning strategy."""
        try:
            # Get phenomena for abductive reasoning
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} explanation hypothesis",
                include_categories=['phenomenological', 'theoretical']
            )

            # Apply NARS abductive reasoning
            abductive_result = await self.nars_reasoning._abductive_reasoning(
                concept, attention_buffer
            )

            if not abductive_result:
                return None

            candidate = LVReasoningCandidate(
                strategy_name="abductive_hypothesis",
                reasoning_pattern="abductive",
                truth_approach="hypothesis",
                content=f"Abductive hypothesis for {concept}: {abductive_result.conclusion}",
                reasoning_result=abductive_result,
                truth_values=[abductive_result.truth],
                evidence_items=abductive_result.evidence
            )

            # Abductive reasoning balances novelty and explanatory power
            candidate.quality_score = abductive_result.truth.confidence * 0.7
            candidate.novelty_score = min(0.95, 0.7 + entropy * 0.25)  # High novelty potential
            candidate.coherence_score = 0.6  # Moderate coherence (hypothetical)
            candidate.epistemic_value = abductive_result.truth.expectation * 1.1  # High epistemic value
            candidate.truth_confidence = abductive_result.truth.confidence
            candidate.evidence_strength = len(abductive_result.evidence) / 8.0
            candidate.inference_depth = len(abductive_result.inference_path)

            return candidate

        except Exception as e:
            logger.warning(f"Abductive hypothesis strategy failed: {e}")
            return None

    async def _analogical_mapping_strategy(self,
                                         concept: str,
                                         context: dict[str, Any],
                                         perspectives: list[str],
                                         entropy: float) -> LVReasoningCandidate | None:
        """Generate analogical mapping reasoning strategy."""
        try:
            # Get similar concepts for analogical reasoning
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} similar analogy",
                include_categories=['metaphysical', 'comparative']
            )

            # Apply NARS analogical reasoning
            analogical_result = await self.nars_reasoning._analogical_reasoning(
                concept, attention_buffer
            )

            if not analogical_result:
                return None

            candidate = LVReasoningCandidate(
                strategy_name="analogical_mapping",
                reasoning_pattern="analogical",
                truth_approach="similarity",
                content=f"Analogical mapping for {concept}: {analogical_result.conclusion}",
                reasoning_result=analogical_result,
                truth_values=[analogical_result.truth],
                evidence_items=analogical_result.evidence
            )

            # Analogical reasoning is creative but uncertain
            candidate.quality_score = analogical_result.truth.confidence * 0.6
            candidate.novelty_score = min(0.9, 0.8 + entropy * 0.1)  # High creativity
            candidate.coherence_score = 0.5  # Lower logical coherence
            candidate.epistemic_value = analogical_result.truth.expectation * 0.8
            candidate.truth_confidence = analogical_result.truth.confidence
            candidate.evidence_strength = len(analogical_result.evidence) / 12.0  # Weaker evidence
            candidate.inference_depth = len(analogical_result.inference_path)

            return candidate

        except Exception as e:
            logger.warning(f"Analogical mapping strategy failed: {e}")
            return None

    async def _dialectical_integration_strategy(self,
                                              concept: str,
                                              context: dict[str, Any],
                                              perspectives: list[str],
                                              entropy: float) -> LVReasoningCandidate | None:
        """Generate dialectical integration reasoning strategy."""
        try:
            # Get contradictory beliefs for dialectical reasoning
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} contradiction dialectic",
                include_categories=['ethical', 'philosophical']
            )

            # Apply NARS dialectical reasoning
            dialectical_result = await self.nars_reasoning._dialectical_reasoning(
                concept, attention_buffer
            )

            if not dialectical_result:
                return None

            candidate = LVReasoningCandidate(
                strategy_name="dialectical_integration",
                reasoning_pattern="dialectical",
                truth_approach="synthesis",
                content=f"Dialectical integration of {concept}: {dialectical_result.conclusion}",
                reasoning_result=dialectical_result,
                truth_values=[dialectical_result.truth],
                evidence_items=dialectical_result.evidence
            )

            # Dialectical reasoning excels in high-entropy contexts
            candidate.quality_score = dialectical_result.truth.confidence * 0.8
            candidate.novelty_score = min(0.95, 0.5 + entropy * 0.45)  # Entropy-adaptive novelty
            candidate.coherence_score = 0.8  # High synthesis coherence
            candidate.diversity_score = 0.9  # Inherently diverse (thesis + antithesis)
            candidate.epistemic_value = dialectical_result.truth.expectation * 1.2  # High epistemic value
            candidate.truth_confidence = dialectical_result.truth.confidence
            candidate.evidence_strength = len(dialectical_result.evidence) / 6.0
            candidate.inference_depth = len(dialectical_result.inference_path)

            return candidate

        except Exception as e:
            logger.warning(f"Dialectical integration strategy failed: {e}")
            return None

    async def _truth_value_ecology_strategy(self,
                                          concept: str,
                                          context: dict[str, Any],
                                          perspectives: list[str],
                                          entropy: float) -> LVReasoningCandidate | None:
        """Generate truth value ecology reasoning strategy."""
        try:
            # Get all relevant truth values for ecological analysis
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=concept,
                include_categories=None  # All categories
            )

            if not attention_buffer:
                return None

            # Perform ecological truth value analysis
            truth_values = [item.truth for item in attention_buffer]

            # Apply Truth.revision to combine evidence ecologically
            if len(truth_values) >= 2:
                synthesized_truth = truth_values[0]
                for tv in truth_values[1:]:
                    synthesized_truth = Truth.revision(synthesized_truth, tv)
            else:
                synthesized_truth = truth_values[0] if truth_values else TruthValue(0.5, 0.1)

            # Create ecological truth analysis
            analysis_content = self._generate_truth_ecology_analysis(
                concept, truth_values, synthesized_truth, attention_buffer
            )

            candidate = LVReasoningCandidate(
                strategy_name="truth_value_ecology",
                reasoning_pattern="ecological",
                truth_approach="revision",
                content=analysis_content,
                truth_values=truth_values,
                evidence_items=attention_buffer
            )

            # Truth value ecology emphasizes evidence integration
            candidate.quality_score = synthesized_truth.confidence
            candidate.novelty_score = 0.4 + entropy * 0.3  # Moderate novelty
            candidate.coherence_score = min(0.95, synthesized_truth.expectation)
            candidate.epistemic_value = synthesized_truth.expectation
            candidate.truth_confidence = synthesized_truth.confidence
            candidate.evidence_strength = len(attention_buffer) / 15.0
            candidate.inference_depth = 1  # Direct evidence integration

            return candidate

        except Exception as e:
            logger.warning(f"Truth value ecology strategy failed: {e}")
            return None

    async def _epistemic_revision_strategy(self,
                                         concept: str,
                                         context: dict[str, Any],
                                         perspectives: list[str],
                                         entropy: float) -> LVReasoningCandidate | None:
        """Generate epistemic revision reasoning strategy."""
        try:
            # Get temporal beliefs for revision analysis
            attention_buffer = self.nars_memory.get_attention_buffer(
                query=f"{concept} revision update",
                include_categories=['epistemological', 'temporal']
            )

            if not attention_buffer:
                return None

            # Perform temporal epistemic revision
            temporal_beliefs = [item for item in attention_buffer
                              if hasattr(item, 'occurrence_time') and
                              item.occurrence_time != "eternal"]

            if temporal_beliefs:
                # Apply temporal projection and revision
                current_time = 1.0  # Normalized current time
                projected_truths = []

                for belief in temporal_beliefs:
                    if isinstance(belief.occurrence_time, int | float):
                        projected_truth = Truth.projection(
                            belief.truth,
                            belief.occurrence_time,
                            current_time
                        )
                        projected_truths.append(projected_truth)

                # Combine with eternalization
                if projected_truths:
                    eternalized_truth = Truth.eternalization(projected_truths)
                else:
                    eternalized_truth = TruthValue(0.5, 0.2)
            else:
                eternalized_truth = TruthValue(0.5, 0.2)

            # Generate revision analysis
            revision_content = self._generate_epistemic_revision_analysis(
                concept, temporal_beliefs, eternalized_truth, context
            )

            candidate = LVReasoningCandidate(
                strategy_name="epistemic_revision",
                reasoning_pattern="temporal",
                truth_approach="eternalization",
                content=revision_content,
                truth_values=[eternalized_truth],
                evidence_items=temporal_beliefs or attention_buffer
            )

            # Epistemic revision balances stability and adaptability
            candidate.quality_score = eternalized_truth.confidence * 0.9
            candidate.novelty_score = 0.3 + entropy * 0.4  # Entropy-adaptive
            candidate.coherence_score = 0.85  # High temporal coherence
            candidate.epistemic_value = eternalized_truth.expectation * 1.1
            candidate.truth_confidence = eternalized_truth.confidence
            candidate.evidence_strength = len(temporal_beliefs) / 8.0 if temporal_beliefs else 0.2
            candidate.inference_depth = 2  # Temporal projection + eternalization

            return candidate

        except Exception as e:
            logger.warning(f"Epistemic revision strategy failed: {e}")
            return None

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helper Methods and Analysis Functions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_truth_ecology_analysis(self,
                                        concept: str,
                                        truth_values: list[TruthValue],
                                        synthesized_truth: TruthValue,
                                        evidence_items: list[MemoryItem]) -> str:
        """Generate analysis of truth value ecology."""
        if not truth_values:
            return f"No truth value ecology found for {concept}"

        # Analyze truth value distribution
        frequencies = [tv.frequency for tv in truth_values]
        confidences = [tv.confidence for tv in truth_values]

        avg_frequency = np.mean(frequencies)
        avg_confidence = np.mean(confidences)
        frequency_variance = np.var(frequencies)
        confidence_variance = np.var(confidences)

        analysis = f"""Truth Value Ecology Analysis for '{concept}':

Evidence Base: {len(evidence_items)} belief items with {len(truth_values)} truth values
Average Frequency: {avg_frequency:.3f} (variance: {frequency_variance:.3f})
Average Confidence: {avg_confidence:.3f} (variance: {confidence_variance:.3f})

Synthesized Truth: frequency={synthesized_truth.frequency:.3f}, confidence={synthesized_truth.confidence:.3f}
Expectation: {synthesized_truth.expectation:.3f}

Ecological Assessment: The truth value ecosystem for {concept} shows """

        if frequency_variance < 0.1:
            analysis += "high consensus on truth frequency, "
        else:
            analysis += "significant disagreement on truth frequency, "

        if confidence_variance < 0.1:
            analysis += "consistent confidence levels across evidence."
        else:
            analysis += "varying confidence levels indicating epistemic uncertainty."

        return analysis

    def _generate_epistemic_revision_analysis(self,
                                            concept: str,
                                            temporal_beliefs: list[MemoryItem],
                                            eternalized_truth: TruthValue,
                                            context: dict[str, Any]) -> str:
        """Generate analysis of epistemic revision."""
        if not temporal_beliefs:
            return f"No temporal beliefs found for epistemic revision of {concept}"

        analysis = f"""Epistemic Revision Analysis for '{concept}':

Temporal Beliefs: {len(temporal_beliefs)} time-indexed beliefs
Eternalized Truth: frequency={eternalized_truth.frequency:.3f}, confidence={eternalized_truth.confidence:.3f}

Revision Process: Applied temporal projection and evidence consolidation
Epistemic Status: """

        if eternalized_truth.confidence > 0.7:
            analysis += "High confidence in revised belief"
        elif eternalized_truth.confidence > 0.4:
            analysis += "Moderate confidence with room for further revision"
        else:
            analysis += "Low confidence indicating need for additional evidence"

        analysis += f"\n\nEpistemic Implications: The concept {concept} shows "

        if eternalized_truth.frequency > 0.7:
            analysis += "strong positive support through temporal revision."
        elif eternalized_truth.frequency < 0.3:
            analysis += "weak or negative support through temporal revision."
        else:
            analysis += "balanced or uncertain status requiring further investigation."

        return analysis

    def _generate_ecosystem_meta_insights(self,
                                        selected_strategies: list[LVReasoningCandidate],
                                        entropy: float) -> list[str]:
        """Generate meta-insights about the reasoning ecosystem."""
        insights = []

        if not selected_strategies:
            return ["No reasoning strategies survived ecosystem selection"]

        # Analyze strategy diversity
        patterns = [s.reasoning_pattern for s in selected_strategies]
        unique_patterns = set(patterns)

        if len(unique_patterns) == 1:
            insights.append(f"Ecosystem converged to {patterns[0]} reasoning pattern")
        else:
            insights.append(f"Ecosystem maintained {len(unique_patterns)} diverse reasoning patterns: {', '.join(unique_patterns)}")

        # Analyze population distribution
        populations = [s.population for s in selected_strategies]
        max_pop = max(populations)
        min_pop = min(populations)

        if max_pop / min_pop > 3.0:
            insights.append("Strong ecological stratification with dominant strategy")
        else:
            insights.append("Balanced ecological distribution across strategies")

        # Analyze entropy-strategy relationship
        if entropy > 0.6:
            high_novelty_strategies = [s for s in selected_strategies if s.novelty_score > 0.6]
            insights.append(f"High entropy context ({entropy:.3f}) favored {len(high_novelty_strategies)} high-novelty strategies")
        else:
            conservative_strategies = [s for s in selected_strategies if s.quality_score > 0.7]
            insights.append(f"Low entropy context ({entropy:.3f}) favored {len(conservative_strategies)} conservative strategies")

        # Analyze epistemic value
        avg_epistemic_value = np.mean([s.epistemic_value for s in selected_strategies])
        if avg_epistemic_value > 0.7:
            insights.append("High overall epistemic value indicates robust philosophical analysis")
        else:
            insights.append("Moderate epistemic value suggests need for additional evidence or perspectives")

        return insights

    def _calculate_strategy_diversity(self, strategies: list[LVReasoningCandidate]) -> float:
        """Calculate diversity among selected strategies."""
        if len(strategies) <= 1:
            return 0.0

        # Pattern diversity
        patterns = [s.reasoning_pattern for s in strategies]
        pattern_diversity = len(set(patterns)) / len(patterns)

        # Population entropy
        populations = [s.population for s in strategies]
        total_pop = sum(populations)
        if total_pop > 0:
            probs = [p / total_pop for p in populations]
            population_entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
            normalized_entropy = population_entropy / np.log(len(strategies))
        else:
            normalized_entropy = 0.0

        # Content diversity (semantic)
        if len(strategies) > 1:
            embeddings = [self.embedder.encode([s.content])[0] for s in strategies]
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            content_diversity = 1.0 - np.mean(similarities)
        else:
            content_diversity = 0.0

        # Combined diversity score
        return float(np.mean(np.array([pattern_diversity, normalized_entropy, content_diversity], dtype=float)))

    def _analyze_convergence_patterns(self, strategies: list[LVReasoningCandidate]) -> dict[str, Any]:
        """Analyze convergence patterns in the ecosystem."""
        if not strategies:
            return {"status": "no_strategies"}

        populations = [s.population for s in strategies]

        # Gini coefficient for population inequality
        sorted_pops = sorted(populations)
        n = len(sorted_pops)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_pops)) / (n * np.sum(sorted_pops)) - (n + 1) / n

        # Concentration analysis
        top_strategy_share = max(populations) / sum(populations) if sum(populations) > 0 else 0

        # Convergence assessment
        if gini > 0.5:
            convergence_status = "high_concentration"
        elif gini > 0.3:
            convergence_status = "moderate_concentration"
        else:
            convergence_status = "balanced_distribution"

        return {
            "gini_coefficient": float(gini),
            "top_strategy_share": float(top_strategy_share),
            "convergence_status": convergence_status,
            "diversity_preserved": gini < 0.4,
            "ecosystem_health": "healthy" if 0.2 < gini < 0.6 else "imbalanced"
        }

    def _extract_epistemic_implications(self, strategies: list[LVReasoningCandidate]) -> list[str]:
        """Extract epistemic implications from selected strategies."""
        implications = []

        if not strategies:
            return ["No strategies available for epistemic analysis"]

        # Analyze reasoning pattern implications
        patterns = [s.reasoning_pattern for s in strategies]

        if "deductive" in patterns:
            implications.append("Deductive reasoning provides logical necessity but limited novelty")

        if "inductive" in patterns:
            implications.append("Inductive reasoning offers empirical grounding with probabilistic conclusions")

        if "abductive" in patterns:
            implications.append("Abductive reasoning generates explanatory hypotheses requiring further validation")

        if "dialectical" in patterns:
            implications.append("Dialectical reasoning reveals tensions and enables higher-order synthesis")

        if "analogical" in patterns:
            implications.append("Analogical reasoning transfers insights across domains with structural similarity")

        # Analyze truth confidence distribution
        confidences = [s.truth_confidence for s in strategies]
        avg_confidence = np.mean(confidences)

        if avg_confidence > 0.8:
            implications.append("High average confidence suggests strong evidential support")
        elif avg_confidence < 0.4:
            implications.append("Low average confidence indicates epistemic uncertainty requiring caution")
        else:
            implications.append("Moderate confidence levels suggest balanced but incomplete understanding")

        # Analyze evidence strength
        evidence_strengths = [s.evidence_strength for s in strategies]
        total_evidence = sum(evidence_strengths)

        if total_evidence > 2.0:
            implications.append("Strong evidence base supports reliable philosophical conclusions")
        else:
            implications.append("Limited evidence base suggests need for additional investigation")

        return implications

    async def _fallback_analysis(self,
                                concept: str,
                                context: dict[str, Any],
                                perspectives: list[str]) -> dict[str, Any]:
        """Provide fallback analysis when LV enhancement fails."""
        try:
            # Use standard NARS reasoning as fallback
            standard_result = await self.nars_reasoning.analyze_concept(
                concept, context.get('domain', ''), perspectives
            )

            return {
                "fallback_type": "standard_nars",
                "analysis": standard_result,
                "note": "LV enhancement failed, using standard NARS analysis"
            }

        except Exception as e:
            # Ultimate fallback
            return {
                "fallback_type": "minimal",
                "analysis": {
                    "concept": concept,
                    "basic_response": f"Analysis of {concept} in context {context.get('domain', 'general')}",
                    "perspectives": perspectives,
                    "error": str(e)
                },
                "note": "Both LV and standard NARS analysis failed"
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration Interface for NeoCoder LV Framework
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LVNARSIntegrationManager:
    """
    High-level interface for integrating LV-enhanced NARS reasoning
    with the broader NeoCoder ecosystem.
    """

    def __init__(self,
                 nars_manager: NARSManager,
                 nars_memory: NARSMemory,
                 nars_reasoning: NARSReasoning,
                 neo4j_session=None,
                 qdrant_client=None):
        """Initialize the integration manager."""
        self.lv_nars = LVNARSEcosystem(
            nars_manager, nars_memory, nars_reasoning
        )
        self.neo4j = neo4j_session
        self.qdrant = qdrant_client

        logger.info("LV-NARS Integration Manager initialized")

    async def enhanced_philosophical_reasoning(self,
                                             query: str,
                                             context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Main entry point for LV-enhanced philosophical reasoning.

        Determines whether to use LV enhancement based on entropy analysis,
        then applies appropriate reasoning strategies.
        """
        # Estimate entropy
        entropy = self.lv_nars.entropy_estimator.estimate_philosophical_entropy(
            query, context
        )

        # Determine enhancement strategy
        if entropy > 0.4:
            logger.info(f"High entropy ({entropy:.3f}) - applying LV enhancement")

            # Extract philosophical concept and perspectives
            concept = self._extract_main_concept(query)
            perspectives = self._determine_perspectives(query, context)

            # Apply LV-enhanced analysis
            result = await self.lv_nars.enhanced_philosophical_analysis(
                concept, context or {}, perspectives
            )

            result['enhancement_applied'] = True
            result['enhancement_reason'] = f"High entropy ({entropy:.3f}) warranted diversity preservation"

        else:
            logger.info(f"Low entropy ({entropy:.3f}) - using standard reasoning")

            # Use standard NARS reasoning
            concept = self._extract_main_concept(query)
            perspectives = self._determine_perspectives(query, context)

            result = await self.lv_nars.nars_reasoning.analyze_concept(
                concept, context.get('domain', '') if context else '', perspectives
            )

            result['enhancement_applied'] = False
            result['enhancement_reason'] = f"Low entropy ({entropy:.3f}) - standard reasoning sufficient"

        # Store results if database connections available
        if self.neo4j or self.qdrant:
            await self._store_reasoning_results(query, result, entropy)

        return result

    def _extract_main_concept(self, query: str) -> str:
        """Extract the main philosophical concept from query."""
        # Simple extraction - in production, use NLP
        query_words = query.lower().split()

        # Remove common question words
        filter_words = {'what', 'is', 'how', 'why', 'where', 'when', 'who', 'the', 'a', 'an'}
        concept_words = [w for w in query_words if w not in filter_words]

        if concept_words:
            return concept_words[0]  # Take first significant word
        else:
            return query.split()[0] if query.split() else "concept"

    def _determine_perspectives(self, query: str, context: dict[str, Any] | None) -> list[str]:
        """Determine relevant philosophical perspectives."""
        if context and 'perspectives' in context:
            return context['perspectives']

        # Default perspectives based on query content
        query_lower = query.lower()
        perspectives = []

        if any(term in query_lower for term in ['mind', 'consciousness', 'experience']):
            perspectives.append('phenomenological')

        if any(term in query_lower for term in ['logic', 'reason', 'argument']):
            perspectives.append('analytical')

        if any(term in query_lower for term in ['good', 'right', 'moral', 'ethics']):
            perspectives.append('ethical')

        if any(term in query_lower for term in ['reality', 'existence', 'being']):
            perspectives.append('metaphysical')

        if any(term in query_lower for term in ['know', 'truth', 'belief']):
            perspectives.append('epistemological')

        # Default perspectives if none detected
        if not perspectives:
            perspectives = ['analytical', 'phenomenological']

        return perspectives

    async def _store_reasoning_results(self,
                                     query: str,
                                     result: dict[str, Any],
                                     entropy: float) -> None:
        """Store reasoning results in Neo4j and Qdrant."""
        try:
            # Store in Neo4j if available
            if self.neo4j:
                # Create reasoning session node
                await self._store_in_neo4j(query, result, entropy)

            # Store in Qdrant if available
            if self.qdrant:
                await self._store_in_qdrant(query, result, entropy)

        except Exception as e:
            logger.warning(f"Failed to store reasoning results: {e}")

    async def _store_in_neo4j(self, query: str, result: dict[str, Any], entropy: float) -> None:
        """Store reasoning results in Neo4j knowledge graph."""
        # Implementation would depend on specific Neo4j schema
        # This is a placeholder for the storage logic
        pass

    async def _store_in_qdrant(self, query: str, result: dict[str, Any], entropy: float) -> None:
        """Store reasoning results in Qdrant vector database."""
        # Implementation would depend on specific Qdrant collection schema
        # This is a placeholder for the storage logic
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Truth Value Functions Enhanced with LV Dynamics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LVTruthFunctions:
    """
    Enhanced NARS truth functions that incorporate LV ecosystem dynamics
    for diversity-preserving truth value synthesis.
    """

    @staticmethod
    def ecological_revision(truth_values: list[TruthValue],
                           populations: list[float]) -> TruthValue:
        """
        Revise truth values using ecological weighting from LV populations.

        Unlike standard NARS revision, this preserves minority perspectives
        based on their ecological fitness rather than pure evidence strength.
        """
        if not truth_values or not populations:
            return TruthValue(0.5, 0.0)

        if len(truth_values) != len(populations):
            # Fallback to standard revision
            result = truth_values[0]
            for tv in truth_values[1:]:
                result = Truth.revision(result, tv)
            return result

        # Ecological weighting
        total_population = sum(populations)
        if total_population == 0:
            return truth_values[0]

        # Weighted frequency and confidence
        weighted_freq = sum(tv.frequency * (pop / total_population)
                           for tv, pop in zip(truth_values, populations, strict=False))

        # Confidence synthesis with diversity preservation
        individual_confidences = [tv.confidence * (pop / total_population)
                                for tv, pop in zip(truth_values, populations, strict=False)]

        # Apply diversity bonus to confidence (prevents overconfidence)
        population_variance = np.var(populations) / (np.mean(populations) or 1)
        diversity_factor = min(1.0, 0.8 + 0.2 * float(population_variance))

        ecological_confidence = sum(individual_confidences) * diversity_factor

        return TruthValue(
            np.clip(weighted_freq, 0.0, 1.0),
            np.clip(ecological_confidence, 0.0, 0.95)  # Cap at 0.95 for epistemic humility
        )

    @staticmethod
    def diversity_preserving_synthesis(truth_populations: list[tuple[TruthValue, float]]) -> TruthValue:
        """
        Synthesize truth values while explicitly preserving diversity.

        This method ensures that minority viewpoints contribute to the final
        truth value proportionally to their ecological fitness.
        """
        if not truth_populations:
            return TruthValue(0.5, 0.0)

        truth_values, populations = zip(*truth_populations, strict=False)

        # Apply ecological revision
        return LVTruthFunctions.ecological_revision(
            list(truth_values), list(populations)
        )

    @staticmethod
    def epistemic_uncertainty_synthesis(candidates: list[LVReasoningCandidate]) -> TruthValue:
        """
        Synthesize truth values from LV reasoning candidates with explicit uncertainty tracking.
        """
        if not candidates:
            return TruthValue(0.5, 0.0)

        # Extract truth values and weights
        truth_values = []
        weights = []

        for candidate in candidates:
            if candidate.truth_values:
                truth_values.extend(candidate.truth_values)
                # Weight by population * epistemic value
                weight = candidate.population * candidate.epistemic_value
                weights.extend([weight] * len(candidate.truth_values))

        if not truth_values:
            return TruthValue(0.5, 0.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total_weight for w in weights]

        # Weighted synthesis
        weighted_frequency = sum(tv.frequency * w
                               for tv, w in zip(truth_values, normalized_weights, strict=False))

        # Uncertainty-aware confidence
        weighted_confidence = sum(tv.confidence * w
                                for tv, w in zip(truth_values, normalized_weights, strict=False))

        # Apply epistemic humility reduction
        epistemic_confidence = weighted_confidence * 0.95  # 5% uncertainty for synthesis

        return TruthValue(
            np.clip(weighted_frequency, 0.0, 1.0),
            np.clip(epistemic_confidence, 0.0, 0.95)
        )
