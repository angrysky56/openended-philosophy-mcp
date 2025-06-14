"""
Utility Functions for OpenEnded Philosophy Framework
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module provides supporting utilities for the philosophical
framework, including logging configuration, uncertainty calculations,
and semantic analysis helpers.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_logging(name: str) -> logging.Logger:
    """
    Configure logging with philosophical formatting.

    ### Format Design

    The logging format emphasizes clarity and traceability, including
    timestamps, levels, and source locations to aid in understanding
    the system's reasoning process.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler with philosophical formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Detailed format for debugging epistemic processes
    format_string = (
        "%(asctime)s | %(name)s | %(levelname)s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Optional file handler for persistence
    log_dir = Path("logs")
    if log_dir.exists():
        file_handler = logging.FileHandler(
            log_dir / f"philosophy_{datetime.now():%Y%m%d}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_epistemic_uncertainty(
    evidence_count: int,
    coherence_score: float,
    temporal_factor: float = 1.0,
    domain_complexity: float = 0.5
) -> float:
    """
    Calculate epistemic uncertainty for a given inference.

    ### Uncertainty Model

    U(e,c,t,d) = base_uncertainty × (1/√(1+e)) × (2-c) × t × (1+d)

    Where:
    - e: evidence count
    - c: coherence score [0,1]
    - t: temporal decay factor
    - d: domain complexity [0,1]

    Args:
        evidence_count: Number of supporting evidence items
        coherence_score: Coherence measure [0,1]
        temporal_factor: Time-based decay factor
        domain_complexity: Inherent domain difficulty [0,1]

    Returns:
        Uncertainty value [0,1] where 1 is maximum uncertainty
    """
    base_uncertainty = 0.2  # Irreducible uncertainty

    # Evidence factor - more evidence reduces uncertainty
    evidence_factor = 1.0 / np.sqrt(1.0 + evidence_count)

    # Coherence factor - higher coherence reduces uncertainty
    coherence_factor = 2.0 - coherence_score

    # Combine factors
    uncertainty = (
        base_uncertainty *
        evidence_factor *
        coherence_factor *
        temporal_factor *
        (1.0 + domain_complexity)
    )

    return np.clip(uncertainty, 0.0, 1.0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def semantic_similarity(
    concept1: Dict[str, Any],
    concept2: Dict[str, Any],
    method: str = "jaccard"
) -> float:
    """
    Calculate semantic similarity between concepts.

    ### Methods

    - jaccard: Intersection over union of features
    - cosine: Cosine similarity of feature vectors
    - wittgenstein: Family resemblance calculation

    Args:
        concept1: First concept representation
        concept2: Second concept representation
        method: Similarity calculation method

    Returns:
        Similarity score [0,1]
    """
    if method == "jaccard":
        return _jaccard_similarity(concept1, concept2)
    elif method == "cosine":
        return _cosine_similarity(concept1, concept2)
    elif method == "wittgenstein":
        return _family_resemblance(concept1, concept2)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def _jaccard_similarity(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """Calculate Jaccard similarity."""
    features1 = set(c1.get('features', []))
    features2 = set(c2.get('features', []))

    if not features1 and not features2:
        return 0.0

    intersection = len(features1 & features2)
    union = len(features1 | features2)

    return intersection / union if union > 0 else 0.0

def _cosine_similarity(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """Calculate cosine similarity."""
    # Create feature vectors
    all_features = sorted(set(c1.get('features', [])) | set(c2.get('features', [])))

    if not all_features:
        return 0.0

    vec1 = np.array([1.0 if f in c1.get('features', []) else 0.0 for f in all_features])
    vec2 = np.array([1.0 if f in c2.get('features', []) else 0.0 for f in all_features])

    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    return dot_product / norm_product if norm_product > 0 else 0.0

def _family_resemblance(c1: Dict[str, Any], c2: Dict[str, Any]) -> float:
    """
    Calculate Wittgensteinian family resemblance.

    Family resemblance doesn't require all features to match,
    but looks for overlapping similarities across multiple dimensions.
    """
    dimensions = ['features', 'contexts', 'uses', 'relations']
    similarities = []

    for dim in dimensions:
        dim1 = set(c1.get(dim, []))
        dim2 = set(c2.get(dim, []))

        if dim1 or dim2:
            overlap = len(dim1 & dim2)
            total = len(dim1 | dim2)
            similarities.append(overlap / total if total > 0 else 0.0)

    # Family resemblance is average across dimensions
    return float(np.mean(similarities)) if similarities else 0.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def coherence_metrics(
    propositions: List[Dict[str, Any]],
    relations: List[Tuple[int, int, str]]
) -> Dict[str, float]:
    """
    Calculate coherence metrics for a set of propositions.

    ### Coherence Theory

    Based on Thagard's coherence theory, we calculate:
    - Constraint satisfaction
    - Explanatory breadth
    - Analogical fit

    Args:
        propositions: List of proposition dictionaries
        relations: List of (prop1_idx, prop2_idx, relation_type) tuples

    Returns:
        Dictionary of coherence metrics
    """
    n_props = len(propositions)

    if n_props == 0:
        return {
            'constraint_satisfaction': 0.0,
            'explanatory_breadth': 0.0,
            'analogical_fit': 0.0,
            'overall_coherence': 0.0
        }

    # Build relation matrix
    positive_matrix = np.zeros((n_props, n_props))
    negative_matrix = np.zeros((n_props, n_props))

    for p1_idx, p2_idx, rel_type in relations:
        if rel_type in ['supports', 'explains', 'analogous']:
            positive_matrix[p1_idx, p2_idx] = 1.0
            positive_matrix[p2_idx, p1_idx] = 1.0
        elif rel_type in ['contradicts', 'competes']:
            negative_matrix[p1_idx, p2_idx] = 1.0
            negative_matrix[p2_idx, p1_idx] = 1.0

    # Constraint satisfaction
    positive_satisfied = np.sum(positive_matrix) / 2  # Divide by 2 for symmetry
    negative_violated = np.sum(negative_matrix) / 2
    total_constraints = positive_satisfied + negative_violated

    if total_constraints > 0:
        constraint_satisfaction = positive_satisfied / total_constraints
    else:
        constraint_satisfaction = 1.0  # No constraints = perfect satisfaction

    # Explanatory breadth
    explained_props = set()
    for _p1_idx, p2_idx, rel_type in relations:
        if rel_type == 'explains':
            explained_props.add(p2_idx)

    explanatory_breadth = len(explained_props) / n_props

    # Analogical fit
    analogical_relations = sum(1 for _, _, rel_type in relations if rel_type == 'analogous')
    max_analogies = n_props * (n_props - 1) / 2
    analogical_fit = analogical_relations / max_analogies if max_analogies > 0 else 0.0

    # Overall coherence
    overall = (
        0.5 * constraint_satisfaction +
        0.3 * explanatory_breadth +
        0.2 * analogical_fit
    )

    return {
        'constraint_satisfaction': constraint_satisfaction,
        'explanatory_breadth': explanatory_breadth,
        'analogical_fit': analogical_fit,
        'overall_coherence': overall
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pragmatic_evaluation(
    solution: Dict[str, Any],
    problem_context: Dict[str, Any],
    criteria: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Evaluate solution quality from pragmatist perspective.

    ### Pragmatist Criteria

    Following Dewey and James, we evaluate based on:
    - Practical consequences
    - Problem-solving efficacy
    - Adaptability to context
    - Simplicity/elegance

    Args:
        solution: Proposed solution
        problem_context: Problem description and constraints
        criteria: Optional weighted criteria (default: equal weights)

    Returns:
        Evaluation results with scores and recommendations
    """
    if criteria is None:
        criteria = {
            'efficacy': 0.4,
            'adaptability': 0.3,
            'simplicity': 0.2,
            'consequences': 0.1
        }

    scores = {}

    # Efficacy - does it solve the problem?
    scores['efficacy'] = _evaluate_efficacy(solution, problem_context)

    # Adaptability - can it handle variations?
    scores['adaptability'] = _evaluate_adaptability(solution, problem_context)

    # Simplicity - Occam's razor
    scores['simplicity'] = _evaluate_simplicity(solution)

    # Consequences - what are the impacts?
    scores['consequences'] = _evaluate_consequences(solution, problem_context)

    # Weighted overall score
    overall_score = sum(
        scores[criterion] * weight
        for criterion, weight in criteria.items()
    )

    # Generate recommendations
    recommendations = _generate_pragmatic_recommendations(scores, solution)

    return {
        'scores': scores,
        'overall': overall_score,
        'recommendations': recommendations,
        'pragmatic_value': overall_score > 0.7
    }

def _evaluate_efficacy(solution: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Evaluate problem-solving efficacy."""
    # Check if solution addresses all problem requirements
    requirements = context.get('requirements', [])
    if not requirements:
        return 0.5

    addressed = sum(
        1 for req in requirements
        if req in solution.get('addresses', [])
    )

    return addressed / len(requirements)

def _evaluate_adaptability(solution: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Evaluate solution adaptability."""
    # Check flexibility indicators
    flexibility_score = 0.0

    if solution.get('parameterizable', False):
        flexibility_score += 0.3

    if solution.get('modular', False):
        flexibility_score += 0.3

    if solution.get('context_sensitive', False):
        flexibility_score += 0.4

    return flexibility_score

def _evaluate_simplicity(solution: Dict[str, Any]) -> float:
    """Evaluate solution simplicity."""
    # Inverse of complexity
    complexity_indicators = [
        'dependencies',
        'assumptions',
        'special_cases',
        'parameters'
    ]

    complexity = sum(
        len(solution.get(indicator, []))
        for indicator in complexity_indicators
    )

    # Simple sigmoid to map complexity to simplicity score
    return 1.0 / (1.0 + 0.1 * complexity)

def _evaluate_consequences(solution: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Evaluate solution consequences."""
    positive = len(solution.get('benefits', []))
    negative = len(solution.get('risks', []))

    if positive + negative == 0:
        return 0.5

    return positive / (positive + negative)

def _generate_pragmatic_recommendations(
    scores: Dict[str, float],
    solution: Dict[str, Any]
) -> List[str]:
    """Generate recommendations based on pragmatic evaluation."""
    recommendations = []

    if scores['efficacy'] < 0.7:
        recommendations.append(
            "Consider enhancing solution to address more requirements"
        )

    if scores['adaptability'] < 0.5:
        recommendations.append(
            "Increase solution flexibility through parameterization or modularity"
        )

    if scores['simplicity'] < 0.5:
        recommendations.append(
            "Simplify solution by reducing dependencies or assumptions"
        )

    if scores['consequences'] < 0.6:
        recommendations.append(
            "Mitigate identified risks or enhance benefits"
        )

    return recommendations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_philosophical_output(
    result: Dict[str, Any],
    style: str = "academic"
) -> str:
    """
    Format philosophical analysis results for presentation.

    ### Styles

    - academic: Formal philosophical discourse
    - accessible: Plain language explanation
    - dialogue: Socratic dialogue format

    Args:
        result: Analysis results
        style: Output formatting style

    Returns:
        Formatted string representation
    """
    if style == "academic":
        return _format_academic(result)
    elif style == "accessible":
        return _format_accessible(result)
    elif style == "dialogue":
        return _format_dialogue(result)
    else:
        return json.dumps(result, indent=2)

def _format_academic(result: Dict[str, Any]) -> str:
    """Format in academic philosophical style."""
    output = []

    # Title
    if 'concept' in result:
        output.append(f"### Analysis of '{result['concept']}'\n")
    elif 'phenomenon' in result:
        output.append(f"### Investigation of '{result['phenomenon']}'\n")

    # Abstract
    if 'synthesis' in result:
        output.append("**Abstract**")
        output.append(f"{result['synthesis'].get('summary', 'No summary available.')}\n")

    # Main analysis
    if 'analyses' in result:
        output.append("**Multi-Perspectival Analysis**")
        for analysis in result['analyses']:
            output.append(f"\n*{analysis['perspective']} Perspective*")
            output.append(f"Confidence: {analysis['confidence']:.2f}")
            output.append(f"{analysis['interpretation']}")

    # Epistemic status
    if 'epistemic_status' in result:
        output.append(f"\n**Epistemic Status**: {result['epistemic_status']}")

    # Tensions
    if 'tensions' in result and result['tensions']:
        output.append("\n**Identified Tensions**")
        for tension in result['tensions']:
            output.append(f"- {tension}")

    return "\n".join(output)

def _format_accessible(result: Dict[str, Any]) -> str:
    """Format in accessible plain language."""
    output = []

    # Simple heading
    if 'concept' in result:
        output.append(f"Let's think about: {result['concept']}\n")

    # Main insights
    if 'primary_insights' in result:
        output.append("Here's what we found:")
        for i, insight in enumerate(result['primary_insights'], 1):
            output.append(f"\n{i}. {insight['content']}")
            output.append(f"   (We're {insight['confidence']*100:.0f}% confident about this)")

    # Contradictions in simple terms
    if 'contradictions' in result and result['contradictions']:
        output.append("\nSome interesting tensions:")
        for contradiction in result['contradictions']:
            output.append(f"- {contradiction}")

    # Simple recommendations
    if 'further_questions' in result:
        output.append("\nQuestions to explore next:")
        for question in result['further_questions']:
            output.append(f"- {question}")

    return "\n".join(output)

def _format_dialogue(result: Dict[str, Any]) -> str:
    """Format as Socratic dialogue."""
    output = []

    output.append("**Philosopher**: What shall we examine today?")

    if 'concept' in result:
        output.append(f"\n**Inquirer**: I'm curious about {result['concept']}.")

    if 'analyses' in result:
        output.append("\n**Philosopher**: Let's consider multiple perspectives...")

        for analysis in result['analyses']:
            output.append(f"\n**{analysis['perspective']} Voice**: {analysis['interpretation']}")
            output.append("**Inquirer**: How certain can we be of this view?")
            output.append(f"**Philosopher**: About {analysis['confidence']*100:.0f}% confident, given our current understanding.")

    if 'tensions' in result and result['tensions']:
        output.append("\n**Inquirer**: I notice some tensions here...")
        output.append(f"**Philosopher**: Indeed! {result['tensions'][0]}")

    output.append("\n**Philosopher**: What questions does this raise for you?")

    return "\n".join(output)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_analysis_state(
    analysis: Dict[str, Any],
    filepath: Path,
    include_metadata: bool = True
) -> None:
    """
    Save analysis state for future reference.

    Args:
        analysis: Analysis results to save
        filepath: Path to save file
        include_metadata: Whether to include metadata
    """
    state = {
        'analysis': analysis,
        'saved_at': datetime.now().isoformat()
    }

    if include_metadata:
        state['metadata'] = {
            'version': '0.1.0',
            'framework': 'OpenEnded Philosophy',
            'epistemic_warranty': 'Results subject to revision'
        }

    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

def load_analysis_state(filepath: Path) -> Dict[str, Any]:
    """
    Load previously saved analysis state.

    Args:
        filepath: Path to saved analysis

    Returns:
        Analysis state dictionary
    """
    with open(filepath, 'r') as f:
        state = json.load(f)

    # Check version compatibility
    if state.get('metadata', {}).get('version') != '0.1.0':
        logger = logging.getLogger(__name__)
        logger.warning("Loading analysis from different framework version")

    return state['analysis']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# End of utils.py
