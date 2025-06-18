"""
utils.py - Utility Functions

This module contains stateless helper functions that are used across the entire
NARS Philosophical Reasoning Engine. These utilities handle common, reusable tasks
like semantic similarity calculation, Narsese conversion, and context mapping.
"""
from __future__ import annotations
import re
from typing import List
import numpy as np

# Import the specific type alias from our types module.
from .types import Embedding

def calculate_semantic_similarity(emb1: Embedding, emb2: Embedding) -> float:
    """
    Calculates the cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two vectors,
    providing a score of how similar they are in direction (i.e., meaning).
    A value of 1.0 means identical, 0.0 means unrelated (orthogonal), and
    -1.0 means diametrically opposed.

    Args:
        emb1: The first embedding vector.
        emb2: The second embedding vector.

    Returns:
        The cosine similarity score, or 0.0 if either embedding is invalid
        or has a zero norm, to prevent division by zero errors.
    """
    if emb1 is None or emb2 is None:
        return 0.0
        
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # The dot product of unit vectors is their cosine similarity.
    return np.dot(emb1, emb2) / (norm1 * norm2)

def context_to_categories(context: str) -> List[str]:
    """
    Maps a natural language context string to a list of predefined philosophical
    categories using keyword matching.

    Args:
        context: The context string to analyze (e.g., "the nature of knowledge").

    Returns:
        A list of matching philosophical categories (e.g., ["epistemological"]).
        Returns a default list if no keywords match.
    """
    context_lower = context.lower()
    
    # These mappings can be expanded to make the system more knowledgeable.
    category_mappings = {
        "metaphysical": ["existence", "reality", "being", "ontology", "realism", "idealism", "material"],
        "epistemological": ["knowledge", "belief", "truth", "justification", "science", "certainty"],
        "ethical": ["ethics", "morality", "values", "good", "evil", "right", "wrong", "justice"],
        "phenomenological": ["consciousness", "experience", "mind", "perception", "awareness", "qualia"],
        "logical": ["logic", "reasoning", "inference", "formal", "validity", "soundness"]
    }

    categories = [
        cat for cat, keywords in category_mappings.items()
        if any(keyword in context_lower for keyword in keywords)
    ]
    
    # If no specific category is found, default to a general one.
    return categories if categories else ["metaphysical"]

def hypothesis_to_narsese(hypothesis: str) -> str:
    """
    Converts a simple natural language hypothesis into a Narsese statement
    using a set of common linguistic patterns.

    Note:
        This is a pattern-based converter, not a full NLP parser. It is
        designed to handle common philosophical statement structures.

    Args:
        hypothesis: The hypothesis in natural language.

    Returns:
        A plausible Narsese representation of the hypothesis.
    """
    text = hypothesis.lower().strip().replace(" a ", " ").replace(" an ", " ")
    
    # Pattern 1: "X is Y" (Inheritance) -> <x --> y>
    # e.g., "A cat is an animal" -> <cat --> animal>
    match = re.match(r'^(.+?)\s+is\s+(.+)$', text)
    if match:
        subject, predicate = [s.strip().replace(' ', '_') for s in match.groups()]
        return f"<{subject} --> {predicate}>"

    # Pattern 2: "X causes Y" (Implication) -> <x =/> y>
    # e.g., "Fire causes heat" -> <fire =/> heat>
    match = re.match(r'^(.+?)\s+causes\s+(.+)$', text)
    if match:
        cause, effect = [s.strip().replace(' ', '_') for s in match.groups()]
        return f"<{cause} =/> {effect}>"
        
    # Pattern 3: "All X are Y" (Universal Inheritance) -> <x --> y>
    # e.g., "All humans are mortal" -> <human --> mortal>
    match = re.match(r'^all\s+(.+?)\s+are\s+(.+)$', text)
    if match:
        subject, predicate = [s.strip().replace(' ', '_') for s in match.groups()]
        return f"<{subject} --> {predicate}>"

    # Default case: Treat the statement as a concept being asserted.
    # e.g., "The world is deterministic" -> <the_world_is_deterministic --> true>
    term = text.replace(' ', '_')
    return f"<{term} --> true>"