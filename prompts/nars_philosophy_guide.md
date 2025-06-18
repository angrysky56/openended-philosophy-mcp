# NARS-Integrated Philosophy Tools Usage Guide

## Overview

The OpenEnded Philosophy MCP Server now includes deep integration with NARS (Non-Axiomatic Reasoning System), enhancing philosophical reasoning with formal truth maintenance, temporal logic, and evidence-based belief revision.

## Enhanced Tools with NARS

### 1. analyze_concept_with_nars

Analyzes philosophical concepts using both semantic analysis and NARS reasoning.

**Example Usage:**
```
analyze_concept_with_nars({
  "concept": "consciousness",
  "context": "philosophy of mind",
  "use_nars": true,
  "truth_threshold": 0.7
})
```

**What NARS Adds:**
- Truth values for beliefs about the concept
- Evidence tracking for each assertion
- Automatic contradiction detection
- Temporal evolution of concept understanding

### 2. reason_about_belief

Uses NARS to perform formal reasoning about philosophical beliefs.

**Example Usage:**
```
reason_about_belief({
  "belief": "All humans are mortal",
  "query": "Is Socrates mortal?",
  "context": ["Socrates is human"],
  "inference_depth": 5
})
```

**NARS Features:**
- Multi-step inference chains
- Confidence propagation
- Evidence combination
- Revision when contradictions found

### 3. explore_philosophical_paradox

Examines paradoxes using NARS's paraconsistent logic capabilities.

**Example Usage:**
```
explore_philosophical_paradox({
  "paradox": "Ship of Theseus",
  "perspectives": ["mereological", "identity", "temporal"],
  "use_nars_revision": true
})
```

**Benefits:**
- Handles contradictory beliefs gracefully
- Tracks confidence in competing views
- Suggests resolution strategies

## NARS-Specific Features

### Truth Values

Every belief in NARS has two dimensions:
- **Frequency**: How often something is true (0.0-1.0)
- **Confidence**: How certain we are (0.0-1.0)

Example interpretations:
- `<0.9, 0.8>`: "Usually true with high confidence"
- `<0.5, 0.9>`: "Equally true/false but we're very certain"
- `<0.8, 0.3>`: "Probably true but low confidence"

### Temporal Reasoning

NARS can reason about time-dependent truths:
```
"Socrates was alive" (past)
"Socrates is remembered" (present)
"Philosophy will evolve" (future)
```

### Evidence Tracking

Each belief maintains its evidential support:
- Direct observations
- Inferred from other beliefs
- Revised through contradiction resolution

## Advanced Usage Patterns

### 1. Building Philosophical Knowledge Bases

```
# Add foundational beliefs
add_philosophical_axioms({
  "domain": "ethics",
  "axioms": [
    "Harm reduction is valuable",
    "Autonomy deserves respect",
    "Justice requires fairness"
  ]
})

# Query implications
explore_ethical_implications({
  "scenario": "AI decision making",
  "check_consistency": true
})
```

### 2. Multi-Perspective Analysis

```
analyze_from_perspectives({
  "claim": "Reality is fundamentally mental",
  "perspectives": [
    "materialism",
    "idealism", 
    "dual-aspect theory",
    "neutral monism"
  ],
  "synthesize": true
})
```

### 3. Philosophical Dialogue Simulation

```
philosophical_dialogue({
  "participants": ["Socrates", "Descartes", "Hume"],
  "topic": "The nature of knowledge",
  "rounds": 5,
  "track_belief_evolution": true
})
```

## Best Practices

1. **Start with Clear Premises**: NARS works best with well-defined initial beliefs
2. **Allow Inference Time**: Complex reasoning may need multiple inference cycles
3. **Check Confidence Values**: Low confidence indicates need for more evidence
4. **Use Revision**: When contradictions arise, let NARS revise beliefs
5. **Combine with Semantic Analysis**: Use both NARS logic and language analysis

## Troubleshooting Common Issues

### "No answers returned"
- Increase inference steps
- Add more relevant premises
- Check query syntax

### "Low confidence results"
- Add more supporting evidence
- Use belief revision
- Check for contradictions

### "Timeout errors"
- Reduce inference depth
- Simplify query
- Break into sub-queries

## Example Philosophical Explorations

### Ethics: Trolley Problem
```
explore_ethical_dilemma({
  "dilemma": "trolley problem",
  "variations": ["standard", "fat man", "loop"],
  "ethical_frameworks": ["utilitarian", "deontological", "virtue ethics"],
  "use_nars_for": "consistency checking"
})
```

### Metaphysics: Identity Over Time
```
analyze_persistence_conditions({
  "entity": "person",
  "theories": ["psychological continuity", "biological continuity", "soul theory"],
  "test_cases": ["teleportation", "gradual replacement", "fission"],
  "track_confidence": true
})
```

### Epistemology: Justified True Belief
```
examine_knowledge_conditions({
  "proposition": "I know that P",
  "analyze": ["truth", "belief", "justification", "Gettier cases"],
  "use_nars": true,
  "confidence_threshold": 0.7
})
```

## Integration with Other Tools

NARS reasoning can be combined with:
- **Semantic embedding search**: Find related concepts
- **Coherence landscape mapping**: Visualize belief networks
- **Language game analysis**: Context-dependent truth

## Further Resources

- NARS Theory: https://www.cis.temple.edu/~pwang/NARS-Intro.html
- ONA Documentation: https://github.com/opennars/OpenNARS-for-Applications
- Philosophy + AI: https://plato.stanford.edu/entries/artificial-intelligence/
