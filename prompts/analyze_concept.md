# Analyze Concept Tool - AI Guidance

## Purpose
The `analyze_concept` tool performs deep philosophical analysis of concepts through multiple interpretive lenses, integrating NARS non-axiomatic reasoning with uncertainty quantification.

## Effective Usage Patterns

### Basic Analysis
```
Use analyze_concept to examine "consciousness" in the context of "neuroscience"
```

### Multi-Perspective Analysis
```
Analyze the concept of "free will" in "ethics" using perspectives: ["analytical", "phenomenological", "pragmatist", "existentialist"]
```

### With Custom Confidence Threshold
```
Explore "justice" in "political philosophy" with confidence_threshold: 0.8 to focus on high-confidence insights
```

## Perspectives Available

- **analytical**: Logical rigor, conceptual clarity, definitional precision
- **phenomenological**: Lived experience, consciousness, intentionality
- **pragmatist**: Practical consequences, problem-solving efficacy
- **critical**: Power analysis, social critique, emancipation
- **existentialist**: Freedom, authenticity, situated existence
- **naturalist**: Scientific continuity, empirical grounding
- **hermeneutic**: Interpretation, understanding, tradition
- **virtue_ethics**: Character excellence, human flourishing
- **deontological**: Duty, obligation, moral universality
- **consequentialist**: Outcomes, utility maximization

## Interpreting Results

### Key Output Fields

1. **analyses**: Perspective-specific interpretations with:
   - `interpretation`: Derived meaning and key features
   - `confidence`: How well the perspective applies (0-1)
   - `coherence_score`: Internal consistency of the analysis
   - `uncertainty_bounds`: Confidence intervals

2. **synthesis**: Cross-perspective integration showing:
   - Common themes across perspectives
   - Emergent insights from synthesis
   - Overall conceptual coherence

3. **tensions**: Identified contradictions or conflicts between perspectives

4. **revision_conditions**: When and how the analysis should be reconsidered

5. **uncertainty_profile**: Comprehensive uncertainty assessment including:
   - Epistemic uncertainty factors
   - Confidence distribution
   - Areas requiring further investigation

## Example Prompts for Users

1. "What does analyze_concept reveal about 'truth' in epistemology?"
2. "How do different philosophical perspectives understand 'consciousness'?"
3. "Analyze 'moral responsibility' considering both free will and determinism"
4. "What conceptual tensions exist in our understanding of 'artificial intelligence'?"

## NARS Integration Features

- **Truth Values**: Each insight has frequency/confidence pairs from NARS
- **Evidence Tracking**: Stamp IDs trace evidential support
- **Belief Revision**: Conflicting evidence is resolved through NARS revision rules
- **Temporal Reasoning**: Concepts can be analyzed at specific time points

## Best Practices

1. **Context Specification**: Always provide a clear context (e.g., "ethics", "metaphysics", "cognitive science")
2. **Perspective Selection**: Choose 3-5 perspectives for balanced analysis
3. **Confidence Thresholds**: Adjust based on need for certainty vs. exploration
4. **Iterative Refinement**: Use revision conditions to guide follow-up analyses
5. **Synthesis Focus**: Pay special attention to cross-perspective synthesis for novel insights
