"""
MCP Prompts Implementation for OpenEnded Philosophy Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module implements reusable prompt templates and workflows for philosophical
inquiry, following the Model Context Protocol specification for prompts.

### Key Features:
- Structured prompt templates with proper MCP schema
- Dynamic argument support
- Context-aware prompt generation
- Multi-perspective philosophical workflows
- Integration with NARS reasoning system

### Prompt Categories:
- Conceptual Analysis Workflows
- Coherence Exploration Templates
- Insight Generation Frameworks
- Hypothesis Testing Protocols
- Meaning Contextualization Guides
"""

import json
from datetime import datetime
from typing import Any, Dict, List

import mcp.types as types

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Definitions Following MCP Specification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHILOSOPHICAL_PROMPTS = {
    "philosophical-concept-analysis": types.Prompt(
        name="philosophical-concept-analysis",
        description=(
            "Comprehensive multi-perspective analysis of philosophical concepts using "
            "non-axiomatic reasoning. Guides systematic exploration through various "
            "interpretive lenses with uncertainty quantification."
        ),
        arguments=[
            types.PromptArgument(
                name="concept",
                description="The philosophical concept to analyze (e.g., 'consciousness', 'free will', 'justice')",
                required=True
            ),
            types.PromptArgument(
                name="context",
                description="Contextual domain for analysis (e.g., 'ethics', 'metaphysics', 'epistemology', 'philosophy of mind')",
                required=True
            ),
            types.PromptArgument(
                name="perspectives",
                description="Comma-separated list of philosophical perspectives (e.g., 'analytical,phenomenological,pragmatist')",
                required=False
            ),
            types.PromptArgument(
                name="confidence_threshold",
                description="Minimum confidence level for insights (0.0-1.0, default: 0.7)",
                required=False
            )
        ]
    ),

    "coherence-landscape-exploration": types.Prompt(
        name="coherence-landscape-exploration",
        description=(
            "Map conceptual coherence patterns and identify philosophical structures "
            "within domains. Reveals stable regions, phase transitions, and emergent "
            "organizational principles."
        ),
        arguments=[
            types.PromptArgument(
                name="domain",
                description="Philosophical domain to explore (e.g., 'ethics', 'consciousness', 'knowledge', 'reality')",
                required=True
            ),
            types.PromptArgument(
                name="depth",
                description="Analysis depth level (1-5, where 1=basic, 5=comprehensive meta-theoretical)",
                required=False
            ),
            types.PromptArgument(
                name="focus_area",
                description="Specific aspect to emphasize (e.g., 'foundational concepts', 'relational structures', 'dialectical tensions')",
                required=False
            )
        ]
    ),

    "multi-perspective-insight-generation": types.Prompt(
        name="multi-perspective-insight-generation",
        description=(
            "Generate fallibilistic insights about phenomena through systematic "
            "multi-perspectival synthesis. Includes contradiction mapping and "
            "uncertainty quantification."
        ),
        arguments=[
            types.PromptArgument(
                name="phenomenon",
                description="Phenomenon to investigate (e.g., 'moral progress', 'artificial consciousness', 'emergence')",
                required=True
            ),
            types.PromptArgument(
                name="perspectives",
                description="Specific perspectives to apply (leave empty for automatic selection)",
                required=False
            ),
            types.PromptArgument(
                name="include_contradictions",
                description="Whether to explicitly explore contradictions (true/false, default: true)",
                required=False
            ),
            types.PromptArgument(
                name="analysis_depth",
                description="Depth of analysis (1-5, default: 3)",
                required=False
            )
        ]
    ),

    "hypothesis-testing-protocol": types.Prompt(
        name="hypothesis-testing-protocol",
        description=(
            "Systematic testing of philosophical hypotheses across multiple domains "
            "with coherence analysis and pragmatic evaluation. Provides confidence "
            "metrics and revision recommendations."
        ),
        arguments=[
            types.PromptArgument(
                name="hypothesis",
                description="The philosophical hypothesis to test",
                required=True
            ),
            types.PromptArgument(
                name="test_domains",
                description="Comma-separated domains for testing (e.g., 'ethics,epistemology,metaphysics')",
                required=False
            ),
            types.PromptArgument(
                name="evaluation_criteria",
                description="Specific criteria to emphasize (e.g., 'logical consistency', 'empirical adequacy', 'practical utility')",
                required=False
            )
        ]
    ),

    "meaning-contextualization-guide": types.Prompt(
        name="meaning-contextualization-guide",
        description=(
            "Derive contextual semantics through Wittgensteinian language game "
            "analysis. Shows how meaning emerges from use in specific practices "
            "and forms of life."
        ),
        arguments=[
            types.PromptArgument(
                name="expression",
                description="Expression or concept to contextualize",
                required=True
            ),
            types.PromptArgument(
                name="language_game",
                description="Language game context (scientific_discourse, ethical_deliberation, aesthetic_judgment, ordinary_language)",
                required=True
            ),
            types.PromptArgument(
                name="trace_genealogy",
                description="Include semantic evolution history (true/false, default: false)",
                required=False
            )
        ]
    ),

    "socratic-inquiry-framework": types.Prompt(
        name="socratic-inquiry-framework",
        description=(
            "Structured Socratic dialogue framework for deep philosophical exploration. "
            "Guides systematic questioning to uncover assumptions and clarify thinking."
        ),
        arguments=[
            types.PromptArgument(
                name="initial_claim",
                description="Starting claim or position to examine",
                required=True
            ),
            types.PromptArgument(
                name="inquiry_depth",
                description="Depth of questioning (1-5, default: 3)",
                required=False
            ),
            types.PromptArgument(
                name="focus_assumptions",
                description="Whether to focus on uncovering hidden assumptions (true/false, default: true)",
                required=False
            )
        ]
    ),

    "dialectical-synthesis-workshop": types.Prompt(
        name="dialectical-synthesis-workshop",
        description=(
            "Facilitates dialectical exploration of philosophical tensions, "
            "moving from thesis/antithesis toward synthesis through rational dialogue."
        ),
        arguments=[
            types.PromptArgument(
                name="thesis",
                description="Initial position or claim (thesis)",
                required=True
            ),
            types.PromptArgument(
                name="antithesis",
                description="Opposing position or objection (antithesis)",
                required=True
            ),
            types.PromptArgument(
                name="domain",
                description="Philosophical domain context",
                required=False
            )
        ]
    ),

    "philosophical-case-study-analysis": types.Prompt(
        name="philosophical-case-study-analysis",
        description=(
            "Deep analysis of philosophical thought experiments, historical cases, "
            "or contemporary examples using multiple analytical frameworks."
        ),
        arguments=[
            types.PromptArgument(
                name="case_description",
                description="Description of the case, thought experiment, or example",
                required=True
            ),
            types.PromptArgument(
                name="philosophical_issues",
                description="Key philosophical issues to explore in the case",
                required=True
            ),
            types.PromptArgument(
                name="analytical_frameworks",
                description="Specific frameworks to apply (e.g., 'virtue ethics,utilitarianism,deontology')",
                required=False
            )
        ]
    )
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Message Generators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_concept_analysis_prompt(
    concept: str,
    context: str,
    perspectives: str | None = None,
    confidence_threshold: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for philosophical concept analysis."""

    # Parse parameters
    perspective_list = perspectives.split(",") if perspectives else None
    confidence = float(confidence_threshold) if confidence_threshold else 0.7

    # Build perspective guidance
    perspective_guidance = ""
    if perspective_list:
        perspective_guidance = f"""
Focus your analysis using these specific philosophical perspectives:
{', '.join(perspective_list)}

Each perspective should be applied systematically to reveal different facets of the concept.
"""
    else:
        perspective_guidance = """
The system will automatically select the most relevant philosophical perspectives for this concept and context.
Common perspectives include: analytical, phenomenological, pragmatist, critical, existentialist, virtue ethics, deontological, consequentialist.
"""

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Philosophical Concept Analysis Request

## Concept to Analyze
**{concept}**

## Contextual Domain
**{context}**

## Analysis Framework
Please conduct a comprehensive multi-perspective analysis of this concept using the `analyze_concept` tool.

{perspective_guidance}

## Confidence Threshold
Use a confidence threshold of {confidence} to ensure high-quality insights.

## Instructions for Analysis

1. **Apply the analyze_concept tool** with the following parameters:
   - concept: "{concept}"
   - context: "{context}"
   {"- perspectives: " + str(perspective_list) if perspective_list else ""}
   - confidence_threshold: {confidence}

2. **Interpret the results comprehensively**, paying attention to:
   - Individual perspective analyses and their confidence levels
   - Cross-perspective synthesis and emergent insights
   - Identified tensions and contradictions
   - Uncertainty profiles and epistemic limitations
   - Revision conditions for future inquiry

3. **Provide philosophical commentary** on:
   - What the analysis reveals about the nature of the concept
   - How different perspectives illuminate different aspects
   - Areas where further philosophical work is needed
   - Implications for related philosophical questions

## Expected Outcome
A rich, multi-dimensional understanding of {concept} in the context of {context}, with explicit attention to uncertainty, tensions, and areas for future development.
"""
            )
        )
    ]

def generate_coherence_exploration_prompt(
    domain: str,
    depth: str | None = None,
    focus_area: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for coherence landscape exploration."""

    depth_level = int(depth) if depth else 3
    focus_guidance = f"\n\nPay particular attention to: {focus_area}" if focus_area else ""

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Coherence Landscape Exploration

## Domain
**{domain}**

## Analysis Depth
Level {depth_level} (1=basic foundations, 5=deep meta-theoretical structures)

## Exploration Instructions

Use the `explore_coherence` tool to map the conceptual landscape of {domain}:

**Tool Parameters:**
- domain: "{domain}"
- depth: {depth_level}
- allow_revision: true

## Interpretive Framework

1. **Foundational Layer Analysis**
   - What are the core concepts that ground this domain?
   - How stable and well-defined are these foundations?
   - What definitional challenges exist?

2. **Relational Structure Mapping**
   - How do concepts connect and support each other?
   - What are the key conceptual dependencies?
   - Where are the strongest and weakest connections?

3. **Systematic Integration Assessment**
   - How well does the domain hang together as a whole?
   - What higher-order patterns emerge?
   - Where are the gaps or tensions in integration?

4. **Dynamical Analysis**
   - What are the stable conceptual attractors?
   - Where might phase transitions occur?
   - How might the landscape evolve?

## Focus Areas
{focus_guidance}

## Expected Insights

The exploration should reveal:
- The overall coherence quality of the domain
- Specific strengths and vulnerabilities
- Opportunities for theoretical development
- Connections to other philosophical domains
- Areas needing further conceptual work

Please interpret the results with attention to both the formal coherence metrics and their philosophical significance.
"""
            )
        )
    ]

def generate_insight_generation_prompt(
    phenomenon: str,
    perspectives: str | None = None,
    include_contradictions: str | None = None,
    analysis_depth: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for multi-perspective insight generation."""

    depth = int(analysis_depth) if analysis_depth else 3
    contradictions = include_contradictions != "false" if include_contradictions else True

    perspective_guidance = ""
    if perspectives:
        perspective_guidance = f"""
**Specified Perspectives:** {perspectives}
These perspectives will guide the analysis and should each contribute unique insights.
"""
    else:
        perspective_guidance = """
**Automatic Perspective Selection:** The system will choose the most relevant philosophical perspectives for this phenomenon.
"""

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Multi-Perspective Insight Generation

## Phenomenon to Investigate
**{phenomenon}**

## Analysis Configuration
- **Depth Level:** {depth} (1=surface insights, 5=deep meta-philosophical analysis)
- **Include Contradictions:** {contradictions}
{perspective_guidance}

## Investigation Protocol

1. **Apply the generate_insights tool** with these parameters:
   - phenomenon: "{phenomenon}"
   {"- perspectives: [" + perspectives + "]" if perspectives else ""}
   - depth: {depth}
   - include_contradictions: {contradictions}

2. **Examine the results systematically:**

   **Primary Insights (by perspective)**
   - What unique understanding does each perspective contribute?
   - How confident is each perspective's analysis?
   - What evidence supports each perspective's claims?

   **Synthetic Insights**
   - What new understanding emerges from perspective integration?
   - How do different viewpoints complement each other?
   - What higher-order patterns become visible?

   **Meta-Philosophical Insights**
   - What does this analysis reveal about philosophical methodology?
   - How do different approaches to understanding work?
   - What are the limits of current philosophical frameworks?

   {"**Contradiction Analysis**" if contradictions else ""}
   {"- Where do perspectives genuinely conflict?" if contradictions else ""}
   {"- Which contradictions are productive vs. problematic?" if contradictions else ""}
   {"- How might tensions be resolved or leveraged?" if contradictions else ""}

3. **Philosophical Interpretation**
   - What does this tell us about the nature of {phenomenon}?
   - How does this contribute to ongoing philosophical debates?
   - What new questions or research directions emerge?
   - What practical implications might these insights have?

## Expected Outcomes
A rich, multi-layered understanding of {phenomenon} that:
- Integrates diverse philosophical perspectives
- Acknowledges uncertainty and limitation
- Identifies productive tensions and contradictions
- Suggests directions for further inquiry
- Connects to broader philosophical questions

Please provide thoughtful commentary on both the substantive insights and the methodological lessons from this multi-perspective approach.
"""
            )
        )
    ]

def generate_hypothesis_testing_prompt(
    hypothesis: str,
    test_domains: str | None = None,
    evaluation_criteria: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for philosophical hypothesis testing."""

    domains_guidance = ""
    if test_domains:
        domains_guidance = f"""
**Testing Domains:** {test_domains}
The hypothesis will be evaluated for coherence and adequacy across these specific domains.
"""
    else:
        domains_guidance = """
**Default Testing:** The hypothesis will be tested across general philosophical domains and contexts.
"""

    criteria_guidance = ""
    if evaluation_criteria:
        criteria_guidance = f"""
**Evaluation Emphasis:** Pay particular attention to {evaluation_criteria} in the assessment.
"""

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Philosophical Hypothesis Testing Protocol

## Hypothesis Under Investigation
**"{hypothesis}"**

## Testing Configuration
{domains_guidance}
{criteria_guidance}

## Testing Methodology

1. **Apply the test_philosophical_hypothesis tool:**
   - hypothesis: "{hypothesis}"
   {"- test_domains: [" + test_domains + "]" if test_domains else ""}
   {"- criteria: {" + evaluation_criteria + "}" if evaluation_criteria else ""}

2. **Systematic Evaluation Framework:**

   **Coherence Analysis**
   - How well does the hypothesis fit with established philosophical knowledge?
   - What internal tensions or contradictions emerge?
   - How does it integrate with other theoretical commitments?

   **Domain-Specific Testing**
   - How does the hypothesis fare in each testing domain?
   - What supporting evidence exists in each area?
   - What challenges or counterexamples arise?

   **Pragmatic Assessment**
   - What practical implications follow from the hypothesis?
   - How useful is it for solving philosophical problems?
   - What new research directions does it suggest?

   **Confidence and Uncertainty**
   - What is the overall confidence level in the hypothesis?
   - Where are the main sources of uncertainty?
   - What additional evidence would be most valuable?

3. **Critical Analysis:**
   - What are the hypothesis's main strengths?
   - What are its primary vulnerabilities?
   - How might it be refined or modified?
   - What alternative hypotheses should be considered?

## Revision and Development
Based on the testing results:
- What revisions to the hypothesis are suggested?
- What further investigation is needed?
- How might the hypothesis be strengthened?
- What new hypotheses emerge from this analysis?

## Expected Assessment
A comprehensive evaluation that:
- Tests the hypothesis rigorously across multiple domains
- Identifies both strengths and weaknesses
- Provides clear confidence metrics
- Suggests specific directions for improvement
- Places the hypothesis in broader philosophical context

Please interpret the results with attention to both the formal testing metrics and their substantive philosophical implications.
"""
            )
        )
    ]

def generate_meaning_contextualization_prompt(
    expression: str,
    language_game: str,
    trace_genealogy: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for meaning contextualization analysis."""

    genealogy = trace_genealogy == "true" if trace_genealogy else False

    language_game_descriptions = {
        "scientific_discourse": "the practices and forms of life of scientific inquiry",
        "ethical_deliberation": "moral reasoning and practical ethical decision-making",
        "aesthetic_judgment": "artistic appreciation and aesthetic evaluation",
        "ordinary_language": "everyday communication and common usage"
    }

    game_description = language_game_descriptions.get(language_game, "the specified language game context")

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Meaning Contextualization Analysis

## Expression/Concept
**"{expression}"**

## Language Game Context
**{language_game}** - {game_description}

## Analysis Approach
Following Wittgenstein's insights about meaning as use, this analysis will explore how "{expression}" functions within {game_description}.

## Contextualization Protocol

1. **Apply the contextualize_meaning tool:**
   - expression: "{expression}"
   - language_game: "{language_game}"
   - trace_genealogy: {genealogy}

2. **Interpretive Framework:**

   **Primary Meaning Analysis**
   - How is "{expression}" typically used in this language game?
   - What role does it play in the practices of this context?
   - What are its standard applications and limitations?

   **Usage Pattern Mapping**
   - What are the typical contexts where this expression appears?
   - How do practitioners in this domain employ the term?
   - What variations in usage exist within the language game?

   **Family Resemblances**
   - What related concepts cluster around this expression?
   - How do these concepts support and illuminate each other?
   - What broader conceptual families is this expression part of?

   **Pragmatic Conditions**
   - Under what conditions is use of this expression successful?
   - What background assumptions must be in place?
   - How do contextual factors affect meaning?

   {"**Semantic Genealogy**" if genealogy else ""}
   {"- How has the meaning of this expression evolved?" if genealogy else ""}
   {"- What historical influences have shaped its current usage?" if genealogy else ""}
   {"- How might it continue to develop?" if genealogy else ""}

3. **Cross-Context Comparison**
   - How might this expression function differently in other language games?
   - What aspects of meaning are preserved across contexts?
   - What new meanings emerge in different practices?

## Expected Understanding
This analysis should reveal:
- The contextual specificity of meaning
- How practices shape conceptual content
- The dynamic relationship between use and meaning
- Sources of semantic stability and change
- Implications for philosophical analysis

Please provide commentary on both the specific semantic analysis and its broader implications for understanding meaning, context, and philosophical methodology.
"""
            )
        )
    ]

def generate_socratic_inquiry_prompt(
    initial_claim: str,
    inquiry_depth: str | None = None,
    focus_assumptions: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for Socratic inquiry framework."""

    depth = int(inquiry_depth) if inquiry_depth else 3
    assumptions_focus = focus_assumptions != "false" if focus_assumptions else True

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Socratic Inquiry Framework

## Initial Claim/Position
**"{initial_claim}"**

## Inquiry Configuration
- **Depth Level:** {depth} (1=basic questioning, 5=deep assumption excavation)
- **Focus on Assumptions:** {assumptions_focus}

## Socratic Method Application

This dialogue will follow the classical Socratic approach of systematic questioning to examine the claim and uncover deeper understanding.

### Phase 1: Clarification Questions
- What exactly do you mean by this claim?
- Can you give me an example of what you're talking about?
- How does this relate to what we discussed earlier?
- What are the key terms that need definition?

### Phase 2: Assumption Probing
{"Since assumption focus is enabled, we'll dig deep into:" if assumptions_focus else "We'll examine:"}
- What assumptions are you making here?
- What if the opposite were true?
- Why do you think this assumption is valid?
- What evidence supports this underlying belief?

### Phase 3: Evidence and Reasoning Examination
- What evidence supports this claim?
- How reliable is this evidence?
- What might contradict this evidence?
- Are there alternative ways to interpret this evidence?

### Phase 4: Implications and Consequences
- If this claim is true, what follows?
- What are the broader implications?
- How does this affect other beliefs you hold?
- What would change if this claim were false?

### Phase 5: Meta-Level Reflection
- How confident are you in this claim now?
- What would it take to change your mind?
- What questions does this raise that we haven't considered?
- How has your understanding evolved through this inquiry?

## Dialogue Structure

I will engage with "{initial_claim}" through systematic questioning, following these principles:

1. **Intellectual Humility**: Begin with genuine curiosity rather than refutation
2. **Systematic Progression**: Build questions logically from simple to complex
3. **Assumption Excavation**: Uncover hidden premises and unstated beliefs
4. **Collaborative Discovery**: Work together toward clearer understanding
5. **Productive Uncertainty**: Embrace the value of not-knowing

## Expected Outcomes

Through this Socratic process, we should achieve:
- Greater clarity about the meaning and implications of the claim
- Awareness of previously hidden assumptions
- Understanding of the evidence base and its limitations
- Recognition of alternative perspectives and possibilities
- Deeper appreciation for the complexity of the issue

Let's begin this philosophical inquiry. I'll pose questions designed to help us examine "{initial_claim}" more deeply and systematically.

**First Question:** Can you help me understand more precisely what you mean when you say "{initial_claim}"? What would be a concrete example that illustrates this claim?
"""
            )
        )
    ]

def generate_dialectical_synthesis_prompt(
    thesis: str,
    antithesis: str,
    domain: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for dialectical synthesis workshop."""

    domain_context = f" within the domain of {domain}" if domain else ""

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Dialectical Synthesis Workshop

## Philosophical Tension{domain_context}

### Thesis
**"{thesis}"**

### Antithesis
**"{antithesis}"**

## Dialectical Analysis Framework

This workshop will explore the tension between these positions and work toward a potential synthesis through rational dialogue and critical examination.

### Stage 1: Position Clarification
**Understanding the Thesis:**
- What are the core commitments of this position?
- What arguments and evidence support it?
- What philosophical tradition or framework does it represent?
- What are its key strengths and insights?

**Understanding the Antithesis:**
- How does this position challenge or contradict the thesis?
- What arguments and evidence support this alternative view?
- What philosophical motivation drives this opposition?
- What valuable insights does it preserve?

### Stage 2: Tension Analysis
**Nature of the Conflict:**
- Is this a logical contradiction or a practical tension?
- Are the positions addressing the same question or talking past each other?
- What deeper philosophical issues underlie this disagreement?
- What's at stake in choosing between these positions?

**Common Ground Identification:**
- What do both positions agree on, even if implicitly?
- What shared values or concerns motivate both perspectives?
- Are there underlying assumptions both positions share?
- What philosophical territory is not in dispute?

### Stage 3: Synthesis Development
**Preserving Insights:**
- What truth or insight in the thesis must be preserved?
- What truth or insight in the antithesis must be preserved?
- How might these insights complement rather than contradict?
- What would be lost if we abandoned either perspective entirely?

**Higher-Order Integration:**
- Is there a higher-level perspective that encompasses both views?
- Can the tension be resolved through reframing or recontextualization?
- Does the conflict point toward a more nuanced or complex understanding?
- What new questions or distinctions might help move beyond the deadlock?

### Stage 4: Synthetic Position
**Proposed Synthesis:**
- How might we integrate the valuable elements of both positions?
- What new understanding emerges from this integration?
- How does the synthesis address the original tension?
- What are the strengths and limitations of this synthetic view?

**Testing the Synthesis:**
- Does it genuinely resolve the original tension or merely paper over it?
- What new problems or questions does the synthesis generate?
- How might this synthesis be further developed or refined?
- What objections might be raised against this integration?

## Philosophical Method

This dialectical exploration will employ:
- **Charitable Interpretation**: Understanding each position in its strongest form
- **Rational Dialogue**: Using reason to explore possibilities for integration
- **Creative Synthesis**: Moving beyond mere compromise toward genuine integration
- **Critical Evaluation**: Testing proposed syntheses for coherence and adequacy

## Expected Outcome

A thoughtful exploration that:
- Clarifies the nature and source of the philosophical tension
- Identifies valuable insights in both positions
- Develops a potential synthesis that preserves what's worth keeping
- Advances philosophical understanding beyond the original debate
- Suggests directions for further philosophical development

Let's begin this dialectical journey by examining each position carefully and then working toward a potential synthesis that honors the insights of both while transcending their limitations.
"""
            )
        )
    ]

def generate_case_study_prompt(
    case_description: str,
    philosophical_issues: str,
    analytical_frameworks: str | None = None
) -> List[types.PromptMessage]:
    """Generate messages for philosophical case study analysis."""

    frameworks_guidance = ""
    if analytical_frameworks:
        frameworks_guidance = f"""
**Specified Analytical Frameworks:** {analytical_frameworks}
These frameworks will guide the analysis and should each contribute unique insights.
"""
    else:
        frameworks_guidance = """
**Framework Selection:** We'll select the most appropriate philosophical frameworks based on the nature of the case and issues involved.
"""

    framework_list_str = ""
    if analytical_frameworks:
        framework_list_str = "- " + analytical_frameworks.replace(",", "\n- ")
    else:
        framework_list_str = (
            "- Virtue Ethics: Character, flourishing, and moral development\n"
            "- Deontological Ethics: Duties, rights, and universal principles\n"
            "- Consequentialism: Outcomes, utility, and aggregate welfare\n"
            "- Care Ethics: Relationships, responsibility, and contextual response\n"
            "- Existentialism: Freedom, authenticity, and personal responsibility"
        )

    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text=f"""# Philosophical Case Study Analysis

## Case Description
{case_description}

## Key Philosophical Issues
{philosophical_issues}

## Analytical Approach
{frameworks_guidance}

## Comprehensive Analysis Framework

### Stage 1: Case Contextualization
**Background Understanding:**
- What are the essential facts and circumstances of this case?
- What makes this case philosophically interesting or significant?
- How does this case relate to broader philosophical questions?
- What real-world implications does this case have?

**Initial Philosophical Mapping:**
- What philosophical concepts are clearly relevant to this case?
- What ethical, metaphysical, or epistemological issues arise?
- How does this case connect to existing philosophical literature?
- What makes this case particularly challenging or illuminating?

### Stage 2: Multi-Framework Analysis
{"We'll analyze this case through the lens of each specified framework:" if analytical_frameworks else "We'll apply multiple philosophical frameworks to gain comprehensive understanding:"}

{"**Framework Applications:**" if analytical_frameworks else "**Potential Frameworks:**"}
{framework_list_str}

**For Each Framework:**
- How does this framework interpret the key issues in the case?
- What aspects of the case does this framework illuminate most clearly?
- What solutions or approaches does this framework suggest?
- What are the strengths and limitations of this framework's analysis?

### Stage 3: Comparative Framework Analysis
**Framework Integration:**
- Where do different frameworks converge in their analysis?
- What are the most significant areas of disagreement?
- How do different frameworks prioritize different values or considerations?
- What unique insights does each framework contribute?

**Tension Resolution:**
- How might conflicts between frameworks be resolved?
- Are there higher-order principles that could integrate different approaches?
- What does the disagreement between frameworks tell us about the case?
- How might we synthesize the best insights from multiple approaches?

### Stage 4: Philosophical Implications
**Broader Significance:**
- What does this case teach us about the philosophical issues involved?
- How does this case challenge or confirm existing philosophical theories?
- What new questions or problems does this case raise?
- How might this case influence future philosophical thinking?

**Practical Applications:**
- What practical guidance emerges from this philosophical analysis?
- How might these insights apply to similar cases or situations?
- What policies or practices might be informed by this analysis?
- How can philosophical insight be translated into actionable guidance?

### Stage 5: Critical Reflection
**Analytical Assessment:**
- What are the strongest insights from this analysis?
- Where are the remaining areas of uncertainty or disagreement?
- What additional information or perspective would be valuable?
- How has this analysis deepened our understanding of the issues?

**Methodological Lessons:**
- What does this case teach us about philosophical methodology?
- How effective were the different analytical frameworks?
- What are the benefits and limitations of case study analysis?
- How might this approach be improved or refined?

## Expected Outcomes

This analysis should produce:
- Deep understanding of the philosophical dimensions of the case
- Clear articulation of how different frameworks approach the issues
- Synthesis of insights from multiple philosophical perspectives
- Practical guidance informed by philosophical reflection
- Broader insights about the philosophical issues involved
- Methodological lessons about case study analysis

Let's proceed with this systematic philosophical examination of the case, bringing rigorous philosophical analysis to bear on this concrete situation.
"""
            )
        )
    ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Handler Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def list_available_prompts() -> List[types.Prompt]:
    """Return list of all available philosophical prompts."""
    return list(PHILOSOPHICAL_PROMPTS.values())

def get_prompt_by_name(name: str, arguments: Dict[str, str] | None = None) -> types.GetPromptResult:
    """Get specific prompt with processed arguments."""

    if name not in PHILOSOPHICAL_PROMPTS:
        raise ValueError(f"Unknown prompt: {name}")

    # Get the base prompt definition
    prompt_def = PHILOSOPHICAL_PROMPTS[name]

    # Prepare arguments with defaults
    args = arguments or {}

    # Generate appropriate messages based on prompt type
    if name == "philosophical-concept-analysis":
        messages = generate_concept_analysis_prompt(
            concept=args.get("concept", "consciousness"),
            context=args.get("context", "philosophy of mind"),
            perspectives=args.get("perspectives"),
            confidence_threshold=args.get("confidence_threshold")
        )
    elif name == "coherence-landscape-exploration":
        messages = generate_coherence_exploration_prompt(
            domain=args.get("domain", "ethics"),
            depth=args.get("depth"),
            focus_area=args.get("focus_area")
        )
    elif name == "multi-perspective-insight-generation":
        messages = generate_insight_generation_prompt(
            phenomenon=args.get("phenomenon", "emergence"),
            perspectives=args.get("perspectives"),
            include_contradictions=args.get("include_contradictions"),
            analysis_depth=args.get("analysis_depth")
        )
    elif name == "hypothesis-testing-protocol":
        messages = generate_hypothesis_testing_prompt(
            hypothesis=args.get("hypothesis", "Consciousness is an emergent property"),
            test_domains=args.get("test_domains"),
            evaluation_criteria=args.get("evaluation_criteria")
        )
    elif name == "meaning-contextualization-guide":
        messages = generate_meaning_contextualization_prompt(
            expression=args.get("expression", "knowledge"),
            language_game=args.get("language_game", "ordinary_language"),
            trace_genealogy=args.get("trace_genealogy")
        )
    elif name == "socratic-inquiry-framework":
        messages = generate_socratic_inquiry_prompt(
            initial_claim=args.get("initial_claim", "Knowledge is justified true belief"),
            inquiry_depth=args.get("inquiry_depth"),
            focus_assumptions=args.get("focus_assumptions")
        )
    elif name == "dialectical-synthesis-workshop":
        messages = generate_dialectical_synthesis_prompt(
            thesis=args.get("thesis", "Free will exists"),
            antithesis=args.get("antithesis", "Determinism is true"),
            domain=args.get("domain")
        )
    elif name == "philosophical-case-study-analysis":
        messages = generate_case_study_prompt(
            case_description=args.get("case_description", "Trolley problem thought experiment"),
            philosophical_issues=args.get("philosophical_issues", "Moral permissibility of action vs. inaction"),
            analytical_frameworks=args.get("analytical_frameworks")
        )
    else:
        # Fallback for unknown prompts
        messages = [
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Please use the philosophical analysis tools to explore the topic specified in the {name} prompt."
                )
            )
        ]

    return types.GetPromptResult(
        description=prompt_def.description,
        messages=messages
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Export Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    'PHILOSOPHICAL_PROMPTS',
    'list_available_prompts',
    'get_prompt_by_name'
]
