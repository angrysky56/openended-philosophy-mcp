# OpenEnded Philosophy MCP Server with NARS Integration

A sophisticated philosophical reasoning system that combines OpenEnded Philosophy with Non-Axiomatic Reasoning System (NARS) for enhanced epistemic analysis, truth maintenance, and multi-perspective synthesis.

## Core Integration: Philosophy + NARS

This server uniquely integrates:
- **NARS/ONA**: Non-axiomatic reasoning with truth maintenance and belief revision
- **Philosophical Pluralism**: Multi-perspective analysis without privileging any single view
- **Epistemic Humility**: Built-in uncertainty quantification and revision conditions
- **Coherence Dynamics**: Emergent conceptual landscapes with stability analysis

## Theoretical Foundation

**Core Philosophical Architecture**:
* **Epistemic Humility**: Every insight carries inherent uncertainty metrics
* **Contextual Semantics**: Meaning emerges through language games and forms of life
* **Dynamic Pluralism**: Multiple interpretive schemas coexist without hierarchical privileging
* **Pragmatic Orientation**: Efficacy measured through problem-solving capability

### Computational Framework

#### 1. **Emergent Coherence Dynamics**
```
C(t) = Σ_{regions} (R_i(t) × Stability_i) + Perturbation_Response(t)
```

Where:
- `C(t)`: Coherence landscape at time t
- `R_i(t)`: Regional coherence patterns
- `Stability_i`: Local stability coefficients
- `Perturbation_Response(t)`: Adaptive response to new experiences

#### 2. **Fallibilistic Inference Engine**
```
P(insight|evidence) = Confidence × (1 - Uncertainty_Propagation)
```

**Key Components**:
- Evidence limitation assessment
- Context dependence calculation
- Unknown unknown estimation
- Revision trigger identification

### System Architecture

```
┌─────────────────────────────────────────┐
│      OpenEnded Philosophy Server        │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐  │
│  │  Coherence  │  │    Language     │  │
│  │  Landscape  │  │     Games       │  │
│  └──────┬──────┘  └────────┬────────┘  │
│         │                   │           │
│  ┌──────▼──────────────────▼────────┐  │
│  │   Dynamic Pluralism Framework    │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │   Fallibilistic Inference Core   │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## NARS Integration Features

### Non-Axiomatic Logic (NAL)
- **Truth Values**: (frequency, confidence) pairs for nuanced belief representation
- **Evidence-Based Reasoning**: Beliefs strengthen with converging evidence
- **Temporal Reasoning**: Handle time-dependent truths and belief projection
- **Inference Rules**: Deduction, induction, abduction, analogy, and revision

### Enhanced Capabilities
- **Truth Maintenance**: Automatic belief revision when contradictions arise
- **Memory System**: Semantic embeddings + NARS attention buffer
- **Reasoning Patterns**: Multiple inference types for comprehensive analysis
- **Uncertainty Tracking**: Epistemic uncertainty propagation through inference chains

## Installation

```bash
git clone https://github.com/angrysky56/openended-philosophy-mcp
cd openended-philosophy-mcp
./setup.sh
```

The setup script will:
1. Install Python dependencies via uv
2. Download required NLTK data
3. Check for ONA (OpenNARS for Applications)
4. Provide configuration instructions
=======

Via Claude Desktop or other MCP client config json-
Adapt the paths to your own:

```json
{
  "mcpServers": {
    "openended-philosophy": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/openended-philosophy-mcp",
        "run",
        "openended-philosophy-server"
      ],
      "env": {
        "PYTHONPATH": "/home/ty/Repositories/ai_workspace/openended-philosophy-mcp",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### For Direct Usage (without MCP client)

If you want to run the philosophy server directly using uv:

#### Prerequisites

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Restart your shell or run:
   ```bash
   source $HOME/.cargo/env
   ```

#### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/angrysky56/openended-philosophy-mcp
   cd openended-philosophy-mcp
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

#### Running the Server

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the MCP server:
   ```bash
   python -m openended_philosophy_mcp
   ```

The server will start and listen for MCP protocol messages on stdin/stdout. You can interact with it programmatically or integrate it with other MCP-compatible tools.

#### Available Tools

- `ask_philosophical_question`: Ask deep philosophical questions and receive thoughtful responses
- `explore_philosophical_topic`: Explore philosophical topics in depth with guided discussion

## Usage via MCP

### Available Tools

#### 1. **analyze_concept**
Analyzes a concept through multiple interpretive lenses without claiming ontological priority.

```json
{
  "concept": "consciousness",
  "context": "neuroscience",
  "confidence_threshold": 0.7
}
```

#### 2. **explore_coherence**
Maps provisional coherence patterns in conceptual space.

```json
{
  "domain": "ethics",
  "depth": 3,
  "allow_revision": true
}
```

#### 3. **contextualize_meaning**
Derives contextual semantics through language game analysis.

```json
{
  "expression": "truth",
  "language_game": "scientific_discourse",
  "form_of_life": "research_community"
}
```

#### 4. **generate_insights**
Produces fallibilistic insights with built-in uncertainty quantification.

```json
{
  "phenomenon": "quantum_consciousness",
  "perspectives": ["physics", "philosophy_of_mind", "information_theory"],
  "openness_coefficient": 0.9
}
```

## Philosophical Methodology

### Wittgensteinian Therapeutic Approach
- **Dissolve Rather Than Solve**: Recognizes category mistakes
- **Language Game Awareness**: Context-dependent semantics
- **Family Resemblance**: Non-essentialist categorization

### Pragmatist Orientation
- **Instrumental Truth**: Measured by problem-solving efficacy
- **Fallibilism**: All knowledge provisional
- **Pluralism**: Multiple valid perspectives

### Information-Theoretic Substrate
- **Pattern Recognition**: Without ontological commitment
- **Emergence**: Novel properties from interactions
- **Complexity**: Irreducible to simple principles

## Development Philosophy

This server embodies its own philosophical commitments:
- **Open Source**: Knowledge emerges through community
- **Iterative Development**: Understanding grows through use
- **Bug-as-Feature**: Errors provide learning opportunities
- **Fork-Friendly**: Multiple development paths encouraged

## NARS Configuration & Setup

### Flexible Installation Support

The server now supports multiple ONA installation methods:

1. **pip-installed ONA** (Recommended)
   ```bash
   uv add ona
   ```

2. **Local executable** via environment variable
   ```bash
   # Set in .env file:
   ONA_PATH=/path/to/your/NAR
   ```

3. **Automatic detection** in common locations

### Configuration via Environment Variables

Create a `.env` file from the template:
```bash
cp .env.example .env
```

Key configuration options:
- `ONA_PATH`: Path to NAR executable (optional if pip-installed)
- `NARS_MEMORY_SIZE`: Concept memory size (default: 1000)
- `NARS_INFERENCE_STEPS`: Inference depth (default: 50)
- `NARS_SILENT_MODE`: Suppress ONA output (default: true)
- `NARS_DECISION_THRESHOLD`: Decision confidence threshold (default: 0.6)

### Testing NARS Integration

Verify your installation:
```bash
# Run comprehensive test suite
./test_nars_integration.py

# Or with uv
uv run python test_nars_integration.py
```

### Process Management

The improved NARS manager includes:
- **Robust cleanup patterns** preventing process leaks
- **Signal handling** for graceful shutdown (SIGTERM, SIGINT)
- **Automatic recovery** from subprocess failures
- **Cross-platform support** (Linux, macOS, Windows)

### Troubleshooting

See `docs/NARS_INSTALLATION.md` for detailed troubleshooting guide.

## Contributing

We welcome contributions that:
- Enhance epistemic humility features
- Add new interpretive schemas
- Improve contextual understanding
- Challenge existing assumptions
- Strengthen NARS integration capabilities

## License

MIT License - In the spirit of open-ended inquiry
