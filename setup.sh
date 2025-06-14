#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenEnded Philosophy MCP Server Setup Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OpenEnded Philosophy MCP Server Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p logs
mkdir -p examples
mkdir -p tests
mkdir -p data
echo "✓ Directories created"

# Run basic tests to verify installation
echo ""
echo "Verifying installation..."
python -c "
from openended_philosophy import OpenEndedPhilosophyServer
print('✓ Core imports successful')

from openended_philosophy.core import (
    EmergentCoherenceNode,
    DynamicPluralismFramework,
    LanguageGameProcessor,
    CoherenceLandscape,
    FallibilisticInference
)
print('✓ All core components accessible')

import numpy as np
import networkx as nx
print('✓ Scientific dependencies loaded')
"

# Create a simple test script
echo ""
echo "Creating test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify installation."""

import asyncio
from openended_philosophy import (
    EmergentCoherenceNode,
    DynamicPluralismFramework,
    calculate_epistemic_uncertainty
)

async def test_basic_functionality():
    """Test basic framework functionality."""
    print("\n### Testing Basic Functionality ###\n")
    
    # Test coherence node
    node = EmergentCoherenceNode(
        initial_pattern={"concept": "truth", "domain": "epistemology"},
        confidence=0.7
    )
    print(f"✓ Created coherence node: {node.pattern.pattern_id}")
    
    # Test pluralism framework
    framework = DynamicPluralismFramework(openness_coefficient=0.9)
    schema_id = framework.integrate_perspective({
        'name': 'pragmatist',
        'concepts': ['utility', 'consequences', 'practice']
    })
    print(f"✓ Integrated perspective: {schema_id}")
    
    # Test uncertainty calculation
    uncertainty = calculate_epistemic_uncertainty(
        evidence_count=5,
        coherence_score=0.8,
        temporal_factor=1.0,
        domain_complexity=0.5
    )
    print(f"✓ Calculated epistemic uncertainty: {uncertainty:.3f}")
    
    print("\n### All tests passed! ###")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
EOF

python test_installation.py

# Setup completion message
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete! 🎉"
echo ""
echo "To run the server:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the server: python -m openended_philosophy"
echo ""
echo "To use as an MCP tool:"
echo "  1. Add the mcp_config.json to your MCP client configuration"
echo "  2. The server will be available as 'openended-philosophy'"
echo ""
echo "For examples and documentation, see the README.md file."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
