#!/usr/bin/env bash

# Enhanced OpenEnded Philosophy MCP Server Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

echo "🔬 Initializing OpenEnded Philosophy MCP Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv package manager not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv package manager detected"

# Create virtual environment with uv
echo "🔧 Creating virtual environment..."
uv venv --python 3.9 --seed

# Activate virtual environment
echo "🔧 Activating virtual environment..."
# shellcheck source=.venv/bin/activate
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --dev

# Create logs directory
echo "📁 Creating logs directory..."
mkdir -p logs

# Set up development environment
echo "🛠️  Setting up development environment..."
uv run ruff format .
uv run ruff check . --fix

# Run basic tests
echo "🧪 Running basic validation..."
if uv run python -c "from openended_philosophy.server import OpenEndedPhilosophyServer; print('✅ Server imports successfully')"; then
    echo "✅ Basic validation passed"
else
    echo "❌ Basic validation failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Copy example_mcp_config.json to your Claude Desktop configuration"
echo "   2. Update the path in the config to match your installation"
echo "   3. Restart Claude Desktop"
echo "   4. Test the philosophical analysis tools"
echo ""
echo "🔍 Development commands:"
echo "   • uv run openended-philosophy-server  # Start server directly"
echo "   • uv run pytest                       # Run tests"
echo "   • uv run ruff format .                # Format code"
echo "   • uv run mypy openended_philosophy/   # Type checking"
echo ""
echo "📖 For more information, see README.md"
