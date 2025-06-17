#!/bin/bash
# Setup script for OpenEnded Philosophy MCP with NARS Integration

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OpenEnded Philosophy MCP Server Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "✗ uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
else
    echo "✓ uv is installed"
fi

# Install Python dependencies
echo -e "\nInstalling Python dependencies..."
uv sync

# Download NLTK data
echo -e "\nDownloading required NLTK data..."
uv run python -c "
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print('✓ NLTK data downloaded')
"

# Check for ONA
echo -e "\nChecking for ONA (OpenNARS for Applications)..."
uv run python check_ona.py

# Instructions
echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete!"
echo ""
echo "To use with Claude Desktop:"
echo "1. Copy the configuration from example_mcp_config.json"
echo "2. Add it to your Claude Desktop config"
echo "3. Restart Claude Desktop"
echo ""
echo "To test the server directly:"
echo "  uv run openended-philosophy-server"
echo ""
echo "For development:"
echo "  uv run python -m openended_philosophy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
