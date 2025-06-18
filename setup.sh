#!/bin/bash
# setup.sh - Comprehensive setup script for OpenEnded Philosophy MCP with NARS

set -e  # Exit on error

echo "================================================================"
echo "OpenEnded Philosophy MCP Server Setup with NARS Integration"
echo "================================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    echo "Error: Python 3.10+ is required"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your shell or run: source \$HOME/.cargo/env"
    echo "Then run this script again."
    exit 0
fi

# Install Python dependencies
echo "Installing Python dependencies with uv..."
uv sync

# Download NLTK data
echo "Downloading required NLTK data..."
uv run python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
"

# Check for ONA installation
echo ""
echo "Checking for ONA (OpenNARS for Applications)..."

# Try to find ONA
if uv run python -c "import ona" 2>/dev/null; then
    echo "✓ ONA is installed via pip"
elif which NAR &> /dev/null; then
    echo "✓ NAR executable found in PATH: $(which NAR)"
elif [ -n "$ONA_PATH" ] && [ -f "$ONA_PATH" ]; then
    echo "✓ ONA_PATH environment variable is set: $ONA_PATH"
else
    echo ""
    echo "⚠️  ONA not found. You have several installation options:"
    echo ""
    echo "Option 1: Install ONA via pip (Recommended)"
    echo "  uv add ona"
    echo ""
    echo "Option 2: Clone and build from source"
    echo "  git clone https://github.com/opennars/OpenNARS-for-Applications.git"
    echo "  cd OpenNARS-for-Applications"
    echo "  ./build.sh"
    echo ""
    echo "Option 3: Use existing installation"
    echo "  Set ONA_PATH in .env file to point to your NAR executable"
    echo ""
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file. Please review and configure as needed."
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data

echo ""
echo "================================================================"
echo "Setup complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo "1. Configure your .env file if needed"
echo "2. Install ONA if not already installed (see options above)"
echo "3. Add the server to your Claude Desktop config using:"
echo "   - See example_mcp_config.json for the configuration"
echo ""
echo "To test the server locally:"
echo "  uv run openended-philosophy-server"
echo ""
