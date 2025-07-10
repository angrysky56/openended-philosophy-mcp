# NARS/ONA Installation Guide

This guide provides comprehensive instructions for installing and configuring NARS (Non-Axiomatic Reasoning System) using ONA (OpenNARS for Applications) for the OpenEnded Philosophy MCP Server.

## Installation Guide

This guide provides comprehensive instructions for installing and configuring NARS (Non-Axiomatic Reasoning System) using ONA (OpenNARS for Applications) for the OpenEnded Philosophy MCP Server.

## Installation

### Python Package (Recommended)

The simplest and only supported method is to install ONA as a Python package:

```bash
# Using uv (recommended)
uv add ona

# Or using pip
pip install ona
```

This method automatically handles dependencies and places the NAR executable in your Python environment.

## Configuration

### Environment Variables

Configure NARS behavior through environment variables in your `.env` file:

```bash
# Memory configuration
NARS_MEMORY_SIZE=1000          # Number of concepts in memory
NARS_INFERENCE_STEPS=50        # Steps per inference cycle

# Operational settings
NARS_SILENT_MODE=true          # Suppress ONA output
NARS_DECISION_THRESHOLD=0.6    # Threshold for decision making

# Logging
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/philosophy_mcp.log
```

### Truth Value Interpretation

NARS uses two-dimensional truth values:
- **Frequency**: How often something is true (0.0 to 1.0)
- **Confidence**: How certain we are about the frequency (0.0 to 1.0)

Example interpretations:
- `<0.9, 0.8>`: "Usually true with high confidence"
- `<0.5, 0.9>`: "Equally true/false with high confidence" 
- `<0.8, 0.3>`: "Often true but low confidence"

## Verification

After installation, verify ONA is working:

```bash
uv run python -c "from openended_philosophy.nars import NARSManager; print('NARS integration ready')"
```

## Troubleshooting

### ONA Not Found

If you encounter a `FileNotFoundError` related to the ONA executable, ensure that the `ona` Python package is correctly installed in your environment. You can try reinstalling it:

```bash
uv pip install ona --force-reinstall
```

### Process Management Issues

If ONA processes accumulate:

1. Check for zombie processes:
   ```bash
   ps aux | grep NAR
   ```

2. Kill orphaned processes:
   ```bash
   pkill -f NAR
   ```

### Memory Issues

For large-scale reasoning:
- Increase `NARS_MEMORY_SIZE` for more concepts
- Adjust `NARS_INFERENCE_STEPS` for deeper reasoning
- Monitor system memory usage

## Integration with Philosophy MCP

The NARS integration enhances philosophical reasoning by:

1. **Truth Maintenance**: Automatic belief revision when contradictions arise
2. **Temporal Reasoning**: Handle time-dependent philosophical concepts
3. **Evidence Tracking**: Maintain provenance of beliefs and conclusions
4. **Uncertainty Quantification**: Represent epistemic uncertainty explicitly

## Advanced Configuration

### Custom NARS Rules

Add custom inference rules by modifying NAL (Non-Axiomatic Logic) settings:

```python
# In your code
async with NARSManager() as nars:
    # Add custom rules
    await nars.query("*stampstoresize=1000")
    await nars.query("*volume=50")  # Partial output
```

### Performance Tuning

For optimal performance:
- Use `NARS_SILENT_MODE=true` in production
- Adjust `NARS_MEMORY_SIZE` based on domain complexity
- Monitor inference times and adjust `NARS_INFERENCE_STEPS`

## Resources

- [ONA Documentation](https://github.com/opennars/OpenNARS-for-Applications/wiki)
- [NARS Theory](https://www.cis.temple.edu/~pwang/NARS-Intro.html)
- [NAL Specification](https://github.com/opennars/opennars/wiki/Non-Axiomatic-Logic-(NAL),-Logic-Rules)
