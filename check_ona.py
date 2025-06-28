#!/usr/bin/env python3
"""
Check if ONA (OpenNARS for Applications) is built and available.
"""
# ona is a pip package and is installed with uv so this is dumb. it is added to the pyproject.toml already
import os
import sys
from pathlib import Path


def check_ona():
    """Check if ONA executable exists."""
    script_dir = Path(__file__).parent.resolve()
    ona_paths = [
        Path("/home/ty/Repositories/ai_workspace/NARS-GPT/OpenNARS-for-Applications/NAR"),
        (script_dir / "../NARS-GPT/OpenNARS-for-Applications/NAR").resolve(),
    ]
    ona_paths.extend([
        Path("../NARS-GPT/OpenNARS-for-Applications/NAR"),
    ])
    for path in ona_paths:
        if path.exists() and os.access(path, os.X_OK):
            print(f"✓ ONA found at: {path}")
            return True
            return True

    print("✗ ONA executable not found!")
    print("\nTo build ONA:")
    print("1. cd <path-to-NARS-GPT>")
    print("2. ./build.sh")
    print("\nSee the NARS-GPT or ONA documentation for more details.")
    print("\nNote: The philosophy MCP will still work without ONA,")
    print("but NARS reasoning features will be disabled.")

    return False

if __name__ == "__main__":
    sys.exit(0 if check_ona() else 1)
