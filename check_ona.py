#!/usr/bin/env python3
"""
Check if ONA (OpenNARS for Applications) is built and available.
"""

import os
import sys
from pathlib import Path

def check_ona():
    """Check if ONA executable exists."""
    ona_paths = [
        Path("/home/ty/Repositories/ai_workspace/NARS-GPT/OpenNARS-for-Applications/NAR"),
        Path("../NARS-GPT/OpenNARS-for-Applications/NAR"),
    ]
    
    for path in ona_paths:
        if path.exists() and path.is_file():
            print(f"✓ ONA found at: {path}")
            return True
    
    print("✗ ONA executable not found!")
    print("\nTo build ONA:")
    print("1. cd /home/ty/Repositories/ai_workspace/NARS-GPT")
    print("2. ./build.sh")
    print("\nNote: The philosophy MCP will still work without ONA,")
    print("but NARS reasoning features will be disabled.")
    
    return False

if __name__ == "__main__":
    sys.exit(0 if check_ona() else 1)
