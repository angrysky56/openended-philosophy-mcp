"""
OpenEnded Philosophy MCP Server Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Main entry point for running the philosophical framework as an MCP server.
This module handles server initialization and lifecycle management.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from openended_philosophy.server import OpenEndedPhilosophyServer, main

if __name__ == "__main__":
    """
    Execute the OpenEnded Philosophy MCP Server.
    
    ### Execution Flow
    1. Initialize server instance
    2. Configure MCP handlers
    3. Start async event loop
    4. Handle graceful shutdown
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nServer shutdown initiated by user")
    except Exception as e:
        print(f"\n\nServer error: {e}")
        sys.exit(1)
