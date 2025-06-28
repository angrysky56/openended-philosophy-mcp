#!/usr/bin/env python3
"""
Test script for OpenEnded Philosophy MCP with NARS Integration
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.server import OpenEndedPhilosophyServer
from openended_philosophy.utils import setup_logging

logger = setup_logging(__name__)


async def test_server():
    """Test basic server functionality."""
    logger.info("Starting OpenEnded Philosophy MCP Server test...")
    
    # Create server instance
    server = OpenEndedPhilosophyServer()
    server.setup_handlers()
    
    logger.info("Server created and handlers configured")
    
    # Test analyze_concept
    logger.info("\n1. Testing analyze_concept...")
    result = await server._analyze_concept(
        concept="consciousness",
        context="neuroscience",
        perspectives=["phenomenological", "naturalist", "functionalist"],
        confidence_threshold=0.6
    )
    
    logger.info(f"Analysis complete. Found {len(result.get('analyses', []))} perspective analyses")
    
    # Test explore_coherence  
    logger.info("\n2. Testing explore_coherence...")
    result = await server._explore_coherence(
        domain="ethics",
        depth=2,
        allow_revision=True
    )
    
    logger.info(f"Coherence exploration complete. Overall coherence: {result.get('overall_coherence', 0):.2f}")
    
    # Test contextualize_meaning
    logger.info("\n3. Testing contextualize_meaning...")
    result = await server._contextualize_meaning(
        expression="truth",
        language_game="scientific_discourse",
        trace_genealogy=False
    )
    
    logger.info(f"Meaning contextualization complete. Stability: {result.get('meaning_stability', 0):.2f}")
    
    # Test generate_insights
    logger.info("\n4. Testing generate_insights...")
    result = await server._generate_insights(
        phenomenon="artificial intelligence",
        perspectives=["analytical", "critical", "pragmatist"],
        depth=2,
        include_contradictions=True
    )
    
    logger.info(f"Insight generation complete. Found {len(result.get('perspective_insights', {}))} perspectives")
    
    # Test hypothesis testing
    logger.info("\n5. Testing test_hypothesis...")
    result = await server._test_hypothesis(
        hypothesis="consciousness emerges from complex information processing",
        test_domains=["neuroscience", "philosophy_of_mind"],
        criteria={"logical_consistency": 0.8, "empirical_adequacy": 0.7}
    )
    
    logger.info(f"Hypothesis testing complete. Overall coherence: {result.get('overall_coherence', 0):.2f}")
    
    logger.info("\n✓ All tests completed successfully!")
    
    # Check NARS status
    if server._nars_initialized:
        logger.info("✓ NARS integration is active")
    else:
        logger.info("✗ NARS integration not available (ONA not found or failed to start)")


async def main():
    """Main test runner."""
    try:
        await test_server()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print("OpenEnded Philosophy MCP Server Test")
    print("=" * 80)
    asyncio.run(main())
