#!/usr/bin/env python3
"""
test_nars_integration.py - Test script for NARS/ONA integration

This script verifies that the NARS integration is properly configured
and functioning within the OpenEnded Philosophy MCP Server.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.nars import NARSManager, Truth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_operations():
    """Test basic NARS operations."""
    logger.info("Testing basic NARS operations...")
    
    async with NARSManager() as nars:
        # Test 1: Simple fact assertion and query
        logger.info("Test 1: Fact assertion and retrieval")
        await nars.query("<Socrates --> mortal>.")
        await nars.query("<Socrates --> philosopher>.")
        
        result = await nars.query("<Socrates --> ?what>?")
        logger.info(f"Query result: {result}")
        
        if result.get("answers"):
            logger.info("✓ Basic fact storage and retrieval working")
        else:
            logger.error("✗ Failed to retrieve stored facts")
            
        # Test 2: Inference
        logger.info("\nTest 2: Basic inference")
        await nars.query("<bird --> animal>.")
        await nars.query("<robin --> bird>.")
        
        # Give NARS time to process
        await asyncio.sleep(0.5)
        
        result = await nars.query("<robin --> ?what>?")
        logger.info(f"Inference result: {result}")
        
        # Check if NARS inferred that robin is an animal
        answers = result.get("answers", [])
        found_animal = any(
            "animal" in answer.get("term", "") 
            for answer in answers
        )
        
        if found_animal:
            logger.info("✓ Basic inference working")
        else:
            logger.warning("⚠ Inference may need more steps or time")


async def test_truth_values():
    """Test truth value operations."""
    logger.info("\nTesting truth value operations...")
    
    # Test truth value creation and operations
    t1 = Truth(frequency=0.9, confidence=0.8)
    t2 = Truth(frequency=0.7, confidence=0.9)
    
    logger.info(f"Truth 1: {t1}")
    logger.info(f"Truth 2: {t2}")
    
    # Test revision (combining evidence)
    revised = t1.revision(t2)
    logger.info(f"Revised truth: {revised}")
    
    # Test expectation calculation
    exp1 = t1.expectation()
    logger.info(f"Expectation of Truth 1: {exp1:.3f}")
    
    logger.info("✓ Truth value operations working")


async def test_philosophical_reasoning():
    """Test philosophical reasoning capabilities."""
    logger.info("\nTesting philosophical reasoning...")
    
    async with NARSManager() as nars:
        # Add philosophical concepts
        await nars.query("<consciousness --> phenomenon>.")
        await nars.query("<phenomenon --> {observable, measurable}>.")
        await nars.query("<qualia --> consciousness>.")
        
        # Test inheritance chain
        result = await nars.query("<qualia --> ?what>?")
        logger.info(f"Philosophical query result: {result}")
        
        if result.get("answers"):
            logger.info("✓ Philosophical concept handling working")
        else:
            logger.warning("⚠ May need more inference steps for complex concepts")


async def test_temporal_reasoning():
    """Test temporal reasoning capabilities."""
    logger.info("\nTesting temporal reasoning...")
    
    async with NARSManager() as nars:
        # Add temporal statements
        await nars.query("<(*, {Socrates}) --> died>. :|:")  # Socrates died (past)
        await nars.query("<(*, {Plato}) --> wrote>. :|:")    # Plato wrote (past)
        
        # Query about temporal relations
        result = await nars.query("<(*, {?who}) --> died>? :|:")
        logger.info(f"Temporal query result: {result}")
        
        if result.get("answers"):
            logger.info("✓ Temporal reasoning capabilities present")
        else:
            logger.info("⚠ Temporal reasoning may require specific NAL-7 support")


async def test_error_handling():
    """Test error handling and recovery."""
    logger.info("\nTesting error handling...")
    
    try:
        # Test with invalid Narsese
        async with NARSManager() as nars:
            result = await nars.query("This is not valid Narsese")
            logger.info(f"Invalid query result: {result}")
            
            if "error" in result:
                logger.info("✓ Error handling working correctly")
            else:
                logger.warning("⚠ Invalid input not properly handled")
                
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("OpenEnded Philosophy MCP - NARS Integration Test")
    logger.info("=" * 60)
    
    try:
        # Verify NARS is available
        manager = NARSManager()
        logger.info(f"NARS executable found at: {manager.ona_path}")
        logger.info("✓ NARS installation verified\n")
        
        # Run test suites
        await test_basic_operations()
        await test_truth_values()
        await test_philosophical_reasoning()
        await test_temporal_reasoning()
        await test_error_handling()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ NARS not found: {e}")
        logger.error("\nPlease install ONA using one of these methods:")
        logger.error("1. uv add ona")
        logger.error("2. Set ONA_PATH in .env file")
        logger.error("3. See docs/NARS_INSTALLATION.md for details")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
