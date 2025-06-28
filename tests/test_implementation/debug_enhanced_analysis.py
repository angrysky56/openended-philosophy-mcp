#!/usr/bin/env python3
"""
Debug Enhanced Analysis Error
============================

This script will help identify exactly where the 'str' object has no attribute 'value' error is occurring.
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openended_philosophy.core import (
    DynamicPluralismFramework,
    CoherenceLandscape,
    FallibilisticInference,
    LanguageGameProcessor
)

from openended_philosophy.nars import (
    NARSManager,
    NARSMemory,
    NARSReasoning
)

from openended_philosophy.operations import PhilosophicalOperations

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def debug_enhanced_analysis():
    """Debug the enhanced concept analysis step by step."""
    
    logger.info("Starting debug of enhanced concept analysis...")
    
    try:
        # Create full system
        pluralism_framework = DynamicPluralismFramework()
        coherence_landscape = CoherenceLandscape()
        inference_engine = FallibilisticInference()
        
        language_games = {
            "scientific_discourse": LanguageGameProcessor(
                "scientific",
                {"empirical_verification": True}
            )
        }
        
        nars_manager = NARSManager()
        memory_file = Path("debug_memory.json")
        nars_memory = NARSMemory(memory_file=memory_file)
        nars_reasoning = NARSReasoning(nars_manager, nars_memory)
        
        operations = PhilosophicalOperations(
            pluralism_framework=pluralism_framework,
            coherence_landscape=coherence_landscape,
            inference_engine=inference_engine,
            language_games=language_games,
            nars_manager=nars_manager,
            nars_memory=nars_memory,
            nars_reasoning=nars_reasoning
        )
        
        logger.info("✓ Operations created successfully")
        
        # Test step 1: Create philosophical context
        logger.info("Step 1: Creating philosophical context...")
        context = operations._create_philosophical_context("philosophy_of_mind", ["materialist", "dualist"])
        logger.info(f"✓ Context created: {context.domain}")
        
        # Test step 2: Perform semantic analysis
        logger.info("Step 2: Performing semantic analysis...")
        semantic_analysis = await operations._perform_semantic_analysis("consciousness", context)
        if semantic_analysis:
            logger.info(f"✓ Semantic analysis completed, found {len(semantic_analysis.primary_concepts)} concepts")
        else:
            logger.warning("⚠ Semantic analysis returned None")
        
        # Test step 3: Philosophical categorization
        logger.info("Step 3: Performing philosophical categorization...")
        try:
            philosophical_category = operations._categorize_philosophically(semantic_analysis)
            if philosophical_category:
                logger.info(f"✓ Categorization completed: {philosophical_category.primary}")
            else:
                logger.warning("⚠ Categorization returned None")
        except Exception as e:
            logger.error(f"✗ Categorization failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # Test step 4: NARS integration
        logger.info("Step 4: NARS memory integration...")
        try:
            memory_item = await operations._integrate_with_nars_memory(
                "consciousness", semantic_analysis, philosophical_category, context
            )
            if memory_item:
                logger.info("✓ NARS integration successful")
            else:
                logger.warning("⚠ NARS integration returned None")
        except Exception as e:
            logger.error(f"✗ NARS integration failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # Test step 5: Multi-perspectival synthesis
        logger.info("Step 5: Multi-perspectival synthesis...")
        try:
            insights = await operations._synthesize_multi_perspectival_insights(
                "consciousness", ["materialist", "dualist"], semantic_analysis, context
            )
            logger.info(f"✓ Generated {len(insights)} insights")
        except Exception as e:
            logger.error(f"✗ Multi-perspectival synthesis failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info("🎉 All steps completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Debug failed: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cleanup
        Path("debug_memory.json").unlink(missing_ok=True)

async def main():
    """Run debug analysis."""
    success = await debug_enhanced_analysis()
    if success:
        logger.info("✓ Debug completed successfully")
    else:
        logger.error("✗ Debug failed - check logs above")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
