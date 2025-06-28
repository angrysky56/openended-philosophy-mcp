#!/usr/bin/env python3
"""
Debug to_dict() Conversion Issues
===============================

This script will test the specific to_dict() conversions that might be causing the error.
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

async def test_to_dict_conversions():
    """Test specific to_dict() conversions that might be failing."""
    
    logger.info("Testing to_dict() conversions...")
    
    try:
        # Create operations system
        pluralism_framework = DynamicPluralismFramework()
        coherence_landscape = CoherenceLandscape()
        inference_engine = FallibilisticInference()
        language_games = {"scientific_discourse": LanguageGameProcessor("scientific", {"empirical_verification": True})}
        nars_manager = NARSManager()
        memory_file = Path("test_dict_memory.json")
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
        
        # Get the components
        context = operations._create_philosophical_context("philosophy_of_mind", ["materialist"])
        semantic_analysis = await operations._perform_semantic_analysis("consciousness", context)
        philosophical_category = operations._categorize_philosophically(semantic_analysis)
        memory_item = await operations._integrate_with_nars_memory("consciousness", semantic_analysis, philosophical_category, context)
        
        # Test 1: semantic_analysis.to_dict()
        logger.info("Testing semantic_analysis.to_dict()...")
        try:
            if semantic_analysis:
                result = semantic_analysis.to_dict()
                logger.info("✓ semantic_analysis.to_dict() successful")
            else:
                logger.warning("⚠ semantic_analysis is None")
        except Exception as e:
            logger.error(f"✗ semantic_analysis.to_dict() failed: {e}")
            logger.error(traceback.format_exc())
        
        # Test 2: philosophical_category.to_dict()
        logger.info("Testing philosophical_category.to_dict()...")
        try:
            if philosophical_category:
                result = philosophical_category.to_dict()
                logger.info("✓ philosophical_category.to_dict() successful")
            else:
                logger.warning("⚠ philosophical_category is None")
        except Exception as e:
            logger.error(f"✗ philosophical_category.to_dict() failed: {e}")
            logger.error(traceback.format_exc())
        
        # Test 3: memory_item.to_dict()
        logger.info("Testing memory_item.to_dict()...")
        try:
            if memory_item:
                result = memory_item.to_dict()
                logger.info("✓ memory_item.to_dict() successful")
            else:
                logger.warning("⚠ memory_item is None")
        except Exception as e:
            logger.error(f"✗ memory_item.to_dict() failed: {e}")
            logger.error(traceback.format_exc())
        
        # Test 4: Try the full analysis_result construction
        logger.info("Testing full analysis_result construction...")
        try:
            insights = []  # Empty for test
            
            analysis_result = {
                'concept': "consciousness",
                'context': "philosophy_of_mind",
                'semantic_analysis': semantic_analysis.to_dict() if semantic_analysis else {},
                'philosophical_category': philosophical_category.to_dict() if philosophical_category else {},
                'nars_memory_item': memory_item.to_dict() if memory_item else {},
                'multi_perspectival_insights': [insight.content for insight in insights] if insights else [],
                'epistemic_uncertainty': semantic_analysis.epistemic_uncertainty if semantic_analysis else {'general': 0.5},
                'revision_triggers': semantic_analysis.revision_triggers if semantic_analysis else [],
                'confidence_assessment': operations._assess_overall_confidence(semantic_analysis, insights),
                'practical_implications': operations._generate_practical_implications(semantic_analysis, insights),
                'further_inquiry_directions': operations._suggest_further_inquiries("consciousness", semantic_analysis),
                'nars_reasoning_trace': await operations._generate_nars_reasoning_trace("consciousness", memory_item)
            }
            
            logger.info("✓ Full analysis_result construction successful")
            logger.info(f"Result keys: {list(analysis_result.keys())}")
            
        except Exception as e:
            logger.error(f"✗ Full analysis_result construction failed: {e}")
            logger.error(traceback.format_exc())
        
        logger.info("🎉 to_dict() testing completed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test setup failed: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cleanup
        Path("test_dict_memory.json").unlink(missing_ok=True)

async def main():
    """Run to_dict() tests."""
    success = await test_to_dict_conversions()
    if success:
        logger.info("✓ to_dict() tests completed")
    else:
        logger.error("✗ to_dict() tests failed")
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
