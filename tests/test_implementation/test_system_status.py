#!/usr/bin/env python3
"""
Test Implementation Status and Fix Issues
==========================================

This script tests the current state of the OpenEnded Philosophy MCP system
and identifies what needs to be implemented or fixed.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openended_philosophy.core import (
    LanguageGameProcessor,
    DynamicPluralismFramework,
    CoherenceLandscape,
    FallibilisticInference
)

from openended_philosophy.nars import (
    NARSManager,
    NARSMemory,
    NARSReasoning
)

from openended_philosophy.operations import PhilosophicalOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_components():
    """Test basic component creation and functionality."""
    logger.info("Testing basic component creation...")
    
    # Test LanguageGameProcessor
    try:
        processor = LanguageGameProcessor(
            "scientific_discourse",
            {"empirical_verification": True, "peer_review": True}
        )
        logger.info("✓ LanguageGameProcessor created successfully")
    except Exception as e:
        logger.error(f"✗ LanguageGameProcessor failed: {e}")
        return False
    
    # Test DynamicPluralismFramework
    try:
        framework = DynamicPluralismFramework(openness_coefficient=0.9)
        logger.info("✓ DynamicPluralismFramework created successfully")
    except Exception as e:
        logger.error(f"✗ DynamicPluralismFramework failed: {e}")
        return False
    
    # Test CoherenceLandscape
    try:
        landscape = CoherenceLandscape(dimensionality='variable')
        logger.info("✓ CoherenceLandscape created successfully")
    except Exception as e:
        logger.error(f"✗ CoherenceLandscape failed: {e}")
        return False
    
    # Test FallibilisticInference
    try:
        inference = FallibilisticInference()
        logger.info("✓ FallibilisticInference created successfully")
    except Exception as e:
        logger.error(f"✗ FallibilisticInference failed: {e}")
        return False
    
    return True

async def test_nars_components():
    """Test NARS component creation and basic functionality."""
    logger.info("Testing NARS components...")
    
    # Test NARSManager
    try:
        nars_manager = NARSManager()
        logger.info("✓ NARSManager created successfully")
    except Exception as e:
        logger.error(f"✗ NARSManager failed: {e}")
        return False
    
    # Test NARSMemory
    try:
        memory_file = Path("test_memory.json")
        nars_memory = NARSMemory(
            memory_file=memory_file,
            attention_size=30,
            recency_size=10
        )
        logger.info("✓ NARSMemory created successfully")
    except Exception as e:
        logger.error(f"✗ NARSMemory failed: {e}")
        return False
    
    # Test NARSReasoning
    try:
        nars_reasoning = NARSReasoning(
            nars_manager=nars_manager,
            nars_memory=nars_memory
        )
        logger.info("✓ NARSReasoning created successfully")
    except Exception as e:
        logger.error(f"✗ NARSReasoning failed: {e}")
        return False
    
    return True

async def test_philosophical_operations():
    """Test PhilosophicalOperations integration."""
    logger.info("Testing PhilosophicalOperations...")
    
    try:
        # Create base components
        pluralism_framework = DynamicPluralismFramework(openness_coefficient=0.9)
        coherence_landscape = CoherenceLandscape(dimensionality='variable')
        inference_engine = FallibilisticInference()
        
        # Create language games
        language_games = {
            "scientific_discourse": LanguageGameProcessor(
                "scientific",
                {"empirical_verification": True, "peer_review": True}
            ),
            "ethical_deliberation": LanguageGameProcessor(
                "ethical",
                {"normative_reasoning": True, "value_pluralism": True}
            )
        }
        
        # Create NARS components
        nars_manager = NARSManager()
        memory_file = Path("test_operations_memory.json")
        nars_memory = NARSMemory(memory_file=memory_file)
        nars_reasoning = NARSReasoning(
            nars_manager=nars_manager,
            nars_memory=nars_memory
        )
        
        # Create PhilosophicalOperations
        operations = PhilosophicalOperations(
            pluralism_framework=pluralism_framework,
            coherence_landscape=coherence_landscape,
            inference_engine=inference_engine,
            language_games=language_games,
            nars_manager=nars_manager,
            nars_memory=nars_memory,
            nars_reasoning=nars_reasoning
        )
        
        logger.info("✓ PhilosophicalOperations created successfully")
        
        # Test if enhanced modules are available
        if operations.llm_processor:
            logger.info("✓ Enhanced LLM processor available")
        else:
            logger.warning("⚠ Enhanced LLM processor not available")
        
        if operations.philosophical_ontology:
            logger.info("✓ Philosophical ontology available")
        else:
            logger.warning("⚠ Philosophical ontology not available")
        
        if operations.semantic_embedding_space:
            logger.info("✓ Semantic embedding space available")
        else:
            logger.warning("⚠ Semantic embedding space not available")
        
        if operations.insight_synthesis:
            logger.info("✓ Insight synthesis available")
        else:
            logger.warning("⚠ Insight synthesis not available")
        
        if operations.recursive_analyzer:
            logger.info("✓ Recursive analyzer available")
        else:
            logger.warning("⚠ Recursive analyzer not available")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PhilosophicalOperations failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_basic_analysis():
    """Test basic concept analysis functionality."""
    logger.info("Testing basic concept analysis...")
    
    try:
        # Create minimal system
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
        memory_file = Path("test_analysis_memory.json")
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
        
        # Test basic concept analysis (fallback mode)
        result = await operations._fallback_concept_analysis(
            "consciousness",
            "philosophy_of_mind"
        )
        
        logger.info("✓ Basic concept analysis completed")
        logger.info(f"Result: {result['status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def test_enhanced_analysis():
    """Test enhanced concept analysis if available."""
    logger.info("Testing enhanced concept analysis...")
    
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
        memory_file = Path("test_enhanced_memory.json")
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
        
        # Test enhanced analysis if available
        if hasattr(operations, 'analyze_concept_enhanced'):
            logger.info("Enhanced analysis method available, testing...")
            
            result = await operations.analyze_concept_enhanced(
                concept="consciousness",
                context="philosophy_of_mind",
                perspectives=["materialist", "dualist"],
                confidence_threshold=0.7,
                enable_recursive_analysis=False  # Disable for initial test
            )
            
            logger.info("✓ Enhanced concept analysis completed")
            logger.info(f"Result keys: {list(result.keys())}")
            
            if 'error' in result:
                logger.warning(f"⚠ Enhanced analysis had error: {result['error']}")
                return False
            else:
                logger.info("✓ Enhanced analysis successful")
                return True
        else:
            logger.warning("⚠ Enhanced analysis method not available")
            return False
        
    except Exception as e:
        logger.error(f"✗ Enhanced analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run all tests and report status."""
    logger.info("Starting OpenEnded Philosophy implementation test...")
    
    tests = [
        ("Basic Components", test_basic_components()),
        ("NARS Components", test_nars_components()),
        ("Philosophical Operations", test_philosophical_operations()),
        ("Basic Analysis", test_basic_analysis()),
        ("Enhanced Analysis", test_enhanced_analysis())
    ]
    
    results = {}
    for test_name, test_coro in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_coro
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name}: PASSED")
            else:
                logger.warning(f"⚠ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is functional.")
    else:
        logger.warning(f"⚠ {total - passed} tests failed. Some functionality needs work.")
    
    # Cleanup test files
    for test_file in ["test_memory.json", "test_operations_memory.json", 
                      "test_analysis_memory.json", "test_enhanced_memory.json"]:
        Path(test_file).unlink(missing_ok=True)
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
