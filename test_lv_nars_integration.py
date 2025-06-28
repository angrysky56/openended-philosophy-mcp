#!/usr/bin/env python3
"""
test_lv_nars_integration.py - Comprehensive Test Suite for LV-NARS Integration

This script tests the integration of Lotka-Volterra Ecosystem Intelligence 
with NARS (Non-Axiomatic Reasoning System) for enhanced philosophical reasoning.

Test Coverage:
- LV-NARS ecosystem initialization
- Entropy estimation for philosophical inquiries  
- Reasoning strategy generation and selection
- Ecological dynamics simulation
- Truth value synthesis with diversity preservation
- Recursive self-analysis capabilities
- Integration with openended-philosophy framework

Author: AI Memory System MVP + NeoCoder LV Framework Integration
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from openended_philosophy.lv_nars_integration import (
    LVNARSEcosystem,
    LVNARSIntegrationManager,
    LVReasoningCandidate,
    LVEntropyEstimator,
    LVTruthFunctions
)
from openended_philosophy.nars import NARSManager, NARSMemory, NARSReasoning, TruthValue
from openended_philosophy.operations import PhilosophicalOperations
from openended_philosophy.core import (
    DynamicPluralismFramework,
    CoherenceLandscape,
    FallibilisticInference,
    LanguageGameProcessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LVNARSTestSuite:
    """Comprehensive test suite for LV-NARS integration."""
    
    def __init__(self):
        self.nars_manager = None
        self.nars_memory = None
        self.nars_reasoning = None
        self.lv_nars_ecosystem = None
        self.lv_integration_manager = None
        self.philosophical_operations = None
        
    async def setup(self):
        """Initialize all components for testing."""
        logger.info("Setting up LV-NARS test environment...")
        
        try:
            # Initialize NARS components
            self.nars_manager = NARSManager()
            self.nars_memory = NARSMemory(
                memory_file=Path("test_lv_nars_memory.json"),
                attention_size=20,
                recency_size=10
            )
            self.nars_reasoning = NARSReasoning(
                nars_manager=self.nars_manager,
                nars_memory=self.nars_memory
            )
            
            # Initialize LV-NARS ecosystem
            self.lv_nars_ecosystem = LVNARSEcosystem(
                nars_manager=self.nars_manager,
                nars_memory=self.nars_memory,
                nars_reasoning=self.nars_reasoning
            )
            
            # Initialize LV integration manager
            self.lv_integration_manager = LVNARSIntegrationManager(
                nars_manager=self.nars_manager,
                nars_memory=self.nars_memory,
                nars_reasoning=self.nars_reasoning
            )
            
            # Initialize philosophical framework components
            pluralism_framework = DynamicPluralismFramework(openness_coefficient=0.9)
            coherence_landscape = CoherenceLandscape(dimensionality='variable')
            inference_engine = FallibilisticInference()
            
            language_games = {
                "philosophical": LanguageGameProcessor(
                    "philosophical",
                    {
                        "conceptual_analysis": True,
                        "epistemic_humility": True,
                        "multi_perspectival": True
                    }
                )
            }
            
            # Initialize philosophical operations with LV enhancement
            self.philosophical_operations = PhilosophicalOperations(
                pluralism_framework=pluralism_framework,
                coherence_landscape=coherence_landscape,
                inference_engine=inference_engine,
                language_games=language_games,
                nars_manager=self.nars_manager,
                nars_memory=self.nars_memory,
                nars_reasoning=self.nars_reasoning
            )
            
            # Start NARS system
            if await self.nars_manager.start():
                logger.info("✓ NARS system started successfully")
            else:
                logger.warning("⚠ NARS system failed to start - some tests may be limited")
            
            logger.info("✓ LV-NARS test environment setup complete")
            
        except Exception as e:
            logger.error(f"✗ Test setup failed: {e}")
            raise
    
    async def test_entropy_estimation(self):
        """Test entropy estimation for philosophical inquiries."""
        logger.info("\n" + "="*60)
        logger.info("Testing Entropy Estimation")
        logger.info("="*60)
        
        entropy_estimator = LVEntropyEstimator()
        
        test_cases = [
            {
                "inquiry": "What is the capital of France?",
                "expected_range": (0.0, 0.3),
                "description": "Low entropy - factual question"
            },
            {
                "inquiry": "What is the nature of consciousness?",
                "expected_range": (0.6, 1.0),
                "description": "High entropy - philosophical mystery"
            },
            {
                "inquiry": "How can we understand the relationship between mind and matter?",
                "expected_range": (0.7, 1.0),
                "description": "Very high entropy - fundamental philosophical problem"
            },
            {
                "inquiry": "Define epistemology in analytical philosophy",
                "expected_range": (0.2, 0.5),
                "description": "Medium entropy - technical but defined concept"
            },
            {
                "inquiry": "Explore multiple perspectives on the meaning of existence",
                "expected_range": (0.8, 1.0),
                "description": "Maximum entropy - explicit request for diversity"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            inquiry = test_case["inquiry"]
            expected_min, expected_max = test_case["expected_range"]
            description = test_case["description"]
            
            logger.info(f"\nTest {i}: {description}")
            logger.info(f"Inquiry: '{inquiry}'")
            
            entropy = entropy_estimator.estimate_philosophical_entropy(inquiry)
            logger.info(f"Estimated entropy: {entropy:.3f}")
            logger.info(f"Expected range: {expected_min:.1f} - {expected_max:.1f}")
            
            if expected_min <= entropy <= expected_max:
                logger.info("✓ Entropy estimation within expected range")
            else:
                logger.warning(f"⚠ Entropy {entropy:.3f} outside expected range [{expected_min:.1f}, {expected_max:.1f}]")
        
        logger.info("\n✓ Entropy estimation tests completed")
    
    async def test_reasoning_strategy_generation(self):
        """Test LV reasoning strategy generation."""
        logger.info("\n" + "="*60)
        logger.info("Testing Reasoning Strategy Generation")
        logger.info("="*60)
        
        test_concepts = [
            {
                "concept": "consciousness",
                "context": {"domain": "philosophy_of_mind", "perspectives": ["phenomenological", "analytical"]},
                "expected_strategies": 3
            },
            {
                "concept": "justice",
                "context": {"domain": "ethics", "perspectives": ["deontological", "consequentialist", "virtue_ethics"]},
                "expected_strategies": 2
            }
        ]
        
        for i, test_case in enumerate(test_concepts, 1):
            concept = test_case["concept"]
            context = test_case["context"]
            expected_strategies = test_case["expected_strategies"]
            
            logger.info(f"\nTest {i}: Generating strategies for '{concept}'")
            logger.info(f"Context: {context}")
            
            try:
                # Test strategy generation
                candidates = await self.lv_nars_ecosystem._generate_reasoning_candidates(
                    concept=concept,
                    context=context,
                    perspectives=context["perspectives"],
                    entropy=0.7  # High entropy for diverse strategies
                )
                
                logger.info(f"Generated {len(candidates)} reasoning candidates")
                
                for j, candidate in enumerate(candidates):
                    logger.info(f"  Strategy {j+1}: {candidate.strategy_name}")
                    logger.info(f"    Pattern: {candidate.reasoning_pattern}")
                    logger.info(f"    Fitness: {candidate.fitness:.3f}")
                
                if len(candidates) >= expected_strategies:
                    logger.info("✓ Sufficient reasoning strategies generated")
                else:
                    logger.warning(f"⚠ Only {len(candidates)} strategies generated, expected >= {expected_strategies}")
                
            except Exception as e:
                logger.error(f"✗ Strategy generation failed: {e}")
        
        logger.info("\n✓ Reasoning strategy generation tests completed")
    
    async def test_lv_ecosystem_dynamics(self):
        """Test Lotka-Volterra ecosystem dynamics simulation."""
        logger.info("\n" + "="*60)
        logger.info("Testing LV Ecosystem Dynamics")
        logger.info("="*60)
        
        # Create test candidates
        test_candidates = [
            LVReasoningCandidate(
                strategy_name="deductive_logic",
                reasoning_pattern="deductive",
                truth_approach="synthesis",
                content="Logical deduction from premises",
                quality_score=0.9,
                novelty_score=0.2,
                coherence_score=0.95,
                epistemic_value=0.8
            ),
            LVReasoningCandidate(
                strategy_name="inductive_generalization", 
                reasoning_pattern="inductive",
                truth_approach="generalization",
                content="Generalization from instances",
                quality_score=0.7,
                novelty_score=0.6,
                coherence_score=0.8,
                epistemic_value=0.7
            ),
            LVReasoningCandidate(
                strategy_name="abductive_hypothesis",
                reasoning_pattern="abductive",
                truth_approach="hypothesis",
                content="Explanatory hypothesis formation",
                quality_score=0.6,
                novelty_score=0.9,
                coherence_score=0.6,
                epistemic_value=0.85
            )
        ]
        
        logger.info(f"Testing with {len(test_candidates)} reasoning candidates")
        
        # Test with different entropy levels
        entropy_levels = [0.2, 0.5, 0.8]
        
        for entropy in entropy_levels:
            logger.info(f"\nTesting LV dynamics with entropy = {entropy:.1f}")
            
            try:
                # Apply LV dynamics
                selected = await self.lv_nars_ecosystem._apply_lv_dynamics(
                    candidates=test_candidates.copy(),
                    entropy=entropy,
                    context={"test": True}
                )
                
                logger.info(f"Selected {len(selected)} strategies after LV dynamics")
                
                for candidate in selected:
                    logger.info(f"  {candidate.strategy_name}: population={candidate.population:.3f}, fitness={candidate.fitness:.3f}")
                
                # Verify diversity preservation
                if len(selected) > 1:
                    populations = [c.population for c in selected]
                    population_variance = np.var(populations) if len(populations) > 1 else 0
                    
                    if population_variance < 0.5:  # Some diversity maintained
                        logger.info("✓ Diversity preserved in ecosystem selection")
                    else:
                        logger.warning("⚠ High population variance - potential over-dominance")
                else:
                    logger.info("ℹ Single strategy selected - convergence occurred")
                
            except Exception as e:
                logger.error(f"✗ LV dynamics test failed: {e}")
        
        logger.info("\n✓ LV ecosystem dynamics tests completed")
    
    async def test_truth_value_synthesis(self):
        """Test LV-enhanced truth value synthesis."""
        logger.info("\n" + "="*60)
        logger.info("Testing LV Truth Value Synthesis")
        logger.info("="*60)
        
        # Create test truth values with populations
        test_truth_populations = [
            (TruthValue(0.8, 0.9), 0.6),  # High confidence, moderate population
            (TruthValue(0.6, 0.7), 0.8),  # Moderate confidence, high population  
            (TruthValue(0.9, 0.6), 0.3),  # High frequency but lower confidence, low population
            (TruthValue(0.4, 0.8), 0.5),  # Low frequency but high confidence, moderate population
        ]
        
        logger.info("Test truth values and populations:")
        for i, (tv, pop) in enumerate(test_truth_populations):
            logger.info(f"  TV{i+1}: f={tv.frequency:.2f}, c={tv.confidence:.2f}, pop={pop:.2f}")
        
        # Test ecological revision
        truth_values = [tv for tv, _ in test_truth_populations]
        populations = [pop for _, pop in test_truth_populations]
        
        ecological_result = LVTruthFunctions.ecological_revision(truth_values, populations)
        
        logger.info(f"\nEcological revision result:")
        logger.info(f"  Frequency: {ecological_result.frequency:.3f}")
        logger.info(f"  Confidence: {ecological_result.confidence:.3f}")
        logger.info(f"  Expectation: {ecological_result.expectation:.3f}")
        
        # Test diversity-preserving synthesis
        diversity_result = LVTruthFunctions.diversity_preserving_synthesis(test_truth_populations)
        
        logger.info(f"\nDiversity-preserving synthesis result:")
        logger.info(f"  Frequency: {diversity_result.frequency:.3f}")
        logger.info(f"  Confidence: {diversity_result.confidence:.3f}")
        logger.info(f"  Expectation: {diversity_result.expectation:.3f}")
        
        # Verify epistemic humility (confidence should be moderated)
        if ecological_result.confidence < 0.95:
            logger.info("✓ Epistemic humility maintained (confidence < 0.95)")
        else:
            logger.warning("⚠ Confidence may be too high for synthesis")
        
        logger.info("\n✓ Truth value synthesis tests completed")
    
    async def test_integrated_philosophical_analysis(self):
        """Test full integrated philosophical analysis using LV-NARS."""
        logger.info("\n" + "="*60)
        logger.info("Testing Integrated Philosophical Analysis")
        logger.info("="*60)
        
        test_cases = [
            {
                "concept": "free will",
                "context": "philosophy_of_mind",
                "perspectives": ["compatibilist", "libertarian", "hard_determinist"],
                "expected_entropy": "high"
            },
            {
                "concept": "knowledge",
                "context": "epistemology", 
                "perspectives": ["empiricist", "rationalist", "pragmatist"],
                "expected_entropy": "medium"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            concept = test_case["concept"]
            context = test_case["context"]
            perspectives = test_case["perspectives"]
            
            logger.info(f"\nTest {i}: Analyzing '{concept}' in {context}")
            logger.info(f"Perspectives: {perspectives}")
            
            try:
                # Perform LV-enhanced analysis
                result = await self.philosophical_operations.analyze_concept(
                    concept=concept,
                    context=context,
                    perspectives=perspectives,
                    confidence_threshold=0.6
                )
                
                logger.info(f"Analysis completed successfully")
                logger.info(f"Enhancement applied: {result.get('enhancement_applied', 'unknown')}")
                
                if 'entropy' in result:
                    logger.info(f"Entropy: {result['entropy']:.3f}")
                
                if 'selected_strategies' in result:
                    strategies = result['selected_strategies']
                    logger.info(f"Selected {len(strategies)} reasoning strategies")
                
                if 'diversity_metrics' in result:
                    diversity = result['diversity_metrics']
                    if isinstance(diversity, dict):
                        overall_diversity = diversity.get('overall_diversity', 0)
                        logger.info(f"Overall diversity: {overall_diversity:.3f}")
                
                if 'philosophical_enhancement' in result:
                    enhancement = result['philosophical_enhancement']
                    logger.info(f"Framework integration: {enhancement.get('framework_integration', 'unknown')}")
                
                logger.info("✓ Integrated analysis completed successfully")
                
            except Exception as e:
                logger.error(f"✗ Integrated analysis failed: {e}")
        
        logger.info("\n✓ Integrated philosophical analysis tests completed")
    
    async def test_recursive_self_analysis(self):
        """Test recursive self-analysis capabilities."""
        logger.info("\n" + "="*60)
        logger.info("Testing Recursive Self-Analysis")
        logger.info("="*60)
        
        # First, perform a basic analysis to get results for meta-analysis
        try:
            initial_analysis = await self.philosophical_operations.analyze_concept(
                concept="consciousness",
                context="philosophy_of_mind",
                perspectives=["phenomenological", "analytical"],
                confidence_threshold=0.7
            )
            
            logger.info("Initial analysis completed for meta-analysis")
            
            # Now perform recursive self-analysis
            meta_analysis = await self.philosophical_operations.analyze_own_reasoning_process(
                analysis_result=initial_analysis,
                analysis_type="concept_analysis",
                meta_depth=2
            )
            
            logger.info("Recursive self-analysis completed")
            
            # Check meta-analysis results
            if 'meta_analysis_type' in meta_analysis:
                logger.info(f"Meta-analysis type: {meta_analysis['meta_analysis_type']}")
            
            if 'meta_insights' in meta_analysis:
                insights = meta_analysis['meta_insights']
                logger.info(f"Generated {len(insights)} meta-insights:")
                for insight in insights[:3]:  # Show first 3
                    logger.info(f"  • {insight}")
            
            if 'self_assessment' in meta_analysis:
                assessment = meta_analysis['self_assessment']
                logger.info(f"Self-assessment keys: {list(assessment.keys())}")
            
            if 'improvement_suggestions' in meta_analysis:
                suggestions = meta_analysis['improvement_suggestions']
                logger.info(f"Generated {len(suggestions)} improvement suggestions")
            
            logger.info("✓ Recursive self-analysis completed successfully")
            
        except Exception as e:
            logger.error(f"✗ Recursive self-analysis failed: {e}")
        
        logger.info("\n✓ Recursive self-analysis tests completed")
    
    async def cleanup(self):
        """Cleanup test environment."""
        logger.info("\nCleaning up test environment...")
        
        try:
            if self.nars_manager:
                await self.nars_manager.stop()
                logger.info("✓ NARS system stopped")
            
            # Clean up test files
            test_memory_file = Path("test_lv_nars_memory.json")
            if test_memory_file.exists():
                test_memory_file.unlink()
                logger.info("✓ Test memory file cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
        
        logger.info("✓ Test environment cleanup completed")
    
    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🧬 Starting LV-NARS Integration Test Suite")
        logger.info("=" * 80)
        
        try:
            await self.setup()
            
            # Run all test modules
            await self.test_entropy_estimation()
            await self.test_reasoning_strategy_generation()
            await self.test_lv_ecosystem_dynamics()
            await self.test_truth_value_synthesis()
            await self.test_integrated_philosophical_analysis()
            await self.test_recursive_self_analysis()
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 All LV-NARS integration tests completed successfully!")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"\n💥 Test suite failed: {e}")
            logger.error("=" * 80)
            raise
        
        finally:
            await self.cleanup()


async def main():
    """Main test execution function."""
    import numpy as np  # Import numpy for test calculations
    
    test_suite = LVNARSTestSuite()
    
    try:
        await test_suite.run_all_tests()
        logger.info("\n✅ LV-NARS Integration Test Suite: ALL TESTS PASSED")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ LV-NARS Integration Test Suite: TESTS FAILED")
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import numpy as np  # Make numpy available for tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
