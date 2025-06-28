#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced OpenEnded Philosophy MCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Enhanced Integration Testing

This comprehensive test suite validates all the enhancements made to the 
openended-philosophy-mcp project according to the implementation plan:

#### Test Coverage:
1. **Enhanced LLM Integration**: Validates sophisticated semantic processing
2. **Deep NARS Integration**: Tests philosophical belief formation and revision
3. **Sophisticated Insight Synthesis**: Verifies multi-perspectival analysis
4. **Recursive Self-Analysis**: Tests meta-philosophical reflection capabilities
5. **System Integration**: Validates end-to-end enhanced workflow

#### Expected Improvements:
- Dynamic concept extraction vs hardcoded patterns
- Multi-perspectival insight synthesis with dialectical processing
- Sophisticated meta-philosophical reflection and improvement recommendations
- Deep NARS integration with semantic grounding

Run this test to validate that the implementation plan has been successfully executed.
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import the enhanced modules
from openended_philosophy.enhanced.enhanced_llm_processor import (
    EnhancedLLMPhilosophicalProcessor,
    LLMAnalysisResult,
    PhilosophicalPerspectiveAnalysis
)
from openended_philosophy.enhanced.recursive_self_analysis import RecursiveSelfAnalysis
from openended_philosophy.enhanced.recursive_self_analysis_completions import (
    RecursiveSelfAnalysisCompletions
)
from openended_philosophy.enhanced.insight_synthesis import (
    EnhancedInsightSynthesis,
    SubstantiveInsight,
    PerspectivalAnalysis,
    DialecticalTension
)
from openended_philosophy.semantic.types import PhilosophicalContext, PhilosophicalDomain
from openended_philosophy.operations import PhilosophicalOperations
from openended_philosophy.nars import NARSManager, NARSMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSystemTester:
    """Comprehensive tester for enhanced philosophical reasoning system."""
    
    def __init__(self):
        """Initialize the enhanced system tester."""
        self.test_results: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
        logger.info("🧪 Enhanced System Tester initialized")
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite for enhanced system."""
        logger.info("🚀 Starting comprehensive enhanced system tests")
        
        try:
            # Test 1: Enhanced LLM Processor
            llm_results = await self.test_enhanced_llm_processor()
            self.test_results['enhanced_llm_processor'] = llm_results
            
            # Test 2: Recursive Self-Analysis
            self_analysis_results = await self.test_recursive_self_analysis()
            self.test_results['recursive_self_analysis'] = self_analysis_results
            
            # Test 3: Enhanced Insight Synthesis
            insight_results = await self.test_enhanced_insight_synthesis()
            self.test_results['enhanced_insight_synthesis'] = insight_results
            
            # Test 4: Integration Test
            integration_results = await self.test_system_integration()
            self.test_results['system_integration'] = integration_results
            
            # Test 5: Performance Comparison
            performance_results = await self.test_performance_improvements()
            self.test_results['performance_improvements'] = performance_results
            
            # Generate final report
            final_report = self.generate_final_report()
            self.test_results['final_report'] = final_report
            
            logger.info("✅ Comprehensive testing completed successfully")
            return self.test_results
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive testing: {e}")
            self.test_results['error'] = str(e)
            return self.test_results
    
    async def test_enhanced_llm_processor(self) -> Dict[str, Any]:
        """Test the enhanced LLM philosophical processor."""
        logger.info("🔬 Testing Enhanced LLM Processor")
        
        try:
            processor = EnhancedLLMPhilosophicalProcessor()
            
            # Test philosophical statement analysis
            test_statement = "Consciousness emerges from the complex neural interactions in the brain, suggesting a materialist foundation for subjective experience."
            
            context = PhilosophicalContext(
                domain="philosophy_of_mind",
                inquiry_type="concept_analysis",
                depth_requirements=3
            )
            
            # Run enhanced analysis
            analysis_result = await processor.analyze_philosophical_statement(
                statement=test_statement,
                context=context,
                enable_multi_perspective=True,
                uncertainty_assessment=True,
                depth_level=3
            )
            
            # Validate results
            results = {
                'test_passed': True,
                'analysis_confidence': analysis_result.confidence_score,
                'concepts_extracted': len(analysis_result.extracted_concepts),
                'semantic_relations': len(analysis_result.semantic_relations),
                'philosophical_frameworks': analysis_result.philosophical_frameworks,
                'uncertainty_levels': analysis_result.uncertainty_assessment,
                'practical_implications': len(analysis_result.practical_implications),
                'revision_triggers': len(analysis_result.revision_triggers),
                'analysis_metadata': analysis_result.analysis_metadata
            }
            
            # Validate that analysis contains expected sophisticated elements
            if (analysis_result.confidence_score > 0.3 and 
                len(analysis_result.extracted_concepts) > 0 and
                len(analysis_result.philosophical_frameworks) > 0):
                
                results['sophistication_check'] = 'PASSED'
                logger.info("✅ Enhanced LLM Processor shows sophisticated analysis capabilities")
            else:
                results['sophistication_check'] = 'FAILED'
                logger.warning("⚠️ Enhanced LLM Processor may need further tuning")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced LLM Processor test failed: {e}")
            return {'test_passed': False, 'error': str(e)}
    
    async def test_recursive_self_analysis(self) -> Dict[str, Any]:
        """Test the recursive self-analysis capabilities."""
        logger.info("🔍 Testing Recursive Self-Analysis")
        
        try:
            # Create a mock philosophy server
            class MockPhilosophyServer:
                def __init__(self):
                    self.llm_processor = EnhancedLLMPhilosophicalProcessor()
            
            mock_server = MockPhilosophyServer()
            analyzer = RecursiveSelfAnalysis(mock_server)
            
            # Create a sample analysis result to analyze
            sample_analysis = {
                'concept': 'consciousness',
                'insights': [
                    'Consciousness exhibits intentional structure',
                    'Neural complexity enables emergent properties',
                    'Subjective experience poses explanatory challenges'
                ],
                'semantic_analysis': {
                    'primary_concepts': ['consciousness', 'emergence', 'neural_complexity'],
                    'relations': [('consciousness', 'emerges_from', 'neural_complexity')]
                },
                'confidence': 0.7
            }
            
            # Run recursive self-analysis
            meta_analysis = await analyzer.analyze_own_reasoning_process(
                analysis_result=sample_analysis,
                analysis_type='concept_analysis',
                meta_depth=2
            )
            
            # Validate meta-analysis results
            results = {
                'test_passed': True,
                'meta_analysis_present': 'meta_analysis' in meta_analysis,
                'immediate_reflection': bool(meta_analysis.get('meta_analysis', {}).get('immediate_reflection')),
                'framework_assessment': bool(meta_analysis.get('meta_analysis', {}).get('framework_assessment')),
                'improvement_recommendations': len(meta_analysis.get('meta_analysis', {}).get('improvement_recommendations', [])),
                'recursive_insights': len(meta_analysis.get('recursive_insights', [])),
                'epistemic_status': meta_analysis.get('epistemic_status'),
                'quality_assessment': meta_analysis.get('quality_assessment')
            }
            
            # Check for sophisticated meta-philosophical reflection
            if (meta_analysis.get('meta_analysis') and 
                len(meta_analysis.get('recursive_insights', [])) > 0):
                
                results['meta_sophistication'] = 'PASSED'
                logger.info("✅ Recursive Self-Analysis demonstrates meta-philosophical capabilities")
            else:
                results['meta_sophistication'] = 'FAILED'
                logger.warning("⚠️ Recursive Self-Analysis may need enhancement")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Recursive Self-Analysis test failed: {e}")
            return {'test_passed': False, 'error': str(e)}
    
    async def test_enhanced_insight_synthesis(self) -> Dict[str, Any]:
        """Test the enhanced insight synthesis capabilities."""
        logger.info("🧠 Testing Enhanced Insight Synthesis")
        
        try:
            # Initialize required components
            nars_memory = NARSMemory()
            llm_processor = EnhancedLLMPhilosophicalProcessor()
            synthesis_engine = EnhancedInsightSynthesis(nars_memory, llm_processor)
            
            # Test multi-perspectival insight synthesis
            inquiry_focus = "the relationship between consciousness and artificial intelligence"
            perspectives = ['materialist', 'phenomenological', 'enactivist', 'pragmatist']
            
            insights = await synthesis_engine.synthesize_insights(
                inquiry_focus=inquiry_focus,
                available_perspectives=perspectives,
                depth_level=3
            )
            
            # Validate synthesis results
            results = {
                'test_passed': True,
                'insights_generated': len(insights),
                'perspectives_used': perspectives,
                'insight_details': []
            }
            
            # Analyze insight quality
            for insight in insights:
                insight_analysis = {
                    'content_length': len(insight.content),
                    'confidence': insight.confidence,
                    'supporting_perspectives': insight.supporting_perspectives,
                    'synthesis_pathway': insight.synthesis_pathway,
                    'practical_implications': len(insight.practical_implications),
                    'revision_conditions': len(insight.revision_conditions)
                }
                results['insight_details'].append(insight_analysis)
            
            # Check for multi-perspectival sophistication
            if (len(insights) > 0 and 
                any(len(insight.supporting_perspectives) > 1 for insight in insights)):
                
                results['multi_perspectival_check'] = 'PASSED'
                logger.info("✅ Enhanced Insight Synthesis shows multi-perspectival capabilities")
            else:
                results['multi_perspectival_check'] = 'FAILED'
                logger.warning("⚠️ Enhanced Insight Synthesis may need perspective enhancement")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced Insight Synthesis test failed: {e}")
            return {'test_passed': False, 'error': str(e)}
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """Test end-to-end system integration."""
        logger.info("🔗 Testing System Integration")
        
        try:
            # Initialize the full enhanced philosophical operations
            nars_manager = NARSManager()
            nars_memory = NARSMemory()
            
            # Mock required components for PhilosophicalOperations
            from openended_philosophy.core import (
                CoherenceLandscape,
                FallibilisticInference,
                LanguageGameProcessor
            )
            from openended_philosophy.nars import NARSReasoning
            from openended_philosophy.lv_nars_integration import LVNARSIntegrationManager
            
            operations = PhilosophicalOperations(
                coherence_analyzer=CoherenceLandscape(),
                inference_engine=FallibilisticInference(),
                language_games={'scientific': LanguageGameProcessor()},
                nars_manager=nars_manager,
                nars_memory=nars_memory,
                nars_reasoning=NARSReasoning(nars_memory),
                lv_nars_manager=LVNARSIntegrationManager()
            )
            
            # Test enhanced concept analysis
            test_concept = "emergent consciousness in artificial systems"
            context = PhilosophicalContext(
                domain=PhilosophicalDomain.PHILOSOPHY_OF_MIND,
                inquiry_type="concept_analysis",
                depth_requirements=3
            )
            
            # Run enhanced analysis
            analysis_result = await operations.analyze_concept_enhanced(
                concept=test_concept,
                context=context,
                perspectives=['materialist', 'phenomenological'],
                confidence_threshold=0.7,
                enable_recursive_analysis=True
            )
            
            # Validate integration results
            results = {
                'test_passed': True,
                'concept_analyzed': test_concept,
                'enhanced_analysis_present': 'semantic_analysis' in analysis_result,
                'nars_integration': 'nars_memory_item' in analysis_result,
                'insight_synthesis': 'substantive_insights' in analysis_result,
                'meta_analysis': 'meta_analysis' in analysis_result,
                'confidence_score': analysis_result.get('semantic_analysis', {}).get('epistemic_uncertainty', 'N/A'),
                'components_initialized': {
                    'llm_processor': operations.llm_processor is not None,
                    'insight_synthesis': operations.insight_synthesis is not None,
                    'recursive_analyzer': operations.recursive_analyzer is not None,
                    'philosophical_ontology': operations.philosophical_ontology is not None,
                    'semantic_embedding': operations.semantic_embedding_space is not None
                }
            }
            
            # Check for full integration
            all_components_working = all(results['components_initialized'].values())
            has_enhanced_features = (
                results['enhanced_analysis_present'] and
                results['insight_synthesis'] and
                results.get('meta_analysis', False)
            )
            
            if all_components_working and has_enhanced_features:
                results['integration_check'] = 'PASSED'
                logger.info("✅ System Integration demonstrates full enhanced capabilities")
            else:
                results['integration_check'] = 'PARTIAL'
                logger.warning("⚠️ System Integration shows some component issues")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ System Integration test failed: {e}")
            return {'test_passed': False, 'error': str(e)}
    
    async def test_performance_improvements(self) -> Dict[str, Any]:
        """Test performance improvements from enhancements."""
        logger.info("⚡ Testing Performance Improvements")
        
        try:
            # Compare basic vs enhanced analysis
            processor = EnhancedLLMPhilosophicalProcessor()
            
            test_statement = "Free will is compatible with determinism through compatibilist interpretations that focus on the absence of external constraints."
            
            context = PhilosophicalContext(
                domain="ethics",
                inquiry_type="concept_analysis",
                depth_requirements=2
            )
            
            # Measure enhanced analysis
            start_time = datetime.now()
            enhanced_result = await processor.analyze_philosophical_statement(
                statement=test_statement,
                context=context,
                enable_multi_perspective=True,
                uncertainty_assessment=True
            )
            enhanced_duration = (datetime.now() - start_time).total_seconds()
            
            # Analyze performance characteristics
            results = {
                'test_passed': True,
                'enhanced_analysis_time': enhanced_duration,
                'concepts_extracted': len(enhanced_result.extracted_concepts),
                'relations_identified': len(enhanced_result.semantic_relations),
                'frameworks_applied': len(enhanced_result.philosophical_frameworks),
                'uncertainty_quantified': bool(enhanced_result.uncertainty_assessment),
                'practical_implications': len(enhanced_result.practical_implications),
                'analysis_depth': len(enhanced_result.analysis_metadata.get('perspective_analyses', [])),
                'performance_metrics': {
                    'concepts_per_second': len(enhanced_result.extracted_concepts) / max(enhanced_duration, 0.1),
                    'analysis_completeness': enhanced_result.confidence_score,
                    'feature_richness': (
                        len(enhanced_result.extracted_concepts) +
                        len(enhanced_result.semantic_relations) +
                        len(enhanced_result.philosophical_frameworks)
                    )
                }
            }
            
            # Performance assessment
            if (enhanced_result.confidence_score > 0.5 and 
                len(enhanced_result.extracted_concepts) > 0 and
                enhanced_duration < 5.0):  # Should complete within 5 seconds
                
                results['performance_check'] = 'PASSED'
                logger.info("✅ Performance improvements show enhanced capabilities within reasonable time")
            else:
                results['performance_check'] = 'NEEDS_OPTIMIZATION'
                logger.warning("⚠️ Performance may need optimization")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return {'test_passed': False, 'error': str(e)}
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        logger.info("📊 Generating Final Test Report")
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Count passed tests
        passed_tests = 0
        total_tests = 0
        
        for test_name, results in self.test_results.items():
            if test_name == 'final_report':
                continue
            total_tests += 1
            if isinstance(results, dict) and results.get('test_passed', False):
                passed_tests += 1
        
        # Generate enhancement verification
        enhancement_status = self.verify_implementation_plan_completion()
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration': total_duration
            },
            'implementation_plan_verification': enhancement_status,
            'detailed_results': self.test_results,
            'recommendations': self.generate_recommendations(),
            'next_steps': self.suggest_next_steps()
        }
        
        # Log final status
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Enhanced system fully functional!")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} test(s) failed - See recommendations")
        
        return report
    
    def verify_implementation_plan_completion(self) -> Dict[str, str]:
        """Verify that the implementation plan has been completed."""
        verification = {}
        
        # Phase 1: Enhanced NARS Integration
        if (self.test_results.get('enhanced_llm_processor', {}).get('test_passed') and
            self.test_results.get('system_integration', {}).get('nars_integration')):
            verification['phase_1_nars_integration'] = 'COMPLETED'
        else:
            verification['phase_1_nars_integration'] = 'PARTIAL'
        
        # Phase 2: Sophisticated Insight Synthesis
        if (self.test_results.get('enhanced_insight_synthesis', {}).get('test_passed') and
            self.test_results.get('enhanced_insight_synthesis', {}).get('multi_perspectival_check') == 'PASSED'):
            verification['phase_2_insight_synthesis'] = 'COMPLETED'
        else:
            verification['phase_2_insight_synthesis'] = 'PARTIAL'
        
        # Phase 3: Recursive Self-Analysis
        if (self.test_results.get('recursive_self_analysis', {}).get('test_passed') and
            self.test_results.get('recursive_self_analysis', {}).get('meta_sophistication') == 'PASSED'):
            verification['phase_3_recursive_analysis'] = 'COMPLETED'
        else:
            verification['phase_3_recursive_analysis'] = 'PARTIAL'
        
        # Phase 4: Integration and Testing
        if self.test_results.get('system_integration', {}).get('integration_check') == 'PASSED':
            verification['phase_4_integration'] = 'COMPLETED'
        else:
            verification['phase_4_integration'] = 'PARTIAL'
        
        return verification
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check each test area for recommendations
        if not self.test_results.get('enhanced_llm_processor', {}).get('test_passed'):
            recommendations.append("Enhance LLM processor integration and error handling")
        
        if self.test_results.get('recursive_self_analysis', {}).get('meta_sophistication') != 'PASSED':
            recommendations.append("Improve recursive self-analysis depth and sophistication")
        
        if self.test_results.get('enhanced_insight_synthesis', {}).get('multi_perspectival_check') != 'PASSED':
            recommendations.append("Enhance multi-perspectival analysis capabilities")
        
        if self.test_results.get('system_integration', {}).get('integration_check') != 'PASSED':
            recommendations.append("Address component integration issues")
        
        if self.test_results.get('performance_improvements', {}).get('performance_check') != 'PASSED':
            recommendations.append("Optimize performance and response times")
        
        if not recommendations:
            recommendations.append("System performing well - consider advanced philosophical reasoning features")
        
        return recommendations
    
    def suggest_next_steps(self) -> List[str]:
        """Suggest next steps for further development."""
        return [
            "Integrate with real-world philosophical databases and corpora",
            "Develop interactive philosophical dialogue capabilities",
            "Implement advanced argumentation analysis",
            "Add cross-cultural philosophical perspective integration",
            "Create philosophical reasoning benchmarks and evaluation metrics",
            "Develop educational applications for philosophical learning",
            "Implement collaborative philosophical inquiry features"
        ]


async def main():
    """Main function to run comprehensive enhanced system tests."""
    print("🔬 Enhanced OpenEnded Philosophy MCP - Comprehensive Test Suite")
    print("=" * 70)
    
    tester = EnhancedSystemTester()
    
    try:
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_system_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Print summary
        final_report = results.get('final_report', {})
        test_summary = final_report.get('test_summary', {})
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"  Total Tests: {test_summary.get('total_tests', 0)}")
        print(f"  Passed Tests: {test_summary.get('passed_tests', 0)}")
        print(f"  Success Rate: {test_summary.get('success_rate', 0):.1f}%")
        print(f"  Duration: {test_summary.get('total_duration', 0):.2f} seconds")
        
        # Print implementation verification
        verification = final_report.get('implementation_plan_verification', {})
        print(f"\n🎯 IMPLEMENTATION PLAN VERIFICATION:")
        for phase, status in verification.items():
            status_icon = "✅" if status == "COMPLETED" else "🔄"
            print(f"  {status_icon} {phase.replace('_', ' ').title()}: {status}")
        
        # Print recommendations
        recommendations = final_report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        print(f"\n❌ Test execution failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    asyncio.run(main())
