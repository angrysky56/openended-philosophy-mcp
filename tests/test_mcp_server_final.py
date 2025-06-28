#!/usr/bin/env python3
"""
Final test to demonstrate that the MCP server works correctly with fixed tools.
This simulates the exact MCP tool calls that were previously failing.
"""

import asyncio
import json
import logging
from openended_philosophy.server import OpenEndedPhilosophyServer

async def test_mcp_server_tools():
    """Test the MCP server with the exact tool calls that were failing."""
    
    print("🧠 Testing Fixed OpenEnded Philosophy MCP Server Tools")
    print("=" * 60)
    
    # Initialize the server
    server = OpenEndedPhilosophyServer()
    
    # Test tool calls with the exact parameters that were failing before
    test_cases = [
        {
            "name": "analyze_concept",
            "arguments": {
                "concept": "consciousness",
                "context": "philosophy_of_mind",
                "perspectives": ["materialist", "phenomenological"],
                "confidence_threshold": 0.7
            }
        },
        {
            "name": "explore_coherence", 
            "arguments": {
                "domain": "metaphysics",
                "depth": 3,
                "allow_revision": True
            }
        },
        {
            "name": "generate_insights",
            "arguments": {
                "phenomenon": "consciousness and emergence",
                "perspectives": ["materialist", "phenomenological", "enactivist"],
                "depth": 3,
                "include_contradictions": True
            }
        },
        {
            "name": "contextualize_meaning",
            "arguments": {
                "expression": "justice",
                "language_game": "ethical_deliberation",
                "trace_genealogy": False
            }
        },
        {
            "name": "test_philosophical_hypothesis",
            "arguments": {
                "hypothesis": "Free will is compatible with soft determinism",
                "test_domains": ["metaphysics", "neuroscience", "psychology"],
                "criteria": {
                    "logical_consistency": 0.8,
                    "empirical_support": 0.6,
                    "explanatory_power": 0.7
                }
            }
        },
        {
            "name": "recursive_self_analysis",
            "arguments": {
                "analysis_result": {
                    "concept": "consciousness",
                    "analysis_type": "concept_analysis",
                    "confidence": 0.75
                },
                "analysis_type": "concept_analysis",
                "meta_depth": 2
            }
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        tool_name = test_case["name"]
        arguments = test_case["arguments"]
        
        print(f"\n🎯 Test {i}: {tool_name}")
        print(f"   Arguments: {list(arguments.keys())}")
        
        try:
            # Call the tool through the operations interface
            if tool_name == "analyze_concept":
                result = await server.operations.analyze_concept(**arguments)
            elif tool_name == "explore_coherence":
                result = await server.operations.explore_coherence(**arguments)
            elif tool_name == "generate_insights":
                result = await server.operations.generate_insights(**arguments)
            elif tool_name == "contextualize_meaning":
                result = await server.operations.contextualize_meaning(**arguments)
            elif tool_name == "test_philosophical_hypothesis":
                result = await server.operations.test_philosophical_hypothesis(**arguments)
            elif tool_name == "recursive_self_analysis":
                result = await server.operations.recursive_self_analysis(**arguments)
            
            # Check if result contains an error
            if "error" in result:
                print(f"❌ {tool_name}: ERROR - {result['error']}")
            else:
                print(f"✅ {tool_name}: SUCCESS")
                success_count += 1
                
                # Show some key result information
                if tool_name == "analyze_concept":
                    print(f"   → Concept analyzed: {result.get('concept')}")
                    confidence = result.get('confidence_assessment', {}).get('overall', 'N/A')
                    print(f"   → Overall confidence: {confidence}")
                    
                elif tool_name == "explore_coherence":
                    print(f"   → Domain: {result.get('domain')}")
                    concepts = len(result.get('concepts_analyzed', []))
                    print(f"   → Concepts analyzed: {concepts}")
                    coherence = result.get('overall_coherence_score', 'N/A')
                    print(f"   → Coherence score: {coherence}")
                    
                elif tool_name == "generate_insights":
                    print(f"   → Phenomenon: {result.get('phenomenon')}")
                    insights = len(result.get('substantive_insights', []))
                    print(f"   → Insights generated: {insights}")
                    
                elif tool_name == "contextualize_meaning":
                    print(f"   → Expression: {result.get('expression')}")
                    print(f"   → Language game: {result.get('language_game')}")
                    
                elif tool_name == "test_philosophical_hypothesis":
                    hypothesis = result.get('hypothesis', '')
                    print(f"   → Hypothesis: {hypothesis[:50]}...")
                    confidence = result.get('posterior_confidence', 'N/A')
                    print(f"   → Posterior confidence: {confidence}")
                    
                elif tool_name == "recursive_self_analysis":
                    meta_depth = result.get('meta_depth', 'N/A')
                    print(f"   → Meta depth: {meta_depth}")
                    insights = len(result.get('recursive_insights', []))
                    print(f"   → Recursive insights: {insights}")
                
        except Exception as e:
            print(f"❌ {tool_name}: EXCEPTION - {e}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Testing Results: {success_count}/{len(test_cases)} tools working correctly!")
    
    if success_count == len(test_cases):
        print("✅ ALL TOOLS FIXED! The OpenEnded Philosophy MCP server is now fully functional.")
    else:
        print(f"⚠️  {len(test_cases) - success_count} tools still need attention.")
    
    return success_count == len(test_cases)

if __name__ == "__main__":
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    asyncio.run(test_mcp_server_tools())
