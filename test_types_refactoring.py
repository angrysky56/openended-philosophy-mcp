#!/usr/bin/env python3
"""
Test script to verify types.py integration and proper import resolution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script tests that all type imports work correctly after refactoring
and that the system can still instantiate and use the core data structures.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test importing from types module directly
        from openended_philosophy.nars.types import TruthValue, MemoryItem, ReasoningResult
        print("✓ Direct imports from types.py successful")
        
        # Test importing from nars module
        from openended_philosophy.nars import (
            NARSManager, NARSMemory, NARSReasoning, 
            Truth, TruthValue, MemoryItem, ReasoningResult
        )
        print("✓ Imports from nars module successful")
        
        # Test importing from specific modules
        from openended_philosophy.nars.truth_functions import Truth
        from openended_philosophy.nars.nars_memory import NARSMemory
        from openended_philosophy.nars.nars_reasoning import NARSReasoning
        print("✓ Module-specific imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_type_instantiation():
    """Test that types can be instantiated and used."""
    print("\nTesting type instantiation...")
    
    try:
        from openended_philosophy.nars.types import TruthValue, MemoryItem, ReasoningResult
        import numpy as np
        
        # Test TruthValue
        tv = TruthValue(frequency=0.8, confidence=0.9)
        print(f"✓ TruthValue created: freq={tv.frequency}, conf={tv.confidence}, exp={tv.expectation:.2f}")
        
        # Test serialization
        tv_dict = tv.to_dict()
        print(f"✓ TruthValue serialization: {tv_dict}")
        
        # Test MemoryItem
        mi = MemoryItem(
            term="<consciousness --> experience>",
            truth=tv,
            stamp=[1, 2, 3],
            occurrence_time="eternal",
            philosophical_category="phenomenological"
        )
        print(f"✓ MemoryItem created: {mi.term} (category: {mi.philosophical_category})")
        
        # Test ReasoningResult
        rr = ReasoningResult(
            conclusion="Consciousness implies subjective experience",
            truth=tv,
            evidence=[mi],
            inference_path=["premise", "deduction", "conclusion"],
            uncertainty_factors={"epistemic": 0.1, "semantic": 0.05},
            philosophical_implications=["Hard problem of consciousness remains"]
        )
        print(f"✓ ReasoningResult created: {rr.conclusion}")
        
        # Test that frozen dataclass works
        try:
            tv.frequency = 0.5  # This should fail
            print("✗ TruthValue mutability test failed - object is mutable!")
            return False
        except Exception:
            print("✓ TruthValue immutability verified")
            
        return True
        
    except Exception as e:
        print(f"✗ Type instantiation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test that the refactored types work with other modules."""
    print("\nTesting module integration...")
    
    try:
        from openended_philosophy.nars import Truth, TruthValue
        
        # Test truth functions with new TruthValue
        tv1 = TruthValue(0.8, 0.9)
        tv2 = TruthValue(0.7, 0.8)
        
        # Test revision
        revised = Truth.revision(tv1, tv2)
        print(f"✓ Truth revision: ({tv1.frequency:.2f}, {tv1.confidence:.2f}) + "
              f"({tv2.frequency:.2f}, {tv2.confidence:.2f}) = "
              f"({revised.frequency:.2f}, {revised.confidence:.2f})")
        
        # Test other truth functions
        deduced = Truth.deduction(tv1, tv2)
        print(f"✓ Truth deduction works: ({deduced.frequency:.2f}, {deduced.confidence:.2f})")
        
        negated = Truth.negation(tv1)
        print(f"✓ Truth negation works: ({negated.frequency:.2f}, {negated.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("Types Refactoring Verification Test")
    print("="*70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Type Instantiation Tests", test_type_instantiation),
        ("Integration Tests", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*70)
    print("Test Summary:")
    print("="*70)
    
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed! The types refactoring is successful.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
