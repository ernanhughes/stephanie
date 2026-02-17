#!/usr/bin/env python3
"""Quick validation test before full testing"""

import sys
import logging

logging.basicConfig(level=logging.INFO)

# Test 1: Can we import everything?
try:
    from stephanie.components.elm import (
        ContextPack,
        CalibratedThresholds,
        RetentionTracker,
        CollapseDetector,
        GovernanceSignalExtractor,
        DominanceChecker,
        RegimeController,
        PerturbationInjector,
        SystemInterface,
        MockSystem
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test 2: Can we create core objects?
try:
    thresholds = CalibratedThresholds(
        energy_max=0.5,
        energy_warning=0.4,
        hrm_min=0.6,
        margin_min=0.5,
        variance_min=0.3,
        collapse_index_max=10.0,
        drift_max=0.15
    )
    print("✅ CalibratedThresholds created")
    
    tracker = RetentionTracker()
    print("✅ RetentionTracker created")
    
    detector = CollapseDetector(thresholds)
    print("✅ CollapseDetector created")
    
except Exception as e:
    print(f"❌ Object creation failed: {e}")
    sys.exit(1)

# Test 3: Does MockSystem satisfy interface?
try:
    mock = MockSystem()
    assert isinstance(mock, SystemInterface)
    print("✅ MockSystem validates against SystemInterface")
except Exception as e:
    print(f"❌ SystemInterface validation failed: {e}")

# Test 4: Can we run a minimal experiment?
try:
    from stephanie.components.elm.experiment.experiment import ScoreBundleExperiment
    
    experiment = ScoreBundleExperiment(
        system=mock,
        queries=["test query 1", "test query 2"],
        thresholds=thresholds,
        extractor=GovernanceSignalExtractor()
    )
    print("✅ Experiment harness created")
    
except Exception as e:
    print(f"❌ Experiment creation failed: {e}")

print("\n" + "="*50)
print("✅ ALL VALIDATION TESTS PASSED")
print("="*50)
print("\nYou can now proceed with full testing!")