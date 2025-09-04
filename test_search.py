#!/usr/bin/env python3
"""
Test script for the hyperparameter search implementation.
Runs a quick dry-run to validate the setup.
"""

import subprocess
import sys
from pathlib import Path


def test_dry_run():
    """Run a quick dry-run test of the hyperparameter search."""
    print("Running dry-run test of hyperparameter search...")
    
    # Test command
    cmd = [
        sys.executable, "search_phase1.py",
        "--env", "dmc",
        "--full-steps", "20000",
        "--fidelity-frac", "0.1",
        "--n-trials", "2",
        "--eval-every", "2000",
        "--eval-episodes", "3",
        "--output-dir", "./test_runs",
        "--seed0", "42",
        "--dry-run"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Dry-run test passed!")
            
            # Check if output files were created
            output_dir = Path("./test_runs")
            expected_files = ["results.csv", "topk.json", "best_config.yaml", "SUMMARY.md"]
            
            for file in expected_files:
                if (output_dir / file).exists():
                    print(f"✅ {file} created successfully")
                else:
                    print(f"❌ {file} not found")
            
            return True
        else:
            print(f"❌ Dry-run test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Dry-run test timed out")
        return False
    except Exception as e:
        print(f"❌ Dry-run test failed with error: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from search_phase1 import load_base_config
        cfg = load_base_config("dmc")
        
        # Check required fields
        required_fields = [
            "algo.jepa_coef",
            "algo.jepa_ema", 
            "algo.jepa_mask.erase_frac",
            "algo.jepa_mask.vec_dropout",
            "algo.jepa_proj_dim",
            "algo.jepa_hidden",
            "env.id"
        ]
        
        for field in required_fields:
            keys = field.split(".")
            value = cfg
            for key in keys:
                value = value[key]
            print(f"✅ {field} = {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_optuna_setup():
    """Test Optuna setup."""
    print("\nTesting Optuna setup...")
    
    try:
        import optuna
        
        # Test sampler creation
        from search_phase1 import create_sampler, create_pruner
        sampler = create_sampler("random")
        pruner = create_pruner("hyperband", 10000)
        
        print(f"✅ Sampler created: {type(sampler).__name__}")
        print(f"✅ Pruner created: {type(pruner).__name__}")
        
        # Test study creation
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        print("✅ Study created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Optuna setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HYPERPARAMETER SEARCH TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Optuna Setup", test_optuna_setup),
        ("Dry Run", test_dry_run),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! The hyperparameter search is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
