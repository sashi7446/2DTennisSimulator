#!/usr/bin/env python3
"""
Test runner for 2D Tennis Simulator.

Runs all tests and provides a summary of what's working and what's not.
"""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests_with_details():
    """Run all tests and show detailed results."""
    # Discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"\nTotal tests: {total}")
    print(f"  ✓ Passed:  {passed}")
    print(f"  ✗ Failed:  {failures}")
    print(f"  ! Errors:  {errors}")

    if failures > 0:
        print("\n" + "-" * 70)
        print("FAILURES (tests that ran but produced wrong results):")
        print("-" * 70)
        for test, traceback in result.failures:
            print(f"\n✗ {test}")
            # Extract just the assertion error
            lines = traceback.strip().split("\n")
            for line in lines[-3:]:
                print(f"  {line}")

    if errors > 0:
        print("\n" + "-" * 70)
        print("ERRORS (tests that crashed):")
        print("-" * 70)
        for test, traceback in result.errors:
            print(f"\n! {test}")
            lines = traceback.strip().split("\n")
            for line in lines[-3:]:
                print(f"  {line}")

    print("\n" + "=" * 70)

    if failures == 0 and errors == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - See details above")

    print("=" * 70)

    return result.wasSuccessful()


def run_single_component(component_name):
    """Run tests for a single component."""
    print(f"\nRunning tests for: {component_name}")
    print("-" * 50)

    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromName(f"tests.test_{component_name}")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific component
        component = sys.argv[1]
        success = run_single_component(component)
    else:
        # Run all tests
        success = run_tests_with_details()

    sys.exit(0 if success else 1)
