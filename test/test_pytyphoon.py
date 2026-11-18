#!/usr/bin/env python
"""Test script for PyTyphoon package.

This script tests the basic functionality of the refactored PyTyphoon package.
"""

import numpy as np
import sys
import os

# Add the current directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import pytyphoon
        from pytyphoon import Typhoon, OpticalFlowCore, RMSE
        from pytyphoon.regularizers import HighOrderRegularizerConv, HighOrderRegularizerMat
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_optical_flow_core():
    """Test OpticalFlowCore class."""
    print("\nTesting OpticalFlowCore...")
    try:
        from pytyphoon.core import OpticalFlowCore

        # Create a simple test case
        shape = (32, 32)
        core = OpticalFlowCore(shape, dtype=np.float32)

        # Test basic properties
        assert core.shape == shape
        assert core.ndim == 2
        assert core.dtype == np.float32

        # Test DFD computation
        im0 = np.random.rand(*shape).astype(np.float32)
        im1 = np.random.rand(*shape).astype(np.float32)
        U = (np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32))

        dfd = core.DFD(im0, im1, U)
        assert dfd.shape == shape
        print("✓ OpticalFlowCore tests passed")
        return True
    except Exception as e:
        print(f"✗ OpticalFlowCore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_typhoon_basic():
    """Test basic Typhoon functionality."""
    print("\nTesting Typhoon class...")
    try:
        from pytyphoon.typhoon import Typhoon

        # Create simple synthetic images with known motion
        size = 32
        im0 = np.random.rand(size, size).astype(np.float32)
        im1 = np.roll(im0, shift=1, axis=1)  # Simple horizontal shift

        # Create Typhoon instance and solve
        typhoon = Typhoon()
        U = typhoon.solve_fast(im0, im1, wav='haar', levels_decomp=3, levels_estim=2)

        # Check results
        assert len(U) == 2  # Should have 2 components for 2D
        assert U[0].shape == (size, size)
        assert U[1].shape == (size, size)

        print(f"✓ Typhoon basic test passed")
        print(f"  - Estimated motion field shapes: {[u.shape for u in U]}")
        print(f"  - Mean displacement: U1={U[0].mean():.4f}, U2={U[1].mean():.4f}")
        return True
    except Exception as e:
        print(f"✗ Typhoon test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rmse():
    """Test RMSE function."""
    print("\nTesting RMSE function...")
    try:
        from pytyphoon.utils import RMSE

        # Create test data
        U1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        U2 = np.array([[1.1, 2.1], [2.9, 4.2]], dtype=np.float32)
        V1 = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        V2 = np.array([[0.55, 0.65], [0.75, 0.85]], dtype=np.float32)

        rmse = RMSE((U1, U2), (V1, V2))
        assert isinstance(rmse, float)
        assert rmse >= 0
        print(f"✓ RMSE test passed (value: {rmse:.4f})")
        return True
    except Exception as e:
        print(f"✗ RMSE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_regularizers():
    """Test high-order regularizers."""
    print("\nTesting high-order regularizers...")
    try:
        import pywt
        from pytyphoon.regularizers import HighOrderRegularizerConv

        # Create test data
        wav = pywt.Wavelet('db3')
        hor = HighOrderRegularizerConv(wav)

        # Create simple test coefficients
        size = 32
        U1 = np.random.rand(size, size).astype(np.float64)
        U2 = np.random.rand(size, size).astype(np.float64)

        C1 = pywt.wavedecn(U1, wav, level=2, mode='periodization')
        C2 = pywt.wavedecn(U2, wav, level=2, mode='periodization')

        # Test l2norm regularizer
        value, grad = hor.evaluate(C1, C2, 'l2norm')
        assert isinstance(value, float)
        assert isinstance(grad, tuple)
        print(f"✓ High-order regularizer test passed (l2norm value: {value:.4f})")
        return True
    except Exception as e:
        print(f"✗ Regularizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI module."""
    print("\nTesting CLI module...")
    try:
        from pytyphoon.cli import create_parser, main, print_versions

        # Test parser creation
        parser = create_parser()
        assert parser is not None

        # Test version printing (should not raise)
        print_versions()

        print("✓ CLI module test passed")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_test():
    """Run all tests."""
    print("=" * 60)
    print("PyTyphoon Package Test Suite")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("OpticalFlowCore", test_optical_flow_core()))
    results.append(("Typhoon basic", test_typhoon_basic()))
    results.append(("RMSE", test_rmse()))
    results.append(("Regularizers", test_regularizers()))
    results.append(("CLI", test_cli()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main_test())