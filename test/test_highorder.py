#!/usr/bin/env python
"""Test script for PyTyphoon's high order regularizers.

"""
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.ndimage as ndimage

import pytyphoon.regularizers.highorder as hor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    DEMO_DATA_PATH = Path(__file__).parent.parent / 'demo'
    print(DEMO_DATA_PATH)
    import sys
    sys.path.insert(0, str(DEMO_DATA_PATH))
    import inr
    HAS_INR = True
    U_INR = Path(DEMO_DATA_PATH) / 'UVtruth.inr'
except ImportError:
    HAS_INR = False
    U_INR = None
    logger.error("Demo data not found, tests will be limited")


# Filters for gradient computation (finite differences, periodic bc)
# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
D1_FILTERS = {
    2: [-1./2., 0., 1./2.],
    4: [1./12., -2./3., 0., 2/3., -1./12.],
    6: [-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.],
    8: [1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.],
    }
D2_FILTERS = {
    2: [1., -2., 1.],
    4: [-1./12., 4./3., -5./2., 4/3., -1./12.],
    6: [1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.],   
    8: [-1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.],
    }

def l2norm(U1, U2):
    return 0.5*np.sum(U1**2 + U2**2)

def hornschunk(U1, U2, d1):
    result = 0.
    g11 = ndimage.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
    g12 = ndimage.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
    result += np.sum(g11**2 + g12**2)
    g21 = ndimage.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
    g22 = ndimage.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
    result += np.sum(g21**2 + g22**2)
    return 0.5*result

def divergence(U1, U2, d1):
    g11 = ndimage.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
    g22 = ndimage.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
    result = np.sum((g11 + g22)**2)
    return 0.5*result       
    
def curl(U1, U2, d1):
    g12 = ndimage.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
    g21 = ndimage.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
    result = np.sum((g21 - g12)**2)
    return 0.5*result         
    
def laplacian(U1, U2, d2):  
    result = 0.
    g11 = ndimage.convolve1d(U1, d2, axis=1, mode='wrap', origin=0)
    g12 = ndimage.convolve1d(U1, d2, axis=1, mode='wrap', origin=0)
    result += np.sum(g11**2 + g12**2)
    g21 = ndimage.convolve1d(U2, d2, axis=0, mode='wrap', origin=0)
    g22 = ndimage.convolve1d(U2, d2, axis=1, mode='wrap', origin=0)  
    result += np.sum(g21**2 + g22**2)
    return 0.5*result

def graddiv(U1, U2, d1):
    g11 = ndimage.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
    g22 = ndimage.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
    div = g11 + g22
    g1 = ndimage.convolve1d(div, d1, axis=0, mode='wrap', origin=0)
    g2 = ndimage.convolve1d(div, d1, axis=1, mode='wrap', origin=0)
    return 0.5*np.sum(g1**2 + g2**2)
    
def gradcurl(U1, U2, d1):
    g12 = ndimage.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
    g21 = ndimage.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
    curl = g21 - g12
    g1 = ndimage.convolve1d(curl, d1, axis=0, mode='wrap', origin=0)
    g2 = ndimage.convolve1d(curl, d1, axis=1, mode='wrap', origin=0)
    return 0.5*np.sum(g1**2 + g2**2)
    
def test_mass_matrix(display=False):
    """

    Written by P. DERIAN 2019-02-18.
    """
    try:
        print('\n** Building mass matrix')

        size = 256  # Image size [px]
        levels = 0  # Decomposition levels
        wav = pywt.Wavelet('db10')
        N0 = hor.mass_matrix(0, wav, levels, size)
        N1 = hor.mass_matrix(1, wav, levels, size)
        N2 = hor.mass_matrix(2, wav, levels, size)

        # Display matrices
        if display:
            eps = 1e-15
            fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(abs(N0) > eps)
            # axes[1].imshow(abs(N1) > eps)
            # axes[2].imshow(abs(N2) > eps)
            axes[0].imshow(N0, cmap='RdYlBu_r', vmin=-1, vmax=1)
            axes[1].imshow(N1, cmap='RdYlBu_r', vmin=-1, vmax=1)
            axes[2].imshow(N2, cmap='RdYlBu_r', vmin=-1, vmax=1)
            plt.show()

        print("✓ Mass matrix test passed")
        return True
    except Exception as e:
        print(f"✗ Mass matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
def test_highorder_conv():
    """

    Written by P. DERIAN 2018-01-09.
    """
    try:
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 3  # Note: levels have no influence with the convolution-based form.
        fd_order = 8  # Order of finite-differences filters for truth
        U2, U1 = inr.readMotion(U_INR)
        # Wavelet-based computations
        hor_conv = hor.HighOrderRegularizerConv(wav)
        C1 = pywt.wavedecn(U1, hor_conv.wav, level=wav_levels, mode=hor_conv.mode)
        C2 = pywt.wavedecn(U2, hor_conv.wav, level=wav_levels, mode=hor_conv.mode)
        l2norm_hor, _ = hor_conv.evaluate(C1, C2, 'l2norm')
        hornschunk_hor, _ = hor_conv.evaluate(C1, C2, 'hornschunck')
        divergence_hor, _ = hor_conv.evaluate(C1, C2, 'div')
        curl_hor, _ = hor_conv.evaluate(C1, C2, 'curl')
        laplacian_hor, _ = hor_conv.evaluate(C1, C2, 'laplacian')
        graddiv_hor, _ = hor_conv.evaluate(C1, C2, 'graddiv')
        gradcurl_hor, _ = hor_conv.evaluate(C1, C2, 'gradcurl')
        # Compute the truth
        # Note: not real "truth" since gradients are computed by finite differences.
        d1 = D1_FILTERS[fd_order]
        d2 = D2_FILTERS[fd_order]
        l2norm_truth = l2norm(U1, U2)
        hornschunk_truth = hornschunk(U1, U2, d1)
        divergence_truth = divergence(U1, U2, d1)
        curl_truth = curl(U1, U2, d1)
        laplacian_truth = laplacian(U1, U2, d2)
        graddiv_truth = graddiv(U1, U2, d1)
        gradcurl_truth = gradcurl(U1, U2, d1)
        # Print out
        print('\n** Convolution-based high-order computations')
        print('Wavelet: {}'.format(wav))
        print('Truth: centered finite-differences, order-{} accuracy'.format(fd_order))
        print('High-order terms:')
        for label, test, truth in zip(
            ['l2-norm', 'H&S', 'divergence', 'curl', 'laplacian', 'graddiv', 'gradcurl'],
            [l2norm_hor, hornschunk_hor, divergence_hor, curl_hor, laplacian_hor, graddiv_hor, gradcurl_hor],
            [l2norm_truth, hornschunk_truth, divergence_truth, curl_truth, laplacian_truth, graddiv_truth, gradcurl_truth],
            ):
            print('\t{:>10}: rel. err={:.3f} % (wav={:.3f}, ref={:.3f})'.format(
                label, 100.*np.abs(test-truth)/truth, test, truth))

        print("✓ High-order convolution test passed")
        return True
    except Exception as e:
        print(f"✗ High-order convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
def test_highorder_mat():
    """

    Written by P. DERIAN 2019-02-18.
    """
    try:
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 2  # Note: levels have no influence with the convolution-based form.
        fd_order = 8  # Order of finite-differences filters for truth
        U2, U1 = inr.readMotion(U_INR)
        # Wavelet-based computations
        hor_mat = hor.HighOrderRegularizerMat(wav, wav_levels, U1.shape)
        C1, _ = pywt.coeffs_to_array(pywt.wavedec2(
            U1, hor_mat.wav, level=wav_levels, mode=hor_mat.mode))
        C2, _ = pywt.coeffs_to_array(pywt.wavedec2(
            U2, hor_mat.wav, level=wav_levels, mode=hor_mat.mode))
        l2norm_hor, _ = hor_mat.evaluate(C1, C2, 'l2norm')
        hornschunk_hor, _ = hor_mat.evaluate(C1, C2, 'hornschunck')
        divergence_hor, _ = hor_mat.evaluate(C1, C2, 'div')
        curl_hor, _ = hor_mat.evaluate(C1, C2, 'curl')
        laplacian_hor, _ = hor_mat.evaluate(C1, C2, 'laplacian')
        graddiv_hor, _ = hor_mat.evaluate(C1, C2, 'graddiv')
        gradcurl_hor, _ = hor_mat.evaluate(C1, C2, 'gradcurl')
        # Note: not actual "truth" since gradients are computed by finite differences.
        d1 = D1_FILTERS[fd_order]
        d2 = D2_FILTERS[fd_order]
        l2norm_truth = l2norm(U1, U2)
        hornschunk_truth = hornschunk(U1, U2, d1)
        divergence_truth = divergence(U1, U2, d1)
        curl_truth = curl(U1, U2, d1)
        laplacian_truth = laplacian(U1, U2, d2)
        graddiv_truth = graddiv(U1, U2, d1)
        gradcurl_truth = gradcurl(U1, U2, d1)
        # Print out
        print('\n** Matrix-based high-order computations')
        print('Wavelet: {}'.format(wav))
        print('Truth: centered finite-differences, order-{} accuracy'.format(fd_order))
        print('High-order terms:')
        for label, test, truth in zip(
            ['l2-norm', 'H&S', 'divergence', 'curl', 'laplacian', 'grad(div)', 'grad(curl)'],
            [l2norm_hor, hornschunk_hor, divergence_hor, curl_hor, laplacian_hor, graddiv_hor, gradcurl_hor],
            [l2norm_truth, hornschunk_truth, divergence_truth, curl_truth, laplacian_truth, graddiv_truth, gradcurl_truth],
            ):
            print('\t{:>10}: rel. err={:.3f} % (wav={:.3f}, ref={:.3f})'.format(
                label, 100.*np.abs(test-truth)/truth, test, truth))

        print("✓ High-order matrix test passed")
        return True
    except Exception as e:
        print(f"✗ High-order matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_highorders(name='laplacian', display=False):
    """
    """
    try:
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 3  # Note: levels have no influence with the convolution-based form.
        U2, U1 = inr.readMotion(U_INR)
        # Regularizers
        hor_conv = hor.HighOrderRegularizerConv(wav)
        hor_mat = hor.HighOrderRegularizerMat(wav, wav_levels, U1.shape)
        # Wavelet coeffs
        C1 = pywt.wavedecn(U1.astype(np.float64), wav, level=wav_levels, mode='periodization')
        C2 = pywt.wavedecn(U2.astype(np.float64), wav, level=wav_levels, mode='periodization')
        C1c, _ = pywt.coeffs_to_array(C1)
        C2c, _ = pywt.coeffs_to_array(C2)
        # Compute terms
        value_conv, grad_conv = hor_conv.evaluate(C1, C2, name)
        value_mat, grad_mat = hor_mat.evaluate(C1c, C2c, name)
        # Display
        print('\n** Comparison of convolution vs Matrix high-order computations')
        print('{}: conv={:.3f}, mat={:.3f}'.format(name, value_conv, value_mat))
        if display :
            fig, axes = plt.subplots(2, 4)
            vmin = -0.5
            vmax = 0.5
            # Conv-based
            g1_conv, _ = pywt.coeffs_to_array(grad_conv[0])
            g2_conv, _ = pywt.coeffs_to_array(grad_conv[1])
            axes[0, 0].imshow(g1_conv, vmin=vmin, vmax=vmax)
            axes[1, 0].imshow(g2_conv, vmin=vmin, vmax=vmax)
            # Matrix-based
            g1_mat, g2_mat = grad_mat
            axes[0, 1].imshow(g1_mat, vmin=vmin, vmax=vmax)
            axes[1, 1].imshow(g2_mat, vmin=vmin, vmax=vmax)
            # Errors
            axes[0, 2].imshow(np.log10(np.abs(g1_conv-g1_mat)/abs(g1_mat)), cmap='RdYlBu_r', vmin=-5, vmax=5)
            axes[1, 2].imshow(np.log10(np.abs(g2_conv-g2_mat)/abs(g2_mat)), cmap='RdYlBu_r', vmin=-5, vmax=5.)
            # Ratios
            axes[0, 3].imshow(np.abs(g1_conv/g1_mat), cmap='RdYlBu_r', vmin=0.5, vmax=2.)
            axes[1, 3].imshow(np.abs(g2_conv/g2_mat), cmap='RdYlBu_r', vmin=0.5, vmax=2.)
            plt.show()

        print(f"✓ High-orders test passed for {name}")
        return True
    except Exception as e:
        print(f"✗ High-orders test failed for {name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_conv(display=False):
    """
    """
    try:
        print('\n** Comparison of convolution / quadrature l2-norms')
        # Data parameters
        fine_scale = 8
        nx = 1
        ny = 3
        sizex = nx * (2**fine_scale)
        sizey = ny * (2**fine_scale)
        wx = 3.  # x-freq
        wy = 10.  # y-freq
        # Wavelet parameters
        wav = pywt.Wavelet('db10')
        wav_levels = 3  # Note: levels have no influence with the convolution-based form.
        # The grid
        x, dx = np.linspace(0, 2.*np.pi*nx, sizex, endpoint=False, retstep=True)
        y, dy = np.linspace(0, 2.*np.pi*ny, sizey, endpoint=False, retstep=True)
        yy, xx = np.meshgrid(y, x)
        print(f'grid size: {x.shape, y.shape}')
        # The data
        f = np.cos(wx*xx) + np.sin(wy*yy)
        fx = -wx*np.sin(wx*xx)
        fy = wy*np.cos(wy*yy)
        fxx = -(wx**2)*np.cos(wx*xx)
        fyy = -(wy**2)*np.sin(wy*yy)
        fxy = np.zeros_like(xx)
        # Theoretical norms
        l2t = {
            'f': 4. * (np.pi**2) * nx *ny,
            'fx': 2 * (np.pi**2) * nx * ny * (wx**2),
            'fy': 2 * (np.pi**2) * nx * ny * (wy**2),
            'fxx': 2 * (np.pi**2) * nx * ny * (wx**4),
            'fyy': 2 * (np.pi**2) * nx * ny * (wy**4),
            'fxy': 0.,
        }
        # Rectangle quadrature formulas
        def l2norm_rectangle(field):
            return (dx*dy)*np.square(field).sum()
        l2r_f = l2norm_rectangle(f)
        l2r_fx = l2norm_rectangle(fx)
        l2r_fy = l2norm_rectangle(fy)
        l2r_fxx = l2norm_rectangle(fxx)
        l2r_fyy = l2norm_rectangle(fyy)
        l2r_fxy = l2norm_rectangle(fxy)
        # Wavelet
        hor_conv = hor.HighOrderRegularizerConv(wav)
        cf = pywt.wavedecn(f, hor_conv.wav, level=wav_levels, mode=hor_conv.mode)
        l2w_f = hor_conv.norm(cf, 0, 0)
        l2w_fx = hor_conv.norm(cf, 1, 0)
        l2w_fy = hor_conv.norm(cf, 0, 1)
        l2w_fxx = hor_conv.norm(cf, 2, 0)
        l2w_fyy = hor_conv.norm(cf, 0, 2)
        l2w_fxy = hor_conv.norm(cf, 1, 1)
        # Print out
        for label, l2r, l2w in zip(
            ['f', 'fx', 'fy', 'fxx', 'fyy', 'fxy'],
            [l2r_f, l2r_fx, l2r_fy, l2r_fxx, l2r_fyy, l2r_fxy],
            [l2w_f, l2w_fx, l2w_fy, l2w_fxx, l2w_fyy, l2w_fxy],
            ):
            print('{:>3}: exact={:.3f}, rectangle approx.={:.3f} ; wavelet approx.={:.3f}'.format(
                label,
                l2t[label],
                l2r,
                l2w,
            ))
            print('\trect/exact={:.3f} ; wav/exact={:.3f}'.format(
                l2r / l2t[label],
                l2w / l2t[label],
            ))
        # Display
        if display:
            fig, ax  = plt.subplots()
            ax.imshow(f, extent=[y[0], y[-1], x[0], x[-1]], origin='lower')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
            ax.set_title('f(x, y)')
            plt.show()

        print("✓ Convolution test passed")
        return True
    except Exception as e:
        print(f"✗ Convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    

def main_test(display=False):
    """Run all tests."""
    print("=" * 60)
    print("PyTyphoon Highorder Test Suite")
    print("=" * 60)

    results = []
    results.append(("Mass matrix", test_mass_matrix(display=display)))
    results.append(("High-order convolution", test_highorder_conv()))
    results.append(("High-order matrix", test_highorder_mat()))
    results.append(("L2-norm comparison", test_highorders('l2norm', display=display)))
    results.append(("Horn & Schunck comparison", test_highorders('hornschunck', display=display)))
    results.append(("Divergence comparison", test_highorders('div', display=display)))
    results.append(("Curl comparison", test_highorders('curl', display=display)))
    results.append(("Laplacian comparison", test_highorders('laplacian', display=display)))
    results.append(("High-order convolutions", test_conv(display=display)))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:25s}: {status}")

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