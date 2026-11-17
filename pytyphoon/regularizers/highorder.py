"""High-order regularization terms for PyTyphoon.

This module implements wavelet-based high-order regularization terms for optical flow.
It provides both convolution-based and matrix-based implementations.

References:
    [1] BEYLKIN, Gregory.
    On the representation of operators in bases of compactly supported wavelets.
    SIAM Journal on Numerical Analysis, 1992, vol. 29, no 6, p. 1716-1740.

    [2] KADRI-HAROUNA, S., DÉRIAN, Pierre, HÉAS, Patrick, et al.
    Divergence-free wavelets and high order regularization.
    International journal of computer vision, 2013, vol. 103, no 1, p. 80-99.

    [3] DÉRIAN, Pierre.
    Wavelets and Fluid Motion Estimation.
    PhD thesis, MATISSE doctoral school, Université Rennes 1, 2012.
"""

import logging
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import pywt
import scipy.ndimage

logger = logging.getLogger(__name__)

def connection_coefficients(wav: pywt.Wavelet, order: int) -> Optional[np.ndarray]:
    """Find the connection coefficients of the wavelet at given order.

    This computes the L2 dot-products of the form:
        ∫[ Φ(x) (d^(n)/dx^n){Φ}(x) ]dx
    where Φ is the mother wavelet.

    Args:
        wav: A pywt.Wavelet object
        order: The derivation order

    Returns:
        A vector of coefficients, or None if not found

    Notes:
        This function uses eigenvalue decomposition to find the connection coefficients.
        The coefficients are normalized using Beylkin normalization.

    Written by P. DERIAN 2018-01-09.
    Updated by P. DERIAN 2019-02-15: using reconstruction filter.
    """
    ctol = 1e-15  # Tolerance for coefficients
    etol = 1e-4  # Tolerance for eigenvalues

    # Get the low-pass reconstruction filter
    lo = wav.rec_lo
    len_lo = len(lo)

    # Create the matrix
    dim = 2 * len_lo - 3
    matrix = np.zeros((dim, dim))

    for m in range(dim):
        for n in range(dim):
            tmp = 0.
            # For each filter value
            for p, lo_p in enumerate(lo):
                idx = m - 2 * n + p + len_lo - 2
                if 0 <= idx < len_lo:
                    tmp += lo_p * lo[idx]
            # Store only if above threshold
            if np.abs(tmp) > ctol:
                matrix[n, m] = tmp

    # Find coefficient vector: solve eigenvalue problem
    evals, evecs = np.linalg.eig(matrix)

    # Check if any REAL eigenvalue matches our order
    sigma = 1.0 / float(2 ** order)
    ev_found = False
    coeff_idx = None

    for i, ev in enumerate(evals):
        if (np.abs(np.real(ev) - sigma) < etol) and (np.abs(np.imag(ev)) < 1e-14):
            ev_found = True
            coeff_idx = i
            break

    # [TODO] warn if not found?
    if not ev_found:
        logger.warning(f"Connection coefficients not found for wavelet {wav.name}, order {order}")
        return None

    # Get the associated eigen vector
    coeffs = np.real(evecs[:, coeff_idx])

    # If the derivation order is odd, force mid value to be exactly zero
    # as it should be by construction.
    if order % 2:
        coeffs[dim // 2] = 0.

    # Apply Beylkin normalization
    norm_factors = np.array([float(-len_lo + 2 + p) for p in range(dim)]) ** order
    coeffs /= np.sum(norm_factors * coeffs)
    tmp = (np.prod(np.arange(1, order + 1))) * np.power(-1., order)
    coeffs *= tmp

    return coeffs

def mass_matrix(order: int, wav: pywt.Wavelet, levels: int, size: int) -> np.ndarray:
    """Compute the mass matrix.

    Args:
        order: Total derivation order
        wav: A pywt.Wavelet object
        levels: Wavelet decomposition levels
        size: Size of the matrix

    Returns:
        A (size, size) mass matrix

    Raises:
        ValueError: If size is not compatible with levels
    """
    # Check that size is compatible with level
    if size % (2 ** levels):
        raise ValueError(f"size {size} is not compatible with levels {levels}")

    # Swap the wavelet for the decomposition
    wavswap = pywt.Wavelet(
        name=f'{wav.name}_swap',
        filter_bank=wav.inverse_filter_bank,
    )

    # Compute connection coefficients
    coeffs = connection_coefficients(wav, order)
    if coeffs is None:
        raise ValueError(f"Could not compute connection coefficients for order {order}")

    # Fill the square matrix, periodizing coefficients
    matrix = np.zeros((size, size), dtype=np.float64)
    mid = (coeffs.size - 1) // 2

    for p in range(size):
        for k, c in enumerate(coeffs, -mid):
            matrix[p, (p + k) % size] += c

    # Perform anisotropic (aka fully-separable transform)
    result = pywt.fswavedecn(matrix, wavswap, mode='periodization', levels=levels)
    return result.coeffs

class HighOrderRegularizerConv:
    """Implements high-order wavelet-based regularization terms using convolutions.

    This implementation is likely closer to what was introduced in [1].

    Attributes:
        ORDER_MAX: Maximum order for coefficients computation
        wav: The main wavelet
        wavswap: Swapped version of the wavelet
        coeffs: Connection coefficients for different orders
        regularizers: Dictionary of available regularizers
        mode: Boundary condition mode
    """

    ORDER_MAX = 5  # Max order for coefficients computation

    def __init__(self, wav: pywt.Wavelet):
        """Initialize the high-order regularizer.

        Args:
            wav: A pywt.Wavelet object
        """
        self.mode = 'periodization'
        self.wav = wav

        # Get a swapped version (primal / dual filters are swapped)
        self.wavswap = pywt.Wavelet(
            name=f'{self.wav.name}_swap',
            filter_bank=self.wav.inverse_filter_bank,
        )

        # Compute the connection coefficients up to order max
        self.coeffs: Dict[int, np.ndarray] = {}
        for k in range(self.ORDER_MAX):
            c = connection_coefficients(wav, k)
            if c is not None:
                self.coeffs[k] = c

        # The set of available regularizers
        self.regularizers = {
            'l2norm': self._l2norm_gradient,
            'hornschunck': self._hornschunck_gradient,
            'div': self._div_gradient,
            'curl': self._curl_gradient,
            'laplacian': self._laplacian_gradient,
            'graddiv': self._graddiv_gradient,
            'gradcurl': self._gradcurl_gradient,
        }

    def evaluate(self, C1: List, C2: List, regul_type: str = 'l2norm'):
        """Evaluate the value and gradient of given regularizer.

        Args:
            C1: Wavelet coefficients for first component
            C2: Wavelet coefficients for second component
            regul_type: Name of the regularizer to use

        Returns:
            tuple: (value, (grad1, grad2))
                - value: The regularizer value
                - (grad1, grad2): Gradients for each component
        """
        # Infer levels for further decomposition
        levels = len(C1) - 1

        # Rebuild the fields
        U1 = pywt.waverecn(C1, self.wav, mode=self.mode)
        U2 = pywt.waverecn(C2, self.wav, mode=self.mode)

        # Evaluate gradient
        grad1, grad2 = self.regularizers[regul_type](U1, U2)

        # Decompose gradient to complete its computation
        grad1 = pywt.wavedecn(grad1, self.wavswap, level=levels, mode=self.mode)
        grad2 = pywt.wavedecn(grad2, self.wavswap, level=levels, mode=self.mode)

        # Compute the functional value
        result = 0.0
        for c, g in zip([C1, C2], [grad1, grad2]):
            # Add contribution of approx
            result += np.dot(c[0].ravel(), g[0].ravel())
            # And details
            for cd, gd in zip(c[1:], g[1:]):
                for k in cd.keys():
                    result += np.dot(cd[k].ravel(), gd[k].ravel())

        return 0.5 * result, (grad1, grad2)

    def inner_product(self, A: List, B: List, order_d1: int, order_d2: int) -> float:
        """Compute inner product with derivation orders."""
        # Infer levels for further decomposition
        levels = len(A) - 1

        # Rebuild the field
        Aa = pywt.waverecn(A, self.wav, mode=self.mode)

        # Evaluate the convolution
        tmp = self.convolve_separable(Aa, self.coeffs[order_d1], self.coeffs[order_d2])

        # Decompose with wavswap
        C = pywt.wavedecn(tmp, self.wavswap, level=levels, mode=self.mode)

        # Compute the inner product
        result = np.dot(B[0].ravel(), C[0].ravel())
        for bd, cd in zip(B[1:], C[1:]):
            for k in cd.keys():
                result += np.dot(bd[k].ravel(), cd[k].ravel())

        return result

    def norm(self, A: List, order_d1: int, order_d2: int) -> float:
        """Evaluate the l2-norm of coefficients A with derivation orders.

        Computes: ∫[ |d1d2{A}(x)|^2 ]dx

        Args:
            A: Wavelet coefficients
            order_d1: First derivation order
            order_d2: Second derivation order

        Returns:
            The norm value

        WARNING: This function has normalization issues (as noted in original code)
        """
        logger.warning("norm() has normalization issues...")
        order_total = order_d1 + order_d2
        sign = (-1) ** order_total
        return sign * self.inner_product(A, A, 2 * order_d1, 2 * order_d2)

    def _l2norm_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of l2-norm (order 0) regularizer.

        Computes: 0.5∫[ |U1(x)|^2 + |U2(x)|^2 ]dx
        """
        c0 = self.coeffs[0]
        result = [self.convolve_separable(U, c0, c0) for U in [U1, U2]]
        return result

    def _hornschunck_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of Horn&Schunck (order 1) regularizer.

        Computes: 0.5∫[ |grad{U1}(x)|^2 + |grad{U2}(x)|^2 ]dx
        """
        c0 = self.coeffs[0]
        c2 = self.coeffs[2]
        result = [-(self.convolve_separable(U, c2, c0) + self.convolve_separable(U, c0, c2))
                  for U in [U1, U2]]
        return result

    def _div_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of divergence (order 1) regularizer.

        Computes: 0.5∫[ |div{U1, U2}(x)|^2 ]dx, with div{U1, U2} = d1(U1) + d2(U2)
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        grad1 = -(self.convolve_separable(U2, c1, c1) + self.convolve_separable(U1, c2, c0))
        grad2 = -(self.convolve_separable(U1, c1, c1) + self.convolve_separable(U2, c0, c2))
        return grad1, grad2

    def _curl_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of curl (order 1) regularizer.

        Computes: 0.5∫[ |curl{U1, U2}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        grad1 = self.convolve_separable(U2, c1, c1) - self.convolve_separable(U1, c0, c2)
        grad2 = self.convolve_separable(U1, c1, c1) - self.convolve_separable(U2, c2, c0)
        return grad1, grad2

    def _laplacian_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of laplacian (order 2) regularizer.

        Computes: 0.5∫[ |laplacian{U1}(x)|^2 + |laplacian{U2}(x)|^2 ]dx
        with laplacian{Ui} = d1^2(Ui) + d2^2(Ui)
        """
        c0 = self.coeffs[0]
        c2 = self.coeffs[2]
        c4 = self.coeffs[4]
        grad1 = (self.convolve_separable(U1, c4, c0) +
                 self.convolve_separable(U1, c0, c4) +
                 2. * self.convolve_separable(U1, c2, c2))
        grad2 = (self.convolve_separable(U2, c4, c0) +
                 self.convolve_separable(U2, c0, c4) +
                 2. * self.convolve_separable(U2, c2, c2))
        return grad1, grad2

    def _graddiv_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of grad(div) regularizer.

        Computes: 0.5∫[ |grad{div{U1, U2}}(x)|^2 ]dx, with div{U1, U2} = d1(U1) + d2(U2)
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        c3 = self.coeffs[3]
        c4 = self.coeffs[4]

        grad1 = (self.convolve_separable(U1, c4, c0) +
                 self.convolve_separable(U1, c2, c2) +
                 self.convolve_separable(U2, c3, c1) +
                 self.convolve_separable(U2, c1, c3))
        grad2 = (self.convolve_separable(U2, c0, c4) +
                 self.convolve_separable(U2, c2, c2) +
                 self.convolve_separable(U1, c3, c1) +
                 self.convolve_separable(U1, c1, c3))
        return grad1, grad2

    def _gradcurl_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of grad(curl) regularizer.

        Computes: 0.5∫[ |grad{curl{U1, U2}}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        c3 = self.coeffs[3]
        c4 = self.coeffs[4]

        grad1 = (self.convolve_separable(U1, c0, c4) +
                 self.convolve_separable(U1, c2, c2) -
                 self.convolve_separable(U2, c3, c1) -
                 self.convolve_separable(U2, c1, c3))
        grad2 = (self.convolve_separable(U2, c4, c0) +
                 self.convolve_separable(U2, c2, c2) -
                 self.convolve_separable(U1, c3, c1) -
                 self.convolve_separable(U1, c1, c3))
        return grad1, grad2

    @staticmethod
    def convolve_separable(x: np.ndarray, filter1: np.ndarray, filter2: np.ndarray,
                          origin1: int = 0, origin2: int = 0) -> np.ndarray:
        """Separable convolution of x by filter1 along first axis and filter2 along second axis.

        Args:
            x: Input array
            filter1: Filter along 1st axis (rows)
            filter2: Filter along 2nd axis (columns)
            origin1: Origin for first filter
            origin2: Origin for second filter

        Returns:
            The convolved array
        """
        tmp = scipy.ndimage.convolve1d(x, filter1, axis=0, mode='wrap', origin=origin1)
        return scipy.ndimage.convolve1d(tmp, filter2, axis=1, mode='wrap', origin=origin2)

class HighOrderRegularizerMat:
    """Implements high-order wavelet-based regularization terms using matrices.

    This is a matrix-based implementation as in [2], [3].
    """

    ORDER_MAX = 5  # Max order for coefficients computation

    def __init__(self, wav: pywt.Wavelet, levels: int, image_shape: Tuple[int, ...]):
        """Initialize the matrix-based high-order regularizer.

        Args:
            wav: A pywt.Wavelet object
            levels: Wavelet decomposition levels
            image_shape: Shape of the input images
        """
        self.wav = wav
        self.mode = 'periodization'

        # Compute mass matrices
        width, height = image_shape[:2]
        self.width_matrices: Dict[int, np.ndarray] = {
            o: mass_matrix(o, wav, levels, width) for o in range(self.ORDER_MAX)
        }
        self.height_matrices: Dict[int, np.ndarray] = {
            o: mass_matrix(o, wav, levels, height) for o in range(self.ORDER_MAX)
        }

        # The set of available regularizers
        self.regularizers = {
            'l2norm': self._l2norm_gradient,
            'hornschunck': self._hornschunck_gradient,
            'div': self._divergence_gradient,
            'curl': self._curl_gradient,
            'laplacian': self._laplacian_gradient,
            'graddiv': self._graddiv_gradient,
            'gradcurl': self._gradcurl_gradient,
        }

    def evaluate(self, C1: np.ndarray, C2: np.ndarray, regul_type: str = 'l2norm'):
        """Evaluate the value and gradient of given regularizer.

        Args:
            C1: Wavelet coefficients for first component (as array)
            C2: Wavelet coefficients for second component (as array)
            regul_type: Name of the regularizer to use

        Returns:
            tuple: (value, [grad1, grad2])
        """
        grad1, grad2 = self.regularizers[regul_type](C1, C2)
        value = 0.5 * (np.dot(C1.ravel(), grad1.ravel()) + np.dot(C2.ravel(), grad2.ravel()))
        return value, [grad1, grad2]

    def _l2norm_gradient(self, C1: np.ndarray, C2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of l2-norm (order 0) regularizer."""
        # Note: transpose operators are omitted in even-order matrices
        # as they are symmetric by construction.
        return (self._triple_product(self.height_matrices[0], C, self.width_matrices[0])
                for C in [C1, C2])

    def _hornschunck_gradient(self, C1: np.ndarray, C2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of Horn&Schunck (order 1) regularizer."""
        return (-(self._triple_product(self.height_matrices[2], C, self.width_matrices[0]) +
                  self._triple_product(self.height_matrices[0], C, self.width_matrices[2]))
                for C in [C1, C2])

    def _divergence_gradient(self, C1: np.ndarray, C2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of Divergence (order 1) regularizer."""
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]

        grad1 = -(self._triple_product(n1h, C2, n1w.T) + self._triple_product(n2h, C1, n0w))
        grad2 = -(self._triple_product(n1h, C1, n1w.T) + self._triple_product(n0h, C2, n2w))
        return grad1, grad2

    def _curl_gradient(self, C1: np.ndarray, C2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of curl (order 1) regularizer."""
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]

        grad1 = self._triple_product(n1h, C2, n1w.T) - self._triple_product(n0h, C1, n2w.T)
        grad2 = self._triple_product(n1h, C1, n1w.T) - self._triple_product(n2h, C2, n0w.T)
        return grad1, grad2

    def _laplacian_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of laplacian (order 2) regularizer."""
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        n4h = self.height_matrices[4]
        n4w = self.width_matrices[4]

        grad1 = (self._triple_product(n4h, U1, n0w) +
                 self._triple_product(n0h, U1, n4w) +
                 2. * self._triple_product(n2h, U1, n2w))
        grad2 = (self._triple_product(n4h, U2, n0w) +
                 self._triple_product(n0h, U2, n4w) +
                 2. * self._triple_product(n2h, U2, n2w))
        return grad1, grad2

    def _graddiv_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of grad(div) regularizer."""
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        n3h = self.height_matrices[3]
        n3w = self.width_matrices[3]
        n4h = self.height_matrices[4]
        n4w = self.width_matrices[4]

        grad1 = (self._triple_product(n4h, U1, n0w) +
                 self._triple_product(n2h, U1, n2w) +
                 self._triple_product(n1h, U2, n3w.T) +
                 self._triple_product(n3h, U2, n1w.T))
        grad2 = (self._triple_product(n0h, U2, n4w) +
                 self._triple_product(n2h, U2, n2w) +
                 self._triple_product(n1h, U1, n3w.T) +
                 self._triple_product(n3h, U1, n1w.T))
        return grad1, grad2

    def _gradcurl_gradient(self, U1: np.ndarray, U2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the gradient of grad(curl) regularizer."""
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        n3h = self.height_matrices[3]
        n3w = self.width_matrices[3]
        n4h = self.height_matrices[4]
        n4w = self.width_matrices[4]

        grad1 = (self._triple_product(n0h, U1, n4w) +
                 self._triple_product(n2h, U1, n2w) -
                 self._triple_product(n1h, U2, n3w.T) -
                 self._triple_product(n3h, U2, n1w.T))
        grad2 = (self._triple_product(n4h, U2, n0w) +
                 self._triple_product(n2h, U2, n2w) -
                 self._triple_product(n1h, U1, n3w.T) -
                 self._triple_product(n3h, U1, n1w.T))
        return grad1, grad2

    @staticmethod
    def _triple_product(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Compute A*B*C matrix product."""
        tmp = np.matmul(B, C)
        return np.matmul(A, tmp)