"""[WIP] High-order regularization terms for PyTyphoon.

Note:
    - the computation of the regularization terms in HighOrderRegularizerConv
        is a convolution-based one, likely closer to what was introduced in [1].
    - the computation of the regularization terms in HighOrderRegularizerMat
        is a matrix-based one as in [2], [3].
    - largely untested, use with care.
        
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
# Standard
import logging

# Third-party
import numpy as np
import pywt
import scipy.ndimage as ndimage
import scipy.optimize as optimize


LOGGER = logging.getLogger(__name__)


def connection_coefficients(wav, order):
    """Find the connection coefficients of the wavelet at given order.

    :param wav: a pywt.Wavelet;
    :param order: the derivation order;
    :param return: a vector of coefficients.

    This is the evaluation of L2 dot-products of the form:
         \int[ Phi(x) (d^(n)/dx^n){Phi}(x) ]dx
    where Phi is the mother wavelet.

    Written by P. DERIAN 2018-01-09.
    Updated by P. DERIAN 2019-02-15: using reconstruction filter.
    """
    ctol = 1e-15  # Tolerance for coefficients
    etol = 1e-4  # Tolerance for eigenvalues    
    # Get the low-pass reconstruction filter
    lo = wav.rec_lo
    len_lo = len(lo)
    # Create the matrix
    dim = 2*len_lo - 3
    matrix = np.zeros((dim, dim))
    for m in range(dim):
        for n in range(dim):
            tmp = 0.
            # for each filter value
            for p, lo_p in enumerate(lo):
                idx = m - 2*n + p + len_lo - 2
                if (idx>=0 and idx<len_lo):
                    tmp += lo_p*lo[idx]
            # store only if above threshold
            if np.abs(tmp)>ctol:
                matrix[n,m] = tmp
    # Find coefficient vector: solve eigenvalue problem
    eval, evec = np.linalg.eig(matrix)
    # Check if any REAL eigenvalue matches our order
    sigma = 1./float(2**order)
    ev_found = False
    for i, ev in enumerate(eval):
        if (np.abs(np.real(ev)-sigma)<etol) and (np.abs(np.imag(ev)<1e-14)):
            ev_found = True
            break
    # [TODO] warn if not found?
    coeffs = None
    if ev_found:
        # Get the associated eigen vector
        coeffs = np.real(evec[:,i])
        # If the derivation order is odd, force mid value to be exactly zero
        # as it should be by construction.
        if order % 2:
            coeffs[dim//2] = 0.
        # Apply Beylkin normalization
        # [TODO] document
        norm_factors = np.array([float(-len_lo + 2 + p) for p in range(dim)])**order
        coeffs /= np.sum(norm_factors*coeffs)
        tmp = (np.prod(np.arange(1, order+1)))*np.power(-1.,order)
        coeffs *= tmp
    return coeffs

    
def mass_matrix(order, wav, levels, size):
    """Compute the mass matrix.
    
    :param order: total derivation order;
    :param wav: a pywt.Wavelet;
    :param levels: wavelet decomposition levels;
    :param size: size of the matrix;
    :return: a (size, size) mass matrix.
    """
    # Check that size is compatible with level
    if size % (2**levels):
        raise ValueError("size {} is not compatible with levels {}".format(size, levels))
    # Swap the wavelet for the decomposition
    wavswap = pywt.Wavelet(
            name='{}_swap'.format(wav.name),
            filter_bank=wav.inverse_filter_bank,
            )
    # Compute connection coefficients
    coeffs = connection_coefficients(wav, order)
    # Fill the square matrix, periodizing coefficients
    matrix = np.zeros((size, size), dtype=np.float64) 
    mid = (coeffs.size - 1)//2
    for p in range(size):
        for k, c in enumerate(coeffs, -mid):
            matrix[p, (p + k) % size] += c
    # Perform anisotropic (aka fully-separable transform)
    result = pywt.fswavedecn(matrix, wavswap, mode='periodization', levels=levels)
    result = result.coeffs
    return result
            

class HighOrderRegularizerConv:
    """Implements high-order wavelet-based regularization terms,
    here implemented with convolutions.

    Written by P. DERIAN 2018-01-09.
    """
    ORDER_MAX = 5  # Max order for coefficients computation.

    def __init__(self, wav):
        """Instance constructor.

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2019-02-15: fixed biorthogonal case.
        """
        self.mode = 'periodization'
        # The main wavelet
        self.wav = wav
        # Get a swapped version (primal / dual filters are swapped)
        # Note: if the wav is orthogonal, wavswap is in practice the same as wav.
        self.wavswap = pywt.Wavelet(
            name='{}_swap'.format(self.wav.name),
            filter_bank=self.wav.inverse_filter_bank,
            )
        # Compute the connection coefficients up to order max. 
        self.coeffs = {k: connection_coefficients(wav, k) for k in range(self.ORDER_MAX)}
        # The set of available regularizers
        self.regularizers = {'l2norm': self._l2norm_gradient,
                             'hornschunck': self._hornschunck_gradient,
                             'div': self._div_gradient,
                             'curl': self._curl_gradient,
                             'laplacian': self._laplacian_gradient,
                             'graddiv': self._graddiv_gradient,
                             'gradcurl': self._gradcurl_gradient,
                             }

    def evaluate(self, C1, C2, regul_type='l2norm'):
        """Evaluate the value and gradient of given regularizer.

        :param C1: wavelet coefficients.
        :param C2: wavelet coefficients.
        :param regul_type:
        :return: value, (grad1, grad2)

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2019-02-15: fixed biorthogonal case.
        """
        # Infer levels for further decomposition
        levels = len(C1) - 1
        # Rebuild the fields
        U1 = pywt.waverecn(C1, self.wav, mode=self.mode)
        U2 = pywt.waverecn(C2, self.wav, mode=self.mode)
        # Evaluate gradient
        [grad1, grad2] = self.regularizers[regul_type](U1, U2)
        # Decompose gradient to complete its computation 
        # Note: use wavswap here!
        grad1 = pywt.wavedecn(grad1, self.wavswap, level=levels, mode=self.mode)
        grad2 = pywt.wavedecn(grad2, self.wavswap, level=levels, mode=self.mode)
        # Compute the functional value
        result = 0.
        for c, g in zip([C1, C2], [grad1, grad2]):
            # Add contribution of approx
            result += np.dot(c[0].ravel(), g[0].ravel())
            # And details
            for cd, gd in zip(c[1:], g[1:]):
                for k in cd.keys():
                    result += np.dot(cd[k].ravel(), gd[k].ravel())
        return 0.5*result, (grad1, grad2)

    def inner_product(self, A, B, order_d1, order_d2):
        """
        """
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
           
    def norm(self, A, order_d1, order_d2):
        """Evaluates the l2-norm of coefficents A with derivation orders:
            \int[ |d1d2{A}(x)|^2 ]dx
            
        WARNING: 
            
        Written by P. DERIAN 2019-02-19
        """
        LOGGER.warning("norm() has normalization issues...")
        order_total = order_d1 + order_d2
        sign = (-1)**order_total
        return sign * self.inner_product(A, A, 2*order_d1, 2*order_d2)
           
    def _l2norm_gradient(self, U1, U2):
        """Evaluates the gradient of l2-norm (order 0) regularizer:
            0.5\int[ |U1(x)|^2 + |U2(x)|^2 ]dx

        Written by P. DERIAN 2018-01-09.
        """
        c0 = self.coeffs[0]
        result = [self.convolve_separable(U, c0, c0) for U in [U1, U2]]
        return result

    def _hornschunck_gradient(self, U1, U2):
        """Evaluates the gradient of Horn&Schunck (order 1) regularizer:
            0.5\int[ |grad{U1}(x)|^2 + |grad{U2}(x)|^2 ]dx

        Written by P. DERIAN 2018-01-09.
        """
        c0 = self.coeffs[0]
        c2 = self.coeffs[2]
        result = [-(self.convolve_separable(U, c2, c0) + self.convolve_separable(U, c0, c2))
                  for U in [U1, U2]]
        return result

    def _div_gradient(self, U1, U2):
        """Evaluates the gradient of divergence (order 1) regularizer:
            0.5\int[ |div{U1, U2}(x)|^2 ]dx, with div{U1, U2} = d1(U1) + d2(U2)

        Written by P. DERIAN 2019-02-19.
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        grad1 = -(self.convolve_separable(U2, c1, c1) + self.convolve_separable(U1, c2, c0))
        grad2 = -(self.convolve_separable(U1, c1, c1) + self.convolve_separable(U2, c0, c2))
        return [grad1, grad2]
      
    def _curl_gradient(self, U1, U2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |curl{U1, U2}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)

        Written by P. DERIAN 2019-02-19.
        """
        c0 = self.coeffs[0]
        c1 = self.coeffs[1]
        c2 = self.coeffs[2]
        grad1 = self.convolve_separable(U2, c1, c1) - self.convolve_separable(U1, c0, c2)
        grad2 = self.convolve_separable(U1, c1, c1) - self.convolve_separable(U2, c2, c0)
        return [grad1, grad2]    
      
    def _laplacian_gradient(self, U1, U2):
        """Evaluates the gradient of laplacian (order 2) regularizer:
            0.5\int[ |laplacian{U1}(x)|^2 + |laplacian{U2}(x)|^2 ]dx ,
            with laplacian{Ui} = d1^2(Ui) + d2^2(Ui)
            
        Written by P. DERIAN 2019-02-19.
        """    
        c0 = self.coeffs[0]
        c2 = self.coeffs[2]
        c4 = self.coeffs[4]
        grad1 = (self.convolve_separable(U1, c4, c0) + 
                 self.convolve_separable(U1, c0, c4) +
                 2.*self.convolve_separable(U1, c2, c2))
        grad2 = (self.convolve_separable(U2, c4, c0) + 
                 self.convolve_separable(U2, c0, c4) +
                 2.*self.convolve_separable(U2, c2, c2))
        return [grad1, grad2]

    def _graddiv_gradient(self, U1, U2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |grad{div{U1, U2}}(x)|^2 ]dx, with div{U1, U2} = d1(U1) + d2(U2)

        Written by P. DERIAN 2019-02-19.
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
        return [grad1, grad2]    
        
    def _gradcurl_gradient(self, U1, U2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |grad{curl{U1, U2}}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)

        Written by P. DERIAN 2019-02-19.
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
        return [grad1, grad2]    
            
    @staticmethod
    def convolve_separable(x, filter1, filter2, origin1=0, origin2=0):
        """Separable convolution of x by filter1 along first axis and filter2 along second axis.

        :param filter1: filter along 1st axis (rows);
        :param filter2: filter along 2nd axis (columns);

        Written by P. DERIAN 2018-01-09.
        """
        tmp = ndimage.filters.convolve1d(x, filter1, axis=0, mode='wrap', origin=origin1)
        return ndimage.filters.convolve1d(tmp, filter2, axis=1, mode='wrap', origin=origin2)


class HighOrderRegularizerMat:
    """Implements high-order wavelet-based regularization terms,
    here implemented with matrices.

    Written by P. DERIAN 2018-02-18.
    """
    ORDER_MAX = 5  # Max order for coefficients computation.

    def __init__(self, wav, levels, image_shape):
        """Instance constructor.

        Written by P. DERIAN 2018-02-19.
        """
        # Keep reference to the wavelet
        self.wav = wav
        self.mode = 'periodization'
        # Compute mass matrices
        width, height = image_shape[:2]
        self.width_matrices = {o: mass_matrix(o, wav, levels, width) for o in range(self.ORDER_MAX)}
        self.height_matrices = {o: mass_matrix(o, wav, levels, height) for o in range(self.ORDER_MAX)}        
        # The set of available regularizers
        self.regularizers = {'l2norm': self._l2norm_gradient,
                             'hornschunck': self._hornschunck_gradient,
                             'div': self._divergence_gradient,
                             'curl': self._curl_gradient,
                             'laplacian': self._laplacian_gradient,
                             'graddiv': self._graddiv_gradient,
                             'gradcurl': self._gradcurl_gradient,
                             }
 
    def evaluate(self, C1, C2, regul_type='l2norm'):
        """Evaluate the value and gradient of given regularizer.

        :param C1: wavelet coefficients;
        :param C2: wavelet coefficients;
        :param regul_type: name of the relarizer;
        :return: value, (grad1, grad2).

        Written by P. DERIAN 2019-02-18.
        """
        grad1, grad2 = self.regularizers[regul_type](C1, C2)
        value = 0.5*(np.dot(C1.ravel(), grad1.ravel()) + np.dot(C2.ravel(), grad2.ravel()))
        return value, [grad1, grad2] 

    def _l2norm_gradient(self, C1, C2):
        """Evaluates the gradient of l2-norm (order 0) regularizer:
            0.5\int[ |U1(x)|^2 + |U2(x)|^2 ]dx

        Written by P. DERIAN 2019-02-18.
        """
        # Note: transpose operators are omitted in even-order matrices
        # as they are symmetric by construction.
        return (self._triple_product(self.height_matrices[0], C, self.width_matrices[0])
                for C in [C1, C2])

    def _hornschunck_gradient(self, C1, C2):
        """Evaluates the gradient of Horn&Schunck (order 1) regularizer:
            0.5\int[ |grad{U1}(x)|^2 + |grad{U2}(x)|^2 ]dx

        Written by P. DERIAN 2019-02-18.
        """
        return (-(self._triple_product(self.height_matrices[2], C, self.width_matrices[0]) +
                  self._triple_product(self.height_matrices[0], C, self.width_matrices[2]))
                for C in [C1, C2])
   
    def _divergence_gradient(self, C1, C2):
        """Evaluates the gradient of Divergence (order 1) regularizer:
            0.5\int[ |div{U1, U2}(x)|^2 ]dx, with div{1, U2} = d1(U1) + d2(U2)

        Written by P. DERIAN 2019-02-18.
        """
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        grad1 = -(self._triple_product(n1h, C2, n1w.T) + self._triple_product(n2h, C1, n0w))
        grad2 = -(self._triple_product(n1h, C1, n1w.T) + self._triple_product(n0h, C2, n2w))
        return grad1, grad2

    def _curl_gradient(self, C1, C2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |curl{U1, U2}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)

        Written by P. DERIAN 2019-02-19.
        """
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n1h = self.height_matrices[1]
        n1w = self.width_matrices[1]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        grad1 = self._triple_product(n1h, C2, n1w.T) - self._triple_product(n0h, C1, n2w.T)
        grad2 = self._triple_product(n1h, C1, n1w.T) - self._triple_product(n2h, C2, n0w.T)
        return grad1, grad2
    
    def _laplacian_gradient(self, U1, U2):
        """Evaluates the gradient of laplacian (order 2) regularizer:
            0.5\int[ |laplacian{U1}(x)|^2 + |laplacian{U2}(x)|^2 ]dx ,
            with laplacian{Ui} = d1^2(Ui) + d2^2(Ui)
            
        Written by P. DERIAN 2019-02-19.
        """        
        n0h = self.height_matrices[0]
        n0w = self.width_matrices[0]
        n2h = self.height_matrices[2]
        n2w = self.width_matrices[2]
        n4h = self.height_matrices[4]
        n4w = self.width_matrices[4]
        grad1 = (self._triple_product(n4h, U1, n0w) + 
                 self._triple_product(n0h, U1, n4w) +
                 2.*self._triple_product(n2h, U1, n2w))
        grad2 = (self._triple_product(n4h, U2, n0w) + 
                 self._triple_product(n0h, U2, n4w) +
                 2.*self._triple_product(n2h, U2, n2w))
        return grad1, grad2

    def _graddiv_gradient(self, U1, U2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |grad{div{U1, U2}}(x)|^2 ]dx, with div{U1, U2} = d1(U1) + d2(U2)

        Written by P. DERIAN 2019-02-19.
        """
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

    def _gradcurl_gradient(self, U1, U2):
        """Evaluates the gradient of curl (order 1) regularizer:
            0.5\int[ |grad{curl{U1, U2}}(x)|^2 ]dx, with curl{U1, U2} = d1(U2) - d2(U1)

        Written by P. DERIAN 2019-02-19.
        """
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
    def _triple_product(A, B, C):
        """compute A*B*C.
        """
        tmp = np.matmul(B, C)
        return np.matmul(A, tmp)   

        
### Demonstrations ###

if __name__=="__main__":
    # Standard library
    import matplotlib.pyplot as pyplot
    # Custom
    import sys
    sys.path.append('..')
    import demo.inr as inr
    import matplotlib.pyplot as plt

    # Filters for gradient computation (finite differences, periodic bc)
    # see https://en.wikipedia.org/wiki/Finite_difference_coefficient
    d1_filters = {
        2: [-1./2., 0., 1./2.],
        4: [1./12., -2./3., 0., 2/3., -1./12.],
        6: [-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.],
        8: [1./280., -4./105., 1./5., -4./5., 0., 4./5., -1./5., 4./105., -1./280.],
        }
    d2_filters = {
        2: [1., -2., 1.],
        4: [-1./12., 4./3., -5./2., 4/3., -1./12.],
        6: [1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90.],   
        8: [-1./560., 8./315., -1./5., 8./5., -205./72., 8./5., -1./5., 8./315., -1./560.],
        }
    
    def l2norm(U1, U2):
        return 0.5*np.sum(U1**2 + U2**2)
 
    def hornschunk(U1, U2, d1):
        result = 0.
        g11 = ndimage.filters.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
        g12 = ndimage.filters.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
        result += np.sum(g11**2 + g12**2)
        g21 = ndimage.filters.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
        g22 = ndimage.filters.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
        result += np.sum(g21**2 + g22**2)
        return 0.5*result
  
    def divergence(U1, U2, d1):
        g11 = ndimage.filters.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
        g22 = ndimage.filters.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
        result = np.sum((g11 + g22)**2)
        return 0.5*result       
      
    def curl(U1, U2, d1):
        g12 = ndimage.filters.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
        g21 = ndimage.filters.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
        result = np.sum((g21 - g12)**2)
        return 0.5*result         
      
    def laplacian(U1, U2, d2):  
        result = 0.
        g11 = ndimage.filters.convolve1d(U1, d2, axis=1, mode='wrap', origin=0)
        g12 = ndimage.filters.convolve1d(U1, d2, axis=1, mode='wrap', origin=0)
        result += np.sum(g11**2 + g12**2)
        g21 = ndimage.filters.convolve1d(U2, d2, axis=0, mode='wrap', origin=0)
        g22 = ndimage.filters.convolve1d(U2, d2, axis=1, mode='wrap', origin=0)  
        result += np.sum(g21**2 + g22**2)
        return 0.5*result

    def graddiv(U1, U2, d1):
        g11 = ndimage.filters.convolve1d(U1, d1, axis=0, mode='wrap', origin=0)
        g22 = ndimage.filters.convolve1d(U2, d1, axis=1, mode='wrap', origin=0)
        div = g11 + g22
        g1 = ndimage.filters.convolve1d(div, d1, axis=0, mode='wrap', origin=0)
        g2 = ndimage.filters.convolve1d(div, d1, axis=1, mode='wrap', origin=0)
        return 0.5*np.sum(g1**2 + g2**2)
        
    def gradcurl(U1, U2, d1):
        g12 = ndimage.filters.convolve1d(U1, d1, axis=1, mode='wrap', origin=0)
        g21 = ndimage.filters.convolve1d(U2, d1, axis=0, mode='wrap', origin=0)
        curl = g21 - g12
        g1 = ndimage.filters.convolve1d(curl, d1, axis=0, mode='wrap', origin=0)
        g2 = ndimage.filters.convolve1d(curl, d1, axis=1, mode='wrap', origin=0)
        return 0.5*np.sum(g1**2 + g2**2)
      
    def demo_mass_matrix():
        """

        Written by P. DERIAN 2019-02-18.
        """    
        size = 256  # Image size [px]
        levels = 0  # Decomposition levels
        wav = pywt.Wavelet('db10')
        N0 = mass_matrix(0, wav, levels, size)
        N1 = mass_matrix(1, wav, levels, size)
        N2 = mass_matrix(2, wav, levels, size)
        
        # Display matrices
        eps = 1e-15
        fig, axes = plt.subplots(1, 3)
        # axes[0].imshow(abs(N0) > eps)
        # axes[1].imshow(abs(N1) > eps)
        # axes[2].imshow(abs(N2) > eps)
        axes[0].imshow(N0, cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[1].imshow(N1, cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[2].imshow(N2, cmap='RdYlBu_r', vmin=-1, vmax=1)        
        plt.show()
            
    def demo_highorder_conv():
        """

        Written by P. DERIAN 2018-01-09.
        """         
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 3  # Note: levels have no influence with the convolution-based form.
        fd_order = 8  # Order of finite-differences filters for truth
        U2, U1 = inr.readMotion('../demo/UVtruth.inr')
        # Wavelet-based computations
        hor_conv = HighOrderRegularizerConv(wav)
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
        d1 = d1_filters[fd_order]
        d2 = d2_filters[fd_order]
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
            
    def demo_highorder_mat():
        """

        Written by P. DERIAN 2019-02-18.
        """
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 2  # Note: levels have no influence with the convolution-based form.
        fd_order = 8  # Order of finite-differences filters for truth
        U2, U1 = inr.readMotion('../demo/UVtruth.inr')
        # Wavelet-based computations
        hor_mat = HighOrderRegularizerMat(wav, wav_levels, U1.shape)
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
        d1 = d1_filters[fd_order]
        d2 = d2_filters[fd_order]
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

    def compare_highorders(name='laplacian'):
        """
        """
        # Parameters
        wav = pywt.Wavelet('bior6.8')
        wav_levels = 3  # Note: levels have no influence with the convolution-based form.
        fd_order = 8  # Order of finite-differences filters for truth
        U2, U1 = inr.readMotion('../demo/UVtruth.inr')
        # Regularizers
        hor_conv = HighOrderRegularizerConv(wav)
        hor_mat = HighOrderRegularizerMat(wav, wav_levels, U1.shape)
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
        
    def check_conv():
        """
        """
        # Data parameters
        fine_scale = 7
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
        print(yy.size, np.sqrt(yy.size))
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
        hor_conv = HighOrderRegularizerConv(wav)
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
        fig, ax  = plt.subplots()
        ax.imshow(f, extent=[y[0], y[-1], x[0], x[-1]], origin='lower')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_title('f(x, y)')
        plt.show()
        
    
    # demo_mass_matrix()
    demo_highorder_conv()
    demo_highorder_mat()
    compare_highorders('l2norm')
    compare_highorders('hornschunck')
    compare_highorders('div')
    compare_highorders('curl')
    compare_highorders('laplacian')
    check_conv()