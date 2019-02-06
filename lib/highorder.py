"""[WIP] High-order regularization terms for PyTyphoon.

Note:
    - largely untested, use with care.
    - the computation of the regularization terms is a convolution-based one, likely
        closer to what was introduced in [1], rather than the matrix-based one in [2] and [3].

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
###
import numpy as np
import pywt
import scipy.ndimage as ndimage
import scipy.optimize as optimize
###

class HighOrderRegularizer:
    """Implements high-order wavelet-based regularization terms.

    Written by P. DERIAN 2018-01-09.
    """
    ORDER_MAX = 6 #max order for coefficients computation.

    def __init__(self, wav):
        """Instance constructor.

        Written by P. DERIAN 2018-01-09.
        """
        self.mode = 'periodization'
        self.wav = wav #keep reference
        self.coeffs = {k: self.connection_coefficients(wav, k) for k in range(self.ORDER_MAX)}
        self.regularizers = {'l2norm': self._l2norm_gradient,
                             'hornschunck': self._hornschunck_gradient,
                             }

    def evaluate(self, C1, C2, regul_type='l2norm'):
        """Evaluate the value and gradient of given regularizer.

        :param C1: wavelet coefficients.
        :param C2: wavelet coefficients.
        :param regul_type:
        :return: value, (grad1, grad2)

        Written by P. DERIAN 2018-01-09.
        """
        ### infer levels for further decomposition
        levels = len(C1) - 1
        ### rebuild the fields
        U1 = pywt.waverec2(C1, self.wav, mode=self.mode)
        U2 = pywt.waverec2(C2, self.wav, mode=self.mode)
        #### evaluate
        [grad1, grad2] = self.regularizers[regul_type](U1, U2)
        # decompose gradient to complete its computation
        grad1 = pywt.wavedec2(grad1, self.wav, level=levels, mode=self.mode)
        grad2 = pywt.wavedec2(grad2, self.wav, level=levels, mode=self.mode)
        # compute the functional value
        result = 0.
        for c, g in zip([C1, C2], [grad1, grad2]):
            # add contribution of approx
            result += np.dot(c[0].ravel(), g[0].ravel())
            # and details
            for cd, gd in zip(c[1:], g[1:]):
                for cdd, gdd in zip(cd, gd):
                    result += np.dot(cdd.ravel(), gdd.ravel())
        return 0.5*result, (grad1, grad2)

    def _l2norm_gradient(self, U1, U2):
        """Evaluates the gradient of l2-norm (order 0) regularizer:
            0.5\int[ |U1(x)|^2 + |U2(x)|^2 ]dx

        Written by P. DERIAN 2018-01-09.
        """
        grad = []
        result = 0.
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
        result = [ -(self.convolve_separable(U, c2, c0) + self.convolve_separable(U, c0, c2))
                   for U in [U1, U2]]
        return result

    @staticmethod
    def connection_coefficients(wav, order):
        """Find the connection coefficients of the wavelet at given order.

        :param wav: a pywt.Wavelet;
        :param order: the derivation order;
        :param return: a vector of coefficients.

        This is the evaluation of L2 dot-products of the form:
             \int[ Phi(x) (d^(n)/dx^n){Phi}(x) ]dx
        where Phi is the mother wavelet.

        Written by P. DERIAN 2018-01-09.
        """
        ctol = 1e-15 #tolerance for coefficients
        etol = 1e-4 #tolerance for eigenvalues
        ### get the low-pass filter
        lo = wav.dec_lo
        len_lo = len(lo)
        ### create the matrix
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
        ### Find coefficient vector
        # solve eigen values
        eval, evec = np.linalg.eig(matrix)
        #print('eigs:', eval)
        # check if any REAL eigenvalue matches our order
        sigma = 1./float(2**order)
        ev_found = False
        for i, ev in enumerate(eval):
            if (np.abs(np.real(ev)-sigma)<etol) and (np.abs(np.imag(ev)<1e-14)):
                ev_found = True
                break
        # [TODO] warn if not found?
        coeffs = None
        if ev_found:
            coeffs = evec[:,i]
            #print('found:', ev_found, i, ev, coeffs)
            ### Process
            # if the order is odd, force mid value to be exactly zero
            if order%2:
                coeffs[dim//2] = 0.
            #print(coeffs)
            # apply Beylkin normalization
            norm_factors = np.array([float(-len_lo + 2 + p) for p in range(dim)])**order
            coeffs /= np.sum(norm_factors*coeffs)
            tmp = (np.prod(np.arange(1, order+1)))*np.power(-1.,order)
            coeffs *= tmp
        return coeffs

    @staticmethod
    def convolve_separable(x, filter1, filter2, origin1=0, origin2=0):
        """Separable convolution of x by filter1 along first axis and filter2 along second axis.

        :param filter1: filter along 1st axis;
        :param filter2: filter along 2nd axis;

        Written by P. DERIAN 2018-01-09.
        """
        tmp = ndimage.filters.convolve(x, filter1.reshape(1,-1), mode='wrap', origin=origin1)
        return ndimage.filters.convolve(tmp, filter2.reshape(-1,1), mode='wrap', origin=origin2)

### Demonstrations ###

if __name__=="__main__":
    ###
    import matplotlib.pyplot as pyplot
    ###
    import sys
    sys.path.append('..')
    import demo.inr as inr
    ###

    def demo_connection_coeff():
        """

        Written by P. DERIAN 2018-01-09.
        """

        def l2norm(U1, U2):
            return 0.5*np.sum(U1**2 + U2**2)

        def hornschunk(U1, U2):
            result = 0.
            g1, g2 = np.gradient(U1)
            result += np.sum(g1**2 + g2**2)
            g1, g2 = np.gradient(U2)
            result += np.sum(g1**2 + g2**2)
            return 0.5*result

        levels = 4
        wav = pywt.Wavelet('db5')
        hor = HighOrderRegularizer(wav)
        U2, U1 = inr.readMotion('../demo/UVtruth.inr')
        C1 = pywt.wavedec2(U1, hor.wav, level=levels, mode=hor.mode)
        C2 = pywt.wavedec2(U2, hor.wav, level=levels, mode=hor.mode)
        l2norm_hor, (grad1, grad2) = hor.evaluate(C1, C2, 'l2norm')
        hornschunk_hor, (grad1, grad2) = hor.evaluate(C1, C2, 'hornschunck')
        # Note: not real "truth" since gradients are computed by finite differences.
        l2norm_truth = l2norm(U1, U2)
        hornschunk_truth = hornschunk(U1, U2)
        #
        for test, truth in zip([l2norm_hor, hornschunk_hor],
                               [l2norm_truth, hornschunk_truth]):
            print('{:.6f}, {:.3f} {:.3f}'.format(test/truth, test, truth))

    demo_connection_coeff()
