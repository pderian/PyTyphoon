"""Python implementation of the Typhoon algorithm solving dense 2D optical flow problems.

Disclaimer: this is a Python implementation of the Typhoon algorithm. The reference implementation
    used in [4], [5] is written in C++ and GPU-accelerated with CUDA. It is the property of
    Inria (FR) and the CSU Chico Research Foundation (Ca, USA).
    This Python implementation is *not* exactly the same as the reference for many reasons,
    and it is obviously much slower.

Dependencies:
    - numpy, scipy;
    - pywavelets.

References:
    [1] Data-term & basic algorithm:
        Derian, P.; Heas, P.; Herzet, C. & Memin, E.
        "Wavelets and Optical Flow Motion Estimation".
        Numerical Mathematics: Theory, Method and Applications, Vol. 6, pp. 116-137, 2013.
    [2] High-order regularization terms:
        Kadri Harouna, S.; Derian, P.; Heas, P. & Memin, E.
        "Divergence-free Wavelets and High Order Regularization".
        International Journal of Computer Vision, Vol. 103, pp. 80-99, 2013.
    [3] Application to wind estimation from lidar images:
        Derian, P.; Mauzey, C. F. & Mayor, S. D.
        "Wavelet-based optical flow for two-component wind field estimation from single aerosol lidar data"
        Journal of Atmospheric and Oceanic Technology, Vol. 32, pp. 1759-1778, 2015.
    [4] Application to near-shore surface current estimation from shore and UAV cameras:
        Derian, P. & Almar, R.
        "Wavelet-based Optical Flow Estimation of Instant Surface Currents from Shore-based and UAV Video"
        IEEE Transactions on Geoscience and Remote Sensing, Vol. 55, pp. 5790-5797, 2017.

Written by P. DERIAN 2018-01-09.
www.pierrederian.net - contact@pierrederian.net
"""
__version__ = 0.1
###
import numpy
import pywt
import scipy.ndimage as ndimage
import scipy.optimize as optimize
###

class OpticalFlowCore:
    """Core functions for optical flow.

    Written by P. DERIAN 2018-01-09.
    """

    def __init__(self, shape, dtype=numpy.float32):
        """Constructor.

        :param shape: the grid shape;
        :param dtype: numpy.float32 or numpy.float64.

        Written by P. DERIAN 2018-01-09.
        """
        self.dtype = dtype
        ### grid coordinates
        self.shape = shape
        # 1D
        self.x1 = numpy.arange(shape[0], dtype=dtype)
        self.x2 = numpy.arange(shape[1], dtype=dtype)
        # 2D
        self.X1, self.X2 = numpy.indices(self.shape, dtype=dtype)
        self.X12 = numpy.vstack((self.X1.ravel(), self.X2.ravel()))
        ### Misc parameters
        self.sigma_blur = 0.5 #gaussian blur sigma before spatial gradient computation
        self.interpolation_order = 3 #pixel interpolation order
        self.boundary_condition = 'wrap' #boundary condition
        ### Buffer
        self.buffer = numpy.zeros(numpy.prod(self.shape), dtype=dtype)

    def DFD(self, im0, im1, U1, U2):
        """Compute the DFD.

        :return: the DFD.

        Written by P. DERIAN 2018-01-09.
        """
        # warp im1
        map_coords = self.X12.copy()
        map_coords[0] += U1.ravel()
        map_coords[1] += U2.ravel()
        im1_warp = ndimage.interpolation.map_coordinates(im1, map_coords,
                                                         order=self.interpolation_order,
                                                         mode=self.boundary_condition)
        # difference
        return im1_warp.reshape(self.shape) - im0

    def DFD_gradient(self, im0, im1, U1, U2):
        """Compute the displaced frame difference (DFD) functional value and its gradients.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param U1: displacment along the first axis;
        :param U2: displacement along the second axis;

        :return: dfd, (grad1, grad2)
            - DFD: the value of the functional;
            - grad1, grad2: the gradient of the DFD functional w.r.t. U1, U2.

        Written by P. DERIAN 2018-01-09.
        """
        # warp im1->buffer
        map_coords = self.X12.copy()
        map_coords[0] += U1.ravel()
        map_coords[1] += U2.ravel()
        ndimage.interpolation.map_coordinates(im1, map_coords,
                                              output=self.buffer,
                                              order=self.interpolation_order,
                                              mode=self.boundary_condition)
        im1_warp = self.buffer.reshape(self.shape)
        # difference
        dfd = im1_warp - im0
        # spatial gradients
        # [TODO] use derivative of gaussians?
        if self.sigma_blur>0:
            im1_warp = ndimage.filters.gaussian_filter(im1_warp, self.sigma_blur)
        grad1, grad2 = numpy.gradient(im1_warp)
        grad1 *= dfd
        grad2 *= dfd
        # return
        return 0.5*numpy.sum(dfd**2), (grad1, grad2)

class Typhoon:
    """Implements the Typhoon algorithm: dense optical flow estimation on wavelet bases.

    Written by P. DERIAN 2018-01-09.
    """

    def __init__(self, shape=None):
        """Instance constructor with optional shape.

        :param shape: optional shape of the problem.

        Written by P. DERIAN 2018-01-09.
        """
        self.core = OpticalFlowCore(shape) if (shape is not None) else None

    def solve(self, im0, im1, wav='haar', levels_decomp=3, levels_estim=None,
              U1_0=None, U2_0=None):
        """Solve the optical flow problem for given images and wavelet.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param wav: the name of the wavelet;
        :param levels_decomp: number of decomposition levels;
        :param levels_estim: number of estimation levels (<=levels_decomp);
        :param U1_0: optional first guess for U1 (same shape as images);
        :param U2_0: optional first guess for U2 (same shape as images);
        :return: U1, U2 the estimated displacement along the first and second axes.

        Note: without explicit regularization terms, it is necessary to set
            levels_estim<levels_decomp in order to close the estimation problem.

        Written by P. DERIAN 2018-01-09.
        """
        ### Core
        # create a new core if the image shape is not compatible
        if (self.core is None) or (not numpy.testing.assert_equal(self.core.shape, im0.shape)):
            self.core = OpticalFlowCore(im0.shape)
        ### Images
        self.im0 = im0.astype(self.core.dtype)
        self.im1 = im1.astype(self.core.dtype)
        ### Wavelets
        self.levels_decomp = levels_decomp
        self.levels_estim = (min(levels_estim, self.levels_decomp) if (levels_estim is not None)
                             else self.levels_decomp-1)
        self.wav = pywt.Wavelet(wav)
        self.wav_boundary_condition = 'periodization'
        ### Motion fields
        # initialize with given fields, if any, otherwise with zeros.
        U1 = U1_0 if (U1_0 is not None) else numpy.zeros(self.core.shape, dtype=self.core.dtype)
        U2 = U2_0 if (U2_0 is not None) else numpy.zeros(self.core.shape, dtype=self.core.dtype)
        # the corresponding wavelet coefficients
        self.C1_list = pywt.wavedec2(U1, self.wav, level=self.levels_decomp,
                                     mode=self.wav_boundary_condition)
        self.C2_list = pywt.wavedec2(U2, self.wav, level=self.levels_decomp,
                                     mode=self.wav_boundary_condition)
        # which we reshape as arrays to get the slices for future manipulations.
        _, self.slices1 = pywt.coeffs_to_array(self.C1_list)
        _, self.slices2 = pywt.coeffs_to_array(self.C2_list)
        ### Solve
        print('Decomposition over {} scales of details, {} estimated'.format(
            self.levels_decomp, self.levels_estim))
        # for each level
        # [TODO] set last level
        for level in range(self.levels_estim+1):
            print('details ({})'.format(level) if level else 'approx. (0)')
            # the initial condition, as array (Note: flattened by l-bfgs)
            C1_array, _ = pywt.coeffs_to_array(self.C1_list[:level+1])
            C2_array, _ = pywt.coeffs_to_array(self.C2_list[:level+1])
            C12_array = numpy.vstack((C1_array[numpy.newaxis,...], C2_array[numpy.newaxis,...]))
            C12_shape = C12_array.shape
            # create the cost function for this step
            f_g = self.create_cost_function(level, C12_shape)
            # minimize
            C12_array, min_value, optim_info = optimize.fmin_l_bfgs_b(f_g,
                                                                      C12_array.astype(numpy.float64),
                                                                      factr=1000.,
                                                                      iprint=0)
            print('\tl-bfgs completed with status {warnflag} - {nit} iterations, {funcalls} calls'.format(**optim_info))
            print('\tcurrent functional value: {:.2f}'.format(min_value))
            # store result
            C12_array = C12_array.reshape(C12_shape)
            C1_list = pywt.array_to_coeffs(C12_array[0], self.slices1[:level+1],
                                           output_format='wavedec2')
            C2_list = pywt.array_to_coeffs(C12_array[1], self.slices2[:level+1],
                                            output_format='wavedec2')
            for l in range(level+1):
                self.C1_list[l] = C1_list[l]
                self.C2_list[l] = C2_list[l]
        ### Rebuild
        U1 = pywt.waverec2(self.C1_list, self.wav, mode=self.wav_boundary_condition)
        U2 = pywt.waverec2(self.C2_list, self.wav, mode=self.wav_boundary_condition)
        return U1, U2

    def create_cost_function(self, step, shape):
        """The cost function takes wavelet coefficient as input parameters;
        and returns (f, grad).

        :param step: 0 (coarse), 1 (first level of details, etc);
        :param shape: the shape to reshape x.
        :return: the cost function for l-bfgs.

        Written by P. DERIAN 2018-01-09.
        """
        def f_g(x):
            """Compute the coast function and its gradient for given input x.

            :param x: input point, ndarray of float64.
            :return: f, g
                - f the function value, scalar;
                - g the gradient, ndarray of same size and type as x.

            Notes:
                - We cast from (at input) and to (at output) float64, as l-bgs wants 8-byte floats.
                - There are losts of shape manipulations, some of them could possibly be
                  avoided. This is due to l-bfgs using 1d arrays whereas pywt relies on lists
                  of (tupples of) arrays. Here it was chosen to be as explicit and clear
                  as possible, possibly sacrificing some speed along the way.

            Written by P. DERIAN 2018-01-09.
            """
            ### rebuild motion field
            # reshape 1d vector to 3d array
            x = x.reshape(shape).astype(self.core.dtype)
            # extract coefficients, reshape for pywt
            C1_list = pywt.array_to_coeffs(x[0], self.slices1[:step+1],
                                           output_format='wavedec2') + self.C1_list[step+1:]
            C2_list = pywt.array_to_coeffs(x[1], self.slices2[:step+1],
                                            output_format='wavedec2') + self.C2_list[step+1:]
            # rebuild motion field
            U1 = pywt.waverec2(C1_list, self.wav, mode=self.wav_boundary_condition)
            U2 = pywt.waverec2(C2_list, self.wav, mode=self.wav_boundary_condition)
            ### evaluate DFD and gradient
            dfd, (grad1, grad2) = self.core.DFD_gradient(self.im0, self.im1, U1, U2)
            ### decompose gradient over wavelet basis, keep only up to current step
            C1_list = pywt.wavedec2(grad1, self.wav, level=self.levels_decomp,
                                    mode=self.wav_boundary_condition)[:step+1]
            C2_list = pywt.wavedec2(grad2, self.wav, level=self.levels_decomp,
                                    mode=self.wav_boundary_condition)[:step+1]
            # reshape as array
            C1_array, _ = pywt.coeffs_to_array(C1_list)
            C2_array, _ = pywt.coeffs_to_array(C2_list)
            # flatten and concatenate for l-bfgs
            C12_array = numpy.concatenate((C1_array.ravel(), C2_array.ravel()))
            # return DFD value and its gradient w.r.t. x
            return dfd, C12_array.astype(numpy.float64)
        return f_g

if __name__=="__main__":
    ###
    import matplotlib.pyplot as pyplot
    ###
    import demo.inr as inr
    ###

    def demo_particles():
        """Simple demo with the synthetic particle images.

        Written by P. DERIAN 2018-01-09.
        """
        im0 = ndimage.imread('demo/run010050000.tif', flatten=True).astype(float)/255.
        im1 = ndimage.imread('demo/run010050010.tif', flatten=True).astype(float)/255.
        V2, V1 = inr.readMotion('demo/UVtruth.inr')
        U1, U2 = Typhoon().solve(im0, im1, levels_decomp=3, levels_estim=None,
                                 U1_0=None, U2_0=None)
        fig, axes = pyplot.subplots(2,3)
        for ax, var, label in zip(axes.T.flat,
                                  [U1, V1, U2, V2],
                                  ['estim U1', 'true U1', 'estim U2', 'true U2']):
            ax.imshow(var, vmin=-3., vmax=3., interpolation='nearest', cmap='RdYlBu_r')
            ax.set_title(label)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        for ax, var, label in zip(axes[:,-1],
                                  [V1-U1, V2-U2],
                                  ['abs(error U1)', 'abs(error U2)']):
            ax.imshow(numpy.abs(var), vmin=0, vmax=0.3, interpolation='nearest')
            ax.set_title(label)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        pyplot.show()

    demo_particles()
