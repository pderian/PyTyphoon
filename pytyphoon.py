"""Python implementation of the Typhoon algorithm solving dense 2D optical flow problems.

Description: This module provides pure Python implementation of Typhoon.  At the moment,
    only the data term [1] is provided. The high-order regularizers [2] are not implemented
     in this module.

Disclaimer: The reference implementation used in [3], [4] is written in C++ and
    GPU-accelerated with CUDA. It is the property of Inria (FR) and the CSU Chico Research
    Foundation (Ca, USA). This Python implementation is *not* exactly the same as the
    reference for many reasons, and it is obviously much slower.

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
import scipy
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import matplotlib.pyplot as pyplot #TMP
###

class OpticalFlowCore:
    """Core functions for optical flow.

    Written by P. DERIAN 2018-01-09.
    Updated by P. DERIAN 2018-01-11: generic n-d version.
    """

    def __init__(self, shape, dtype=numpy.float32, interpolation_order=3):
        """Constructor.

        :param shape: the grid shape;
        :param dtype: numpy.float32 or numpy.float64.
        :param interpolation_order: spline interpolation order>0, faster when lower.

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2018-01-11: generic n-d version, changed boundary condition.
        """
        self.dtype = dtype
        ### grid coordinates
        self.shape = shape
        self.ndim = len(self.shape)
        # 1D
        self.x = tuple(numpy.arange(s, dtype=self.dtype) for s in self.shape)
        # N-D
        self.X = numpy.indices(self.shape, dtype=self.dtype)
        self.Xcoords = numpy.vstack((X.ravel() for X in self.X))
        ### Misc parameters
        self.sigma_blur = 0.5 #gaussian blur sigma before spatial gradient computation
        self.interpolation_order = interpolation_order #pixel interpolation order
        self.boundary_condition = 'mirror' #boundary condition
        # Note: setting the bc to 'constant', 'reflect' or 'wrap' seems to cause issues...?
        # while 'nearest', 'mirror' are OK.
        ### Buffer
        self.buffer = numpy.zeros(numpy.prod(self.shape), dtype=dtype)

    def DFD(self, im0, im1, U):
        """Compute the DFD.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param U: displacements (U1, U2, ...) along the first, second, ... axis;
        :return: the DFD.

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2018-01-11: generic n-d version.
        """
        # warp im1
        map_coords = self.Xcoords.copy()
        for n, Ui in enumerate(U):
            map_coords[n] += Ui.ravel()
        im1_warp = ndimage.interpolation.map_coordinates(im1, map_coords,
                                                         order=self.interpolation_order,
                                                         mode=self.boundary_condition)
        # difference
        return im1_warp.reshape(self.shape) - im0

    def DFD_gradient(self, im0, im1, U):
        """Compute the displaced frame difference (DFD) functional value and its gradients.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param U: displacements (U1, U2, ...) along the first, second, ... axis;
        :return: dfd, (grad1, grad2, ...)
            - DFD: the value of the functional;
            - (grad1, grad2, ...): the gradient of the DFD functional w.r.t. component U1, U2, ....

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2018-01-11: generic n-d version.
        """
        # warp im1->buffer
        map_coords = self.Xcoords.copy()
        for n, Ui in enumerate(U):
            map_coords[n] += Ui.ravel()
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
        grad = numpy.gradient(im1_warp)
        grad = (g*dfd for g in grad)
        # return
        return 0.5*numpy.sum(dfd**2), grad

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

    def solve(self, im0, im1, wav='haar', mode=None,
              levels_decomp=3, levels_estim=None,
              U0=None):
        """Solve the optical flow problem for given images and wavelet.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param wav: the name of the wavelet;
        :param mode: the signal extension mode ('zero', 'periodization'), see [a].
        :param levels_decomp: number of decomposition levels;
        :param levels_estim: number of estimation levels (<=levels_decomp);
        :param U0: optional first guess for U as (U1_0, U2_0, ...).
        :return: U=(U1, U2, ...) the estimated displacement along the first, second, ... axes.

        Notes:
            - without explicit regularization terms, it is necessary to set
              levels_estim<levels_decomp in order to close the estimation problem.
            - 'periodization' mode is much faster, but creates issues near edges.

        References:
            [a] https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2018-01-11: generic n-d version, checked levels.
        """
        ### Core
        # create a new core if the image shape is not compatible
        if (self.core is None) or (not numpy.testing.assert_equal(self.core.shape, im0.shape)):
            self.core = OpticalFlowCore(im0.shape)
        ### Wavelets
        self.wav = pywt.Wavelet(wav)
        if (mode is None) or (mode not in pywt.Modes.modes):
            mode = 'periodization'
            print('[!] set mode to "{}".'.format(mode))
        self.wav_boundary_condition = mode
        # check levels_decomp w.r.t. pywt
        levels_max = min([pywt.dwt_max_level(s, self.wav) for s in self.core.shape])
        if levels_decomp>levels_max:
            print('[!] too many decomposition levels ({}) requested for given wavelet/shape, set to {}.'.format(
                  levels_decomp, levels_max))
            levels_decomp = levels_max
        # check levels_estim w.r.t levels_decomp, if not None
        if (levels_estim is not None) and (levels_estim>levels_decomp):
            print('[!] too many estimation levels ({}) requested for decomposition, set to {}.'.format(
                  levels_estim, levels_decomp))
            levels_estim = levels_decomp
        # set the final levels values
        self.levels_decomp = levels_decomp
        self.levels_estim = levels_estim if (levels_estim is not None) else self.levels_decomp-1
        ### Images
        # make sure the images have size compatible with decomposition levels, padding
        # if necessary
        self.im0 = im0.astype(self.core.dtype)
        self.im1 = im1.astype(self.core.dtype)
        ### Motion fields
        # initialize with given fields, if any, otherwise with zeros.
        if U0 is None:
            U0 = [None,]*self.core.ndim
        U = (Ui.astype(self.core.dtype) if (Ui is not None)
             else numpy.zeros(self.core.shape, dtype=self.core.dtype) for Ui in U0)
        # the corresponding wavelet coefficients
        self.C_list = [pywt.wavedecn(Ui, self.wav, level=self.levels_decomp,
                                     mode=self.wav_boundary_condition) for Ui in U]
        # which we reshape as arrays to get the slices for future manipulations.
        self.slices = tuple(pywt.coeffs_to_array(Ci)[1] for Ci in self.C_list)
        ### Solve
        print('Decomposition over {} scales of details, {} estimated'.format(
            self.levels_decomp, self.levels_estim))
        # for each level
        for level in range(self.levels_estim+1):
            print('details ({})'.format(level) if level else 'approx. (0)')
            # the initial condition, as array (Note: flattened by l-bfgs)
            C_array = (pywt.coeffs_to_array(Ci[:level+1])[0] for Ci in self.C_list)
            C_array = numpy.vstack((Ci[numpy.newaxis,...] for Ci in C_array))
            # so that C_array[i] contains all coefficients of component i.
            # we remember the shape for future manipulations.
            C_shape = C_array.shape
            # create the cost function for this step
            f_g = self.create_cost_function(level, C_shape)
            # minimize
            C_array, min_value, optim_info = optimize.fmin_l_bfgs_b(f_g,
                                                                    C_array.astype(numpy.float64),
                                                                    factr=1000.,
                                                                    iprint=0)
            print('\tl-bfgs completed with status {warnflag} - {nit} iterations, {funcalls} calls'.format(**optim_info))
            print('\tcurrent functional value: {:.2f}'.format(min_value))
            # store result in main coefficients
            C_array = C_array.reshape(C_shape)
            C_list = (pywt.array_to_coeffs(Ci, self.slices[i][:level+1], output_format='wavedecn')
                      for i, Ci in enumerate(C_array))
            for n, Ci in enumerate(C_list):
                for l in range(level+1):
                    self.C_list[n][l] = Ci[l]
        ### Rebuild field and return
        U = tuple(pywt.waverecn(Ci, self.wav, mode=self.wav_boundary_condition) for Ci in self.C_list)
        return U

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
            Updated by P. DERIAN 2018-01-11: generic n-d version.
            """
            ### rebuild motion field
            # reshape 1d vector to 3d array
            x = x.reshape(shape).astype(self.core.dtype)
            # extract coefficients, complement with finer scales and reshape for pywt
            # Note: ideally we would not complement, as finer scales are zeros. This is made
            #   necessary by pywt.
            C_list = (pywt.array_to_coeffs(xi, self.slices[i][:step+1],
                                           output_format='wavedecn')
                      + self.C_list[i][step+1:] for i, xi in enumerate(x))
            # rebuild motion field
            U = (pywt.waverecn(Ci, self.wav, mode=self.wav_boundary_condition) for Ci in C_list)
            ### evaluate DFD and gradient
            func_value, G = self.core.DFD_gradient(self.im0, self.im1, U)
            # decompose gradient over wavelet basis, keep coefficients only up to current step
            G_list = (pywt.wavedecn(Gi, self.wav, level=self.levels_decomp,
                                    mode=self.wav_boundary_condition)[:step+1]
                      for Gi in G)
            # reshape as array, flatten and concatenate for l-bfgs
            G_array = numpy.hstack(
                (pywt.coeffs_to_array(Gi)[0].ravel() for Gi in G_list))
            ### evaluate regularizer and its gradient
            # [TODO]
            ### return DFD (+ regul) value, and its gradient w.r.t. x
            return func_value, G_array.astype(numpy.float64)
        return f_g

### Helpers ###

def RMSE(uvA, uvB):
    """Root Mean Squared Error (RMSE).

    :param uvA: (uA, vA)
    :param uvB: (uB, vB)
    :return: the RMSE.

    Written by P. DERIAN 2018-01-09.
    """
    return numpy.sqrt(numpy.mean(numpy.sum((
        numpy.asarray(uvA) - numpy.asarray(uvB))**2, axis=0)))

### Demonstrations ###

if __name__=="__main__":
    import sys
    import time
    ###
    import matplotlib
    import matplotlib.pyplot as pyplot
    ###
    import demo.inr as inr
    ###

    def print_versions():
        print("\n* Module versions:")
        print('Python:', sys.version)
        print('Numpy:', numpy.__version__)
        print('Scipy:', scipy.__version__)
        print('PyWavelet:', pywt.__version__)
        print('Matplotlib:', matplotlib.__version__)
        print('PyTyphoon (this):', __version__)

    def demo_particles():
        """Simple demo with the synthetic particle images.

        Written by P. DERIAN 2018-01-09.
        """
        print("\n* PyTyphoon {} ({}) – demo".format(__version__, __file__))
        ### load data
        im0 = ndimage.imread('demo/run010050000.tif', flatten=True).astype(float)/255.
        im1 = ndimage.imread('demo/run010050010.tif', flatten=True).astype(float)/255.
        # Note:
        #   - U1, V1 are vertical (1st axis) components;
        #   - U2, V2 are horizontal (2nd axis) components.
        #   - inr.ReadMotion() returns (horizontal, vertical).
        V2, V1 = inr.readMotion('demo/UVtruth.inr')
        ### solve OF
        typhoon = Typhoon()
        tstart = time.clock()
        U1, U2 = typhoon.solve(im0, im1, wav='db2',
                               levels_decomp=3, levels_estim=1,
                               U0=None)
        tend = time.clock()
        print('Estimation completed in {:.2f} s.'.format(tend-tstart))
        ### post-process & display
        rmse = RMSE((U1, U2), (V1, V2))
        dpi = 72.
        fig, axes = pyplot.subplots(2,4, figsize=(800./dpi, 450./dpi))
        # images
        for ax, var, label in zip(axes[:,0], [im0, im1], ['image #0', 'image #1']):
            pi = ax.imshow(var, vmin=0., vmax=1., interpolation='nearest', cmap='gray')
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
        # velocity fields
        for ax, var, label in zip(axes[:,1:-1].flat,
                                  [U1, V1, U2, V2],
                                  ['estim U1', 'true U1', 'estim U2', 'true U2']):
            pf = ax.imshow(var, vmin=-3., vmax=3., interpolation='nearest', cmap='RdYlBu_r')
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
        # error maps
        for ax, var, label in zip(axes[:,-1],
                                  [V1-U1, V2-U2],
                                  ['abs(error U1)', 'abs(error U2)']):
            pe = ax.imshow(numpy.abs(var), vmin=0, vmax=0.3, interpolation='nearest')
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
        # colormaps
        pyplot.subplots_adjust(bottom=0.25, top=0.94, left=.05, right=.95)
        axf = fig.add_axes([1.25/8., 0.16, 2.5/8., 0.03])
        pyplot.colorbar(pf, cax=axf, orientation='horizontal')
        axf.set_title('displacement (pixel)', fontdict={'fontsize':'medium'})
        axe = fig.add_axes([4.5/8., 0.16, 2.5/8., 0.03])
        pyplot.colorbar(pe, cax=axe, orientation='horizontal')
        axe.set_title('error (pixel)', fontdict={'fontsize':'medium'})
        # labels
        pyplot.figtext(
            0.05, 0.015, 'PyTyphoon {} demo\n"{}" wavelet, {} scales decomp., {} scales estim., no regularizer.'.format(
                __version__, typhoon.wav.name, typhoon.levels_decomp, typhoon.levels_estim),
            size='medium', ha='left', va='bottom')
        pyplot.figtext(
            0.95, 0.015, 'RMSE={:.2f}'.format(rmse),
            size='medium', ha='right', va='bottom')
        pyplot.show()

    print_versions()
    demo_particles()
