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
    - Pillow;
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
###

class OpticalFlowCore:
    """Core functions for optical flow.

    Written by P. DERIAN 2018-01-09.
    Updated by P. DERIAN 2018-01-11: generic n-d version, added __str__().
    """

    def __init__(self, shape, dtype=numpy.float32, interpolation_order=3, sigma_blur=0.5):
        """Constructor.

        :param shape: the grid shape;
        :param dtype: numpy.float32 or numpy.float64.
        :param interpolation_order: spline interpolation order>0, faster when lower.
        :param sigma_blur: sigma of gaussian blur applied before spatial gradient computation.

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
        self.sigma_blur = sigma_blur #gaussian blur sigma before spatial gradient computation
        self.interpolation_order = interpolation_order #pixel interpolation order
        self.boundary_condition = 'mirror' #boundary condition
        # Note: setting the bc to 'constant', 'reflect' or 'wrap' seems to cause issues...?
        # while 'nearest', 'mirror' are OK.
        ### Buffer
        self.buffer = numpy.zeros(numpy.prod(self.shape), dtype=dtype)

    def __str__(self):
        """
        Written by P. DERIAN 2018-10-11.
        """
        param_str = '\n\tshape={}'.format(self.shape)
        param_str += '\n\tdtype={}'.format(self.dtype.__name__)
        param_str += '\n\tinterpolation order={}'.format(self.interpolation_order)
        param_str += '\n\tsigma blur={}'.format(self.sigma_blur)
        return self.__class__.__name__ + param_str

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
    Updated by P. DERIAN 2018-01-10: added default values and solve_fast().
    """

    DEFAULT_WAV = 'haar' #default wavelet name
    DEFAULT_MODE = 'zero' #default wavelet signal extension mode

    def __init__(self, shape=None):
        """Instance constructor with optional shape.

        :param shape: optional shape of the problem.

        Written by P. DERIAN 2018-01-09.
        """
        self.core = OpticalFlowCore(shape) if (shape is not None) else None

    def solve(self, im0, im1, wav=None, mode=None,
              levels_decomp=3, levels_estim=None, U0=None,
              interpolation_order=3, sigma_blur=0.5,
              ):
        """Solve the optical flow problem for given images and wavelet.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param wav: the name of the wavelet;
        :param mode: the signal extension mode ('zero', 'periodization'), see [a].
        :param levels_decomp: number of decomposition levels;
        :param levels_estim: number of estimation levels (<=levels_decomp);
        :param U0: optional first guess for U as (U1_0, U2_0, ...).
        :param interpolation_order: spline interpolation order>0, faster when lower.
        :param sigma_blur: sigma of gaussian blur applied before spatial gradient computation.
        :return: U=(U1, U2, ...) the estimated displacement along the first, second, ... axes.

        Notes:
            - without explicit regularization terms, it is necessary to set
              levels_estim<levels_decomp in order to close the estimation problem.
            - 'periodization' mode is much faster, but creates larger errors near edges.

        References:
            [a] https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html

        Written by P. DERIAN 2018-01-09.
        Updated by P. DERIAN 2018-01-11: generic n-d version, checked levels, checked shapes.
        Updated by P. DERIAN 2018-06-11: fixed first-guess motion (U0) which was not paded when needed.
        """
        ### Wavelets
        # check arguments
        if (wav is None) or (wav not in pywt.wavelist(kind='discrete')):
            wav = self.DEFAULT_WAV
            print('[!] set wavelet to default "{}".'.format(wav))
        if (mode is None) or (mode not in pywt.Modes.modes):
            mode = self.DEFAULT_MODE
            print('[!] set mode to default "{}".'.format(mode))
        # set final values
        self.wav = pywt.Wavelet(wav)
        self.wav_boundary_condition = mode
        # check levels_decomp argument w.r.t. pywt
        levels_max = min([pywt.dwt_max_level(s, self.wav) for s in im0.shape])
        if levels_decomp>levels_max:
            print('[!] too many decomposition levels ({}) requested for given wavelet/shape, set to {}.'.format(
                  levels_decomp, levels_max))
            levels_decomp = levels_max
        # check levels_estim argument w.r.t levels_decomp, if not None
        if (levels_estim is not None) and (levels_estim>levels_decomp):
            print('[!] too many estimation levels ({}) requested for decomposition, set to {}.'.format(
                  levels_estim, levels_decomp))
            levels_estim = levels_decomp
        # set the final levels values
        self.levels_decomp = levels_decomp
        self.levels_estim = levels_estim if (levels_estim is not None) else self.levels_decomp-1

        ### Images and core
        # make sure the images have size compatible with decomposition levels, padding
        # if necessary with zeros
        block_size = 2**self.levels_decomp
        # how much is missing in each axis
        self.pad_size = tuple((block_size - (s%block_size))%block_size for s in im0.shape)
        # the slice to crop back the original area
        self.crop_slice = tuple(slice(None, -p if p else None, None) for p in self.pad_size)
        # padd if necessary
        if any(self.pad_size):
            padding = [(0, p) for p in self.pad_size]
            im0 = numpy.pad(im0, padding, mode='constant')
            im1 = numpy.pad(im1, padding, mode='constant')
        # create a new core if the image shape is not compatible
        if (self.core is None) or (not numpy.testing.assert_equal(self.core.shape, im0.shape)):
            self.core = OpticalFlowCore(im0.shape, interpolation_order=interpolation_order,
                                        sigma_blur=sigma_blur)
        print(self.core)
        # and the images
        self.im0 = im0.astype(self.core.dtype)
        self.im1 = im1.astype(self.core.dtype)

        ### Motion fields
        # get a sequence of the proper length
        if U0 is None:
            U0 = [None,]*self.core.ndim
        # for each component
        U = []
        for Ui in U0:
            # if None, initialize with zeros
            if Ui is None:
                Ui = numpy.zeros(self.core.shape, dtype=self.core.dtype)
            # else pad the given field if necessary
            elif any(self.pad_size):
                padding = [(0, p) for p in self.pad_size]
                Ui = numpy.pad(Ui, padding, mode='constant')
            U.append(Ui)
        # as a tuple
        U = tuple(U)
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
        # cropping the relevant area
        U = tuple(pywt.waverecn(Ci, self.wav, mode=self.wav_boundary_condition)[self.crop_slice]
                  for Ci in self.C_list)
        return U

    def solve_fast(self, im0, im1, wav='haar',
                   levels_decomp=3, levels_estim=None, U0=None):
        """Fastest estimation (but lower accuracy).

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param wav: the name of the wavelet;
        :param levels_decomp: number of decomposition levels;
        :param levels_estim: number of estimation levels (<=levels_decomp);
        :param U0: optional first guess for U as (U1_0, U2_0, ...).
        :return: U=(U1, U2, ...) the estimated displacement along the first, second, ... axes.

        Note: uses linear (order 1) interpolation, no blurring and 'periodization' mode.

        Written by P. DERIAN 2018-01-11.
        """
        return self.solve(im0, im1, wav=wav, mode='periodization',
                          levels_decomp=levels_decomp, levels_estim=levels_estim,
                          interpolation_order=1, sigma_blur=0.)

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

    @staticmethod
    def solve_pyramid(im0, im1, levels_pyr=1, solve_fast=False,
                      wav='haar', levels_decomp=3, levels_estim=None,
                      **kwargs):
        """Wrapper for pyramidal estimation.

        When very large displacements are involved, the usual pyramidal approach
        can be employed to help achieving a correct estimation.

        :param im0: the first (grayscale) image;
        :param im1: the second (grayscale) image;
        :param levels_pyr: number of pyramid levels;
        :param fast: if True, use faster (but less accurate) estimation;
        :param wav: the name of the wavelet;
        :param levels_decomp: number of decomposition levels;
        :param levels_estim: number of estimation levels (<=levels_decomp);
        :param **kwargs: remaining arguments passed to Typhoon.solve().
        :return: U, typhoon
            U=(U1, U2, ...) the estimated displacement along the first, second, ... axes.
            typhoon the instance of Typhoon used for the last (finest) pyramid level.

        Written by P. DERIAN 2018-01-11.
        """
        downscale = 0.5
        upscale = 1./downscale
        ### create the pyramid of images
        pyr_im0 = [im0,]
        pyr_im1 = [im1,]
        for l in range(levels_pyr):
            # filter image and interpolate
            tmp_im = ndimage.gaussian_filter(pyr_im0[0], 2./3.)
            pyr_im0.insert(0, ndimage.interpolation.zoom(tmp_im, downscale))
            tmp_im = ndimage.gaussian_filter(pyr_im1[0], 2./3.)
            pyr_im1.insert(0, ndimage.interpolation.zoom(tmp_im, downscale))
        ### for each level:
        for l, (im0, im1) in enumerate(zip(pyr_im0, pyr_im1)):
            # if not the first, use previous motion as first guess
            if l:
                U0 = [ndimage.interpolation.zoom(Ui, upscale) for Ui in U]
            else:
                U0 = None
            typhoon = Typhoon()
            if solve_fast:
                U = typhoon.solve_fast(im0, im1, wav=wav,
                                       levels_decomp=levels_decomp,
                                       levels_estim=levels_estim)
            else:
                U = typhoon.solve(im0, im1, U0=U0, wav=wav,
                                  levels_decomp=levels_decomp,
                                  levels_estim=levels_estim,
                                  **kwargs)
        ### return the last estimates
        return U, typhoon

### Helpers ###

def RMSE(Ua, Ub):
    """Root Mean Squared Error (RMSE).

    :param Ua: (Ua1, Ua2, ...) first vector field;
    :param Ub: (Ub1, Ub2, ...) second vector field.
    :return: the RMSE.

    Written by P. DERIAN 2018-01-09.
    """
    return numpy.sqrt(numpy.mean(numpy.sum((
        numpy.asarray(Ua) - numpy.asarray(Ub))**2, axis=0)))

### Demonstrations ###

if __name__=="__main__":
    ###
    import argparse
    import os.path
    import sys
    import time
    ###
    import matplotlib
    import matplotlib.pyplot as pyplot
    from PIL import Image
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
        """Demo with the synthetic particle images.

        Written by P. DERIAN 2018-01-09.
        """
        print('\n* PyTyphoon {} ({}) – "particles" demo'.format(__version__, __file__))

        ### load data
        im0 = numpy.array(Image.open('demo/run010050000.tif').convert(mode='L')).astype(float)/255.
        im1 = numpy.array(Image.open('demo/run010050010.tif').convert(mode='L')).astype(float)/255.

        # Note:
        #   - U1, V1 are vertical (1st axis) components;
        #   - U2, V2 are horizontal (2nd axis) components.
        #   - inr.ReadMotion() returns (horizontal, vertical).
        V2, V1 = inr.readMotion('demo/UVtruth.inr')

        ### solve OF
        typhoon = Typhoon()
        tstart = time.clock()
        U1, U2 = typhoon.solve(im0, im1, wav='db2', mode='periodization',
                               levels_decomp=3, levels_estim=1)
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
            0.05, 0.015, 'PyTyphoon {} "particles" demo\n"{}" wavelet, {} scales decomp., {} scales estim., no regularizer.'.format(
                __version__, typhoon.wav.name, typhoon.levels_decomp, typhoon.levels_estim),
            size='medium', ha='left', va='bottom')
        pyplot.figtext(
            0.95, 0.015, 'RMSE={:.2f} pixel'.format(rmse),
            size='medium', ha='right', va='bottom')
        pyplot.show()

    def demo_3dshift(seed=123456789):
        """Demo of 3D estimation with simple shift.

        :param seed: seed for images generation. Pass None for random images.

        Synthetic images are generated by filtering gaussian noise.
        The true motion is a simple integer shift in all directions.
        Estimation is performed with the basic "fast" solver.

        Written by P. DERIAN 2018-10-12.
        """
        print('\n* PyTyphoon {} ({}) – "3dshift" demo'.format(__version__, __file__))

        ### create synthetic images
        print('Generating images...')
        # sum of random normal noise filtered by different gaussian kernels.
        size = 64
        sigma_list = [1., 3., 9.]
        im0 = numpy.zeros((size, size, size))
        numpy.random.seed(seed) #seed the generator for repeatability
        for sigma in sigma_list:
            tmp = numpy.random.randn(size, size, size)
            im0 += ndimage.gaussian_filter(tmp, sigma, mode='wrap')
        # rescale to [0, 1]
        min = im0.min()
        max = im0.max()
        im0 = (im0-min)/(max-min)
        # second image: simply shift
        shift = (-1, 2, 3)
        im1 = numpy.roll(im0, shift=shift, axis=(0,1,2))

        ### perform estimation
        typhoon = Typhoon()
        tstart = time.clock()
        U = typhoon.solve_fast(im0, im1, wav='db2', levels_decomp=3, levels_estim=1)
        tend = time.clock()
        print('Estimation completed in {:.2f} s.'.format(tend-tstart))

        ### post-process and display
        truth_label = 'true shift: ({}) pixel'.format(', '.join('{:.2f}'.format(s) for s in shift))
        estim_label = 'estimated mean displacement: ({}) pixel'.format(', '.join('{:.2f}'.format(Ui.mean()) for Ui in U))
        dpi = 72.
        fig, axes = pyplot.subplots(3, 5, figsize=(800./dpi, 450./dpi))
        # show the 3 component in the middle plane along each axis
        slice_all = slice(None, None, None)
        for i in range(3):
            # slice of the mid-plane along this axis
            slice_mid = [slice_all, slice_all, slice_all]
            slice_mid[i] = size//2
            # for each image
            for j, (ax, imj) in enumerate(zip(axes[i,:2], [im0, im1])):
                ax.imshow(imj[slice_mid], interpolation='nearest', cmap='gray',
                          vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('im{} @ mid-ax{}'.format(j, i+1))
            # for each component
            for j, (ax, Uj) in enumerate(zip(axes[i,2:], U), 1):
                pc = ax.imshow(Uj[slice_mid], interpolation='nearest', cmap='RdYlBu_r',
                               vmin=-3, vmax=3)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('U{} @ mid-ax{}'.format(j, i+1))
        pyplot.subplots_adjust(left=.1, right=.9, bottom=.14, top=.9, hspace=.3)
        # colorbar
        cax = fig.add_axes([.75, 0.05, .2, 0.03])
        cb = pyplot.colorbar(pc, cax=cax, orientation='horizontal')
        cb.ax.set_title('displacement (pixel)', fontdict={'fontsize':'medium'})
        # labels
        pyplot.figtext(0.5, 0.98, truth_label+' - '+estim_label,
                       ha='center', va='top', size='medium')
        pyplot.figtext(
            0.05, 0.015, 'PyTyphoon {} "3dshift" demo\n"{}"wavelet, {} scales decomp., {} scales estim., no regularizer, "fast" solver.'.format(
                __version__, typhoon.wav.name, typhoon.levels_decomp, typhoon.levels_estim),
            size='medium', ha='left', va='bottom')
        pyplot.show()

    def demo_3dvortex(seed=123456789):
        """Demo of 3D estimation with vortex motion.

        :param seed: seed for images generation. Pass None for random images.

        Synthetic images are generated by filtering gaussian noise.
        Estimation is performed with the basic "fast" solver.

        Written by P. DERIAN 2018-10-12.
        """
        print('\n* PyTyphoon {} ({}) – "3dvortex" demo'.format(__version__, __file__))

        ### create synthetic images
        print('Generating images...')
        # sum of random normal noise filtered by different gaussian kernels.
        size = 96
        margin = 4 #add a margin on every dimension
        sigma_list = [1., 3., 9.] #a list of gaussian blur sigma
        size_tmp = size + 2*margin
        im0 = numpy.zeros((size_tmp, size_tmp, size_tmp))
        numpy.random.seed(seed) #seed the generator for repeatability
        for sigma in sigma_list:
            tmp = numpy.random.randn(size_tmp, size_tmp, size_tmp)
            im0 += ndimage.gaussian_filter(tmp, sigma, mode='wrap')
        # rescale to [0, 1]
        min = im0.min()
        max = im0.max()
        im0 = (im0-min)/(max-min)
        ### create synthetic motion
        # the coordinates
        X1, X2, X3 = numpy.indices(im0.shape, dtype=float)
        # tubular vortex: from streamfunction
        sigma = float(size)/3. #make sure it decays enough near edges
        alpha = float(size) #amplitude factor
        beta = 3. #amplitude max of 3rd component
        x0 = [size_tmp//2 for _ in range(3)] #coordinates of center of volume
        SF = alpha*numpy.exp((-1./sigma**2)*( (X1-x0[0])**2 + (X2-x0[1])**2))
        G1, G2, _ = numpy.gradient(SF)
        # first 2 components are vortex, third increase linearly from zero to beta
        U_truth = (G2,
                   -G1,
                   (X3 - margin)*(beta/float(size))) #divide by n_adv due to multiple steps
        # displacement coordinates
        n_adv = 30 #number of advection steps
        map_coords = numpy.vstack((X.ravel() for X in (X1, X2, X3))) #for interpolation
        for n, Ui in enumerate(U_truth):
            map_coords[n] -= (1./float(n_adv))*Ui.ravel()
        # very crude advection
        im1 = im0
        for _ in range(n_adv):
            im1 = ndimage.interpolation.map_coordinates(im1, map_coords, order=3,
                                                        mode='constant').reshape(im0.shape)
        # finally retain only the working area
        if margin>0:
            im0 = im0[margin:-margin, margin:-margin, margin:-margin]
            im1 = im1[margin:-margin, margin:-margin, margin:-margin]
            U_truth = tuple(Ui[margin:-margin, margin:-margin, margin:-margin] for Ui in U_truth)

        ### perform estimation
        # instance typhoon so we can use its internal coordinates for simplicity.
        typhoon = Typhoon()
        tstart = time.clock()
        U = typhoon.solve_fast(im0, im1, wav='db3', levels_decomp=3, levels_estim=0)
        tend = time.clock()
        print('Estimation completed in {:.2f} s.'.format(tend-tstart))

        ### post-process and display
        # show the 3 component in the middle plane along each axis
        rmse = RMSE(U_truth, U)
        dpi = 72.
        fig, axes = pyplot.subplots(3, 5, figsize=(800./dpi, 450./dpi))
        # for each axis
        slice_all = slice(None, None, None)
        for i in range(3):
            # slice of the mid-plane along this axis
            slice_mid = [slice_all, slice_all, slice_all]
            slice_mid[i] = size//2
            # for each image
            for j, (ax, imj) in enumerate(zip(axes[i,:2], [im0, im1])):
                ax.imshow(imj[slice_mid], interpolation='nearest', cmap='gray',
                          vmin=0, vmax=1)
                ax.autoscale(tight=True)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('im{} @ mid-ax{}'.format(j, i+1))
            # for each component
            for j, (ax, Uj) in enumerate(zip(axes[i,2:], U), 1):
                pc = ax.imshow(Uj[slice_mid], interpolation='nearest', cmap='RdYlBu_r',
                               vmin=-3, vmax=3)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('U{} @ mid-ax{}'.format(j, i+1))
        pyplot.subplots_adjust(left=.1, right=.9, bottom=.14, top=.95, hspace=.3)
        # colorbar
        cax = fig.add_axes([.565, 0.05, .2, 0.03])
        cb = pyplot.colorbar(pc, cax=cax, orientation='horizontal')
        cb.ax.set_title('displacement (pixel)', fontdict={'fontsize':'medium'})
        # labels
        pyplot.figtext(
            0.05, 0.015, 'PyTyphoon {} "3dvortex" demo\n"{}"wavelet, {} scales decomp., {} scales estim., no regularizer, "fast" solver.'.format(
                __version__, typhoon.wav.name, typhoon.levels_decomp, typhoon.levels_estim),
            size='medium', ha='left', va='bottom')
        pyplot.figtext(
            0.95, 0.015, 'RMSE={:.2f} pixel'.format(rmse),
            size='medium', ha='right', va='bottom')
        pyplot.show()

    def main(argv):
        """Entry point.

        Written by P. DERIAN 2018-01-11.
        """
        ### Arguments
        # create parser
        parser = argparse.ArgumentParser(
            description="Python implementation of Typhoon motion estimator.",
            )
        # estimation parameters
        parser.add_argument('-i0', dest="im0", type=str, default='',
                            help="first image")
        parser.add_argument('-i1', dest="im1", type=str, default='',
                            help="second image")
        parser.add_argument('-w', '--wav', dest="wav", type=str, default=None,
                            help="wavelet name")
        parser.add_argument('-d', '--decomp', dest="levels_decomp", type=int, default=3,
                            help="number of decomposition levels")
        parser.add_argument('-e', '--estim', dest="levels_estim", type=int, default=None,
                            help="number of estimation levels")
        parser.add_argument('-p', '--pyr', dest="levels_pyr", type=int, default=0,
                            help="pyramidal estimation levels")
        parser.add_argument('--order', dest="interpolation_order", type=int, default=3,
                            help="image interpolation order")
        parser.add_argument('--blur', dest="sigma_blur", type=float, default=0.5,
                            help="gaussian blur sigma")
        parser.add_argument('--mode', dest="mode", type=str, default=None,
                            help="signal extension mode")
        parser.add_argument('--fast', dest="solve_fast", action='store_true',
                            help="faster estimation (less accurate)")
        parser.add_argument('--display', dest="display_result", action='store_true',
                            help="display estimation results")
        # misc arguments
        parser.add_argument('--demo', dest="demo_name", type=str, default='',
                            help="name of the demo")
        parser.add_argument('--version', dest="print_version", action='store_true',
                            help="print module versions")
        #parser.add_argument('-q', "--quiet", dest="verbose", action='store_false',
        #                    help="set quiet mode")
        # set defaults and parse
        parser.set_defaults(verbose=True, print_versions=False, display_result=False,
                            solve_fast=False)
        args = parser.parse_args(argv)

        ### Versions
        if args.print_version:
            print_versions()
            return

        ### Demos
        # list of available demos
        demos = {'particles': demo_particles,
                 '3dshift': demo_3dshift,
                 '3dvortex': demo_3dvortex,
                 }
        default_demo = lambda : print("[!] Unkown demo '{}', valid demos: {}".format(
            args.demo_name, list(demos.keys())))
        # start a demo
        if args.demo_name:
            demos.get(args.demo_name, default_demo)()
            return

        ### Single estimation
        if args.im0 and args.im1:
            print('\nEstimation:\n\t{}\n\t{}'.format(args.im0, args.im1))
            ### load images
            im0 = numpy.array(Image.open(args.im0).convert(mode='L')).astype(numpy.float32)/255.
            im1 = numpy.array(Image.open(args.im1).convert(mode='L')).astype(numpy.float32)/255.
            ### Solve problem
            tstart = time.clock()
            (U1, U2), typhoon = Typhoon.solve_pyramid(im0, im1, levels_pyr=args.levels_pyr,
                                                      solve_fast=args.solve_fast,
                                                      wav=args.wav, mode=args.mode,
                                                      levels_decomp=args.levels_decomp,
                                                      levels_estim=args.levels_estim,
                                                      interpolation_order=args.interpolation_order,
                                                      sigma_blur=args.sigma_blur)
            tend = time.clock()
            print('Estimation completed in {:.2f} s.'.format(tend-tstart))
            # export
            # [TODO] retrieve estim parameters and save as well.
            # display
            # [TODO] move to function?
            if args.display_result:
                dpi = 72.
                fig, axes = pyplot.subplots(2,3, figsize=(800./dpi, 600./dpi))
                # images
                for ax, var, label in zip(axes[0,1:], [im0, im1], ['image #0', 'image #1']):
                    pi = ax.imshow(var, vmin=0., vmax=1., interpolation='nearest', cmap='gray')
                    ax.set_title(label)
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[0,0].remove()
                # velocity fields
                pf = []
                for ax, var, label in zip(axes[1,1:], [U1, U2], ['estim U1', 'estim U2']):
                    pf.append(ax.imshow(var, interpolation='nearest', cmap='RdYlBu_r'))
                    ax.set_title(label)
                    ax.set_xticks([])
                    ax.set_yticks([])
                # colorbars?
                # [TODO]
                # quiver - note that array axes order is reversed
                ax = axes[1,0]
                ax.set_aspect('equal')
                qstep = 8
                ax.quiver(typhoon.core.x[1][::qstep], typhoon.core.x[0][::qstep],
                          U2[::qstep, ::qstep], U1[::qstep, ::qstep],
                          pivot='middle', color='b', units='xy', angles='xy')
                ax.set_xlim(typhoon.core.x[1][0], typhoon.core.x[1][-1])
                ax.set_ylim(typhoon.core.x[0][0], typhoon.core.x[0][-1])
                ax.invert_yaxis()
                ax.set_title('motion field')
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.subplots_adjust(left=.05, right=.95, top=.95, bottom=.1)
                # labels
                pyplot.figtext(0.05, 0.015, 'PyTyphoon {}'.format(__version__),
                               size='medium', ha='left', va='bottom')
                pyplot.figtext(0.05, 0.95,
                               'im0: {}\nim1: {}\nlevels pyramid: {}\nwavelet: {}\nwav. decomp. lvls: {}\nwav. estim. lvls: {}'.format(
                                    os.path.basename(args.im0),
                                    os.path.basename(args.im1),
                                    args.levels_pyr,
                                    typhoon.wav.name,
                                    typhoon.levels_decomp,
                                    typhoon.levels_estim),
                               size='medium', ha='left', va='top')
                pyplot.show()

    main(sys.argv[1:])
