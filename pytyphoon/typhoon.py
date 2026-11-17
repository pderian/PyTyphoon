"""Main Typhoon algorithm implementation for optical flow estimation.

This module contains the Typhoon class which implements the wavelet-based
optical flow estimation algorithm.
"""

import logging
from typing import Tuple, Optional, Union, List, Dict, Any
import numpy as np
import pywt
import scipy.ndimage
import scipy.optimize

from .core import OpticalFlowCore
from .utils import RMSE

logger = logging.getLogger(__name__)

class Typhoon:
    """Implements the Typhoon algorithm: dense optical flow estimation on wavelet bases.

    This class provides the main interface for estimating optical flow using
    wavelet-based multiscale decomposition.

    Attributes:
        DEFAULT_WAV: Default wavelet name
        DEFAULT_MODE: Default wavelet signal extension mode
        core: OpticalFlowCore instance
        wav: Wavelet object
        wav_boundary_condition: Wavelet boundary condition mode
        levels_decomp: Number of decomposition levels
        levels_estim: Number of estimation levels
        pad_size: Padding size for each dimension
        crop_slice: Slice object to crop back to original size
        im0: First image (normalized)
        im1: Second image (normalized)
        C_list: List of wavelet coefficients for motion field components
        slices: Tuple of slices for coefficient manipulation
    """

    DEFAULT_WAV = 'haar'  # Default wavelet name
    DEFAULT_MODE = 'zero'  # Default wavelet signal extension mode

    def __init__(self, shape: Optional[Tuple[int, ...]] = None):
        """Initialize Typhoon instance with optional shape.

        Args:
            shape: Optional shape of the problem. If provided, creates a core instance.
        """
        self.core: Optional[OpticalFlowCore] = OpticalFlowCore(shape) if shape is not None else None
        self.wav: Optional[pywt.Wavelet] = None
        self.wav_boundary_condition: str = self.DEFAULT_MODE
        self.levels_decomp: int = 3
        self.levels_estim: int = 2
        self.pad_size: Tuple[int, ...] = ()
        self.crop_slice: Tuple[slice, ...] = ()
        self.im0: Optional[np.ndarray] = None
        self.im1: Optional[np.ndarray] = None
        self.C_list: List = []
        self.slices: Tuple = ()

    def solve(self, im0: np.ndarray, im1: np.ndarray,
              wav: Optional[str] = None, mode: Optional[str] = None,
              levels_decomp: int = 3, levels_estim: Optional[int] = None,
              U0: Optional[Tuple[np.ndarray, ...]] = None,
              interpolation_order: int = 3, sigma_blur: float = 0.5) -> Tuple[np.ndarray, ...]:
        """Solve the optical flow problem for given images and wavelet.

        Args:
            im0: The first (grayscale) image
            im1: The second (grayscale) image
            wav: The name of the wavelet (default: DEFAULT_WAV)
            mode: The signal extension mode ('zero', 'periodization', etc.)
            levels_decomp: Number of decomposition levels
            levels_estim: Number of estimation levels (<= levels_decomp)
            U0: Optional first guess for U as (U1_0, U2_0, ...)
            interpolation_order: Spline interpolation order > 0, faster when lower
            sigma_blur: Sigma of Gaussian blur applied before spatial gradient computation

        Returns:
            Tuple of displacement fields (U1, U2, ...) along each axis

        Notes:
            - Without explicit regularization terms, it is necessary to set
              levels_estim < levels_decomp in order to close the estimation problem.
            - 'periodization' mode is much faster, but creates larger errors near edges.

        References:
            [a] https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
        """
        ### Wavelets
        # Check arguments
        if wav is None or wav not in pywt.wavelist(kind='discrete'):
            wav = self.DEFAULT_WAV
            logger.info(f'Set wavelet to default "{wav}".')

        if mode is None or mode not in pywt.Modes.modes:
            mode = self.DEFAULT_MODE
            logger.info(f'Set mode to default "{mode}".')

        # Set final values
        self.wav = pywt.Wavelet(wav)
        self.wav_boundary_condition = mode

        # Check levels_decomp argument w.r.t. pywt
        levels_max = min([pywt.dwt_max_level(s, self.wav) for s in im0.shape])
        if levels_decomp > levels_max:
            logger.warning(
                f'Too many decomposition levels ({levels_decomp}) requested for given '
                f'wavelet/shape, set to {levels_max}.'
            )
            levels_decomp = levels_max

        # Check levels_estim argument w.r.t levels_decomp, if not None
        if levels_estim is not None and levels_estim > levels_decomp:
            logger.warning(
                f'Too many estimation levels ({levels_estim}) requested for decomposition, '
                f'set to {levels_decomp}.'
            )
            levels_estim = levels_decomp

        # Set the final levels values
        self.levels_decomp = levels_decomp
        self.levels_estim = levels_estim if levels_estim is not None else levels_decomp - 1

        ### Images and core
        # Make sure the images have size compatible with decomposition levels, padding
        # if necessary with zeros
        block_size = 2 ** self.levels_decomp
        # How much is missing in each axis
        self.pad_size = tuple((block_size - (s % block_size)) % block_size for s in im0.shape)
        # The slice to crop back the original area
        self.crop_slice = tuple(slice(None, -p if p else None, None) for p in self.pad_size)

        # Pad if necessary
        if any(self.pad_size):
            padding = [(0, p) for p in self.pad_size]
            im0 = np.pad(im0, padding, mode='constant')
            im1 = np.pad(im1, padding, mode='constant')

        # Create a new core if the image shape is not compatible
        if self.core is None or self.core.shape != im0.shape:
            self.core = OpticalFlowCore(
                im0.shape,
                interpolation_order=interpolation_order,
                sigma_blur=sigma_blur
            )

        logger.info(str(self.core))

        # Store the images
        self.im0 = im0.astype(self.core.dtype)
        self.im1 = im1.astype(self.core.dtype)

        ### Motion fields
        # Get a sequence of the proper length
        if U0 is None:
            U0 = [None] * self.core.ndim

        # For each component
        U: List[np.ndarray] = []
        for Ui in U0:
            # If None, initialize with zeros
            if Ui is None:
                Ui = np.zeros(self.core.shape, dtype=self.core.dtype)
            # Else pad the given field if necessary
            elif any(self.pad_size):
                padding = [(0, p) for p in self.pad_size]
                Ui = np.pad(Ui, padding, mode='constant')
            U.append(Ui)

        # As a tuple
        U = tuple(U)

        # The corresponding wavelet coefficients
        self.C_list = [
            pywt.wavedecn(Ui, self.wav, level=self.levels_decomp, mode=self.wav_boundary_condition)
            for Ui in U
        ]

        # Which we reshape as arrays to get the slices for future manipulations
        self.slices = tuple(pywt.coeffs_to_array(Ci)[1] for Ci in self.C_list)

        ### Solve
        logger.info(
            f'Decomposition over {self.levels_decomp} scales of details, '
            f'{self.levels_estim} estimated'
        )

        # For each level
        for level in range(self.levels_estim + 1):
            level_name = 'details' if level else 'approx.'
            logger.info(f'{level_name} ({level})')

            # The initial condition, as array (Note: flattened by l-bfgs)
            C_array = (pywt.coeffs_to_array(Ci[:level + 1])[0] for Ci in self.C_list)
            C_array = np.vstack(list(Ci[np.newaxis, ...] for Ci in C_array))

            # So that C_array[i] contains all coefficients of component i
            # We remember the shape for future manipulations
            C_shape = C_array.shape

            # Create the cost function for this step
            f_g = self.create_cost_function(level, C_shape)

            # Minimize
            C_array, min_value, optim_info = scipy.optimize.fmin_l_bfgs_b(
                f_g,
                C_array.astype(np.float64),
                factr=1000.
            )

            logger.info(
                f'\tl-bfgs completed with status {optim_info["warnflag"]} - '
                f'{optim_info["nit"]} iterations, {optim_info["funcalls"]} calls'
            )
            logger.info(f'\tcurrent functional value: {min_value:.2f}')

            # Store result in main coefficients
            C_array = C_array.reshape(C_shape)
            C_list = (
                pywt.array_to_coeffs(
                    Ci,
                    self.slices[i][:level + 1],
                    output_format='wavedecn'
                )
                for i, Ci in enumerate(C_array)
            )

            for n, Ci in enumerate(C_list):
                for l in range(level + 1):
                    self.C_list[n][l] = Ci[l]

        ### Rebuild field and return
        # Cropping the relevant area
        U = tuple(
            pywt.waverecn(Ci, self.wav, mode=self.wav_boundary_condition)[self.crop_slice]
            for Ci in self.C_list
        )

        return U

    def solve_fast(self, im0: np.ndarray, im1: np.ndarray, wav: str = 'haar',
                   levels_decomp: int = 3, levels_estim: Optional[int] = None,
                   U0: Optional[Tuple[np.ndarray, ...]] = None) -> Tuple[np.ndarray, ...]:
        """Fastest estimation (but lower accuracy).

        Args:
            im0: The first (grayscale) image
            im1: The second (grayscale) image
            wav: The name of the wavelet
            levels_decomp: Number of decomposition levels
            levels_estim: Number of estimation levels (<= levels_decomp)
            U0: Optional first guess for U as (U1_0, U2_0, ...)

        Returns:
            Tuple of displacement fields (U1, U2, ...) along each axis

        Note:
            Uses linear (order 1) interpolation, no blurring and 'periodization' mode.
        """
        return self.solve(
            im0, im1, wav=wav, mode='periodization',
            levels_decomp=levels_decomp, levels_estim=levels_estim,
            interpolation_order=1, sigma_blur=0.
        )

    def create_cost_function(self, step: int, shape: Tuple[int, ...]):
        """Create the cost function for L-BFGS optimization.

        The cost function takes wavelet coefficients as input parameters
        and returns (f, grad).

        Args:
            step: Current level (0 = coarse, 1 = first level of details, etc.)
            shape: The shape to reshape the input parameters

        Returns:
            A cost function suitable for scipy.optimize.fmin_l_bfgs_b
        """
        def f_g(x: np.ndarray) -> Tuple[float, np.ndarray]:
            """Compute the cost function and its gradient for given input x.

            Args:
                x: Input point, 1D ndarray of float64

            Returns:
                tuple: (f, g)
                    - f: The function value (scalar)
                    - g: The gradient (ndarray of same size and type as x)

            Notes:
                - We cast from (at input) and to (at output) float64, as l-bfgs wants 8-byte floats
                - There are lots of shape manipulations, some of which could possibly be
                  avoided. This is due to l-bfgs using 1D arrays whereas pywt relies on lists
                  of (tuples of) arrays. Here it was chosen to be as explicit and clear
                  as possible, possibly sacrificing some speed along the way.
            """
            ### Rebuild motion field
            # Reshape 1D vector to multi-D array
            x = x.reshape(shape).astype(self.core.dtype)

            # Extract coefficients, complement with finer scales and reshape for pywt
            # Note: ideally we would not complement, as finer scales are zeros. This is made
            #   necessary by pywt.
            C_list = (
                pywt.array_to_coeffs(
                    xi,
                    self.slices[i][:step + 1],
                    output_format='wavedecn'
                ) + self.C_list[i][step + 1:]
                for i, xi in enumerate(x)
            )

            # Rebuild motion field
            U = (pywt.waverecn(Ci, self.wav, mode=self.wav_boundary_condition) for Ci in C_list)

            ### Evaluate DFD and gradient
            func_value, G = self.core.DFD_gradient(self.im0, self.im1, U)

            # Decompose gradient over wavelet basis, keep coefficients only up to current step
            G_list = (
                pywt.wavedecn(
                    Gi,
                    self.wav,
                    level=self.levels_decomp,
                    mode=self.wav_boundary_condition
                )[:step + 1]
                for Gi in G
            )

            # Reshape as array, flatten and concatenate for l-bfgs
            G_array = np.hstack(
                list(pywt.coeffs_to_array(Gi)[0].ravel() for Gi in G_list)
            )

            ### Evaluate regularizer and its gradient
            # [TODO] Add regularization term integration

            ### Return DFD (+ regul) value, and its gradient w.r.t. x
            return func_value, G_array.astype(np.float64)

        return f_g

    @staticmethod
    def solve_pyramid(im0: np.ndarray, im1: np.ndarray, levels_pyr: int = 1,
                      solve_fast: bool = False, wav: str = 'haar',
                      levels_decomp: int = 3, levels_estim: Optional[int] = None,
                      **kwargs) -> Tuple[Tuple[np.ndarray, ...], 'Typhoon']:
        """Wrapper for pyramidal estimation.

        When very large displacements are involved, the usual pyramidal approach
        can be employed to help achieving a correct estimation.

        Args:
            im0: The first (grayscale) image
            im1: The second (grayscale) image
            levels_pyr: Number of pyramid levels
            solve_fast: If True, use faster (but less accurate) estimation
            wav: The name of the wavelet
            levels_decomp: Number of decomposition levels
            levels_estim: Number of estimation levels (<= levels_decomp)
            **kwargs: Remaining arguments passed to Typhoon.solve()

        Returns:
            tuple: (U, typhoon)
                - U: Tuple of displacement fields (U1, U2, ...)
                - typhoon: The Typhoon instance used for the last (finest) pyramid level
        """
        downscale = 0.5
        upscale = 1.0 / downscale

        ### Create the pyramid of images
        pyr_im0 = [im0]
        pyr_im1 = [im1]

        for l in range(levels_pyr):
            # Filter image and interpolate
            tmp_im = scipy.ndimage.gaussian_filter(pyr_im0[0], 2.0 / 3.0)
            pyr_im0.insert(0, scipy.ndimage.zoom(tmp_im, downscale))
            tmp_im = scipy.ndimage.gaussian_filter(pyr_im1[0], 2.0 / 3.0)
            pyr_im1.insert(0, scipy.ndimage.zoom(tmp_im, downscale))

        ### For each level:
        U: Optional[Tuple[np.ndarray, ...]] = None
        typhoon: Optional[Typhoon] = None

        for l, (im0, im1) in enumerate(zip(pyr_im0, pyr_im1)):
            # If not the first, use previous motion as first guess
            if l and U is not None:
                U0 = [scipy.ndimage.zoom(Ui, upscale) for Ui in U]
            else:
                U0 = None

            typhoon = Typhoon()

            if solve_fast:
                U = typhoon.solve_fast(
                    im0, im1, wav=wav,
                    levels_decomp=levels_decomp,
                    levels_estim=levels_estim
                )
            else:
                U = typhoon.solve(
                    im0, im1, U0=U0, wav=wav,
                    levels_decomp=levels_decomp,
                    levels_estim=levels_estim,
                    **kwargs
                )

        ### Return the last estimates
        if typhoon is None or U is None:
            raise RuntimeError("Pyramid estimation failed")

        return U, typhoon