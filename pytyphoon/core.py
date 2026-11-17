"""Core optical flow functionality for PyTyphoon.

This module contains the OpticalFlowCore class which handles the fundamental
operations for computing displaced frame differences and their gradients.
"""

import logging
from typing import Tuple, Optional, Union, List
import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)

class OpticalFlowCore:
    """Core functions for optical flow.

    This class handles the fundamental operations for computing
    displaced frame differences (DFD) and their gradients.

    Attributes:
        shape: The grid shape as a tuple
        ndim: Number of dimensions
        dtype: Data type for computations
        interpolation_order: Spline interpolation order
        sigma_blur: Sigma for Gaussian blur before gradient computation
        boundary_condition: Boundary condition for interpolation
    """

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32,
                 interpolation_order: int = 3, sigma_blur: float = 0.5):
        """Initialize the optical flow core.

        Args:
            shape: The grid shape as a tuple of integers
            dtype: Data type for computations (default: np.float32)
            interpolation_order: Spline interpolation order > 0, faster when lower
            sigma_blur: Sigma of Gaussian blur applied before spatial gradient computation
        """
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(self.shape)

        # 1D grid coordinates
        self.x = tuple(np.arange(s, dtype=self.dtype) for s in self.shape)

        # N-D grid coordinates
        self.X = np.indices(self.shape, dtype=self.dtype)
        self.Xcoords = np.vstack(list(X.ravel() for X in self.X))

        # Misc parameters
        self.sigma_blur = sigma_blur
        self.interpolation_order = interpolation_order
        self.boundary_condition = 'mirror'

        # Buffer for intermediate computations
        self.buffer = np.zeros(np.prod(self.shape), dtype=dtype)

    def __str__(self) -> str:
        """Return string representation of the core."""
        param_str = f'\n\tshape={self.shape}'
        param_str += f'\n\tdtype={self.dtype.__name__}'
        param_str += f'\n\tinterpolation order={self.interpolation_order}'
        param_str += f'\n\tsigma blur={self.sigma_blur}'
        return self.__class__.__name__ + param_str

    def DFD(self, im0: np.ndarray, im1: np.ndarray, U: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Compute the Displaced Frame Difference (DFD).

        Args:
            im0: The first (grayscale) image
            im1: The second (grayscale) image
            U: Displacements (U1, U2, ...) along each axis

        Returns:
            The DFD as a numpy array
        """
        # Warp im1
        map_coords = self.Xcoords.copy()
        for n, Ui in enumerate(U):
            map_coords[n] += Ui.ravel()

        im1_warp = scipy.ndimage.map_coordinates(
            im1, map_coords,
            order=self.interpolation_order,
            mode=self.boundary_condition
        )

        # Difference
        return im1_warp.reshape(self.shape) - im0

    def DFD_gradient(self, im0: np.ndarray, im1: np.ndarray, U: Tuple[np.ndarray, ...]):
        """Compute the displaced frame difference (DFD) functional value and its gradients.

        Args:
            im0: The first (grayscale) image
            im1: The second (grayscale) image
            U: Displacements (U1, U2, ...) along each axis

        Returns:
            tuple: (dfd_value, grad_tuple)
                - dfd_value: The value of the functional (0.5 * sum(dfd^2))
                - grad_tuple: Tuple of gradients w.r.t. each component of U
        """
        # Warp im1 -> buffer
        map_coords = self.Xcoords.copy()
        for n, Ui in enumerate(U):
            map_coords[n] += Ui.ravel()

        scipy.ndimage.map_coordinates(
            im1, map_coords,
            output=self.buffer,
            order=self.interpolation_order,
            mode=self.boundary_condition
        )
        im1_warp = self.buffer.reshape(self.shape)

        # Difference
        dfd = im1_warp - im0

        # Spatial gradients
        if self.sigma_blur > 0:
            im1_warp = scipy.ndimage.gaussian_filter(im1_warp, self.sigma_blur)

        grad = np.gradient(im1_warp)

        # FIX: Convert generator to tuple - this was the critical bug
        grad = tuple(g * dfd for g in grad)

        # Return functional value and gradients
        return 0.5 * np.sum(dfd**2), grad