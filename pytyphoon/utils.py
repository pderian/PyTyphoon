"""Utility functions for PyTyphoon.

This module contains helper functions for optical flow computation,
including error metrics and other utilities.
"""

import numpy as np
from typing import Tuple, Union

def RMSE(Ua: Tuple[np.ndarray, ...], Ub: Tuple[np.ndarray, ...]) -> float:
    """Compute Root Mean Squared Error (RMSE) between two vector fields.

    Args:
        Ua: First vector field as tuple of components (Ua1, Ua2, ...)
        Ub: Second vector field as tuple of components (Ub1, Ub2, ...)

    Returns:
        The RMSE as a float

    Example:
        >>> U1 = np.array([[1, 2], [3, 4]])
        >>> U2 = np.array([[1.1, 2.1], [2.9, 4.2]])
        >>> V1 = np.array([[0.5, 0.6], [0.7, 0.8]])
        >>> V2 = np.array([[0.55, 0.65], [0.75, 0.85]])
        >>> RMSE((U1, U2), (V1, V2))
    """
    return float(np.sqrt(np.mean(np.sum(
        (np.asarray(Ua) - np.asarray(Ub))**2, axis=0
    ))))