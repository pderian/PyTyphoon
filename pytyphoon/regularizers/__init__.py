"""Regularization terms for PyTyphoon.

This package contains implementations of various regularization terms
for optical flow estimation, including high-order wavelet-based regularizers.
"""

from .highorder import HighOrderRegularizerConv, HighOrderRegularizerMat

__all__ = ['HighOrderRegularizerConv', 'HighOrderRegularizerMat']