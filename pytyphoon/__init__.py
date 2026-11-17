"""PyTyphoon - Python implementation of the Typhoon optical flow algorithm.

This package provides a pure Python implementation of the Typhoon algorithm for
dense 2D/3D optical flow estimation on wavelet bases, primarily aimed at fluid
motion estimation.

Version: 0.2.0

Changes from original:
- Refactored into modular package structure
- Fixed critical bug in DFD_gradient (generator to tuple conversion)
- Updated deprecated numpy/scipy functions
- Added comprehensive type hints
- Added logging support
- Improved documentation and docstrings
- Separated demos into individual modules
- Integrated high-order regularizers
"""

from .core import OpticalFlowCore
from .typhoon import Typhoon
from .utils import RMSE
from .regularizers import HighOrderRegularizerConv, HighOrderRegularizerMat

__version__ = "0.2.0"
__all__ = [
    'OpticalFlowCore',
    'Typhoon',
    'RMSE',
    'HighOrderRegularizerConv',
    'HighOrderRegularizerMat',
    '__version__'
]
