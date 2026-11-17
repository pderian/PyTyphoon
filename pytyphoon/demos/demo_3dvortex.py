"""Demo of 3D estimation with vortex motion for PyTyphoon.

This demo creates synthetic 3D images by filtering random normal noise at various scales.
The displacement field is a column vortex (first two axes) with an updraft increasing
linearly along the third axis. Estimation is performed with the basic "fast" solver.
"""

import logging
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

from pytyphoon.typhoon import Typhoon
from pytyphoon.utils import RMSE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_3dvortex(seed: int = 123456789):
    """Demo of 3D estimation with vortex motion.

    Args:
        seed: Seed for random number generator. Pass None for random images.
    """
    print('\n* PyTyphoon demo – "3dvortex" demo')

    ### Create synthetic images
    print('Generating images...')

    # Sum of random normal noise filtered by different Gaussian kernels
    size = 96
    margin = 4  # Add a margin on every dimension
    sigma_list = [1., 3., 9.]  # A list of Gaussian blur sigma
    size_tmp = size + 2 * margin

    if seed is not None:
        np.random.seed(seed)  # Seed the generator for repeatability

    im0 = np.zeros((size_tmp, size_tmp, size_tmp))
    for sigma in sigma_list:
        tmp = np.random.randn(size_tmp, size_tmp, size_tmp)
        im0 += scipy.ndimage.gaussian_filter(tmp, sigma, mode='wrap')

    # Rescale to [0, 1]
    min_val = im0.min()
    max_val = im0.max()
    im0 = (im0 - min_val) / (max_val - min_val)

    ### Create synthetic motion
    # The coordinates
    X1, X2, X3 = np.indices(im0.shape, dtype=float)

    # Tubular vortex: from streamfunction
    sigma = float(size) / 3.  # Make sure it decays enough near edges
    alpha = float(size)  # Amplitude factor
    beta = 3.  # Amplitude max of 3rd component
    x0 = [size_tmp // 2 for _ in range(3)]  # Coordinates of center of volume

    SF = alpha * np.exp((-1. / sigma ** 2) * ((X1 - x0[0]) ** 2 + (X2 - x0[1]) ** 2))
    G1, G2, _ = np.gradient(SF)

    # First 2 components are vortex, third increases linearly from zero to beta
    U_truth = (
        G2,
        -G1,
        (X3 - margin) * (beta / float(size))  # Divide by n_adv due to multiple steps
    )

    # Displacement coordinates
    n_adv = 30  # Number of advection steps
    map_coords = np.vstack(list(X.ravel() for X in (X1, X2, X3)))  # For interpolation

    for n, Ui in enumerate(U_truth):
        map_coords[n] -= (1. / float(n_adv)) * Ui.ravel()

    # Very crude advection
    im1 = im0
    for _ in range(n_adv):
        im1 = scipy.ndimage.map_coordinates(
            im1, map_coords, order=3, mode='constant'
        ).reshape(im0.shape)

    # Finally retain only the working area
    if margin > 0:
        im0 = im0[margin:-margin, margin:-margin, margin:-margin]
        im1 = im1[margin:-margin, margin:-margin, margin:-margin]
        U_truth = tuple(
            Ui[margin:-margin, margin:-margin, margin:-margin]
            for Ui in U_truth
        )

    ### Perform estimation
    # Instance typhoon so we can use its internal coordinates for simplicity
    typhoon = Typhoon()
    tstart = time.perf_counter()
    U = typhoon.solve_fast(im0, im1, wav='db3', levels_decomp=3, levels_estim=0)
    tend = time.perf_counter()
    print(f'Estimation completed in {tend - tstart:.2f} s.')

    ### Post-process and display
    # Show the 3 components in the middle plane along each axis
    rmse = RMSE(U_truth, U)
    dpi = 72.
    fig, axes = plt.subplots(3, 5, figsize=(800. / dpi, 450. / dpi))

    # For each axis
    for i in range(3):
        # Slice of the mid-plane along this axis
        slice_mid = [slice(None)] * 3
        slice_mid[i] = size // 2
        slice_mid = tuple(slice_mid)

        # For each image
        for j, (ax, imj) in enumerate(zip(axes[i, :2], [im0, im1])):
            ax.imshow(
                imj[slice_mid],
                interpolation='nearest',
                cmap='gray',
                vmin=0,
                vmax=1
            )
            ax.autoscale(tight=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'im{j} @ mid-ax{i + 1}')

        # For each component
        for j, (ax, Uj) in enumerate(zip(axes[i, 2:], U), 1):
            pc = ax.imshow(
                Uj[slice_mid],
                interpolation='nearest',
                cmap='RdYlBu_r',
                vmin=-3,
                vmax=3
            )
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'U{j} @ mid-ax{i + 1}')

    plt.subplots_adjust(left=.1, right=.9, bottom=.14, top=.95, hspace=.3)

    # Colorbar
    cax = fig.add_axes([.565, 0.05, .2, 0.03])
    cb = plt.colorbar(pc, cax=cax, orientation='horizontal')
    cb.ax.set_title('displacement (pixel)', fontdict={'fontsize': 'medium'})

    # Labels
    plt.figtext(
        0.05, 0.015,
        f'PyTyphoon demo "3dvortex"\n'
        f'"{typhoon.wav.name}" wavelet, {typhoon.levels_decomp} scales decomp., '
        f'{typhoon.levels_estim} scales estim., no regularizer, "fast" solver.',
        size='medium', ha='left', va='bottom'
    )
    plt.figtext(
        0.95, 0.015,
        f'RMSE={rmse:.2f} pixel',
        size='medium', ha='right', va='bottom'
    )
    plt.show()

if __name__ == "__main__":
    demo_3dvortex()