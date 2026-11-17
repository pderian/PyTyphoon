"""Demo of 3D estimation with simple shift for PyTyphoon.

This demo creates synthetic 3D images by filtering random normal noise at various scales.
The true motion is a simple integer shift in all directions.
Estimation is performed with the basic "fast" solver.
"""

import logging
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

from pytyphoon.typhoon import Typhoon

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_3dshift(seed: int = 123456789):
    """Demo of 3D estimation with simple shift.

    Args:
        seed: Seed for random number generator. Pass None for random images.
    """
    print('\n* PyTyphoon demo – "3dshift" demo')

    ### Create synthetic images
    print('Generating images...')

    # Sum of random normal noise filtered by different Gaussian kernels
    size = 64
    sigma_list = [1., 3., 9.]

    if seed is not None:
        np.random.seed(seed)  # Seed the generator for repeatability

    im0 = np.zeros((size, size, size))
    for sigma in sigma_list:
        tmp = np.random.randn(size, size, size)
        im0 += scipy.ndimage.gaussian_filter(tmp, sigma, mode='wrap')

    # Rescale to [0, 1]
    min_val = im0.min()
    max_val = im0.max()
    im0 = (im0 - min_val) / (max_val - min_val)

    # Second image: simply shift
    shift = (-1, 2, 3)
    im1 = np.roll(im0, shift=shift, axis=(0, 1, 2))

    ### Perform estimation
    typhoon = Typhoon()
    tstart = time.perf_counter()
    U = typhoon.solve_fast(im0, im1, wav='db2', levels_decomp=3, levels_estim=1)
    tend = time.perf_counter()
    print(f'Estimation completed in {tend - tstart:.2f} s.')

    ### Post-process and display
    truth_label = f'true shift: ({", ".join(f"{s:.2f}" for s in shift)}) pixel'
    estim_label = f'estimated mean displacement: ({", ".join(f"{Ui.mean():.2f}" for Ui in U)}) pixel'

    dpi = 72.
    fig, axes = plt.subplots(3, 5, figsize=(800. / dpi, 450. / dpi))

    # Show the 3 components in the middle plane along each axis
    for i in range(3):
        # Slice of the mid-plane along this axis
        slice_mid = [slice(None)] * 3
        slice_mid[i] = size // 2
        slice_mid = tuple(slice_mid)

        # For each image
        for j, (ax, imj) in enumerate(zip(axes[i, :2], [im0, im1])):
            ax.imshow(imj[slice_mid], interpolation='nearest', cmap='gray',
                      vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'im{j} @ mid-ax{i + 1}')

        # For each component
        for j, (ax, Uj) in enumerate(zip(axes[i, 2:], U), 1):
            pc = ax.imshow(Uj[slice_mid], interpolation='nearest', cmap='RdYlBu_r',
                           vmin=-3, vmax=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'U{j} @ mid-ax{i + 1}')

    plt.subplots_adjust(left=.1, right=.9, bottom=.14, top=.9, hspace=.3)

    # Colorbar
    cax = fig.add_axes([.75, 0.05, .2, 0.03])
    cb = plt.colorbar(pc, cax=cax, orientation='horizontal')
    cb.ax.set_title('displacement (pixel)', fontdict={'fontsize': 'medium'})

    # Labels
    plt.figtext(0.5, 0.98, truth_label + ' - ' + estim_label,
               ha='center', va='top', size='medium')
    plt.figtext(
        0.05, 0.015,
        f'PyTyphoon demo "3dshift"\n'
        f'"{typhoon.wav.name}" wavelet, {typhoon.levels_decomp} scales decomp., '
        f'{typhoon.levels_estim} scales estim., no regularizer, "fast" solver.',
        size='medium', ha='left', va='bottom'
    )
    plt.show()

if __name__ == "__main__":
    import scipy.ndimage
    demo_3dshift()