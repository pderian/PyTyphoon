"""Demo with synthetic particle images for PyTyphoon.

This demo uses synthetic particle images (256x256 pixels) originally created
for the FLUID project (image database #1).
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytyphoon.typhoon import Typhoon
from pytyphoon.utils import RMSE

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import demo utilities
try:
    demo_path = Path(__file__).parent.parent.parent.parent / 'demo'
    import sys
    sys.path.insert(0, str(demo_path))
    import inr
    HAS_INR = True
except ImportError:
    HAS_INR = False
    logger.warning("inr module not available, demo_particles will be limited")

def demo_particles():
    """Demo with the synthetic particle images."""
    print('\n* PyTyphoon demo – "particles" demo')

    # Get demo data path
    demo_data_path = Path(__file__).parent.parent.parent / 'demo'

    if not demo_data_path.exists():
        print(f"Demo data not found at {demo_data_path}")
        print("Please ensure demo/ directory exists with required files:")
        print("  - run010050000.tif")
        print("  - run010050010.tif")
        print("  - UVtruth.inr")
        return

    ### Load data
    try:
        im0 = np.array(Image.open(demo_data_path / 'run010050000.tif').convert(mode='L')).astype(float) / 255.
        im1 = np.array(Image.open(demo_data_path / 'run010050010.tif').convert(mode='L')).astype(float) / 255.

        # Note:
        #   - U1, V1 are vertical (1st axis) components;
        #   - U2, V2 are horizontal (2nd axis) components.
        #   - inr.ReadMotion() returns (horizontal, vertical).
        if HAS_INR:
            V2, V1 = inr.readMotion(demo_data_path / 'UVtruth.inr')
        else:
            # Create dummy ground truth if inr not available
            V1 = np.zeros_like(im0)
            V2 = np.zeros_like(im0)
            logger.warning("Using dummy ground truth (inr module not available)")
    except FileNotFoundError as e:
        print(f"Error loading demo data: {e}")
        return

    ### Solve OF
    typhoon = Typhoon()
    tstart = time.perf_counter()
    U1, U2 = typhoon.solve(im0, im1, wav='db2', mode='periodization',
                           levels_decomp=3, levels_estim=1)
    tend = time.perf_counter()
    print(f'Estimation completed in {tend - tstart:.2f} s.')

    ### Post-process & display
    if HAS_INR:
        rmse = RMSE((U1, U2), (V1, V2))
    else:
        rmse = 0.0

    dpi = 72.
    fig, axes = plt.subplots(2, 4, figsize=(800. / dpi, 450. / dpi))

    # Images
    for ax, var, label in zip(axes[:, 0], [im0, im1], ['image #0', 'image #1']):
        pi = ax.imshow(var, vmin=0., vmax=1., interpolation='nearest', cmap='gray')
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # Velocity fields
    for ax, var, label in zip(axes[:, 1:-1].flat,
                              [U1, V1, U2, V2],
                              ['estim U1', 'true U1', 'estim U2', 'true U2']):
        pf = ax.imshow(var, vmin=-3., vmax=3., interpolation='nearest', cmap='RdYlBu_r')
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # Error maps
    for ax, var, label in zip(axes[:, -1],
                              [V1 - U1, V2 - U2],
                              ['abs(error U1)', 'abs(error U2)']):
        pe = ax.imshow(np.abs(var), vmin=0, vmax=0.3, interpolation='nearest')
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colormaps
    plt.subplots_adjust(bottom=0.25, top=0.94, left=.05, right=.95)
    axf = fig.add_axes([1.25 / 8., 0.16, 2.5 / 8., 0.03])
    plt.colorbar(pf, cax=axf, orientation='horizontal')
    axf.set_title('displacement (pixel)', fontdict={'fontsize': 'medium'})
    axe = fig.add_axes([4.5 / 8., 0.16, 2.5 / 8., 0.03])
    plt.colorbar(pe, cax=axe, orientation='horizontal')
    axe.set_title('error (pixel)', fontdict={'fontsize': 'medium'})

    # Labels
    plt.figtext(
        0.05, 0.015,
        f'PyTyphoon demo "particles"\n'
        f'"{typhoon.wav.name}" wavelet, {typhoon.levels_decomp} scales decomp., '
        f'{typhoon.levels_estim} scales estim., no regularizer.',
        size='medium', ha='left', va='bottom'
    )
    plt.figtext(
        0.95, 0.015,
        f'RMSE={rmse:.2f} pixel',
        size='medium', ha='right', va='bottom'
    )
    plt.show()

if __name__ == "__main__":
    demo_particles()