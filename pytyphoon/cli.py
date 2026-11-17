"""Command-line interface for PyTyphoon.

This module provides the command-line interface for running PyTyphoon
as a standalone optical flow estimation tool.
"""

import argparse
import logging
import os.path
import sys
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .typhoon import Typhoon
from .utils import RMSE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def print_versions() -> None:
    """Print module versions."""
    import pytyphoon
    import scipy
    import pywt

    print("\n* Module versions:")
    print(f'Python: {sys.version}')
    print(f'Numpy: {np.__version__}')
    print(f'Scipy: {scipy.__version__}')
    print(f'PyWavelet: {pywt.__version__}')
    print(f'PyTyphoon (this): {pytyphoon.__version__}')

def run_single_estimation(args) -> None:
    """Run single optical flow estimation from command line arguments."""
    print(f'\nEstimation:\n\t{args.im0}\n\t{args.im1}')

    ### Load images
    im0 = np.array(Image.open(args.im0).convert(mode='L')).astype(np.float32) / 255.
    im1 = np.array(Image.open(args.im1).convert(mode='L')).astype(np.float32) / 255.

    ### Solve problem
    tstart = time.perf_counter()

    if args.levels_pyr > 0:
        (U1, U2), typhoon = Typhoon.solve_pyramid(
            im0, im1,
            levels_pyr=args.levels_pyr,
            solve_fast=args.solve_fast,
            wav=args.wav,
            mode=args.mode,
            levels_decomp=args.levels_decomp,
            levels_estim=args.levels_estim,
            interpolation_order=args.interpolation_order,
            sigma_blur=args.sigma_blur
        )
    else:
        typhoon = Typhoon()
        U1, U2 = typhoon.solve(
            im0, im1,
            wav=args.wav,
            mode=args.mode,
            levels_decomp=args.levels_decomp,
            levels_estim=args.levels_estim,
            interpolation_order=args.interpolation_order,
            sigma_blur=args.sigma_blur
        )

    tend = time.perf_counter()
    print(f'Estimation completed in {tend - tstart:.2f} s.')

    # Display
    if args.display_result:
        try:
            import matplotlib.pyplot as plt
            dpi = 72.
            fig, axes = plt.subplots(2, 3, figsize=(800. / dpi, 600. / dpi))

            # Images
            for ax, var, label in zip(axes[0, 1:], [im0, im1], ['image #0', 'image #1']):
                pi = ax.imshow(var, vmin=0., vmax=1., interpolation='nearest', cmap='gray')
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0, 0].remove()

            # Velocity fields
            pf = []
            for ax, var, label in zip(axes[1, 1:], [U1, U2], ['estim U1', 'estim U2']):
                pf.append(ax.imshow(var, interpolation='nearest', cmap='RdYlBu_r'))
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])

            # Quiver - note that array axes order is reversed
            ax = axes[1, 0]
            ax.set_aspect('equal')
            qstep = 8
            ax.quiver(
                typhoon.core.x[1][::qstep],
                typhoon.core.x[0][::qstep],
                U2[::qstep, ::qstep],
                U1[::qstep, ::qstep],
                pivot='middle',
                color='b',
                units='xy',
                angles='xy'
            )
            ax.set_xlim(typhoon.core.x[1][0], typhoon.core.x[1][-1])
            ax.set_ylim(typhoon.core.x[0][0], typhoon.core.x[0][-1])
            ax.invert_yaxis()
            ax.set_title('motion field')
            ax.set_xticks([])
            ax.set_yticks([])

            plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.1)

            # Labels
            plt.figtext(
                0.05, 0.015,
                f'PyTyphoon {pytyphoon.__version__}',
                size='medium',
                ha='left',
                va='bottom'
            )
            plt.figtext(
                0.05, 0.95,
                f'im0: {os.path.basename(args.im0)}\n'
                f'im1: {os.path.basename(args.im1)}\n'
                f'levels pyramid: {args.levels_pyr}\n'
                f'wavelet: {typhoon.wav.name}\n'
                f'wav. decomp. lvls: {typhoon.levels_decomp}\n'
                f'wav. estim. lvls: {typhoon.levels_estim}',
                size='medium',
                ha='left',
                va='top'
            )
            plt.show()
        except ImportError:
            logger.error("Matplotlib is required for display. Install with: pip install matplotlib")

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Python implementation of Typhoon motion estimator."
    )

    # Estimation parameters
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

    # Misc arguments
    parser.add_argument('--demo', dest="demo_name", type=str, default='',
                        help="name of the demo")
    parser.add_argument('--version', dest="print_version", action='store_true',
                        help="print module versions")

    # Set defaults
    parser.set_defaults(verbose=True, print_version=False, display_result=False,
                        solve_fast=False)

    return parser

def main(argv: Optional[list] = None) -> None:
    """Entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = create_parser()
    args = parser.parse_args(argv)

    ### Versions
    if args.print_version:
        print_versions()
        return

    ### Demos
    # List of available demos
    demos = {
        'particles': 'demo_particles',
        '3dshift': 'demo_3dshift',
        '3dvortex': 'demo_3dvortex',
    }

    if args.demo_name:
        if args.demo_name in demos:
            try:
                demo_module = __import__(f'pytyphoon.demos.{demos[args.demo_name]}',
                                       fromlist=[demos[args.demo_name]])
                demo_func = getattr(demo_module, args.demo_name)
                demo_func()
            except ImportError as e:
                logger.error(f"Could not load demo '{args.demo_name}': {e}")
        else:
            print(f"[!] Unknown demo '{args.demo_name}', valid demos: {list(demos.keys())}")
        return

    ### Single estimation
    if args.im0 and args.im1:
        run_single_estimation(args)
        return

    # If no arguments provided, show help
    if len(argv) == 0:
        parser.print_help()

if __name__ == "__main__":
    main()