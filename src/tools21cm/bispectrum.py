'''
Contains functions to estimate various three-point statistics.
The Bispectrum class inherits from Polyspectrum (polyspectrum.py) which
implements the general N-point estimator of Watkinson et al. 2017
(arXiv:1705.06284).
'''

import numpy as np, gc
from time import time, sleep
from tqdm import tqdm
from scipy import fftpack, stats
from joblib import Parallel, delayed

from . import const
from . import conv
from .helper_functions import print_msg, get_eval
from .scipy_func import numpy_product
from .power_spectrum import _get_dims, _get_k, _get_kbins, _resolve_k_limit, apply_window
from .polyspectrum import Polyspectrum, fft_nd


class Bispectrum(Polyspectrum):
    '''
    Three-point polyspectrum estimator.  Inherits from Polyspectrum so that
    bispectrum_fast and bispectrum_k are thin wrappers around the general
    N-point algorithm.
    '''

    def __init__(self, input_array, box_dims=None, verbose=True):
        super().__init__(input_array, box_dims, verbose)

    def cosine_rule(self, k1, k2, alpha):
        return np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(alpha))

    def bispectrum_fast(self, input_array_fft,
                        k1, k2, k3,
                        dk1, dk2, dk3,
                        box_dims=None,
                        k_mag=None,
                        return_n_modes=False,
                        binning='log',
                        ):
        '''Estimate B(k1,k2,k3) for a single triangle configuration.'''
        return self.polyspectrum_fast(
            input_array_fft,
            [k1, k2, k3], [dk1, dk2, dk3],
            box_dims=box_dims, k_mag=k_mag,
            return_n_modes=return_n_modes,
        )

    def bispectrum_k1k2(self,
                        k1, k2,
                        dk=0.2,
                        n_bins=10,
                        box_dims=None,
                        return_n_modes=False,
                        binning='linear',
                        verbose=True,
                        window=None,
                        n_jobs=1,
                        ):
        tstart = time()
        input_array_nd = self.input_array.astype(np.float64)
        input_array_nd = apply_window(input_array_nd, window)

        if box_dims is None:
            box_dims = self.box_dims
        box_dims, box_dims_input = _get_dims(box_dims, input_array_nd.shape), box_dims
        if verbose and box_dims_input is None:
            print(f'box_dims set to {box_dims} Mpc')

        if verbose:
            print(f'Computing bispectrum with k1,k2={k1:.2f},{k2:.2f} 1/Mpc...')
        input_array_fft, k_comp, k_mag = fft_nd(input_array_nd, box_dims=box_dims, verbose=verbose)
        if verbose:
            print('FFT of data done')

        if binning == 'linear':
            alphas = np.linspace(0, np.pi, n_bins)
        else:
            alphas = 10**np.linspace(-2, np.log10(np.pi), n_bins)
        k3_list = self.cosine_rule(k1, k2, alphas)

        def run_loop(i):
            k3 = k3_list[i]
            return self.bispectrum_fast(
                input_array_fft, k1, k2, k3, dk, dk, dk,
                k_mag=k_mag, box_dims=box_dims,
                return_n_modes=return_n_modes, binning=binning,
            )

        if n_jobs in (0, 1):
            B_list = np.array([run_loop(i) for i in tqdm(range(n_bins))])
        else:
            B_list = np.array(Parallel(n_jobs=n_jobs)(
                delayed(run_loop)(i) for i in tqdm(range(n_bins))))

        if verbose:
            print(f'...done | Runtime: {time()-tstart:.3f} s')
        return {
            'B': B_list,
            'alpha': alphas,
            'theta': np.pi - alphas,
            'k1': k1,
            'k2': k2,
            'k3': k3_list,
        }

    def bispectrum_k(self,
                     dk=0.2,
                     n_bins=10,
                     box_dims=None,
                     return_n_modes=False,
                     binning='log',
                     verbose=True,
                     window=None,
                     n_jobs=1,
                     k_limit=None,
                     ):
        '''
        Equilateral bispectrum B(k, k, k).

        k_limit : None, 'nyquist', or 'aliasing'
            None       — use k_mag.max()/2 (default grid-safe limit)
            'nyquist'  — cap at k_Nyquist = pi*min(Ni/Li), clamped to k_mag.max()/2
            'aliasing' — cap at k_Nyquist/2, clamped to k_mag.max()/2
        '''
        result = self.polyspectrum_equilateral(
            n_point=3, dk=dk, n_bins=n_bins, box_dims=box_dims,
            return_n_modes=return_n_modes, binning=binning,
            verbose=verbose, window=window, n_jobs=n_jobs, k_limit=k_limit,
        )
        return {'B': result['P'], 'k': result['k']}


class BispectrumPylians:
    def __init__(self, input_array, box_dims=None, verbose=True, n_jobs=1, MAS='CIC'):
        try:
            import Pk_library as PKL
            self.PKL = PKL
        except Exception:
            print('Install the Pylians package to use this class:')
            print('https://pylians3.readthedocs.io/en/master/installation.html')

        box_dims, box_dims_input = _get_dims(box_dims, input_array.shape)[0], box_dims
        if verbose and box_dims_input is None:
            print(f'box_dims set to {box_dims} Mpc')

        self.input_array = input_array
        self.box_dims = box_dims
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.MAS = MAS

    def bispectrum_k1k2(self, k1, k2, n_bins=15):
        threads = self.n_jobs
        MAS = self.MAS
        BoxSize = self.box_dims
        if isinstance(n_bins, (int, float)):
            theta = np.linspace(0, np.pi, n_bins)
        else:
            theta = n_bins
        if self.verbose:
            print(f'k1, k2 = {k1:.2f}/Mpc, {k2:.2f}/Mpc')
            print(f'\\theta bins = {theta}')
        delta = np.ascontiguousarray(self.input_array, dtype=np.float32)

        BBk = self.PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)
        return {
            'k1': k1,
            'k2': k2,
            'k3': BBk.k,
            'theta': theta,
            'Bk': BBk.B,
            'Qk': BBk.Q,
            'Pk': BBk.Pk,
        }


def bispectrum_k1k2(input_array, k1, k2, n_bins=15, box_dims=None, window=None, n_jobs=1, **kwargs):
    """
    Calculate the bispectrum for given k1 and k2 values.

    Parameters:
    -----------
    input_array : numpy.ndarray
    k1, k2 : float
    n_bins : int
    box_dims : tuple, optional
    window : array, optional
    n_jobs : int

    Returns:
    --------
    dict
    """
    input_array = apply_window(input_array, window).astype(np.float32)
    bisp = BispectrumPylians(input_array, box_dims=box_dims,
                             verbose=kwargs.get('verbose', True), n_jobs=n_jobs)
    return bisp.bispectrum_k1k2(k1, k2, n_bins=n_bins)


def bispectrum_equilateral(input_array, n_bins=15, box_dims=None, window=None, n_jobs=1, **kwargs):
    """
    Calculate the equilateral bispectrum.

    Parameters:
    -----------
    input_array : numpy.ndarray
    n_bins : int
    box_dims : tuple, optional
    window : array, optional
    n_jobs : int

    Returns:
    --------
    dict
    """
    tstart = time()
    input_array = apply_window(input_array, window).astype(np.float32)
    binning = kwargs.get('binning', 'log')
    kmin = 2 * np.pi / box_dims
    kmax = kmin * np.min(input_array.shape) / 2
    if binning.lower() in ['linear', 'lin']:
        kbins = np.linspace(kmin, kmax, n_bins + 1)
        kbins = (kbins[1:] + kbins[:-1]) / 2
    elif binning.lower() in ['logarithmic', 'log']:
        kbins = np.linspace(np.log10(kmin), np.log10(kmax), n_bins + 1)
        kbins = 10 ** ((kbins[1:] + kbins[:-1]) / 2)
    verbose = kwargs.get('verbose', True)
    bisp = BispectrumPylians(input_array, box_dims=box_dims, verbose=False, n_jobs=n_jobs)
    outp = {ke: np.array([]) for ke in ['k1', 'k2', 'k3', 'theta', 'Bk', 'Qk', 'Pk']}
    for ii, ki in enumerate(kbins):
        if verbose:
            print(f'{ii + 1}/{len(kbins)} | k={ki:.2f}')
        outi = bisp.bispectrum_k1k2(ki, ki, n_bins=np.array([np.pi / 3]))
        for ke in ['k1', 'k2', 'k3', 'theta', 'Bk', 'Qk', 'Pk']:
            outp[ke] = np.append(outp[ke], outi[ke])
    if verbose:
        print(f'Total Runtime: {time() - tstart:.3f} s')
    return outp
