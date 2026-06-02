'''
General N-point polyspectrum estimator using the FFT-shell method.

Reference: Watkinson et al. 2017 (arXiv:1705.06284).

The estimator for an N-point polyspectrum is:

    P_n(k1,...,kn) = (V^(n-1) / N^n) * sum(d1*...*dn) / sum(s1*...*sn)

where di = IFFT[FFT(delta) * W_i(k)], si = IFFT[W_i(k)], and W_i is a
spherical shell window in Fourier space.  Special cases:
  n=2  ->  power spectrum
  n=3  ->  bispectrum
  n=4  ->  trispectrum
'''

import numpy as np
from time import time
from tqdm import tqdm
from scipy import fftpack
from joblib import Parallel, delayed

from .scipy_func import numpy_product
from .power_spectrum import _get_dims, _get_k, _get_kbins, _resolve_k_limit, apply_window


def fft_nd(input_array, box_dims=None, verbose=False):
    '''Return (fftshifted FFT, k_components, k_magnitude) for input_array.'''
    box_dims = _get_dims(box_dims, input_array.shape)
    k_comp, k = _get_k(input_array, box_dims)
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
    return ft, k_comp, k


class Polyspectrum:
    '''
    General N-point polyspectrum estimator (Watkinson et al. 2017, arXiv:1705.06284).

    The estimator is:
        P_n(k1,...,kn) = (V^(n-1) / N^n) * sum_x[d1*...*dn] / sum_x[s1*...*sn]
    where di = IFFT[FFT(delta)*shell_i(ki)], si = IFFT[shell_i(ki)].

    n=2 recovers the standard power spectrum; n=3 the bispectrum; etc.

    Parameters
    ----------
    input_array : ndarray
        The density contrast field.
    box_dims : float or array-like, optional
        Box dimensions in Mpc.  Defaults to the current conv setting.
    verbose : bool
        Print progress messages.
    '''

    def __init__(self, input_array, box_dims=None, verbose=True):
        box_dims, box_dims_input = _get_dims(box_dims, input_array.shape), box_dims
        if verbose and box_dims_input is None:
            print(f'box_dims set to {box_dims} Mpc')
        self.input_array = input_array
        self.box_dims = box_dims
        self.verbose = verbose

    def spherical_shell_mask(self, array_shape, k_center, k_width, k_comp=None, k_mag=None):
        '''Return a binary mask selecting Fourier modes within k_width/2 of k_center.'''
        assert k_comp is not None or k_mag is not None
        if k_mag is None:
            k_mag = np.sum(np.array(k_comp)**2, axis=0)
        try:
            out = np.zeros(array_shape)
        except Exception:
            out = np.zeros((array_shape[0], array_shape[1], array_shape[2]))
        out[np.abs(k_mag - k_center) < k_width / 2] = 1
        return out

    def polyspectrum_fast(self, input_array_fft, k_list, dk_list,
                          box_dims=None, k_mag=None, return_n_modes=False):
        '''
        Estimate the N-point polyspectrum for a single (k1,...,kn) configuration.

        Parameters
        ----------
        input_array_fft : ndarray
            fftshifted FFT of the input field.
        k_list : sequence of floats
            k values [Mpc^-1], length n.
        dk_list : sequence of floats
            Shell widths [Mpc^-1], length n.
        box_dims : float or array-like
            Box dimensions in Mpc.
        k_mag : ndarray, optional
            Precomputed k-magnitude array (avoids recomputation across calls).

        Returns
        -------
        float : P_n value.
        '''
        n = len(k_list)
        box_vol = float(numpy_product(box_dims))
        n_pixel = int(numpy_product(input_array_fft.shape))

        # Cache by (k, dk) so that equal legs (e.g. equilateral) reuse arrays.
        cache_d, cache_s = {}, {}
        d_prod = s_prod = None

        for k, dk in zip(k_list, dk_list):
            key = (float(k), float(dk))
            if key not in cache_d:
                shell = self.spherical_shell_mask(input_array_fft.shape, k, dk, k_mag=k_mag)
                cache_s[key] = np.fft.ifftn(np.fft.fftshift(shell))
                cache_d[key] = np.fft.ifftn(np.fft.fftshift(input_array_fft * shell))
            if d_prod is None:
                d_prod = cache_d[key].copy()
                s_prod = cache_s[key].copy()
            else:
                d_prod = d_prod * cache_d[key]
                s_prod = s_prod * cache_s[key]

        s_sum = float(np.real(s_prod).sum())
        if s_sum == 0:
            return float('nan')
        p_n = float(np.real(d_prod).sum()) / s_sum * box_vol**(n - 1) / n_pixel**n
        return p_n

    def polyspectrum_equilateral(self, n_point=3, dk=0.2, n_bins=10, box_dims=None,
                                 return_n_modes=False, binning='log', verbose=True,
                                 window=None, n_jobs=1, k_limit=None):
        '''
        Equilateral N-point polyspectrum: P_n(k, k, ..., k).

        Parameters
        ----------
        n_point : int
            Order of the polyspectrum (2 = power spectrum, 3 = bispectrum, …).
        dk : float or array-like
            Shell width(s) [Mpc^-1].
        n_bins : int
            Number of k bins.
        box_dims : float or array-like
            Box dimensions in Mpc.
        binning : str
            'log' or 'linear'.
        k_limit : str or None
            Upper k cut: None (grid max), 'nyquist', or 'aliasing'.

        Returns
        -------
        dict with keys:
            'P' : ndarray, polyspectrum values
            'k' : ndarray, bin centre k values [Mpc^-1]
        '''
        tstart = time()
        input_array_nd = self.input_array.astype(np.float64)
        input_array_nd = apply_window(input_array_nd, window)

        if box_dims is None:
            box_dims = self.box_dims
        box_dims, box_dims_input = _get_dims(box_dims, input_array_nd.shape), box_dims
        if verbose and box_dims_input is None:
            print(f'box_dims set to {box_dims} Mpc')
        if verbose:
            print(f'Computing equilateral {n_point}-point polyspectrum...')

        input_array_fft, k_comp, k_mag = fft_nd(input_array_nd, box_dims=box_dims)
        if verbose:
            print('FFT done')

        if isinstance(n_bins, int):
            # auto-generate bin edges
            # For n>=3 the polygon must close on the grid; n=2 has no such constraint.
            kmax_grid = k_mag.max() / 2 if n_point >= 3 else k_mag.max()
            kmax_limit = _resolve_k_limit(k_limit, box_dims, input_array_nd.shape)
            kmax = kmax_grid if kmax_limit is None else min(kmax_limit, kmax_grid)
            kbins = _get_kbins(n_bins, box_dims, k_mag, binning=binning, kmax=kmax, kmin=None)
        else:
            # user-provided bin edges (array-like): use as-is
            kbins = np.asarray(n_bins, dtype=float)

        n_bins_actual = len(kbins) - 1
        k_centers = kbins[1:] / 2 + kbins[:-1] / 2  # evaluate at bin centres
        if dk == 'auto':
            # match shell width to each bin's width for exact comparison with tophat PS
            dk_arr = kbins[1:] - kbins[:-1]
        else:
            dk_arr = dk * np.ones(n_bins_actual) if isinstance(dk, float) else np.asarray(dk)

        def run_loop(i):
            ki  = float(k_centers[i])
            dki = float(dk_arr[i])
            return self.polyspectrum_fast(
                input_array_fft,
                [ki] * n_point, [dki] * n_point,
                box_dims=box_dims, k_mag=k_mag,
            )

        if n_jobs in (0, 1):
            P_list = np.array([run_loop(i) for i in tqdm(range(n_bins_actual))])
        else:
            P_list = np.array(Parallel(n_jobs=n_jobs)(
                delayed(run_loop)(i) for i in tqdm(range(n_bins_actual))))

        if verbose:
            print(f'...done | Runtime: {time() - tstart:.3f} s')
        return {'P': P_list, 'k': k_centers}
