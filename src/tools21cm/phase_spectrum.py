'''
Contains functions to estimate phase spectra.
'''

import numpy as np
from .scipy_func import numpy_product


def phase_spectrum_nd(input_array, box_dims, verbose=False):
    '''
    Calculate the phase spectrum of input_array and return it as an
    n-dimensional array with the same shape as input_array.

    Parameters:
        input_array (numpy array): array to compute the phase spectrum of.
        box_dims (float or array-like): box size in comoving Mpc. If a scalar,
            the same length is used along all axes.
        verbose (bool): print progress messages (default False).

    Returns:
        Phase spectrum array (radians, range [-pi, pi]).
    '''
    if np.array(box_dims).size != 3:
        box_dims = np.array([box_dims, box_dims, box_dims])
    if verbose:
        print('Calculating phase spectrum...')
    ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
    phase_spectrum = np.angle(ft)
    if verbose:
        print('...done')
    return phase_spectrum
