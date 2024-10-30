'''
Methods:

* simulate the radio telescope observation strategy
* simulate telescope gains
'''

import numpy as np
import sys
from .telescope_functions import *
from .usefuls import *
from . import conv
from . import cosmo as cm
from . import smoothing as sm
import scipy
from glob import glob
from time import time, sleep
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

def get_uv_map_with_gains(ncells, z, gain_model={'name': 'random_uniform', 'min': 0.5, 'max': 1.1}, filename=None, 
                 total_int_time=6., int_time=10., boxsize=None, declination=-30., include_mirror_baselines=False, verbose=True):
    """
    Create the gain and uv maps with individual gain values for each baseline stored per pixel.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the grid.
    z : float
        Redshift.
    gain_model : dict or function
        Gain model parameters or a custom function that returns random gain values.
        If a dictionary with 'name' key, options are:
            - 'random_gaussian': requires 'mu' and 'std' for Gaussian distribution.
    filename : str
        Path to the file with telescope configuration.
    total_int_time : float
        Total observation time per day (in hours).
    int_time : float
        Integration time (in seconds).
    declination : float
        Declination angle in degrees.
    boxsize : float
        Size of the observed sky area in Mpc.
    include_mirror_baselines : bool
        Whether to include mirror baselines.
    verbose : bool
        If True, enables verbose output.

    Returns
    -------
    uv_map : ndarray
        Array of lists, each containing gain values for the pixel.
    N_ant : int
        Number of antennas.
    """
    # Define gain model if passed as a dictionary
    if isinstance(gain_model, dict):
        if gain_model['name'].lower() == 'random_gaussian':
            mu, std = gain_model['mu'], gain_model['std']
            gain_model = lambda N: np.random.normal(mu, std, N)
        elif gain_model['name'].lower() == 'random_uniform':
            mn, mx = gain_model['min'], gain_model['max']
            gain_model = lambda N: np.random.uniform(mn, mx, N)
        else:
            pass
    
    z = float(z)
    Nbase, N_ant = from_antenna_config(filename, z)
    total_observations = int(3600. * total_int_time / int_time)
    
    if verbose: 
        print("Generating UV map with daily observations and gain values...")
    
    # Initialize uv_map as a 2D array of empty lists to store gain values per pixel
    gain_uv_map = np.empty((ncells, ncells), dtype=object)
    for i in range(ncells):
        for j in range(ncells):
            gain_uv_map[i, j] = []  # Each cell will store a list of gain values

    # Vectorize observation process with gain application
    if verbose: 
        print("Modelling the rotation of earth")
    time_indices = np.arange(total_observations) + 1
    all_rotated_Nbase = np.array([earth_rotation_effect(Nbase, i, int_time, declination) for i in time_indices])
    if verbose: 
        print("Getting gains for the baseline using the provided gain_model")
    all_gains = gain_model(np.product(all_rotated_Nbase.shape[:2])).reshape(all_rotated_Nbase.shape[:2])

    # Grid uv tracks and store gain values at each pixel
    for rotated_Nbase, gain_vals in tqdm(zip(all_rotated_Nbase, all_gains), total=total_observations, disable=not verbose, desc="Gridding uv tracks"):
        grid_uv_tracks_with_gains(rotated_Nbase, gain_vals, gain_uv_map, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
    
    if verbose:
        print("Observation with gains complete.")
    
    return gain_uv_map, N_ant


def grid_uv_tracks_with_gains(Nbase, gain_vals, gain_uv_map, z, ncells, boxsize=None, include_mirror_baselines=False, verbose=True):
    """
    Grid uv tracks with gain values on a grid, storing individual gain values for each baseline at each pixel.

    Parameters
    ----------
    Nbase : ndarray
        Array containing ux, uy, uz values of the antenna configuration.
    gain_vals : ndarray
        Array containing gain values for each baseline.
    uv_map : ndarray
        2D array of lists, each containing gain values for the respective grid pixel.
    z : float
        Redshift of the slice observed.
    ncells : int
        Number of cells in the grid.
    boxsize : float, optional
        Comoving size of the sky observed. Defaults to a predefined constant if None.
    include_mirror_baselines : bool, optional
        If True, includes mirror baselines.

    Returns
    -------
    None : Modifies uv_map in-place to store gain values.
    """
    if boxsize is None:
        boxsize = conv.LB  # Default boxsize (assumed defined globally or elsewhere)
    
    # Calculate theta_max and normalize baseline positions
    theta_max = boxsize / cm.z_to_cdist(z)  # Using predefined comoving distance function
    Nb = np.round(Nbase * theta_max)
    
    # Filter baselines within bounds
    in_bounds = (
        (Nb[:, 0] < ncells / 2) & (Nb[:, 1] < ncells / 2) &
        (Nb[:, 0] >= -ncells / 2) & (Nb[:, 1] >= -ncells / 2)
    )
    Nb, gain_vals = Nb[in_bounds], gain_vals[in_bounds]
    
    for (x, y), gain in tqdm(zip(Nb[:, :2], gain_vals), total=len(gain_vals), disable=not verbose):
        gain_uv_map[int(x), int(y)].append(gain)
        if include_mirror_baselines:
            gain_uv_map[-int(x), -int(y)].append(gain)

def apply_uv_with_gains_response_on_image(array, uv_map, verbose=True):
    """
    Apply the effect of radio observation strategy with varied antenna/baseline gains on an image.
    
    Parameters
    ----------
    array : ndarray
        The input image array.
    uv_map : ndarray of lists
        The uv_map, where each entry is a list of gain values for that pixel.
    verbose : bool, optional
        If True, enables verbose progress output using tqdm.
        
    Returns
    -------
    img_map : ndarray
        The resulting radio image after applying the uv_map with gains in the Fourier domain.
    """
    assert array.shape == uv_map.shape, "Array and uv_map must have the same shape"
    
    # Perform the Fourier transform of the input image
    img_arr = np.fft.fft2(array)
    
    # Initialize an array to hold the gain-weighted Fourier components
    gain_applied_ft = np.zeros_like(img_arr, dtype=complex)
    
    # Iterate over each pixel in the Fourier-transformed array with tqdm
    for (i, j), gains in tqdm(np.ndenumerate(uv_map), total=uv_map.size, disable=not verbose, desc="Applying gains"):
        if gains:  # Only process if there are gain values for this pixel
            gain_applied_ft[i, j] = np.mean(gains) * img_arr[i, j] 
    
    # Apply inverse Fourier transform to obtain the final image
    img_map = np.fft.ifft2(gain_applied_ft)
    
    # Return the real part of the transformed image
    return np.real(img_map)
