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


def from_antenna_config_with_gains(filename, z, nu=None, 
                gain_model={'name': 'random_uniform', 'min': 0.5, 'max': 1.1}):
    """
    The function reads the antenna positions (N_ant antennas) from the file given.

    Parameters
    ----------
    filename: str
        Name of the file containing the antenna configurations (text file).
    z       : float
        Redhsift of the slice observed.
    nu      : float
        The frequency observed by the telescope.

    Returns
    -------
    Nbase   : ndarray
        Numpy array (N_ant(N_ant-1)/2 x 3) containing the (ux,uy,uz) values derived 
                from the antenna positions.
    N_ant   : int
        Number of antennas.
    """
    if isinstance(gain_model, dict):
        if 'name' in gain_model.keys():
            if gain_model['name'].lower() == 'random_gaussian':
                mu, std = gain_model['mu'], gain_model['std']
                gain_model_gen = lambda N: np.random.normal(mu, std, N)
            elif gain_model['name'].lower() == 'random_uniform':
                mn, mx = gain_model['min'], gain_model['max']
                gain_model_gen = lambda N: np.random.uniform(mn, mx, N)
            else:
                pass
            all_gains_1 = gain_model_gen(N_ant) #Real or Amplitudes
            all_gains_2 = gain_model_gen(N_ant) #Imag or phases
        elif 'real' in gain_model.keys():
            all_gains_1 = gain_model['real'](N_ant) #Real or Amplitudes
            all_gains_2 = gain_model['imag'](N_ant) #Imag or phases
        elif 'phase' in gain_model.keys():
            all_gains_1 = gain_model['amplitude'](N_ant) #Real or Amplitudes
            all_gains_2 = gain_model['phase'](N_ant) #Imag or phases
        else:
            print('The provided gain_model is not implemented.')
            return None

    z = float(z)
    if filename is None: antll  = SKA1_LowConfig_Sept2016()
    else: antll  = np.loadtxt(filename, dtype=str) if isinstance(filename, str) else filename
    antll  = antll[:,-2:].astype(float)
    Re     = 6.371e6                                        # in m
    pp     = np.pi/180
    if not nu: nu = cm.z_to_nu(z)                           # MHz
    antxyz = np.zeros((antll.shape[0],3))		            # in m
    antxyz[:,0] = Re*np.cos(antll[:,1]*pp)*np.cos(antll[:,0]*pp)
    antxyz[:,1] = Re*np.cos(antll[:,1]*pp)*np.sin(antll[:,0]*pp)
    antxyz[:,2] = Re*np.sin(antll[:,1]*pp)	
    del pp, antll
    N_ant = antxyz.shape[0]
    pair_comb = itertools.combinations(range(N_ant), 2)
    pair_comb = list(pair_comb)	
    lam = c_light/(nu*1e6)/1e2 			            # in m
    Nbase_with_gains = []
    for ii,jj in list(pair_comb):
        # print(ii,jj,len(Nbase_with_gains))
        ux = (antxyz[ii,0]-antxyz[jj,0])/lam
        uy = (antxyz[ii,1]-antxyz[jj,1])/lam
        uz = (antxyz[ii,2]-antxyz[jj,2])/lam
        gr = np.array([all_gains_1[ii], all_gains_1[jj]])
        gi = np.array([all_gains_2[ii], all_gains_2[jj]])
        if ux==0: print(ii,jj)
        Nbase_with_gains.append([ux,uy,uz,gr[0]*gr[1],gi[0]*gi[1]])
    Nbase = np.array(Nbase_with_gains)	
    return Nbase, N_ant


def get_uv_map_with_gains(ncells, z, gain_model={'real': lambda n: np.random.normal(1,0.5,n), 'imag': lambda n: np.random.normal(1,0.5,n)}, 
                          filename=None, total_int_time=6., int_time=10., boxsize=None, declination=-30., 
                          include_mirror_baselines=False, verbose=True):
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

    # Load the antenna configuration with gains only once
    if verbose: 
        print("Loading antenna configuration with gains...")
    Nbase, N_ant = from_antenna_config_with_gains(filename, z, gain_model=gain_model)

    # Calculate the total number of time steps
    total_observations = int(3600. * total_int_time / int_time)

    # Initialize uv_map for storing accumulated gain information
    gain_uv_map = np.zeros((ncells, ncells, 3))
    
    if verbose: 
        print("Starting UV map generation...")

    # Iterate through time slices, performing incremental rotation and gridding
    for time_idx in tqdm(range(total_observations), disable=not verbose, desc="Gridding uv tracks"):
        # Apply incremental rotation for the time step
        rotated_Nbase = earth_rotation_effect(Nbase[:,:3], time_idx, int_time, declination)

        # Grid the rotated baselines with gain values
        grid_uv_tracks_with_gains(rotated_Nbase, Nbase[:,3:], gain_uv_map, z, ncells,
                                  boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    if verbose:
        print("UV map generation complete.")

    return gain_uv_map, N_ant

def gain_uv_map_to_uv_map(gain_uv_map, verbose=True):
    uv_map = np.zeros_like(gain_uv_map)
    for (i, j), gains in tqdm(np.ndenumerate(gain_uv_map), total=gain_uv_map.size, disable=not verbose):
        uv_map[i, j] = len(gains)
    return uv_map.astype(int)

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
    gain_vals1, gain_vals2 = gain_vals[:,0], gain_vals[:,1]
    Nb, gain_vals1, gain_vals2 = Nb[in_bounds], gain_vals1[in_bounds], gain_vals2[in_bounds]
    
    for (x, y), gain1, gain2 in zip(Nb[:, :2], gain_vals1, gain_vals2):
        gain_uv_map[int(x), int(y), 0] += 1
        gain_uv_map[int(x), int(y), 1] += gain1
        gain_uv_map[int(x), int(y), 2] += gain2
        if include_mirror_baselines:
            gain_uv_map[-int(x), -int(y), 0] += 1
            gain_uv_map[-int(x), -int(y), 1] += gain1
            gain_uv_map[-int(x), -int(y), 2] += gain2


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

def from_antenna_config_with_antenna_stamp(filename, z, nu=None):
    """
    The function reads the antenna positions (N_ant antennas) from the file given.

    Parameters
    ----------
    filename: str
        Name of the file containing the antenna configurations (text file).
    z       : float
        Redhsift of the slice observed.
    nu      : float
        The frequency observed by the telescope.

    Returns
    -------
    Nbase   : ndarray
        Numpy array (N_ant(N_ant-1)/2 x 3) containing the (ux,uy,uz) values derived 
                from the antenna positions.
    N_ant   : int
        Number of antennas.
    """
    z = float(z)
    if filename is None: antll  = SKA1_LowConfig_Sept2016()
    else: antll  = np.loadtxt(filename, dtype=str) if isinstance(filename, str) else filename
    antll  = antll[:,-2:].astype(float)
    Re     = 6.371e6                                        # in m
    pp     = np.pi/180
    if not nu: nu = cm.z_to_nu(z)                           # MHz
    antxyz = np.zeros((antll.shape[0],3))		            # in m
    antxyz[:,0] = Re*np.cos(antll[:,1]*pp)*np.cos(antll[:,0]*pp)
    antxyz[:,1] = Re*np.cos(antll[:,1]*pp)*np.sin(antll[:,0]*pp)
    antxyz[:,2] = Re*np.sin(antll[:,1]*pp)	
    del pp, antll
    N_ant = antxyz.shape[0]
    pair_comb = itertools.combinations(range(N_ant), 2)
    pair_comb = list(pair_comb)	
    lam = c_light/(nu*1e6)/1e2 			            # in m
    all_ant_tags = np.arange(N_ant)+1
    Nbase_with_gains = []
    for ii,jj in list(pair_comb):
        # print(ii,jj,len(Nbase_with_gains))
        ux = (antxyz[ii,0]-antxyz[jj,0])/lam
        uy = (antxyz[ii,1]-antxyz[jj,1])/lam
        uz = (antxyz[ii,2]-antxyz[jj,2])/lam
        if ux==0: print(ii,jj)
        Nbase_with_gains.append([ux,uy,uz,all_ant_tags[ii],all_ant_tags[jj]])
    Nbase = np.array(Nbase_with_gains)	
    return Nbase, N_ant

def get_uv_map_with_antenna_stamp(ncells, z, 
                          filename=None, total_int_time=6., int_time=10., boxsize=None, declination=-30., 
                          include_mirror_baselines=False, verbose=True):
    """
    Create the gain and uv maps with individual gain values for each baseline stored per pixel.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the grid.
    z : float
        Redshift.
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

    # Load the antenna configuration with gains only once
    if verbose: 
        print("Loading antenna configuration with antenna stamps...")
    Nbase, N_ant = from_antenna_config_with_antenna_stamp(filename, z)

    # Calculate the total number of time steps
    total_observations = int(3600. * total_int_time / int_time)

    # Initialize uv_map for storing accumulated gain information
    ant_tag_uv_map = np.zeros((ncells, ncells, 2, total_observations))
    
    if verbose: 
        print("Starting UV map generation...")

    # Iterate through time slices, performing incremental rotation and gridding
    for time_idx in tqdm(range(total_observations), disable=not verbose, desc="Gridding uv tracks"):
        # Apply incremental rotation for the time step
        rotated_Nbase = earth_rotation_effect(Nbase[:,:3], time_idx, int_time, declination)

        # Grid the rotated baselines with gain values
        grid_uv_tracks_with_antenna_stamp(rotated_Nbase, Nbase[:,3:], ant_tag_uv_map, z, ncells, time_idx,
                                  boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    if verbose:
        print("UV map generation complete.")

    return ant_tag_uv_map, N_ant

def antenna_stamp_uv_map_to_uv_map(ant_tag_uv_map, verbose=True):
    uv_map = np.zeros_like(ant_tag_uv_map[:,:,0,0])
    for (i, j), gains in tqdm(np.ndenumerate(uv_map), total=uv_map.size, disable=not verbose):
        uv_map[i, j] = (ant_tag_uv_map[i,j,0,:]>0).sum()
    return uv_map.astype(int)

def grid_uv_tracks_with_antenna_stamp(Nbase, ant_tag, ant_tag_uv_map, z, ncells, time_idx,
                                      boxsize=None, include_mirror_baselines=False, verbose=True):
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
    ant_tag1, ant_tag2 = ant_tag[:,0], ant_tag[:,1]
    Nb, ant_tag1, ant_tag2 = Nb[in_bounds], ant_tag1[in_bounds], ant_tag2[in_bounds]
    
    for (x, y), ant1, ant2 in zip(Nb[:, :2], ant_tag1, ant_tag2):
        ant_tag_uv_map[int(x), int(y), 0, time_idx] = ant1
        ant_tag_uv_map[int(x), int(y), 1, time_idx] = ant2
        if include_mirror_baselines:
            print('include_mirror_baselines is not implemented yet.')