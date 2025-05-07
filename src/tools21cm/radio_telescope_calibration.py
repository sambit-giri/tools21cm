'''
Methods:

* simulate the radio telescope observation strategy
* simulate telescope gains
'''

import numpy as np
import sys
from .radio_telescope_sensitivity import *
from .usefuls import *
from . import conv
from . import cosmo as cm
from . import smoothing as sm
import scipy
from glob import glob
from time import time, sleep
import pickle
from joblib import cpu_count, Parallel, delayed
from tqdm import tqdm


def from_antenna_config_with_gains(antxyz, z, nu=None, 
                gain_model={'name': 'random_uniform', 'min': 0.5, 'max': 1.1}):
    """
    The function reads the antenna positions (N_ant antennas) from the file given.

    Parameters
    ----------
    antxyz: ndarray
        The radio telescope antenna configuration.
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
    if antxyz is None: 
        antll  = SKA1_LowConfig_Sept2016()
        antll  = antll[:,-2:].astype(float)
        antxyz = geographic_to_cartesian_coordinate_system(antll)
    else:
        antxyz = antxyz.to('m').value
    if not nu: 
        nu = cm.z_to_nu(z)                           # MHz

    N_ant = antxyz.shape[0]
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
            all_gains_r = gain_model_gen(N_ant) #Real or Amplitudes
            all_gains_i = gain_model_gen(N_ant) #Imag or phases
        elif 'real' in gain_model.keys():
            all_gains_r = gain_model['real'](N_ant) #Real or Amplitudes
            all_gains_i = gain_model['imag'](N_ant) #Imag or phases
        elif 'phase' in gain_model.keys():
            all_gains_r = gain_model['amplitude'](N_ant) #Real or Amplitudes
            all_gains_i = gain_model['phase'](N_ant) #Imag or phases
        else:
            print('The provided gain_model is not implemented.')
            return None
    else:
        all_gains_r, all_gains_i = gain_model[:,0], gain_model[:,1]
    
    pair_comb = itertools.combinations(range(N_ant), 2)
    pair_comb = list(pair_comb)	
    lam = c_light_cgs/(nu*1e6)/1e2 			            # in m
    Nbase_with_gains = []
    for ii,jj in list(pair_comb):
        # print(ii,jj,len(Nbase_with_gains))
        ux = (antxyz[ii,0]-antxyz[jj,0])/lam
        uy = (antxyz[ii,1]-antxyz[jj,1])/lam
        uz = (antxyz[ii,2]-antxyz[jj,2])/lam
        jm = [all_gains_r[ii]*all_gains_r[jj],
              all_gains_i[ii]*all_gains_i[jj],
              all_gains_r[ii]*all_gains_i[jj],
              all_gains_i[ii]*all_gains_r[jj] ]
        if ux==0: print(ii,jj)
        Nbase_with_gains.append([ux,uy,uz,jm[0],jm[1],jm[2],jm[3]])
    Nbase = np.array(Nbase_with_gains)	
    return Nbase, N_ant


def get_uv_map_with_gains(ncells, z, 
                          gain_model={'real': lambda n: np.random.normal(1,0.5,n), 'imag': lambda n: np.random.normal(1,0.5,n)}, 
                          gain_timescale=[10,60],
                          subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., 
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
    gain_timescale : list
        Timescale after which gain values will evolve.
    subarray_type: str
		The name of the SKA-Low layout configuration.
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

    # Calculate the total number of time steps
    total_observations = int(3600. * total_int_time / int_time)

    # Initialize uv_map for storing accumulated gain information
    gain_uv_map = np.zeros((ncells, ncells, 5))

    antxyz, N_ant = subarray_type_to_antxyz(subarray_type)
    
    if verbose: 
        print("Starting UV map generation...")

    # Iterate through time slices, performing incremental rotation and gridding
    for time_idx in tqdm(range(total_observations), disable=not verbose, desc="Gridding uv tracks"):
        if np.any([not time_idx*int_time%tt for tt in gain_timescale]):
            Nbase, N_ant = from_antenna_config_with_gains(antxyz, z, gain_model=gain_model)

        # Apply incremental rotation for the time step
        rotated_Nbase = earth_rotation_effect(Nbase[:,:3], time_idx, int_time, declination)

        # Grid the rotated baselines with gain values
        grid_uv_tracks_with_gains(rotated_Nbase, Nbase[:,3:], gain_uv_map, z, ncells,
                                  boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    if verbose:
        print("UV map generation complete.")

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
    gain_vals = gain_vals[in_bounds,:]
    gain_rr, gain_ii = gain_vals[:,0], gain_vals[:,1]
    gain_ri, gain_ir = gain_vals[:,2], gain_vals[:,3]
    Nb = Nb[in_bounds]
    
    for (x, y), gain1, gain2, gain3, gain4 in zip(Nb[:, :2], gain_rr, gain_ii, gain_ri, gain_ir):
        gain_uv_map[int(x), int(y), 0] += 1
        gain_uv_map[int(x), int(y), 1] += gain1
        gain_uv_map[int(x), int(y), 2] += gain2
        gain_uv_map[int(x), int(y), 3] += gain3
        gain_uv_map[int(x), int(y), 4] += gain4
        if include_mirror_baselines:
            gain_uv_map[-int(x), -int(y), 0] += 1
            gain_uv_map[-int(x), -int(y), 1] += gain1
            gain_uv_map[-int(x), -int(y), 2] += gain2
            gain_uv_map[-int(x), -int(y), 3] += gain3
            gain_uv_map[-int(x), -int(y), 4] += gain4

def gain_uv_map_to_uv_map(gain_uv_map):
    return gain_uv_map[:,:,0].astype(int)

def apply_uv_with_gains_response_on_image(array, gain_uv_map, verbose=True):
    """
    Apply the effect of radio observation strategy with varied antenna/baseline gains on an image.
    
    Parameters
    ----------
    array : ndarray
        The input image array.
    gain_uv_map : ndarray of lists
        The uv_map, where each entry is a list of gain values for that pixel.
    verbose : bool, optional
        If True, enables verbose progress output using tqdm.
        
    Returns
    -------
    img_map : ndarray
        The resulting radio image after applying the uv_map with gains in the Fourier domain.
    """
    assert array.shape == gain_uv_map[:,:,0].shape, "Array and uv_map must have the same shape"
    
    # Perform the Fourier transform of the input image
    arr_ft = np.fft.fft2(array)

    # Apply the games
    gain_applied_ft = arr_ft*np.sqrt(np.nan_to_num((gain_uv_map[:,:,1]+1.j*gain_uv_map[:,:,2])/gain_uv_map[:,:,0]))
    
    # Apply inverse Fourier transform to obtain the final image
    img_map = np.fft.ifft2(gain_applied_ft)
    
    # Return the real part of the transformed image
    return np.real(img_map)

def from_antenna_config_with_antenna_stamp(antxyz, z, nu=None):
    """
    The function reads the antenna positions (N_ant antennas) from the file given.

    Parameters
    ----------
    subarray_type: str
		The name of the SKA-Low layout configuration.
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
    if antxyz is None: 
        antll  = SKA1_LowConfig_Sept2016()
        antll  = antll[:,-2:].astype(float)
        antxyz = geographic_to_cartesian_coordinate_system(antll)
    else:
        antxyz = antxyz.to('m').value
    if not nu: 
        nu = cm.z_to_nu(z)                                 # MHz

    N_ant = antxyz.shape[0]
    pair_comb = itertools.combinations(range(N_ant), 2)
    pair_comb = list(pair_comb)	
    lam = c_light_cgs/(nu*1e6)/1e2 			            # in m
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


def process_chunk_get_full_uv_map_with_antenna_stamp(chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress):
    """
    Process a chunk of time slices.

    Parameters
    ----------
    chunk_start : int
        Start index of the chunk.
    chunk_end : int
        End index of the chunk.
    ncells : int
        Number of cells in each dimension of the grid.
    z : float
        Redshift.
    Nbase : ndarray
        Array containing ux, uy, uz values of the antenna configuration.
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
    show_progress : bool
        If True, shows the progress bar.

    Returns
    -------
    ant_tag_uv_map_chunk : ndarray
        Array of lists, each containing gain values for the pixel for the chunk.
    """
    ant_tag_uv_map_chunk = np.empty((chunk_end - chunk_start, ncells, ncells), dtype=object)
    for t in range(chunk_start, chunk_end):
        for x in range(ncells):
            for y in range(ncells):
                ant_tag_uv_map_chunk[t - chunk_start, x, y] = []

    time_indices = range(chunk_start, chunk_end)
    if show_progress:
        time_indices = tqdm(time_indices, desc="Gridding uv tracks", disable=not verbose)

    for time_idx in time_indices:
        rotated_Nbase = earth_rotation_effect(Nbase[:, :3], time_idx, int_time, declination)
        grid_uv_tracks_with_antenna_stamp(rotated_Nbase, Nbase[:, 3:], ant_tag_uv_map_chunk, z, ncells, time_idx - chunk_start,
                                          boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    return ant_tag_uv_map_chunk

def get_full_uv_map_with_antenna_stamp(ncells, z, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30.,
                                       include_mirror_baselines=False, verbose=True, n_jobs=-1):
    """
    Create the gain and uv maps with individual gain values for each baseline stored per pixel.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the grid.
    z : float
        Redshift.
    subarray_type: str
		The name of the SKA-Low layout configuration.
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
    n_jobs : int
        Number of parallel jobs to run. -1 means using all processors.

    Returns
    -------
    uv_map : ndarray
        Array of lists, each containing gain values for the pixel.
    N_ant : int
        Number of antennas.
    """
    antxyz, N_ant = subarray_type_to_antxyz(subarray_type)

    total_observations = int(3600. * total_int_time / int_time)
    if total_observations*ncells**2>6e6:
        print('CAUTION: This setup will create a huge array that the memory might struggle to handle.')

    if verbose:
        print("Loading antenna configuration with antenna stamps...")
    Nbase, N_ant = from_antenna_config_with_antenna_stamp(antxyz, z)

    if n_jobs == -1:
        n_jobs = cpu_count()

    if verbose:
        print(f"Starting UV map generation on {n_jobs} CPUs...")

    # Split the workload into chunks
    chunk_size = total_observations // n_jobs
    chunks = [(i, min(i + chunk_size, total_observations)) for i in range(0, total_observations, chunk_size)]

    # Parallel processing with progress bar for the first chunk
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk_get_full_uv_map_with_antenna_stamp)(
            chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress=(i == 0))
            for i, (chunk_start, chunk_end) in enumerate(chunks)
    )

    # Combine the results
    ant_tag_uv_map = np.concatenate(results, axis=0)

    if verbose:
        print("UV map generation complete.")

    return ant_tag_uv_map, N_ant

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
    
    for (x, y), ant1, ant2 in zip(Nb[:, :2].astype(int), ant_tag1.astype(int), ant_tag2.astype(int)):
        ant_tag_uv_map[time_idx, x, y].append([ant1, ant2])
        if include_mirror_baselines:
            print('include_mirror_baselines is not implemented yet.')

def full_uv_map_with_antenna_stamp_to_uv_map(ant_tag_uv_map, verbose=True):
    uv_map = np.zeros_like(ant_tag_uv_map[0,:,:])
    for (i, j), gains in tqdm(np.ndenumerate(uv_map), total=uv_map.size, disable=not verbose):
        uv_map[i, j] = np.array([np.array(val).shape[0] for val in ant_tag_uv_map[:,i,j]]).sum()
    return uv_map.astype(int)