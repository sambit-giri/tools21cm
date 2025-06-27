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
from glob import glob
from time import time, sleep
import pickle
from joblib import cpu_count, Parallel, delayed
from tqdm import tqdm

def from_antenna_config_with_gains(antxyz, z, nu=None, 
                gain_model={'name': 'random_uniform', 'min': 0.5, 'max': 1.1}):
    """
    Calculates baselines and complex gain products for an antenna array.

    This function takes antenna positions, generates complex gain values for each
    antenna based on a specified model, and then computes the baseline vectors
    (u,v,w). For each baseline, it also calculates the four products of the
    complex gains (g_i * g_j*) which are g_r*g_r, g_i*g_i, g_r*g_i, g_i*g_r.

    Parameters
    ----------
    antxyz: astropy.Quantity
        An object with antenna positions, expected to have a `.to('m').value` method.
    z : float
        Redshift of the slice observed.
    nu : float, optional
        The frequency observed by the telescope in MHz. If None, it's calculated from z.
    gain_model : dict or numpy.ndarray, optional
        Specifies how to generate antenna gains. Can be a dict defining a
        distribution ('random_gaussian' or 'random_uniform') or a pre-computed
        array of complex gains.

    Returns
    -------
    Nbase : numpy.ndarray
        Array of shape `(n_baselines, 7)` containing `(u,v,w)` and the four
        gain product components for each baseline.
    N_ant : int
        The total number of antennas in the configuration.
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
    Creates a UV map where each grid cell accumulates complex gain information.

    This function simulates a radio observation over time, accounting for
    evolving instrumental gains. It iterates through time steps, periodically
    re-calculating antenna gains, applying Earth rotation, and gridding the
    resulting baseline tracks. The output map stores the number of hits and the
    sum of four gain products for each UV cell.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the grid.
    z : float
        Redshift of the observation.
    gain_model : dict or function
        Model for generating antenna gains. Passed to `from_antenna_config_with_gains`.
    gain_timescale : list of int
        Timescales (in seconds) at which gain values are re-calculated.
    subarray_type : str
        The name of the telescope layout configuration (e.g., "AA4").
    total_int_time : float
        Total observation time, in hours.
    int_time : float
        Integration time per snapshot, in seconds.
    declination : float
        Declination of the pointing center, in degrees.
    boxsize : float, optional
        Comoving size of the observed sky area, in Mpc.
    include_mirror_baselines : bool
        If True, grids both (u,v) and (-u,-v) tracks.
    verbose : bool
        If True, enables progress bars and informational messages.

    Returns
    -------
    gain_uv_map : numpy.ndarray
        A 3D array of shape `(ncells, ncells, 5)`. For each UV cell (i,j),
        `gain_uv_map[i,j,0]` is the hit count, and the other 4 channels are
        the accumulated sums of the gain products.
    N_ant : int
        The total number of antennas.
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
        _grid_uv_tracks_with_gains(rotated_Nbase, Nbase[:,3:], gain_uv_map, z, ncells,
                                  boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    if verbose:
        print("UV map generation complete.")

    return gain_uv_map, N_ant

def _grid_uv_tracks_with_gains(Nbase, gain_vals, gain_uv_map, z, ncells, boxsize=None, include_mirror_baselines=False, verbose=True):
    """
    Grids UV tracks and accumulates gain values for a single time snapshot.

    This function projects rotated baseline coordinates onto a 2D grid. For each
    grid cell, it increments a hit counter and adds the four gain product
    components of any baseline that falls into that cell.

    Parameters
    ----------
    Nbase : numpy.ndarray
        Array `(n_baselines, 3)` of rotated (u, v, w) coordinates.
    gain_vals : numpy.ndarray
        Array `(n_baselines, 4)` of gain products for each baseline.
    gain_uv_map : numpy.ndarray
        The 3D output array `(ncells, ncells, 5)` to be modified in-place.
    z : float
        Redshift of the observation slice.
    ncells : int
        Number of cells in one dimension of the target grid.
    boxsize : float, optional
        Comoving size of the sky observed in Mpc.
    include_mirror_baselines : bool, optional
        If True, also grids the hermitian conjugate baselines at (-u, -v).

    Returns
    -------
    None
        Modifies `gain_uv_map` in-place.
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
    Applies the instrumental response, including gains, to a sky image.

    This function simulates the effect of observing a true sky image with an
    interferometer that has complex antenna gains. It operates in the Fourier
    domain by multiplying the Fourier transform of the sky by an effective
    complex gain derived from the simulated observation.

    Parameters
    ----------
    array : numpy.ndarray
        The 2D input sky image.
    gain_uv_map : numpy.ndarray
        The 3D UV map `(ncells, ncells, 5)` from `get_uv_map_with_gains`.
    verbose : bool, optional
        (Currently unused).

    Returns
    -------
    img_map : numpy.ndarray
        The resulting 2D "dirty" image after applying the instrumental response.
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
    Calculates baselines and attaches unique integer tags for each antenna.

    This function converts antenna positions into baseline vectors (u,v,w) and,
    for each baseline, stores the integer tags of the two antennas that form it.

    Parameters
    ----------
    antxyz: astropy.Quantity
        An object with antenna positions, expected to have a `.to('m').value` method.
    z : float
        Redshift of the slice observed.
    nu : float, optional
        The frequency observed by the telescope in MHz. If None, it's calculated from z.

    Returns
    -------
    Nbase : numpy.ndarray
        Array `(n_baselines, 5)` containing `(u,v,w, ant_tag1, ant_tag2)`.
    N_ant : int
        The total number of antennas.
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

## antenna stamps in uv space ##

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
        delayed(_process_chunk_get_full_uv_map_with_antenna_stamp)(
            chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress=(i == 0))
            for i, (chunk_start, chunk_end) in enumerate(chunks)
    )

    # Combine the results
    ant_tag_uv_map = np.concatenate(results, axis=0)

    if verbose:
        print("UV map generation complete.")

    return ant_tag_uv_map, N_ant

def _process_chunk_get_full_uv_map_with_antenna_stamp(chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress):
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
        _grid_uv_tracks_with_antenna_stamp(rotated_Nbase, Nbase[:, 3:], ant_tag_uv_map_chunk, z, ncells, time_idx - chunk_start,
                                          boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    return ant_tag_uv_map_chunk

def _grid_uv_tracks_with_antenna_stamp(Nbase, ant_tag, ant_tag_uv_map, z, ncells, time_idx,
                                      boxsize=None, include_mirror_baselines=False, verbose=True):
    """
    Grids UV tracks with antenna tags onto a 2D grid for a single time snapshot.

    This function takes the rotated baseline coordinates for a single moment,
    projects them onto a 2D grid, and for each grid cell (pixel), it appends
    the antenna pair tags of all baselines that fall into that cell.

    Parameters
    ----------
    Nbase : numpy.ndarray
        Array of shape `(n_baselines, 3)` containing the rotated (u, v, w)
        baseline coordinates for a single time step.
    ant_tag : numpy.ndarray
        Array of shape `(n_baselines, 2)` containing the integer tags for
        each antenna pair.
    ant_tag_uv_map : numpy.ndarray
        The 3D output array of shape `(n_timesteps, ncells, ncells)` where each
        element is a list. This function appends `[ant1, ant2]` pairs to these
        lists. It is modified in-place.
    z : float
        Redshift of the observation slice.
    ncells : int
        Number of cells in one dimension of the target grid.
    time_idx : int
        The time index within the `ant_tag_uv_map` to populate.
    boxsize : float, optional
        Comoving size of the sky observed in Mpc. Defaults to a predefined
        constant if None.
    include_mirror_baselines : bool, optional
        If True, includes mirror baselines. (Note: Not yet implemented).
    verbose : bool, optional
        Enables verbose output. (Note: Currently unused in this function).

    Returns
    -------
    None
        Modifies `ant_tag_uv_map` in-place.
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

### uv locations for every antenna stamp combination (Lagrangian space) ###

def get_full_uv_lagrangian_with_antenna_stamp(ncells, z, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30.,
                                       include_mirror_baselines=False, verbose=True, n_jobs=-1):
    """
    Creates a time-ordered map of gridded UV coordinates for each baseline.

    This function simulates radio interferometer observations by calculating the
    UV tracks for each antenna baseline over a specified total observation time.
    It parallelizes the computation over the time steps and returns a 3D array
    mapping each baseline to its integer (x, y) grid coordinate at each
    time step, along with the list of antenna pairs.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the simulation grid.
    z : float
        Redshift of the observation.
    subarray_type : str, optional
        The name of the SKA-Low layout configuration (e.g., "AA4").
    total_int_time : float, optional
        Total observation time per day, in hours.
    int_time : float, optional
        Integration time for each snapshot, in seconds.
    boxsize : float, optional
        Comoving size of the observed sky area, in Mpc. If None, a default
        is used.
    declination : float, optional
        Declination of the pointing center, in degrees.
    include_mirror_baselines : bool, optional
        Whether to include mirror baselines (u,v) -> (-u,-v).
    verbose : bool, optional
        If True, enables progress bars and informational messages.
    n_jobs : int, optional
        Number of parallel jobs to run. -1 uses all available CPUs.

    Returns
    -------
    ant_tag_uv_lagr : numpy.ndarray
        A 3D array of shape `(n_observations, n_baselines, 2)` containing the
        integer grid coordinates `(x, y)` for each baseline at each time step.
        Values can be NaN if a baseline falls outside the grid.
    ant_pairs : numpy.ndarray
        A 2D array of shape `(n_baselines, 2)` where each row contains the
        integer tags of the two antennas forming the corresponding baseline.
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
        delayed(_process_chunk_get_full_uv_lagrangian_with_antenna_stamp)(
            chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress=(i == 0))
            for i, (chunk_start, chunk_end) in enumerate(chunks)
    )

    # Combine the results
    ant_tag_uv_lagr = np.concatenate(results, axis=0)

    if verbose:
        print("UV position generation complete.")

    ant_pairs = Nbase[:,-2:]
    return ant_tag_uv_lagr, ant_pairs

def _process_chunk_get_full_uv_lagrangian_with_antenna_stamp(chunk_start, chunk_end, ncells, z, Nbase, int_time, declination, boxsize, include_mirror_baselines, verbose, show_progress):
    """
    Processes a chunk of observation time slices for parallel computation.

    This is a worker function for `get_full_uv_lagrangian_with_antenna_stamp`.
    It iterates through a specified range of time steps, applies Earth rotation
    to the baseline coordinates, and grids the resulting UV tracks.

    Parameters
    ----------
    chunk_start : int
        The starting time index for this chunk.
    chunk_end : int
        The ending time index (exclusive) for this chunk.
    ncells : int
        Number of cells in each dimension of the simulation grid.
    z : float
        Redshift of the observation.
    Nbase : numpy.ndarray
        Array containing initial baseline vectors (ux, uy, uz) and antenna tags.
    int_time : float, optional
        Integration time for each snapshot, in seconds.
    declination : float, optional
        Declination of the pointing center, in degrees.
    boxsize : float, optional
        Comoving size of the observed sky area, in Mpc.
    include_mirror_baselines : bool, optional
        Whether to include mirror baselines.
    verbose : bool, optional
        If True, enables progress bars and informational messages.
    show_progress : bool
        If True, shows a tqdm progress bar for this chunk.

    Returns
    -------
    ant_tag_uv_map_chunk : numpy.ndarray
        The portion of the UV coordinate map for the processed time chunk.
        A 3D array of shape `(chunk_end - chunk_start, n_baselines, 2)`.
    """
    ant_tag_uv_map_chunk = np.full((chunk_end - chunk_start, Nbase.shape[0], 2), np.nan)

    time_indices = range(chunk_start, chunk_end)
    if show_progress:
        time_indices = tqdm(time_indices, desc="Gridding uv tracks", disable=not verbose)

    for time_idx in time_indices:
        rotated_Nbase = earth_rotation_effect(Nbase[:, :3], time_idx, int_time, declination)
        _grid_uv_lagrangian_tracks_with_antenna_stamp(rotated_Nbase, Nbase[:, 3:], ant_tag_uv_map_chunk, z, ncells, time_idx - chunk_start,
                                          boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)

    return ant_tag_uv_map_chunk

def _grid_uv_lagrangian_tracks_with_antenna_stamp(Nbase, ant_tag, ant_tag_uv_map, z, ncells, time_idx,
                                      boxsize=None, include_mirror_baselines=False, verbose=True):
    """
    Grids UV tracks for a single time snapshot onto an integer grid.

    This function takes the rotated baseline coordinates for a single moment,
    projects them onto a 2D grid, filters out baselines that fall outside the
    grid boundaries, and stores the resulting integer (x, y) coordinates in the
    output array `ant_tag_uv_map`.

    Parameters
    ----------
    Nbase : numpy.ndarray
        Array of shape `(n_baselines, 3)` containing the rotated (u, v, w)
        baseline coordinates for a single time step.
    ant_tag : numpy.ndarray
        Array of shape `(n_baselines, 2)` containing the integer tags for
        each antenna pair.
    ant_tag_uv_map : numpy.ndarray
        The 3D output array of shape `(n_timesteps, n_baselines, 2)` that is
        populated by this function. It is modified in-place.
    z : float
        Redshift of the observation slice.
    ncells : int
        Number of cells in one dimension of the target grid.
    time_idx : int
        The time index within the `ant_tag_uv_map` to populate.
    boxsize : float, optional
        Comoving size of the sky observed in Mpc. Defaults to a predefined
        constant if None.
    include_mirror_baselines : bool, optional
        If True, includes mirror baselines. (Note: Not yet implemented).
    verbose : bool, optional
        Enables verbose output. (Note: Currently unused in this function).

    Returns
    -------
    None
        Modifies `ant_tag_uv_map` in-place.
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
    
    count = 0
    for (x, y), ant1, ant2 in zip(Nb[:, :2].astype(int), ant_tag1.astype(int), ant_tag2.astype(int)):
        ant_tag_uv_map[time_idx, count, 0] = x
        ant_tag_uv_map[time_idx, count, 1] = y
        count += 1
        if include_mirror_baselines:
            print('include_mirror_baselines is not implemented yet.')