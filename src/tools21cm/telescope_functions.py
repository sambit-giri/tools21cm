import numpy as np
import itertools
import sys
from .usefuls import *
from . import cosmo as cm
from . import conv
from .radio_telescope_layout import *
from tqdm import tqdm
import time
#from skimage.transform import rescale, resize, downscale_local_mean

KB_SI   = 1.38e-23
c_light = 2.99792458e+10  #in cm/s
janskytowatt = 1e-26

def from_antenna_config(filename, z, nu=None):
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
	if not nu: nu = cm.z_to_nu(z)                           # MHz
	antxyz = geographic_to_cartesian_coordinate_system(antll)
	N_ant = antll.shape[0]
	pair_comb = itertools.combinations(range(N_ant), 2)
	pair_comb = list(pair_comb)	
	lam = c_light/(nu*1e6)/1e2 			            # in m
	Nbase = []
	for ii,jj in list(pair_comb):
		ux = (antxyz[ii,0]-antxyz[jj,0])/lam
		uy = (antxyz[ii,1]-antxyz[jj,1])/lam
		uz = (antxyz[ii,2]-antxyz[jj,2])/lam
		if ux==0: print(ii,jj)
		Nbase.append([ux,uy,uz])
	Nbase = np.array(Nbase)	
	return Nbase, N_ant

def earth_rotation_effect(Nbase, slice_nums, int_time, declination=-30.):
    """
    Computes the effect of Earth's rotation over observation times, adjusting the measured 
    sky coordinates for each antenna configuration at specified time slices.

    Parameters
    ----------
    Nbase       : ndarray
        Array containing the initial ux, uy, uz values of the antenna configuration.
    slice_nums  : ndarray or int
        Array of observed slice numbers corresponding to each integration time, 
        or a single integer slice number if only one is required.
    int_time    : float
        Integration time after which the signal is recorded (in seconds).
    declination : float, optional
        Declination angle (latitude) where the telescope is located (in degrees).
        Default is -30 degrees.

    Returns
    -------
    new_Nbase   : ndarray
        Adjusted Nbase values for each time slice, with shape (len(slice_nums), N, 3).
    """
    # Convert to radians
    p = np.pi / 180.
    delta = p * declination
    
    # Ensure slice_nums is an array for vectorized handling of multiple slices
    slice_nums = np.atleast_1d(slice_nums)
    
    # Compute hour angles (HA) for each slice_num in vectorized manner
    HA = -15.0 * p * (slice_nums - 1) * int_time / 3600. - np.pi / 2 + 2 * np.pi

    # Rotation matrices for each HA and declination delta
    cos_HA, sin_HA = np.cos(HA), np.sin(HA)
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)

    # Pre-allocate new_Nbase for each time slice
    new_Nbase = np.empty((len(slice_nums), *Nbase.shape))
    
    # Apply rotation transformations
    new_Nbase[:, :, 0] = sin_HA[:, None] * Nbase[:, 0] + cos_HA[:, None] * Nbase[:, 1]
    new_Nbase[:, :, 1] = (-sin_delta * cos_HA[:, None] * Nbase[:, 0] +
                          sin_delta * sin_HA[:, None] * Nbase[:, 1] + cos_delta * Nbase[:, 2])
    new_Nbase[:, :, 2] = (cos_delta * cos_HA[:, None] * Nbase[:, 0] -
                          cos_delta * sin_HA[:, None] * Nbase[:, 1] + sin_delta * Nbase[:, 2])
    
    return new_Nbase.squeeze() if len(slice_nums) == 1 else new_Nbase

def get_uv_daily_observation(ncells, z, filename=None, total_int_time=4., int_time=10., boxsize=None, declination=-30., include_mirror_baselines=False, verbose=True):
    """
    Simulates daily radio observations and generates a uv map based on antenna configurations.

    Parameters
    ----------
    ncells         : int
        Number of cells in the image grid.
    z              : float
        Redshift of the observed slice.
    filename       : str
        Name of the file containing antenna configurations.
    total_int_time : float
        Total observation time per day in hours.
    int_time       : float
        Integration time per observation in seconds.
    boxsize        : float, optional
        Comoving size of the sky observed.
    declination    : float
        Declination angle of the SKA in degrees.
    include_mirror_baselines : bool, optional
        If True, includes mirrored baselines on the grid.
    verbose        : bool, optional
        If True, prints progress information.

    Returns
    -------
    uv_map : ndarray
        ncells x ncells array of baseline counts per pixel, averaged over total observations.
    N_ant  : int
        Number of antennas.
    """
    # Load antenna configurations and initialize parameters
    Nbase, N_ant = from_antenna_config(filename, z)
    uv_map = np.zeros((ncells, ncells), dtype=np.float32)
    total_observations = int((total_int_time * 3600) / int_time)
    
    if verbose: 
        print("Generating uv map from daily observations...")
    
    # Vectorize the observation loop by calculating all rotations at once
    time_indices = np.arange(total_observations) + 1
    all_rotated_Nbase = (earth_rotation_effect(Nbase, i, int_time, declination) for i in time_indices)
    
    # Grid uv tracks for each observation without individual loops
    for rotated_Nbase in tqdm(all_rotated_Nbase, disable=not verbose, desc="Gridding uv tracks"):
        uv_map += grid_uv_tracks(rotated_Nbase, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
    
    uv_map /= total_observations  # Normalize by the total number of observations
    
    if verbose:
        print("...done")

    return uv_map, N_ant

def grid_uv_tracks(Nbase, z, ncells, boxsize=None, include_mirror_baselines=False):
    """
    Places uv tracks on a grid.

    Parameters
    ----------
    Nbase   : ndarray
        Array containing ux, uy, uz values of the antenna configuration.
    z       : float
        Redshift of the slice observed.
    ncells  : int
        Number of cells in the image.
    boxsize : float, optional
        Comoving size of the sky observed. Default: determined by simulation constants.
    include_mirror_baselines : bool, optional
        If True, includes mirrored baselines on the grid.

    Returns
    -------
    uv_map  : ndarray
        ncells x ncells array of baseline counts per pixel.
    """
    z = float(z)
    if not boxsize: 
        boxsize = conv.LB  # Assuming conv.LB is defined elsewhere in your code

    uv_map = np.zeros((ncells, ncells), dtype=np.int32)  # Using int32 for faster summing operations
    theta_max = boxsize / cm.z_to_cdist(z)
    
    # Scale and round the coordinates in a single operation for efficiency
    Nb = np.round(Nbase * theta_max).astype(int)

    # Vectorized conditionals to select points within grid bounds, skipping for-loops
    mask = (
        (Nb[:, 0] < ncells / 2) & (Nb[:, 0] >= -ncells / 2) &
        (Nb[:, 1] < ncells / 2) & (Nb[:, 1] >= -ncells / 2)
    )
    Nb = Nb[mask]
    
    # Convert from negative to positive indexing within the grid and use broadcasting
    xx = (Nb[:, 0] + ncells // 2).astype(int)
    yy = (Nb[:, 1] + ncells // 2).astype(int)
    
    # Fast counting using numpy's bincount and reshaping
    uv_map_flat = np.bincount(xx * ncells + yy, minlength=ncells * ncells)
    uv_map = uv_map_flat.reshape(ncells, ncells)

    # Include mirrored baselines if requested
    if include_mirror_baselines:
        uv_map += np.flip(np.flip(uv_map, axis=0), axis=1)  # Flip along both axes for mirror effect
        uv_map /= 2  # Average with mirrored baselines
    
    return np.fft.fftshift(uv_map)

def sigma_noise_radio(z, uv_map, depth_mhz, obs_time, int_time, N_ant=564., verbose=True, T_sys=None):
    """
    Calculate the rms of the noise added by radio interferometers.

    Parameters
    ----------
    z : float
        Redshift of the slice observed.
    uv_map : ndarray
        ncells x ncells numpy array containing the number of baselines observing each pixel.
    depth_mhz : float
        The bandwidth of the observation (in MHz).
    obs_time : float
        The total hours of observation time.
    int_time : float
        The integration time.
    N_ant : float, optional
        Number of antennas in SKA. Default is 564.
    verbose : bool, optional
        If True, print detailed information. Default is True.
    T_sys : callable or None, optional
        System temperature as a function of frequency. If None, a default model is used.

    Returns
    -------
    sigma : float
        The rms of the noise in the image produced by SKA for uniformly distributed antennas (in µJy).
    rms_noise : float
        The rms of the noise due to the antenna positions in the uv field (in µJy).
    """
    z = float(z)
    nuso = 1420.0 / (1.0 + z)
    delnu = depth_mhz * 1e3  # in kHz
    effective_baseline = np.sum(uv_map)

    if T_sys is None:
        # Standard definition of sky temperature
        T_sky_atnu300MHz = 60.0  # K
        T_sky = lambda nu: T_sky_atnu300MHz * (300.0 / nu) ** 2.55
        T_rcvr = 100  # K
        T_sys = lambda nu: T_sky(nu) + T_rcvr

    try:
        T_sys = T_sys(nuso)
    except:
        pass

    ant_radius_ska = 35.0 / 2.0  # in m
    nu_crit = 110.0  # in MHz
    ep = (nu_crit / nuso) ** 2 if nuso > nu_crit else 1.0
    A_ant_ska = ep * np.pi * ant_radius_ska ** 2

    KB_SI = 1.380649e-23  # Boltzmann constant in SI units
    janskytowatt = 1e-26  # Conversion factor from Jansky to Watts

    rms_noise = (1e6 * np.sqrt(2) * KB_SI * T_sys / A_ant_ska /
                 np.sqrt(depth_mhz * 1e6 * int_time) / janskytowatt)  # in µJy
    sigma = (rms_noise / np.sqrt(N_ant * (N_ant - 1) / 2.0) /
             np.sqrt(3600 * obs_time / int_time))  # in µJy

    if verbose:
        print('\nExpected: rms in image in µJy per beam for full =', sigma)
        print('Effective baseline =', sigma * np.sqrt(N_ant * N_ant / 2.0) / np.sqrt(effective_baseline), 'm')
        print('Calculated: rms in the visibility =', rms_noise, 'µJy')

    return sigma, rms_noise

def apply_uv_response(array, uv_map):
	"""
	Parameters
	----------
	array     : A complex 2d array of signal in the uv field.
	uv_map    : Numpy array containing the number of baselines observing each pixel.
	Returns 
	----------
	new_array : It is the 'array' after degrading the resoltion with the baseline configuration.
	"""
	noise_real = np.real(array)
	noise_imag = np.imag(array)
	noise_four = np.zeros(noise_real.shape)+1.j*np.zeros(noise_real.shape)
	ncells     = noise_real.shape[0]
	for i in range(ncells):
		for j in range(ncells):
			if uv_map[i,j] == 0: noise_four[i,j] = 0
			else: noise_four[i,j] = noise_real[i,j]/np.sqrt(uv_map[i,j]) + 1.j*noise_imag[i,j]/np.sqrt(uv_map[i,j])
	return noise_four


def kelvin_jansky_conversion(ncells, z, boxsize=None):
	"""
	Parameters
	----------
	ncells  : int
		Number of cells/pixels in the image.
	z       : float
		Redshift
	boxsize : float
		The comoving size of the sky observed. Default: It is determined from the simulation constants set.	

	Returns
	-------
	The conversion factor multiplied to values in kelvin to get values in jansky.
	"""
	z = float(z)
	if not boxsize: boxsize = conv.LB
	KB_SI       = 1.38e-23
	janskytowatt= 1e-26
	dist_z      = cm.z_to_cdist(z)
	boxsize_pp  = boxsize/dist_z				 #in rad	
	omega_pixel = boxsize_pp**2/ncells**2
	omega_total = boxsize_pp**2.0
	c_light_SI  = 2.99792458e+8                              #in m
	mktomujy_nuc= 2.0*KB_SI/c_light_SI/c_light_SI/janskytowatt*((cm.z_to_nu(z)*1e6)**2.0)*1e3
	con_sol     = mktomujy_nuc*omega_pixel
	return con_sol

def jansky_2_kelvin(array, z, boxsize=None, ncells=None):
	"""
	Parameters
	----------
	array   : ndarray
		Numpy array containing the values in jansky.
	z       : float
		Redshift
	boxsize : float
		The comoving size of the sky observed. Default: It is determined from the simulation constants set.
	ncells  : int
		The number of grid cells. Default: None
	
	Returns
	-------
	A numpy array with values in mK.
	"""
	z = float(z)
	if not ncells: ncells  = array.shape[0]
	con_sol = kelvin_jansky_conversion(ncells, z, boxsize=boxsize)	
	return  array/con_sol

def kelvin_2_jansky(array, z, boxsize=None, ncells=None):
	"""
	Parameters
	----------
	array   : ndarray
		Numpy array containing the values in mK.
	z       : float
		Redshift
	boxsize : float
		The comoving size of the sky observed. Default: It is determined from the simulation constants set.

	Returns
	-------
	A numpy array with values in Jy.
	"""
	z = float(z)
	if not ncells: ncells  = array.shape[0]
	con_sol = kelvin_jansky_conversion(ncells, z, boxsize=boxsize)	
	return  array*con_sol
