import numpy as np
from .usefuls import *
from .scipy_func import *
from . import cosmo as cm
from . import conv
from .const import KB_SI, c_light_cgs, c_light_SI, janskytowatt
from .radio_telescope_layout import *
from .read_files import get_package_resource_path

def get_SEFD(nu_obs, T_sys=None, sefd_data=None, nu_data=None, D_station=40., ep_aperture=None):
	"""Calculates the System Equivalent Flux Density (SEFD) for a radio antenna.

	This function can operate in two modes:
	1.  **Interpolation**: If `sefd_data` and `nu_data` are provided, it
		interpolates from the given data to find the SEFD at `nu_obs`.
	2.  **Direct Calculation**: If `sefd_data` is not provided, it calculates
		the SEFD from fundamental parameters like system temperature (`T_sys`),
		station diameter (`D_station`), and aperture efficiency (`ep_aperture`).

	Parameters
	----------
	nu_obs : float or np.ndarray
		Observing frequency or frequencies in MHz.
	T_sys : float, callable, or None, optional
		System temperature in Kelvin. Can be a single float value or a
		callable function of frequency in MHz. If `None`, a default model
		approximating the SKA-Low sky temperature plus a receiver
		temperature is used. Default is `None`.
	sefd_data : np.ndarray or None, optional
		An array of known SEFD values for interpolation. If this is provided,
		`nu_data` must also be given. Default is `None`.
	nu_data : np.ndarray or None, optional
		An array of frequencies in MHz corresponding to `sefd_data`.
		Default is `None`.
	D_station : float, optional
		Diameter of the antenna station in meters. Default is 40.0.
	ep_aperture : float, callable, or None, optional
		Aperture efficiency. Can be a single float value or a callable
		function of frequency in MHz. If `None`, a default frequency-dependent
		model is used. Default is `None`.

	Returns
	-------
	sefd : float or np.ndarray
		The calculated System Equivalent Flux Density in Janskys (Jy).
		Returns a float or array matching the shape of `nu_obs`.
	"""
	if isinstance(sefd_data, str):
		if sefd_data.upper() in ['SKA1-LOW', 'SKA1', 'SKAO-TEL-0000818']:
			sefd_filename = get_package_resource_path('tools21cm', 'input_data/SEFD_SKAO-TEL-0000818-V2_SKA1.txt')
			table_data = np.loadtxt(sefd_filename)
			nu_data = table_data[:,0]
			sefd_data = table_data[:,2]

	if sefd_data is not None:
		if nu_data is None:
			raise ValueError("If sefd_data is provided, nu_data must also be provided.")
		log10_sefd_fct = interp1d(nu_data, np.log10(sefd_data), fill_value='extrapolate')
		return 10**log10_sefd_fct(nu_obs)

	nu_obs = np.atleast_1d(nu_obs)

	# Default system temperature model
	if T_sys is None:
		T_sky = lambda nu: 60.0 * (300.0 / nu) ** 2.55  # K
		T_rcvr = 100.0  # K
		T_sys = lambda nu: T_sky(nu) + T_rcvr

	# If T_sys is a function, evaluate it; otherwise, use the provided value
	T_sys_val = T_sys(nu_obs) if callable(T_sys) else T_sys

	# Antenna effective area
	ant_radius_ska = D_station / 2.0  # m
	# Default aperture efficiency model
	if ep_aperture is None:
		nu_crit = 110.0  # MHz
		ep_aperture_func = lambda nu: np.where(nu > nu_crit, (nu_crit / nu) ** 2, 1.0)
		ep_aperture_val = ep_aperture_func(nu_obs)
	# If ep_aperture is a function, evaluate it; otherwise, use the provided value
	elif callable(ep_aperture):
		ep_aperture_val = ep_aperture(nu_obs)
	else:
		ep_aperture_val = ep_aperture
	A_eff = ep_aperture_val * np.pi * ant_radius_ska ** 2  # m^2

	# SEFD in Jy
	sefd = 2 * KB_SI * T_sys_val / A_eff / janskytowatt  # Jy

	# Return a scalar if input was scalar
	return sefd[0] if nu_obs.size == 1 else sefd

def sigma_noise_radio(z, depth_mhz, obs_time, int_time, uv_map=None, N_ant=512, verbose=True, 
					  T_sys=None, sefd_data=None, nu_data=None, D_station=40., ep_aperture=None):
	"""
	Calculate the rms of the noise added by radio interferometers.

	Parameters
	----------
	z : float
		Redshift of the slice observed.
	depth_mhz : float
		The bandwidth of the observation (in MHz).
	obs_time : float
		The total hours of observation time.
	int_time : float
		The integration time.
	uv_map : ndarray
		ncells x ncells numpy array containing the number of baselines observing each pixel.
	N_ant : float, optional
		Number of antennas in SKA. Default is 564.
	verbose : bool, optional
		If True, print detailed information. Default is True.
	T_sys : float, callable, or None, optional
        System temperature in Kelvin. Can be a single float value or a
        callable function of frequency in MHz. If `None`, a default model
        approximating the SKA-Low sky temperature plus a receiver
        temperature is used. Default is `None`.
    sefd_data : np.ndarray or None, optional
        An array of known SEFD values for interpolation. If this is provided,
        `nu_data` must also be given. Default is `None`.
    nu_data : np.ndarray or None, optional
        An array of frequencies in MHz corresponding to `sefd_data`.
        Default is `None`.
    D_station : float, optional
        Diameter of the antenna station in meters. Default is 40.0.
    ep_aperture : float, callable, or None, optional
        Aperture efficiency. Can be a single float value or a callable
        function of frequency in MHz. If `None`, a default frequency-dependent
        model is used. Default is `None`.

	Returns
	-------
	sigma : float
		The rms of the noise in the image produced by SKA for uniformly distributed antennas (in µJy).
	rms_noise : float
		The rms of the noise due to the antenna positions in the uv field (in µJy).
	"""
	z = float(z)
	nuso = 1420.0 / (1.0 + z)

	# Compute SEFD
	sefd = get_SEFD(nuso, T_sys, sefd_data=sefd_data, nu_data=nu_data, D_station=D_station, ep_aperture=ep_aperture)  # Jy

	# RMS noise per visibility (converted to µJy)
	rms_noise = 1e6 * sefd / np.sqrt(2 * depth_mhz * 1e6 * int_time)  # µJy

	# Final map noise (µJy/beam)
	sigma = (rms_noise / np.sqrt(N_ant * (N_ant - 1) / 2.0) /
				np.sqrt(3600 * obs_time / int_time))  # µJy

	if verbose:
		print('\nExpected: rms in image in µJy per beam for full =', sigma)
		if uv_map is not None:
			effective_baseline = np.sum(uv_map)
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
	dist_z      = cm.z_to_cdist(z)
	boxsize_pp  = boxsize/dist_z				 #in rad	
	omega_pixel = boxsize_pp**2/ncells**2
	omega_total = boxsize_pp**2.0
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
