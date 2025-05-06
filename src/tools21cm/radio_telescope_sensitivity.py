import numpy as np
import itertools
import sys
from .usefuls import *
from . import cosmo as cm
from . import conv
from .const import KB_SI, c_light_cgs, c_light_SI, janskytowatt
from .radio_telescope_layout import *
from tqdm import tqdm
import time

def get_SEFD(nu_obs, T_sys=None):
	"""
	Compute the System Equivalent Flux Density (SEFD) of radio antenna.

	Parameters
	----------
	nu_obs : float
		Observing frequency in MHz.
	T_sys : float or callable or None
		System temperature in Kelvin or a function of frequency (MHz). If None, a default model is used.

	Returns
	-------
	sefd : float
		System Equivalent Flux Density in Jy.
	"""
	nu_obs = np.atleast_1d(nu_obs)

	# Default system temperature model
	if T_sys is None:
		T_sky = lambda nu: 60.0 * (300.0 / nu) ** 2.55  # K
		T_rcvr = 100.0  # K
		T_sys = lambda nu: T_sky(nu) + T_rcvr

	# If T_sys is a function, evaluate it
	T_sys_val = T_sys(nu_obs) if callable(T_sys) else T_sys

	# Antenna effective area
	ant_radius_ska = 35.0 / 2.0  # m
	nu_crit = 110.0  # MHz
	ep = np.where(nu_obs > nu_crit, (nu_crit / nu_obs) ** 2, 1.0)
	A_eff = ep * np.pi * ant_radius_ska ** 2  # m^2

	# SEFD in Jy
	janskytowatt = 1e-26
	sefd = 2 * KB_SI * T_sys_val / A_eff / janskytowatt  # Jy
	return sefd

def sigma_noise_radio(z, depth_mhz, obs_time, int_time, uv_map=None, N_ant=512, verbose=True, T_sys=None):
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

	# Compute SEFD
	sefd = get_SEFD(nuso, T_sys)  # Jy

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
