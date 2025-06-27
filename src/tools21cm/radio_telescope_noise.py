'''
Methods:

* simulate the radio telescope observation strategy
* simulate telescope noise
'''

import numpy as np
import sys
from .radio_telescope_sensitivity import *
from .usefuls import *
from . import conv
from . import cosmo as cm
from . import smoothing as sm
from .power_spectrum import radial_average, _get_k, _get_dims
from .read_files import read_dictionary_data, write_dictionary_data
import scipy
from .scipy_func import *
from glob import glob
from time import time, sleep
import pickle
import astropy.units as u
from joblib import Parallel, delayed
from tqdm import tqdm

def signal_window(ncells, method, ndim=1, extra_param=30):
	"""
	Generates a 1D, 2D, or 3D window function for signal processing.

	Parameters
	----------
	ncells : int
		The size of the window along each dimension.
	method : str
		The name of the window function to use (e.g., 'blackmanharris', 'gaussian').
	ndim : int, optional
		The number of dimensions for the window (1, 2, or 3).
	extra_param : float, optional
		An additional parameter required by some window functions, like the
		standard deviation for 'gaussian' or beta for 'kaiser'.

	Returns
	-------
	np.ndarray or None
		The generated window array, or None if the method is not implemented
		or the dimension is not supported.
	"""
	if method.lower() in ['blackmanharris', 'blackman-harris']:
		win = windows.blackmanharris(ncells)
	elif method.lower() in ['blackman']:
		win = windows.blackman(ncells)
	elif method.lower() in ['barthann']:
		win = windows.barthann(ncells)
	elif method.lower() in ['bartlett']:
		win = windows.bartlett(ncells)
	elif method.lower() in ['gaussian']:
		win = windows.gaussian(ncells, extra_param)
	elif method.lower() in ['hamming']:
		win = windows.hamming(ncells)
	elif method.lower() in ['hann']:
		win = windows.hann(ncells)
	elif method.lower() in ['kaiser']:
		win = windows.kaiser(ncells, extra_param)
	else:
		print(f'{method} window function is not implemented')
		return None
	if ndim==1:
		pass
	elif ndim==2:
		win = (win[:,None] @ win[None,:])
	elif ndim==3:
		win = (win[:,None] @ win[None,:])[:,:,None] @ win[None,:]
	else:
		print(f'window dimension = {ndim} is not supported yet')
		return None 
	return win

def noise_coeval_power_spectrum_1d(ncells, z, depth_mhz, obs_time=1000, subarray_type="AA4", kbins=10, boxsize=None, binning='log', return_n_modes=False, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None, uv_weighting='natural', sefd_data=None, nu_data=None, fft_wrap=False, verbose=True):
	"""
	Computes the 1D power spectrum of instrumental noise for a coeval observation.

	This function simulates the instrumental noise properties in the Fourier domain
	based on the telescope configuration and observation strategy, and then computes
	the radially averaged 1D power spectrum.

	Parameters
	----------
	ncells : int
		The number of grid cells in each spatial dimension.
	z : float
		The redshift of the observation.
	depth_mhz : float
		The observational bandwidth in MHz.
	obs_time : float, optional
		The total on-source observation time in hours.
	subarray_type : str, optional
		The telescope layout configuration name.
	kbins : int, optional
		The number of bins for radial averaging of the power spectrum.
	boxsize : float, optional
		The comoving size of the simulation box in Mpc.
	binning : {'log', 'linear'}, optional
		The type of binning to use for k-modes.
	return_n_modes : bool, optional
		If True, also returns the number of modes in each k-bin.
	total_int_time : float, optional
		Total observation time per day in hours for UV coverage simulation.
	int_time : float, optional
		Integration time in seconds for UV coverage simulation.
	declination : float, optional
		The pointing declination in degrees.
	uv_map : np.ndarray, optional
		A pre-computed UV coverage map. If None, it will be simulated.
	N_ant : int, optional
		The number of antennas. If None, it will be determined.
	uv_weighting : {'natural', 'uniform'}, optional
		The weighting scheme to apply in the UV plane.
	sefd_data, nu_data : various, optional
		Data for SEFD calculation (passed to `sigma_noise_radio`).
	verbose : bool, optional
		If True, enables verbose output.

	Returns
	-------
	pn : np.ndarray
		The 1D noise power spectrum values.
	kn : np.ndarray
		The central k-mode values for each bin.
	n_modes : np.ndarray, optional
		The number of modes in each k-bin (if `return_n_modes` is True).
	"""
	if boxsize is None:
		boxsize = conv.LB

	antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)

	if uv_map is None: 
		uv_map, N_ant = get_uv_map(ncells, z, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if N_ant is None: 
		N_ant = antxyz.shape[0]

	sigma, rms_noi = sigma_noise_radio(z, depth_mhz, obs_time, int_time, uv_map=uv_map, N_ant=N_ant, verbose=False, sefd_data=sefd_data, nu_data=nu_data)
	box_dims = _get_dims(boxsize, uv_map.shape)
	k_nq = np.pi/boxsize*min(uv_map.shape)

	if uv_weighting.lower() in ['nat', 'natural', 'natural_weighting']:
		out = rms_noi/np.sqrt(uv_map)
	elif uv_weighting.lower() in ['uni', 'uniform', 'uniform_weighting']:
		out = rms_noi
	else:
		print(f'{uv_weighting} scheme is not known or implemented')

	power = np.fft.fftshift(out**2)
	# scale
	boxvol = numpy_product(box_dims)
	pixelsize = boxvol/(numpy_product(uv_map.shape))
	power *= pixelsize**2/boxvol
	pn, kn, n_modes = radial_average(power, box_dims, kbins=kbins, binning=binning)

	if return_n_modes:
		return pn, kn, n_modes
	return pn, kn

def get_uv_map_lightcone(ncells, zs, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., save_uvmap=None, n_jobs=4, verbose=True):
    """
    Generates or loads a lightcone of UV coverage maps, one for each redshift.

    This function handles the creation of UV maps across a range of redshifts,
    using parallel processing for efficiency and providing an option to cache
    the results to a file to speed up subsequent runs.

    Parameters
    ----------
    ncells : int
        The number of grid cells.
    zs : np.ndarray or list
        An array or list of redshift values for the lightcone.
    subarray_type, total_int_time, etc. : various, optional
        Observational parameters passed to `get_uv_map`.
    save_uvmap : str, optional
        File path to save or load the UV map dictionary. If the file exists,
        maps are loaded; otherwise, they are generated and saved.
    n_jobs : int, optional
        Number of CPUs for parallel generation of UV maps.
    verbose : bool, optional
        If True, enables progress bars and informational messages.

    Returns
    -------
    dict
        A dictionary containing the UV maps for each redshift, keyed by the
        redshift value formatted to three decimal places, along with metadata
        about the simulation parameters.
    """
    antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)
    if boxsize is None: boxsize = conv.LB
    if isinstance(zs, list): zs = np.array(zs)

    # Attempt to load existing UV maps or initialize a new dictionary
    if save_uvmap and os.path.exists(save_uvmap):
        uvs = read_dictionary_data(save_uvmap)
    else:
        uvs = {'ncells': ncells, 'boxsize': boxsize, 'total_int_time': total_int_time, 'int_time': int_time, 'declination': declination}
    
    # Identify which redshifts need a UV map to be generated
    z_to_run = [zi for zi in zs if '{:.3f}'.format(zi) not in uvs]

    if z_to_run:
        print(f'Generating {len(z_to_run)} new UV maps...')
        _uvmap_worker = lambda zi: get_uv_map(ncells, zi, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=False)[0]
        
        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_uvmap_worker)(i) for i in z_to_run)
            for zi, uv_map in zip(z_to_run, results):
                uvs['{:.3f}'.format(zi)] = uv_map
        else:
            for zi in tqdm(z_to_run, desc="Generating UV maps sequentially"):
                uvs['{:.3f}'.format(zi)] = _uvmap_worker(zi)
        
        uvs['Nant'] = N_ant
        if save_uvmap:
            print(f"Saving updated UV maps to {save_uvmap}")
            write_dictionary_data(uvs, save_uvmap)
            
    return uvs

def noise_map(ncells, z, depth_mhz, obs_time=1000, subarray_type="AA4", boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None, uv_weighting='natural', sefd_data=None, nu_data=None, fft_wrap=False, verbose=True, suppress_sharp_features_uv_map=False):
	"""
	Creates a 2D map of instrumental noise for a single observation slice.

	This function simulates the thermal noise from a radio interferometer,
	taking into account the UV coverage and weighting scheme, and produces
	a noise map in the image domain.

	Parameters
	----------
	ncells : int
		The number of grid cells in each spatial dimension.
	z : float
		The redshift of the observation.
	depth_mhz : float
		The observational bandwidth in MHz.
	obs_time : float, optional
		The total on-source observation time in hours.
	subarray_type : str, optional
		The telescope layout configuration name.
	boxsize : float, optional
		The comoving size of the simulation box in Mpc.
	total_int_time, int_time, declination : float, optional
		Parameters for `get_uv_map` if `uv_map` is not provided.
	uv_map, N_ant : various, optional
		Pre-computed UV map and number of antennas.
	uv_weighting : {'natural', 'uniform'}, optional
		The weighting scheme to apply.
	sefd_data, nu_data : various, optional
		Data for SEFD calculation.
	fft_wrap : bool, optional
		If True, use a wrapped FFT to handle periodic boundary conditions.
	verbose : bool, optional
		If True, enables verbose output.
	suppress_sharp_features_uv_map : bool or str, optional
		If a string, specifies a method to smooth the UV map to reduce artifacts.

	Returns
	-------
	np.ndarray
		A 2D array representing the noise map in the image domain (in muJy).
	"""	
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)

	if uv_map is None: 
		uv_map, N_ant = get_uv_map(ncells, z, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if N_ant is None: 
		N_ant = antxyz.shape[0]

	sigma, rms_noi = sigma_noise_radio(z, depth_mhz, obs_time, int_time, uv_map=uv_map, N_ant=N_ant, verbose=False, sefd_data=sefd_data, nu_data=nu_data)
	noise_real = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_imag = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_arr  = noise_real + 1.j*noise_imag
	noise_four = apply_uv_response_noise(noise_arr, uv_map, boxsize=boxsize, uv_weighting=uv_weighting, suppress_sharp_features_uv_map=suppress_sharp_features_uv_map)
	# noise_four = apply_uv_response_noise_briggs(noise_arr, uv_map, robust=2.0, epsilon=1e-6)
	win_2d = signal_window(ncells, 'blackmanharris', ndim=2)
	noise_four = noise_four*np.fft.fftshift(win_2d)
	if fft_wrap: noise_map  = ifft2_wrap(noise_four)*np.sqrt(int_time/3600./obs_time)
	else: noise_map  = np.fft.ifft2(noise_four)*np.sqrt(int_time/3600./obs_time)
	return np.real(noise_map)

def _suppress_sharp_features_uv_map(uv_map, boxsize=None, method='gaussian', filter_param=5.):
	"""
	(Internal) Smooths the UV coverage map to mitigate sharp features.

	Parameters
	----------
	uv_map : np.ndarray
		The 2D UV coverage map.
	boxsize : float, optional
		The comoving size of the box in Mpc, required for 'binned' method.
	method : {'gaussian', 'binned'}, optional
		The smoothing method to use.
	filter_param : float or int, optional
		The primary parameter for the smoothing method (e.g., sigma for
		Gaussian, number of bins for binned).

	Returns
	-------
	np.ndarray
		The smoothed UV map.
	"""
	if method.lower()=='gaussian':
		uv_map_smooth = np.sqrt(gaussian_filter(uv_map, filter_param))
		uv_map_smooth *= uv_map.max()/uv_map_smooth.max()
	elif method.lower() in ['binned', 'bin']:
		assert boxsize is not None
		box_dims = [boxsize,boxsize]
		k_nq = np.pi/boxsize*min(uv_map.shape)
		uv_rad_avg, k_rad_avg, n_modes = radial_average(np.fft.fftshift(uv_map), box_dims, kbins=filter_param, binning='log')
		k_comp, k_mag = _get_k(uv_map, [boxsize,boxsize])
		xx, yy = np.log10(k_rad_avg[k_rad_avg<k_nq/2]), np.log10(uv_rad_avg[k_rad_avg<k_nq/2])
		uv_map_smooth = np.fft.fftshift(10**interp1d(xx[np.isfinite(yy)], yy[np.isfinite(yy)], fill_value='extrapolate')(np.log10(k_mag)))
	else:
		print(f'{method} to suppress sharp features in the uv maps is not supported. Choose from "gaussian", "binned".')
	return uv_map_smooth

def apply_uv_response_noise(noise, uv_map, boxsize=None, uv_weighting='natural', uv_map_min=0.01, suppress_sharp_features_uv_map=False):
	"""
	(Internal) Applies the instrumental UV response to a noise array.

	This function weights the noise in the Fourier domain according to the
	UV coverage and the chosen weighting scheme.

	Parameters
	----------
	noise : np.ndarray
		The complex noise array in the Fourier domain.
	uv_map : np.ndarray
		The 2D UV coverage map.
	boxsize : float, optional
		The comoving size of the box, needed if smoothing is applied.
	uv_weighting : {'natural', 'uniform'}, optional
		The weighting scheme.
	uv_map_min : float, optional
		Threshold below which UV cells are considered un-sampled and zeroed out.
	suppress_sharp_features_uv_map : bool or str, optional
		Method to smooth the UV map before applying weights.

	Returns
	-------
	np.ndarray
		The noise array after applying the UV response.
	"""
	if uv_weighting.lower() in ['nat', 'natural', 'natural_weighting']:
		if suppress_sharp_features_uv_map:
			uv_map_smooth = _suppress_sharp_features_uv_map(uv_map, boxsize=boxsize, method=suppress_sharp_features_uv_map, filter_param=15)
		else:
			uv_map_smooth = uv_map
		out = noise/np.sqrt(uv_map_smooth)
	elif uv_weighting.lower() in ['uni', 'uniform', 'uniform_weighting']:
		out = noise
	else:
		print(f'{uv_weighting} scheme is not known or implemented')
	out[uv_map<uv_map_min] = 0.
	return out

def apply_uv_response_noise_briggs(noise, uv_map, robust=2.0, epsilon=1e-6):
	"""
	Applies Briggs weighting to a noise array.

	Parameters
	----------
	noise : np.ndarray
		2D array of base noise values (assumed uniform before weighting).
	uv_map : np.ndarray
		2D array of natural weights (hit counts).
	robust : float, optional
		Briggs robustness parameter (-2 for uniform to +2 for near-natural).
	epsilon : float, optional
		A small value to prevent division by zero.

	Returns
	-------
	np.ndarray
		Noise map after applying Briggs weighting.
	"""
	w_nat = uv_map
	w_mean = np.mean(w_nat[w_nat > 0]) + epsilon

	briggs_weight = w_nat / (1 + (w_nat / w_mean)**2 * 10**(2 * robust))
	noise_out = noise / np.sqrt(briggs_weight + epsilon)
	noise_out[w_nat == 0] = 0.0

	return noise_out

def ifft2_wrap(nn1):
	"""
	Performs a 2D inverse FFT with wrapping to handle boundaries.

	This is useful for avoiding artifacts when the data is not perfectly periodic.

	Parameters
	----------
	nn1 : np.ndarray
		The 2D array in Fourier space.

	Returns
	-------
	np.ndarray
		The central part of the inverse transformed array.
	"""
	assert nn1.ndim==2
	bla0 = np.vstack((nn1,nn1))
	bla1 = np.roll(bla0, nn1.shape[0]/2, 0)
	bla2 = np.hstack((bla1,bla1))
	bla3 = np.roll(bla2, nn1.shape[1]/2, 1)
	imap = np.fft.ifft2(bla3)
	return imap[nn1.shape[0]/2:-nn1.shape[0]/2,nn1.shape[1]/2:-nn1.shape[1]/2]

def apply_uv_response_on_image(array, uv_map):
	"""
	Simulates the effect of incomplete UV coverage on a true sky image.

	This function mimics the observation process by transforming an image to
	the Fourier (UV) domain, setting all un-sampled UV cells to zero, and
	transforming back to the image domain.

	Parameters
	----------
	array : np.ndarray
		The 2D input image array.
	uv_map : np.ndarray
		The 2D UV coverage map, where non-zero values indicate sampled cells.

	Returns
	-------
	np.ndarray
		The resulting "dirty" image after applying the UV mask.
	"""
	assert array.shape == uv_map.shape
	img_arr  = np.fft.fft2(array)
	img_arr[uv_map==0] = 0
	img_map  = np.fft.ifft2(img_arr)
	return np.real(img_map)

def max_baseline_to_max_k(redshift, max_baseline):
	"""
	Converts a maximum physical baseline length to a maximum k-mode.

	Parameters
	----------
	redshift : float
		The redshift of observation.
	max_baseline : float or astropy.units.Quantity
		The maximum baseline length of the telescope, assumed in km if no
		units are given.

	Returns
	-------
	astropy.units.Quantity
		The corresponding maximum k-mode (spatial frequency).
	"""
	if not isinstance(max_baseline, (u.Quantity)):
		max_baseline *= u.km
	lcut = (1 + redshift) * (21 * u.cm / max_baseline).to('') * cm.z_to_cdist(redshift)
	kcut = np.pi / lcut
	return kcut

def get_uv_map(ncells, z, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., include_mirror_baselines=True, verbose=True, max_baseline=None):
	"""
	Generates a 2D UV coverage map for a given observation.

	This function simulates a daily observation to calculate the UV coverage,
	optionally applying a cut based on maximum baseline length.

	Parameters
	----------
	ncells : int
		Number of cells in the grid.
	z : float
		Redshift of observation.
	subarray_type : str, optional
		The telescope layout configuration name.
	total_int_time, int_time, declination : float, optional
		Observation parameters.
	boxsize : float, optional
		Comoving size of the simulation box in Mpc.
	include_mirror_baselines : bool, optional
		If True, includes mirrored baselines.
	verbose : bool, optional
		If True, enables verbose output.
	max_baseline : float, optional
		The maximum baseline length in km. If provided, UV tracks beyond the
		corresponding k-mode are cut.

	Returns
	-------
	uv_map : np.ndarray
		The 2D array of gridded UV coverage.
	N_ant : int
		The number of antennas.
	"""
	if boxsize is None: 
		boxsize = conv.LB  
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)
	uv_map, N_ant  = get_uv_daily_observation(ncells, z, antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, include_mirror_baselines=include_mirror_baselines, verbose=verbose)
	if max_baseline is not None:
		kcut = max_baseline_to_max_k(z, max_baseline)
		kk = np.linspace(-ncells/2, ncells/2, ncells) * np.pi / boxsize
		kx, ky = np.meshgrid(kk, kk)
		ksq = kx**2 + ky**2
		uv_map_shift = np.fft.fftshift(uv_map)
		uv_map_shift[ksq>kcut**2] = 0.0
		uv_map = np.fft.fftshift(uv_map_shift)
	return uv_map, N_ant

def make_uv_map_lightcone(ncells, zs, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
	Creates a 3D cube of UV maps, one for each redshift slice.

	Parameters
	----------
	ncells : int
		Number of cells in the grid.
	zs : np.ndarray
		An array of redshift values for the lightcone.
	subarray_type, total_int_time, int_time, declination, boxsize : various, optional
		Observational parameters passed to `get_uv_map`.
	verbose : bool, optional
		If True, enables verbose output.

	Returns
	-------
	uv_lc : np.ndarray
		A 3D array `(ncells, ncells, n_redshifts)` containing the UV maps.
	N_ant : int
		The number of antennas.
	"""
	uv_lc = np.zeros((ncells,ncells,zs.shape[0]))
	percc = np.round(100./zs.shape[0],decimals=2)
	for i in range(zs.shape[0]):
		z = zs[i]
		uv_map, N_ant = get_uv_map(ncells, z, subarray_type=subarray_type, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
		uv_lc[:,:,i] = uv_map
		print("\nThe lightcone has been constructed upto %.1f per cent." %(i*percc))
	return uv_lc, N_ant

def apply_uv_response_on_coeval(array, z, subarray_type="AA4", boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None):
	"""
	Applies the UV response slice-by-slice to a 3D coeval data cube.

	Parameters
	----------
	array : np.ndarray
		The input 3D data cube `(nx, ny, nz)`.
	z, subarray_type, boxsize, etc. : various, optional
		Parameters passed to `get_uv_map` if `uv_map` is not provided.

	Returns
	-------
	np.ndarray
		The observed 3D data cube after applying the UV mask to each slice.
	"""
	ncells = array.shape[-1]
	if boxsize is None: boxsize = conv.LB
	if uv_map is None: 
		uv_map, N_ant  = get_uv_map(ncells, z, subarray_type=subarray_type, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	data3d = np.zeros(array.shape)
	print("Creating the noise cube")
	for k in range(ncells):
		data2d = apply_uv_response_on_image(array[:,:,k], uv_map=uv_map)
		data3d[:,:,k] = data2d
	return data3d

def noise_cube_coeval(ncells, z, depth_mhz=None, obs_time=1000, subarray_type="AA4", boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None, uv_weighting='natural', verbose=True, fft_wrap=False, sefd_data=None, nu_data=None, suppress_sharp_features_uv_map=False):
	"""
	Generates a 3D coeval cube of instrumental noise.

	This function simulates a noise cube where the noise properties are
	constant along the line-of-sight (frequency) axis.

	Parameters
	----------
	ncells, z, obs_time, etc. : various
		Parameters passed to `noise_map` for each slice.
	depth_mhz : float, optional
		The bandwidth per channel. If None, it is calculated from `boxsize`.

	Returns
	-------
	np.ndarray
		A 3D cube `(ncells, ncells, ncells)` of instrumental noise in mK.
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)

	if boxsize is None: 
		boxsize = conv.LB
	if depth_mhz is None: 
		depth_mhz = (cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)-boxsize/2))-cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)+boxsize/2)))/ncells
	if uv_map is None: 
		uv_map, N_ant  = get_uv_map(ncells, z, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)

	# ncells = int(ncells); print(ncells)
	noise3d = np.zeros((ncells,ncells,ncells))
	if verbose: 
		print("Creating the noise cube...")
	sleep(1)
	for k in tqdm(range(ncells), disable=not verbose):
		noise2d = noise_map(ncells, z, depth_mhz, obs_time=obs_time, subarray_type=antxyz, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, uv_weighting=uv_weighting, fft_wrap=fft_wrap, sefd_data=sefd_data, nu_data=nu_data, suppress_sharp_features_uv_map=suppress_sharp_features_uv_map)
		noise3d[:,:,k] = noise2d
	if verbose: 
		print("...noise cube created.")
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)

def noise_cube_lightcone(ncells, z, obs_time=1000, subarray_type="AA4", boxsize=None, save_uvmap=None, total_int_time=6., int_time=10., declination=-30., N_ant=None, uv_weighting='natural', fft_wrap=False, verbose=True, n_jobs=4, checkpoint=64, sefd_data=None, nu_data=None, suppress_sharp_features_uv_map=False):
	"""
	Generates a 3D lightcone of instrumental noise around a central redshift.

	This function is ideal for single redshift studies with a narrow bandwidth,
	where each slice along the line-of-sight corresponds to a slightly
	different redshift and frequency.

	Parameters
	----------
	ncells : int
		The grid size.
	z : float
		The central redshift of the lightcone.
	obs_time, subarray_type, etc. : various
		Observational parameters.
	save_uvmap : str, optional
		File path to save or load pre-computed UV maps to speed up re-runs.
	n_jobs, checkpoint : int, optional
		Parameters for parallel processing of UV map generation.

	Returns
	-------
	np.ndarray
		A 3D `(ncells, ncells, ncells)` lightcone of instrumental noise in mK.
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type, verbose=verbose)
	if boxsize is None: 
		boxsize = conv.LB
	zs = cm.cdist_to_z(np.linspace(cm.z_to_cdist(z)-boxsize/2, cm.z_to_cdist(z)+boxsize/2, ncells))

	# This function body is very similar to `noise_lightcone`. Consider refactoring.
	return noise_lightcone(ncells, zs, obs_time, subarray_type, boxsize, save_uvmap, total_int_time, int_time, declination, N_ant, uv_weighting, fft_wrap, verbose, n_jobs, checkpoint, sefd_data, nu_data, suppress_sharp_features_uv_map)

def noise_lightcone(ncells, zs, obs_time=1000, subarray_type="AA4", boxsize=None, save_uvmap=None, total_int_time=6., int_time=10., declination=-30., uv_weighting='natural', fft_wrap=False, verbose=True, n_jobs=4, sefd_data=None, nu_data=None, suppress_sharp_features_uv_map=False):
    """
    Generates a 3D lightcone of instrumental noise over a list of redshifts.

    Each slice along the line-of-sight corresponds to a different redshift,
    and noise properties are calculated accordingly. This function first calls
    `get_uv_map_lightcone` to efficiently generate or load all required UV maps.

    Parameters
    ----------
    ncells : int
        The grid size.
    zs : np.ndarray or list
        An array or list of redshift values for each slice of the lightcone.
    obs_time, subarray_type, etc. : various
        Observational parameters.
    save_uvmap : str, optional
        File path passed to `get_uv_map_lightcone` for caching UV maps.
    n_jobs : int, optional
        Number of CPUs for parallel processing of UV maps.
    
    Returns
    -------
    np.ndarray
        A 3D `(ncells, ncells, len(zs))` lightcone of noise in mK.
    """
    if isinstance(zs, list): zs = np.array(zs)
    
    # Step 1: Get all UV maps efficiently.
    uvs = get_uv_map_lightcone(
        ncells, zs, subarray_type=subarray_type, total_int_time=total_int_time, 
        int_time=int_time, boxsize=boxsize, declination=declination, 
        save_uvmap=save_uvmap, n_jobs=n_jobs, verbose=verbose
    )
    N_ant = uvs.get('Nant')

    # Step 2: Create the noise cube using the pre-computed UV maps.
    print('Creating noise lightcone...')
    noise3d = np.zeros((ncells, ncells, zs.size))
    for k, zi in enumerate(tqdm(zs, desc="Generating noise slices")):
        if k + 1 < zs.size: 
            depth_mhz = np.abs(cm.z_to_nu(zs[k+1]) - cm.z_to_nu(zs[k]))
        else: 
            depth_mhz = np.abs(cm.z_to_nu(zs[k]) - cm.z_to_nu(zs[k-1]))
        
        uv_map = uvs['{:.3f}'.format(zi)]
        
        # Note: We pass `subarray_type` but `uv_map` is provided, so a new one won't be generated inside noise_map.
        noise2d = noise_map(
            ncells, zi, depth_mhz, obs_time=obs_time, subarray_type=subarray_type, 
            boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, 
            declination=declination, uv_map=uv_map, N_ant=N_ant, 
            uv_weighting=uv_weighting, verbose=False, fft_wrap=fft_wrap, 
            sefd_data=sefd_data, nu_data=nu_data, 
            suppress_sharp_features_uv_map=suppress_sharp_features_uv_map
        )
        noise3d[:,:,k] = jansky_2_kelvin(noise2d, zi, boxsize=boxsize)
        
    return noise3d

def gauss_kernel_3d(size, sigma=1.0, fwhm=None):
	"""
	Generates a 3D normalized Gaussian kernel.

	Parameters
	----------
	size : int
		Width of the output array in pixels.
	sigma : float, optional
		The standard deviation of the Gaussian.
	fwhm : float, optional
		The Full Width at Half Maximum. Overrides `sigma` if provided.
		
	Returns
	-------
	np.ndarray
		A 3D numpy array with the Gaussian kernel, normalized to sum to 1.
	"""
	if fwhm != None:
		sigma = fwhm/(2.*np.sqrt(2.*np.log(2)))

	if size % 2 == 0:
		size = int(size/2)
		x,y,z = np.mgrid[-size:size, -size:size, -size:size]
	else:
		size = int(size/2)
		x,y,z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]

	g = np.exp(-(x**2 + y**2 + z**2)/(2.*sigma**2))

	return g/g.sum()

def smooth_gauss_3d(array, fwhm):
	"""
	Smooths a 3D array using a Gaussian kernel via FFT convolution.

	Parameters
	----------
	array : np.ndarray
		The 3D input array to be smoothed.
	fwhm : float
		The Full Width at Half Maximum of the Gaussian kernel in pixels.
		
	Returns
	-------
	np.ndarray
		The smoothed 3D array.
	"""
	gg = gauss_kernel_3d(array.shape[0],fwhm=fwhm)
	out = scipy.signal.fftconvolve(array, gg)
	return out
