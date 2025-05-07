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
import scipy
from glob import glob
from time import time, sleep
import pickle
import astropy.units as u
from joblib import Parallel, delayed
from tqdm import tqdm

def noise_map(ncells, z, depth_mhz, obs_time=1000, subarray_type="AA4", boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None, uv_weighting='natural', fft_wrap=False):
	"""
	It creates a noise map by simulating the radio observation strategy (1801.06550).

	Parameters
	----------
	z: float
		Redshift.
	ncells: int
		The grid size.
	depth_mhz: float
		The bandwidth in MHz.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	uv_map: ndarray
		numpy array containing gridded uv coverage. If nothing given, then the uv map 
		will be simulated
	N_ant: int
		Number of antennae
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc
	
	Returns
	-------
	noise_map: ndarray
		A 2D slice of the interferometric noise at that frequency (in muJy).
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type)

	if uv_map is None: 
		uv_map, N_ant = get_uv_map(ncells, z, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if N_ant is None: 
		N_ant = antxyz.shape[0]

	sigma, rms_noi = sigma_noise_radio(z, depth_mhz, obs_time, int_time, uv_map=uv_map, N_ant=N_ant, verbose=False)
	noise_real = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_imag = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_arr  = noise_real + 1.j*noise_imag
	noise_four = apply_uv_response_noise(noise_arr, uv_map, uv_weighting=uv_weighting)
	# noise_four = apply_uv_response_noise_briggs(noise_arr, uv_map, robust=2.0, epsilon=1e-6)
	if fft_wrap: noise_map  = ifft2_wrap(noise_four)*np.sqrt(int_time/3600./obs_time)
	else: noise_map  = np.fft.ifft2(noise_four)*np.sqrt(int_time/3600./obs_time)
	return np.real(noise_map)

def apply_uv_response_noise(noise, uv_map, uv_weighting='natural'):
	'''
	Apply the effect of uv coverage on the noise array.
	'''
	if uv_weighting.lower() in ['nat', 'natural', 'natural_weighting']:
		out = noise/np.sqrt(uv_map)
	elif uv_weighting.lower() in ['uni', 'uniform', 'uniform_weighting']:
		out = noise
	else:
		print(f'{uv_weighting} scheme is not known or implemented')
	out[uv_map==0] = 0.
	return out

def apply_uv_response_noise_briggs(noise, uv_map, robust=2.0, epsilon=1e-6):
    """
    Apply the effect of uv coverage on the noise array using Briggs weighting.

    Parameters
    ----------
    noise : ndarray
        2D array of base noise values (assumed uniform before weighting).
    uv_map : ndarray
        2D array containing number of observations per pixel (natural weights).
    robust : float
        Briggs robustness parameter (-2 to +2).
    epsilon : float
        Small value to avoid division by zero.

    Returns
    -------
    weighted_noise : ndarray
        Noise map after applying Briggs weighting.
    """
    w_nat = uv_map
    w_mean = np.mean(w_nat[w_nat > 0]) + epsilon

    briggs_weight = w_nat / (1 + (w_nat / w_mean)**2 * 10**(2 * robust))
    noise_out = noise / np.sqrt(briggs_weight + epsilon)
    noise_out[w_nat == 0] = 0.0

    return noise_out

def ifft2_wrap(nn1):
	assert nn1.ndim==2
	bla0 = np.vstack((nn1,nn1))
	bla1 = np.roll(bla0, nn1.shape[0]/2, 0)
	bla2 = np.hstack((bla1,bla1))
	bla3 = np.roll(bla2, nn1.shape[1]/2, 1)
	imap = np.fft.ifft2(bla3)
	return imap[nn1.shape[0]/2:-nn1.shape[0]/2,nn1.shape[1]/2:-nn1.shape[1]/2]

def apply_uv_response_on_image(array, uv_map):
	"""
	Parameters
	----------
	array: ndarray
		Image array
	uv_map: ndarray
		numpy array containing gridded uv coverage. 
	
	Returns
	-------
	Radio image after applying the effect of radio observation strategy.
	"""
	assert array.shape == uv_map.shape
	img_arr  = np.fft.fft2(array)
	img_arr[uv_map==0] = 0
	img_map  = np.fft.ifft2(img_arr)
	return np.real(img_map)

def get_uv_map(ncells, z, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
	It creates the uv map at a given redshift (z).

	Parameters
	----------
	ncells: int
		Number of cells
	z: float
		Redshift.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc	
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	uv_map: ndarray
		array of gridded uv coverage.
	N_ant: int
		Number of antennae
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type)
	uv_map, N_ant  = get_uv_daily_observation(ncells, z, antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
	return uv_map, N_ant

def make_uv_map_lightcone(ncells, zs, subarray_type="AA4", total_int_time=6., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
	It creates uv maps at every redshift of the lightcone.

	Parameters
	----------
	ncells: int
		Number of cells
	zs: ndarray
		array of redshift values.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc	
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	uv_lc: ndarray
		array of gridded uv coverage at all the redshifts.
	N_ant: int
		Number of antennae
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

def noise_cube_coeval(ncells, z, depth_mhz=None, obs_time=1000, subarray_type="AA4", boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=None, N_ant=None, uv_weighting='natural', verbose=True, fft_wrap=False):
	"""
	It creates a noise coeval cube by simulating the radio observation strategy (1801.06550).

	Parameters
	----------
	ncells: int
		The grid size.
	z: float
		Redshift.
	depth_mhz: float
		The bandwidth in MHz.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	uv_map: ndarray
		numpy array containing gridded uv coverage. If nothing given, then the uv map 
		will be simulated
	N_ant: int
		Number of antennae
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_cube: ndarray
		A 3D cube of the interferometric noise (in mK).
		The frequency is assumed to be the same along the assumed frequency (last) axis.	
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type)

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
		noise2d = noise_map(ncells, z, depth_mhz, obs_time=obs_time, subarray_type=antxyz, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, uv_weighting=uv_weighting, fft_wrap=fft_wrap)
		noise3d[:,:,k] = noise2d
	if verbose: 
		print("...noise cube created.")
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)

def noise_cube_lightcone(ncells, z, obs_time=1000, subarray_type="AA4", boxsize=None, save_uvmap=None, total_int_time=6., int_time=10., declination=-30., N_ant=None, uv_weighting='natural', fft_wrap=False, verbose=True, n_jobs=4, checkpoint=64):
	"""
	It creates a noise cube by simulating the radio observation strategy (1801.06550)
	considerng the input redshift (z) as the central slice of the cube.
	This function is ideal for single redshift studies (observations with narrow bandwidth).
	We assume the third axis to be along the line-of-sight and therefore 
	each each will correspond to a different redshift.

	Parameters
	----------
	ncells: int
		The grid size.
	z: float
		Central redshift.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	N_ant: int
		Number of antennae
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	save_uvmap: str
		Give the filename of the pickle file of uv maps. If
			
			- the file is absent, then uv maps are created and saved with the given filename.
			- the file is present, then the uv map is read in.
			- the file is present and the uv maps are incomplete, then it is completed.
			- None is given, then the uv maps are not saved.
	n_jobs: int
		Number of CPUs to run in. The calculation is parallelised using joblib.
	checkpoint: int
		Number of iterations after which uv maps are saved if save_uvmap is not None.
	
	Returns
	-------
	noise_lightcone: A 3D cubical lightcone of the interferometric noise with frequency varying 
	along last axis(in mK).	
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type)

	if boxsize is None: 
		boxsize = conv.LB
	zs = cm.cdist_to_z(np.linspace(cm.z_to_cdist(z)-boxsize/2, cm.z_to_cdist(z)+boxsize/2, ncells))
	noise3d = np.zeros((ncells,ncells,ncells))

	if save_uvmap is not None:
		if save_uvmap[-4:]!='.pkl': save_uvmap += '.pkl'
		if len(glob(save_uvmap)):
			uvs = pickle.load(open(save_uvmap, 'rb'))
			if uvs['ncells']!=ncells or uvs['boxsize']!=boxsize or uvs['total_int_time']!=total_int_time or uvs['int_time']!=int_time or uvs['declination']!=declination:
				print('All or some uv maps is read from the given file. Be sure that they were run with the same parameter values as provided now.')
				print('Compare to the values of the parameters in the output dictionary.')
				return uvs
		else:
			uvs = {'ncells':ncells, 'boxsize':boxsize, 'total_int_time':total_int_time, 'int_time':int_time, 'declination':declination}
	else:
		uvs = {'ncells':ncells, 'boxsize':boxsize, 'total_int_time':total_int_time, 'int_time':int_time, 'declination':declination}

	# Create uv maps
	print('Creating the uv maps.')
	if n_jobs<=1:
		tstart = time()
		for k,zi in enumerate(zs):
			print(f'{k+1}/{len(zs)} | z={zi:.3f}')
			if '{:.3f}'.format(zi) not in uvs.keys():
				uv_map, N_ant  = get_uv_map(ncells, zi, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
				uvs['{:.3f}'.format(zi)] = uv_map
				uvs['Nant'] = N_ant
				pickle.dump(uvs, open(save_uvmap, 'wb'))
			tend = time()
			# print('\nz = {:.3f} | {:.2f} % completed | Elapsed time: {:.2f} mins'.format(zi,100*(k+1)/zs.size,(tend-tstart)/60))
	else:
		Nbase, N_ant = from_antenna_config(antxyz, zs[0])
		uvs['Nant'] = N_ant
		_uvmap = lambda zi: get_uv_map(ncells, zi, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=False)[0] 
		if checkpoint<2*n_jobs:
			checkpoint = 4*n_jobs
			print('checkpoint value should be more than 4*n_jobs. checkpoint set to 4*n_jobs.')
		z_run = np.array([])
		for k,zi in enumerate(zs):
			if '{:.3f}'.format(zi) not in uvs.keys():
				z_run = np.append(z_run, zi)
		n_iterations = int(z_run.size/checkpoint)
		if n_iterations>1:
			for ii in range(n_iterations):
				istart, iend = ii*checkpoint, (ii+1)*checkpoint 
				zrs = z_run[istart:iend] if ii+1<n_iterations else z_run[istart:]
				fla = Parallel(n_jobs=n_jobs,verbose=20)(delayed(_uvmap)(i) for i in zrs)
				for jj,zi in enumerate(zrs):
					uvs['{:.3f}'.format(zi)] = fla[jj]
				if save_uvmap is not None: pickle.dump(uvs, open(save_uvmap, 'wb'))
				print('{:.2f} % completed'.format(100*(len(uvs.keys())-1)/zs.size))
		else:
			fla = Parallel(n_jobs=n_jobs,verbose=20)(delayed(_uvmap)(i) for i in z_run)
			for jj,zi in enumerate(z_run):
				uvs['{:.3f}'.format(zi)] = fla[jj]
			if save_uvmap is not None: pickle.dump(uvs, open(save_uvmap, 'wb'))
		print('...done')


	# Calculate noise maps
	print('Creating noise.')
	for k,zi in enumerate(zs):
		if k+1<zs.size: depth_mhz = np.abs(cm.z_to_nu(zs[k+1])-cm.z_to_nu(zs[k]))
		else: depth_mhz = np.abs(cm.z_to_nu(zs[k])-cm.z_to_nu(zs[k-1]))
		uv_map, N_ant  = uvs['{:.3f}'.format(zi)], uvs['Nant']
		noise2d = noise_map(ncells, zi, depth_mhz, obs_time=obs_time, subarray_type=antxyz, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, uv_weighting=uv_weighting, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = jansky_2_kelvin(noise2d, zi, boxsize=boxsize)
		verbose = False
		print('z = {:.3f} | {:.2f} % completed'.format(zi,100*(k+1)/zs.size))
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)


def noise_lightcone(ncells, zs, obs_time=1000, subarray_type="AA4", boxsize=None, save_uvmap=None, total_int_time=6., int_time=10., declination=-30., N_ant=None, uv_weighting='natural', fft_wrap=False, verbose=True, n_jobs=4, checkpoint=64):
	"""
	It creates a noise lightcone by simulating the radio observation strategy (1801.06550).
	We assume the third axis to be along the line-of-sight and therefore 
	each each will correspond to a different redshift.

	Parameters
	----------
	ncells: int
		The grid size.
	zs: ndarray
		List of redshifts.
	obs_time: float
		The observation time in hours.
	total_int_time: float
		Total observation per day time in hours
	int_time: float
		Intergration time in seconds
	declination: float
		Declination angle in deg
	N_ant: int
		Number of antennae
	subarray_type: str
		The name of the SKA-Low layout configuration.
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	save_uvmap: str
		Give the filename of the pickle file of uv maps. If
			
			- the file is absent, then uv maps are created and saved with the given filename.
			- the file is present, then the uv map is read in.
			- the file is present and the uv maps are incomplete, then it is completed.
			- None is given, then the uv maps are not saved.
	n_jobs: int
		Number of CPUs to run in. The calculation is parallelised using joblib.
	checkpoint: int
		Number of iterations after which uv maps are saved if save_uvmap is not None.
	
	Returns
	-------
	noise_lightcone: A 3D lightcone of the interferometric noise with frequency varying 
	along last axis(in mK).	
	"""
	antxyz, N_ant = subarray_type_to_antxyz(subarray_type)

	if boxsize is None: 
		boxsize = conv.LB
	if isinstance(zs, list): 
		zs = np.array(zs)
	noise3d = np.zeros((ncells,ncells,zs.size))

	if save_uvmap is not None:
		if save_uvmap[-4:]!='.pkl': save_uvmap += '.pkl'
		if len(glob(save_uvmap)):
			uvs = pickle.load(open(save_uvmap, 'rb'))
			if uvs['ncells']!=ncells or uvs['boxsize']!=boxsize or uvs['total_int_time']!=total_int_time or uvs['int_time']!=int_time or uvs['declination']!=declination:
				print('All or some uv maps is read from the given file. Be sure that they were run with the same parameter values as provided now.')
				print('Compare to the values of the parameters in the output dictionary.')
				return uvs
		else:
			uvs = {'ncells':ncells, 'boxsize':boxsize, 'total_int_time':total_int_time, 'int_time':int_time, 'declination':declination}
	else:
		uvs = {'ncells':ncells, 'boxsize':boxsize, 'total_int_time':total_int_time, 'int_time':int_time, 'declination':declination}

	# Create uv maps
	print('Creating the uv maps.')
	if n_jobs<=1:
		tstart = time()
		for k,zi in enumerate(zs):
			print(f'{k+1}/{len(zs)} | z={zi:.3f}')
			if '{:.3f}'.format(zi) not in uvs.keys():
				uv_map, N_ant  = get_uv_map(ncells, zi, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
				uvs['{:.3f}'.format(zi)] = uv_map
				uvs['Nant'] = N_ant
				if save_uvmap is not None: pickle.dump(uvs, open(save_uvmap, 'wb'))
			tend = time()
			# print('\nz = {:.3f} | {:.2f} % completed | Elapsed time: {:.2f} mins'.format(zi,100*(k+1)/zs.size,(tend-tstart)/60))
	else:
		Nbase, N_ant = from_antenna_config(antxyz, zs[0])
		uvs['Nant'] = N_ant
		_uvmap = lambda zi: get_uv_map(ncells, zi, subarray_type=antxyz, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=False)[0] 
		if checkpoint<2*n_jobs:
			checkpoint = 4*n_jobs
			print('checkpoint value should be more than 4*n_jobs. checkpoint set to 4*n_jobs.')
		z_run = np.array([])
		for k,zi in enumerate(zs):
			if '{:.3f}'.format(zi) not in uvs.keys():
				z_run = np.append(z_run, zi)
		n_iterations = int(z_run.size/checkpoint)
		if n_iterations>1:
			for ii in range(n_iterations):
				istart, iend = ii*checkpoint, (ii+1)*checkpoint 
				zrs = z_run[istart:iend] if ii+1<n_iterations else z_run[istart:]
				fla = Parallel(n_jobs=n_jobs,verbose=20)(delayed(_uvmap)(i) for i in zrs)
				for jj,zi in enumerate(zrs):
					uvs['{:.3f}'.format(zi)] = fla[jj]
				if save_uvmap is not None: pickle.dump(uvs, open(save_uvmap, 'wb'))
				print('{:.2f} % completed'.format(100*(len(uvs.keys())-1)/zs.size))
		else:
			fla = Parallel(n_jobs=n_jobs,verbose=20)(delayed(_uvmap)(i) for i in z_run)
			for jj,zi in enumerate(z_run):
				uvs['{:.3f}'.format(zi)] = fla[jj]
			if save_uvmap is not None: pickle.dump(uvs, open(save_uvmap, 'wb'))
		print('...done')

	# Calculate noise maps
	print('Creating noise.')
	for k,zi in enumerate(zs):
		if k+1<zs.size: depth_mhz = np.abs(cm.z_to_nu(zs[k+1])-cm.z_to_nu(zs[k]))
		else: depth_mhz = np.abs(cm.z_to_nu(zs[k])-cm.z_to_nu(zs[k-1]))
		uv_map, N_ant  = uvs['{:.3f}'.format(zi)], uvs['Nant']
		noise2d = noise_map(ncells, zi, depth_mhz, obs_time=obs_time, subarray_type=antxyz, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, uv_weighting=uv_weighting, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = jansky_2_kelvin(noise2d, zi, boxsize=boxsize)
		print('\nz = {:.3f} | {:.2f} % completed'.format(zi,100*(k+1)/zs.size))
	return noise3d

def gauss_kernel_3d(size, sigma=1.0, fwhm=None):
	''' 
	Generate a normalized gaussian kernel, defined as
	exp(-(x^2 + y^2 + z^2)/(2sigma^2)).
	
	
	Parameters:
		size (int): Width of output array in pixels.
		sigma = 1.0 (float): The sigma parameter for the Gaussian.
		fwhm = None (float or None): The full width at half maximum.
				If this parameter is given, it overrides sigma.
		
	Returns:
		numpy array with the Gaussian. The dimensions will be
		size x size or size x sizey depending on whether
		sizey is set. The Gaussian is normalized so that its
		integral is 1.	
	'''
	
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
	gg = gauss_kernel_3d(array.shape[0],fwhm=fwhm)
	out = scipy.signal.fftconvolve(array, gg)
	return out
