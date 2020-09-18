'''
Methods:

* simulate the radio telescope observation strategy
* simulate telescope noise
'''

import numpy as np
import sys
from .telescope_functions import *
from .usefuls import *
from . import conv
from . import cosmology as cm
from . import smoothing as sm
import scipy

def noise_map(ncells, z, depth_mhz, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=np.array([]), N_ant=None, verbose=True, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)

	It creates a noise map by simulating the radio observation strategy.

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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_map: ndarray
		A 2D slice of the interferometric noise at that frequency (in muJy).
	"""
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	sigma, rms_noi = kanan_noise_image_ska(z, uv_map, depth_mhz, obs_time, int_time, N_ant_ska=N_ant, verbose=False)
	noise_real = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_imag = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_arr  = noise_real + 1.j*noise_imag
	noise_four = apply_uv_response_noise(noise_arr, uv_map)
	if fft_wrap: noise_map  = ifft2_wrap(noise_four)*np.sqrt(int_time/3600./obs_time)
	else: noise_map  = np.fft.ifft2(noise_four)*np.sqrt(int_time/3600./obs_time)
	return np.real(noise_map)

def apply_uv_response_noise(noise, uv_map):
	'''
	Apply the effect of uv coverage on the noise array.
	'''
	out = noise/np.sqrt(uv_map)
	out[uv_map==0] = 0.
	return out

def ifft2_wrap(nn1):
	assert nn1.ndim==2
	bla0 = np.vstack((nn1,nn1))
	bla1 = np.roll(bla0, nn1.shape[0]/2, 0)
	bla2 = np.hstack((bla1,bla1))
	bla3 = np.roll(bla2, nn1.shape[1]/2, 1)
	imap = np.fft.ifft2(bla3)
	return imap[nn1.shape[0]/2:-nn1.shape[0]/2,nn1.shape[1]/2:-nn1.shape[1]/2]

def telescope_response_on_image(array, z, depth_mhz, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=np.array([]), N_ant=None):
	"""
	Parameters
	----------
	array: ndarray
		Image array
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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	
	Returns
	-------
	Radio image after applying the effect of radio observation strategy.
	"""
	assert array.shape[0] == array.shape[1]
	ncells = array.shape[0]
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination) 
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	img_arr  = np.fft.fft2(array)
	img_arr[uv_map==0] = 0
	img_map  = np.fft.ifft2(img_arr)
	return np.real(img_map)

def get_uv_map(ncells, z, filename=None, total_int_time=6., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
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
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	uv_map, N_ant  = get_uv_daily_observation(ncells, z, filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
	return uv_map, N_ant

def make_uv_map_lightcone(ncells, zs, filename=None, total_int_time=6., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
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
		uv_map, N_ant = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
		uv_lc[:,:,i] = uv_map
		print("\nThe lightcone has been constructed upto %.1f \%" %(i*percc))
	return uv_lc, N_ant

def telescope_response_on_coeval(array, z, depth_mhz=None, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=np.array([]), N_ant=None):
	ncells = array.shape[-1]
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not boxsize: boxsize = conv.LB
	if not depth_mhz: depth_mhz = (cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)-boxsize/2))-cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)+boxsize/2)))/ncells
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	data3d = np.zeros(array.shape)
	print("Creating the noise cube")
	for k in range(ncells):
		data2d = telescope_response_on_image(array[:,:,k], z, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant)
		data3d[:,:,k] = data2d
	return data3d

def noise_cube_coeval(ncells, z, depth_mhz=None, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., uv_map=np.array([]), N_ant=None, verbose=True, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)

	It creates a noise coeval cube by simulating the radio observation strategy.

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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
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
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not boxsize: boxsize = conv.LB
	if not depth_mhz: depth_mhz = (cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)-boxsize/2))-cm.z_to_nu(cm.cdist_to_z(cm.z_to_cdist(z)+boxsize/2)))/ncells
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	print("\nCreating the noise cube...")
	for k in range(ncells):
		noise2d = noise_map(ncells, z, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = noise2d
		verbose = False
		perc = np.round((k+1)*100/ncells, decimals=1) 
		loading_verbose(str(perc)+'%')
	print("\n...Noise cube created.")
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)

def noise_cube_lightcone(ncells, z, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., N_ant=None, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)

	It creates a noise cube by simulating the radio observation strategy. 
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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_lightcone: A 3D cubical lightcone of the interferometric noise with frequency varying 
	along last axis(in mK).	
	"""
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not boxsize: boxsize = conv.LB
	zs = cm.cdist_to_z(np.linspace(cm.z_to_cdist(z)-boxsize/2, cm.z_to_cdist(z)+boxsize/2, ncells))
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	print("Creating the noise cube")
	verbose = True
	for k in range(ncells):
		zi = zs[k]
		if k+1<ncells: depth_mhz = cm.z_to_nu(zs[k+1])-cm.z_to_nu(zs[k])
		else: depth_mhz = cm.z_to_nu(zs[k])-cm.z_to_nu(zs[k-1])
		uv_map, N_ant  = get_uv_map(ncells, zi, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
		noise2d = noise_map(ncells, zi, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = noise2d
		verbose = False
		print('\n {:.2f} % completed'.format(100*(k+1)/zs.size))
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)


def noise_lightcone(ncells, zs, obs_time=1000, filename=None, boxsize=None, total_int_time=6., int_time=10., declination=-30., N_ant=None, fft_wrap=False):
	"""
	@ Ghara et al. (2017), Giri et al. (2018b)

	It creates a noise lightcone by simulating the radio observation strategy.

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
	filename: str
		The path to the file containing the telescope configuration.

			- As a default, it takes the SKA-Low configuration from Sept 2016
			- It is not used if uv_map and N_ant is provided
	boxsize: float
		Boxsize in Mpc
	verbose: bool
		If True, verbose is shown
	
	Returns
	-------
	noise_lightcone: A 3D lightcone of the interferometric noise with frequency varying 
	along last axis(in mK).	
	"""
	if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
	if not boxsize: boxsize = conv.LB
	# zs = cm.cdist_to_z(np.linspace(cm.z_to_cdist(z)-boxsize/2, cm.z_to_cdist(z)+boxsize/2, ncells))
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	print("Creating the noise cube")
	verbose = True
	for k,zi in enumerate(zs):
		if k+1<zs.size: depth_mhz = cm.z_to_nu(zs[k+1])-cm.z_to_nu(zs[k])
		else: depth_mhz = cm.z_to_nu(zs[k])-cm.z_to_nu(zs[k-1])
		uv_map, N_ant  = get_uv_map(ncells, zi, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
		noise2d = noise_map(ncells, zi, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose, fft_wrap=fft_wrap)
		noise3d[:,:,k] = jansky_2_kelvin(noise2d, zi, boxsize=boxsize)
		verbose = False
		print('\nz = {} | {:.2f} % completed'.format(zi,100*(k+1)/zs.size))
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




		

