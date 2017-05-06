import numpy as np
import c2raytools as c2t
from telescope_functions import *
from usefuls import *

def noise_map(ncells, z, depth_mhz, obs_time=1000, filename=None, boxsize=None, total_int_time=4., int_time=10., declination=30., uv_map=np.array([]), N_ant=None, verbose=True):
	"""
	Parameter
	z: 		   Redshift.
	ncells:    The grid size.
	depth_mhz: The bandwidth in MHz.
	obs_time:  The observation time in hours.
	filename:  The path to the file containing the telescope configuration.	
	
	Return
	noise_map: A 2D slice of the interferometric noise at that frequency (in muJy).
	"""
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	sigma, rms_noi = kanan_noise_image_ska(z, uv_map, depth_mhz, obs_time, int_time, N_ant_ska=N_ant, verbose=False)
	noise_real = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_imag = np.random.normal(loc=0.0, scale=rms_noi, size=(ncells, ncells))
	noise_arr  = noise_real + 1.j*noise_imag
	noise_four = apply_uv_response(noise_arr, uv_map)
	noise_map  = np.fft.ifft2(noise_four)*np.sqrt(int_time/3600./obs_time)
	return np.real(noise_map)

def telescope_response_on_image(array, z, depth_mhz, obs_time=1000, filename=None, boxsize=None, total_int_time=4., int_time=10., declination=30., uv_map=np.array([]), N_ant=None):
	assert array.shape[0] == array.shape[1]
	ncells = array.shape[0]
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination) 
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	img_arr  = np.fft.fft2(array)
	img_arr[uv_map==0] = 0
	img_map  = np.fft.ifft2(img_arr)
	return np.real(img_map)

def get_uv_map(ncells, z, filename=None, total_int_time=4., int_time=10., boxsize=None, declination=30., verbose=True):
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	uv_map, N_ant  = get_uv_daily_observation(ncells, z, filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
	return uv_map, N_ant

def make_uv_map_lightcone(ncells, zs, filename=None, total_int_time=4., int_time=10., boxsize=None, declination=30., verbose=True):
	uv_lc = np.zeros((ncells,ncells,zs.shape[0]))
	percc = np.round(100./zs.shape[0],decimals=2)
	for i in xrange(zs.shape[0]):
		z = zs[i]
		uv_map, N_ant = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose)
		uv_lc[:,:,i] = uv_map
		print "\nThe lightcone has been constructed upto", i*percc, "%"
	return uv_lc, N_ant

def telescope_response_on_coeval(array, z, depth_mhz=None, obs_time=1000, filename=None, boxsize=None, total_int_time=4., int_time=10., declination=30., uv_map=np.array([]), N_ant=None):
	ncells = array.shape[-1]
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	if not boxsize: boxsize = c2t.conv.LB
	if not depth_mhz: depth_mhz = (c2t.z_to_nu(c2t.cdist_to_z(c2t.z_to_cdist(z)-boxsize/2))-c2t.z_to_nu(c2t.cdist_to_z(c2t.z_to_cdist(z)+boxsize/2)))/ncells
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	data3d = np.zeros(array.shape)
	print "Creating the noise cube"
	for k in xrange(ncells):
		data2d = telescope_response_on_image(array[:,:,k], z, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant)
		data3d[:,:,k] = data2d
	return data3d

def noise_cube_coeval(ncells, z, depth_mhz=None, obs_time=1000, filename=None, boxsize=None, total_int_time=4., int_time=10., declination=30., uv_map=np.array([]), N_ant=None, verbose = True):
	"""
	Parameter
	z        : Redshift.
	ncells   : The grid size.
	depth_mhz: The bandwidth in MHz.
	obs_time : The observation time in hours.
	filename : The path to the file containing the telescope configuration.	
	
	Return
	noise_map: A 2D slice of the interferometric noise at that frequency (in mK).
	"""
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	if not boxsize: boxsize = c2t.conv.LB
	if not depth_mhz: depth_mhz = (c2t.z_to_nu(c2t.cdist_to_z(c2t.z_to_cdist(z)-boxsize/2))-c2t.z_to_nu(c2t.cdist_to_z(c2t.z_to_cdist(z)+boxsize/2)))/ncells
	if not uv_map.size: uv_map, N_ant  = get_uv_map(ncells, z, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	print "\nCreating the noise cube..."
	for k in xrange(ncells):
		noise2d = noise_map(ncells, z, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose)
		noise3d[:,:,k] = noise2d
		verbose = False
		perc = (k+1)*100/ncells
		loading_verbose(str(perc)+'%')
	print "...Noise cube created."
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)

def noise_cube_lightcone(ncells, z, obs_time=1000, filename=None, boxsize=None, total_int_time=4., int_time=10., declination=30., N_ant=None):
	"""
	Parameter
	z        : Redshift.
	ncells   : The grid size.
	depth_mhz: The bandwidth in MHz.
	obs_time : The observation time in hours.
	filename : The path to the file containing the telescope configuration.	
	
	Return
	noise_map: A 2D slice of the interferometric noise at that frequency (in mK).
	"""
	if not filename: filename = 'input/SKA1_LowConfig_Sept2016.dat'
	if not boxsize: boxsize = c2t.conv.LB
	zs = c2t.cdist_to_z(np.linspace(c2t.z_to_cdist(z)-boxsize/2, c2t.z_to_cdist(z)+boxsize/2, ncells))
	if not N_ant: N_ant = np.loadtxt(filename, dtype=str).shape[0]
	noise3d = np.zeros((ncells,ncells,ncells))
	print "Creating the noise cube"
	verbose = True
	for k in xrange(ncells):
		zi = zs[k]
		if k+1<ncells: depth_mhz = c2t.z_to_nu(zi[k+1])-c2t.z_to_nu(zi[k])
		else: depth_mhz = c2t.z_to_nu(zi[k])-c2t.z_to_nu(zi[k-1])
		uv_map, N_ant  = get_uv_map(ncells, zi, filename=filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination)
		noise2d = noise_map(ncells, zi, depth_mhz, obs_time=obs_time, filename=filename, boxsize=boxsize, total_int_time=total_int_time, int_time=int_time, declination=declination, uv_map=uv_map, N_ant=N_ant, verbose=verbose)
		noise3d[:,:,k] = noise2d
		verbose = False
	return jansky_2_kelvin(noise3d, z, boxsize=boxsize)


def gauss_kernel_3d(size, sigma=1.0, fwhm=None):
	''' 
	Generate a normalized gaussian kernel, defined as
	exp(-(x^2 + y^2 + z^2)/(2sigma^2)).
	
	
	Parameters:
		* size (int): Width of output array in pixels.
		* sigma = 1.0 (float): The sigma parameter for the Gaussian.
		* fwhm = None (float or None): The full width at half maximum.
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
	out = c2t.smooth_with_kernel(array, gg)
	return out




		

