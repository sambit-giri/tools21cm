'''
Methods to simulate the primary beam of radio telescope
'''

import numpy as np
from . import cosmo as cm
from glob import glob
from . import conv

def primary_beam(array, z, nu_axis=2, beam_func='Gaussian', boxsize=None, D=0.35):
	"""
	array    : ndarray
		The array of brightness temperature.
	z        : float
		redshift. With one value, the function will assume the array to be coeval. 
	           In case of light-cone, provide size of z should be equivalent to the length of 
	           frequency axis of the array.
	nu_axis  : int
		The frequency axis of the array (Default: 2) 
	beam_func: str
		The type of function to model the primary beam. The options are 'gaussian', 'sigmoid'
		   and 'step'. Default: 'circular'
	boxsize  : float
		Size of the box in physical units (cMpc). Default: From set simulation constants.
	D        : float
		Diameter of the dish in metres. Default: 0.35
	"""
	assert array.ndim > 1
	if boxsize is None: boxsize = conv.LB
	beam = np.zeros(array.shape)
	if array.ndim == 2: return array*circular_beam(array.shape[0], z, D=D, beam_func=beam_func, boxsize=boxsize)
	if nu_axis!=2 : array = np.swapaxes(array, nu_axis, 2)
	if np.array(z).size == 1: z = z*np.ones(array.shape[2])
	for i in xrange(z.size): 
		beam[:,:,i] = circular_beam(array.shape[0], z[i], D=D, beam_func=beam_func, boxsize=boxsize)
	beamed = array*beam
	if nu_axis!=2 : beamed = np.swapaxes(beamed, 2, nu_axis)
	return beamed

def circular_beam(ncells, z, D=0.35, beam_func='Gaussian', boxsize=None):
	if boxsize is None: boxsize = conv.LB
	HWHM_beam = cm.nu_to_wavel(cm.z_to_nu(z))/D/2
	n_FWHM = ncells*HWHM_beam*2/cm.angular_size_comoving(boxsize, z)
	xx, yy = np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]
	if beam_func.lower()=='step':
		beam = np.zeros((nx,ny))
		beam[xx**2+yy**2<=n_FWHM] = 1
	elif beam_func.lower()=='gaussian':
		sigma = n_FWHM/2/2/np.log(2)
		beam  = np.exp(-(xx**2 + yy**2)/(2.*sigma**2))
	elif beam_func.lower()=='sigmoid':
		r0, b = n_FWHM/2., 0.5
		r2 = xx**2+yy**2
		beam = 1.-1/(1+np.exp(b*r0-b*r2**0.5))
	else:
		print("Please choose between step, gaussian and sigmoid functions.")

	return beam/beam.max()
