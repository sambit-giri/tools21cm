'''
Methods to simulate the primary beam of radio telescope
'''

import numpy as np
from tqdm import tqdm
from scipy.special import j1
from astropy import units

from . import cosmo as cm
from . import conv

def primary_beam(array, z, nu_axis=2, beam_func='Gaussian', boxsize=None, D=40.):
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
		The type of function to model the primary beam. The options are 'gaussian', 'bessel', 
		'sigmoid' and 'step'. Default: 'gaussian'
	boxsize  : float
		Size of the box in physical units (cMpc). Default: From set simulation constants.
	D        : float
		Diameter of the dish in metres. Default: 40.
	"""
	assert array.ndim > 1
	if boxsize is None: 
		boxsize = conv.LB
	beam = np.zeros(array.shape)
	if array.ndim==2: 
		beamed = array*circular_beam(array.shape[0], z, D=D, beam_func=beam_func, boxsize=boxsize)
	else:
		if nu_axis!=2: 
			array = np.swapaxes(array, nu_axis, 2)
		if np.array(z).size==1: 
			z = z*np.ones(array.shape[2])
		for i in tqdm(range(z.size)): 
			beam[:,:,i] = circular_beam(array.shape[0], z[i], D=D, beam_func=beam_func, boxsize=boxsize)
		beamed = array*beam
		if nu_axis!=2: 
			beamed = np.swapaxes(beamed, 2, nu_axis)
	return beamed

def primary_beam_null(z, D=40.):
    """
    Calculates the physical diameter of the first null of an Airy disk
    projected on the sky at redshift z.
    """
    if isinstance(D,units.Quantity):
        D = D.to('m').value
    l_null = cm.z_to_cdist(z)*cm.nu_to_wavel(cm.z_to_nu(z))/D*1.22
    return l_null 

def circular_beam(ncells, z, D=40., beam_func='Gaussian', boxsize=None):
    """
    Generates a 2D circular beam pattern.
    """
    if boxsize is None: 
        boxsize = conv.LB
        
    # Get the diameter of the first null in cMpc
    l_null = primary_beam_null(z, D=D)
    
	# Create a grid of distances from the center
    xx, yy = np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]*boxsize/ncells
    rr = np.sqrt(xx**2+yy**2)
    
    if beam_func.lower()=='step':
        beam = np.zeros_like(rr)
        # Use the radius of the first null (l_null / 2)
        beam[rr<=(l_null/2)] = 1
    elif beam_func.lower()=='gaussian':
        # The FWHM of an Airy disk is ~1.029 * lambda / D, which is l_null / 1.186
        fwhm = l_null / 1.186 
        sigma = fwhm / 2.355 
        beam  = np.exp(-rr**2 / (2. * sigma**2))
    elif beam_func.lower() == 'bessel':
        r_null = l_null / 2.0   # The radius of the first null is half the diameter
        # Calculate the scaling factor alpha for the Airy disk formula
        alpha = 3.8317 / r_null # The first zero of J1(x) is at x approx 3.8317
        argument = alpha * rr   # The argument for the Bessel function
        # Handle the center (r=0) to avoid division by zero.
        # The limit of (2*J1(x)/x)^2 as x->0 is 1.0.
        center_idx = np.where(rr == 0)
        argument[center_idx] = 1e-9 # Use a small number to prevent error
        # Calculate the beam pattern using the Airy disk formula
        beam = (2 * j1(argument) / argument)**2
        beam[center_idx] = 1.0  # Manually set the peak to 1
    elif beam_func.lower()=='sigmoid':
        r0, b = l_null/2., 0.5
        beam = 1.-1/(1+np.exp(b*r0-b*rr**2**0.5))
    else:
        raise ValueError("Please choose between 'step', 'gaussian', 'bessel', and 'sigmoid' for beam_func.")
    
    return beam/beam.max()