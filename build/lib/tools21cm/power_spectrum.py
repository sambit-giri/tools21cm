import numpy as np
import const
import conv
from helper_functions import print_msg, get_eval
from scipy import fftpack


def power_spectrum_nd(input_array, box_dims=None):
	''' 
	Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used
	
	Parameters:
		* input_array (numpy array): the array to calculate the 
			power spectrum of. Can be of any dimensions.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
	
	Returns:
		The power spectrum in the same dimensions as the input array.		
	'''

	box_dims = _get_dims(box_dims, input_array.shape)

	print_msg( 'Calculating power spectrum...')
	ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	print_msg( '...done')

	# scale
	boxvol = np.product(map(float,box_dims))
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol
	
	return power_spectrum


def cross_power_spectrum_nd(input_array1, input_array2, box_dims):
	''' 
	Calculate the cross power spectrum two arrays and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used
	
	Parameters:
		* input_array1 (numpy array): the first array to calculate the 
			power spectrum of. Can be of any dimensions.
		* input_array2 (numpy array): the second array. Must have same 
			dimensions as input_array1.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
	
	Returns:
		The cross power spectrum in the same dimensions as the input arrays.
		
	TODO:
		Also return k values.
	'''

	assert(input_array1.shape == input_array2.shape)

	box_dims = _get_dims(box_dims, input_array1.shape)

	print_msg( 'Calculating power spectrum...')
	ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
	ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
	power_spectrum = np.real(ft1)*np.real(ft2)+np.imag(ft1)*np.imag(ft2)
	print_msg( '...done')

	# scale
	#boxvol = float(box_side)**len(input_array1.shape)
	boxvol = np.product(map(float,box_dims))
	pixelsize = boxvol/(np.product(map(float,input_array1.shape)))
	power_spectrum *= pixelsize**2/boxvol

	return power_spectrum


def radial_average(input_array, box_dims, kbins=10, binning='log', breakpoint=0.1):
	'''
	Radially average data. Mostly for internal use.
	
	Parameters: 
		* input_array (numpy array): the data array
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* kbins = 10 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
			
	Returns:
		A tuple with (data, bins, n_modes), where data is an array with the 
		averaged data, bins is an array with the bin centers and n_modes is the 
		number of modes in each bin

	'''

	k_comp, k = _get_k(input_array, box_dims)

	kbins = _get_kbins(kbins, box_dims, k, binning=binning, breakpoint=breakpoint)
	
	#Bin the data
	print_msg('Binning data...')
	dk = (kbins[1:]-kbins[:-1])/2.
	#Total power in each bin
	outdata = np.histogram(k.flatten(), bins=kbins,
						weights = input_array.flatten())[0]
	#Number of modes in each bin
	n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
	outdata /= n_modes
	
	return outdata, kbins[:-1]+dk, n_modes
	

def power_spectrum_1d(input_array_nd, kbins=100, box_dims=None, return_n_modes=False, binning='log', breakpoint=0.1):
	''' Calculate the spherically averaged power spectrum of an array 
	and return it as a one-dimensional array.
	
	Parameters: 
		* input_array_nd (numpy array): the data array
		* kbins = 100 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* return_n_modes = False (bool): if true, also return the
			number of modes in each bin
		* binning = 'log' : It defines the type of binning in k-space. The other option is 
				    'linear' or 'mixed'.
			
	Returns: 
		A tuple with (Pk, bins), where Pk is an array with the 
		power spectrum and bins is an array with the k bin centers.
	'''

	box_dims = _get_dims(box_dims, input_array_nd.shape)

	input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)	

	ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims=box_dims, binning=binning, breakpoint=breakpoint)
	if return_n_modes:
		return ps, bins, n_modes
	return ps, bins


def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, kbins=100, box_dims=None, return_n_modes=False, binning='log',breakpoint=0.1):
	''' Calculate the spherically averaged cross power spectrum of two arrays 
	and return it as a one-dimensional array.
	
	Parameters: 
		* input_array1_nd (numpy array): the first data array
		* input_array2_nd (numpy array): the second data array
		* kbins = 100 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* return_n_modes = False (bool): if true, also return the
			number of modes in each bin
		* binning = 'log' : It defines the type of binning in k-space. The other option is 
				    'linear' or 'mixed'.
			
	Returns: 
		A tuple with (Pk, bins), where Pk is an array with the 
		cross power spectrum and bins is an array with the k bin centers.
	'''

	box_dims = _get_dims(box_dims, input_array1_nd.shape)

	input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd, box_dims=box_dims)	

	ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims = box_dims, binning=binning, breakpoint=breakpoint)
	if return_n_modes:
		return ps, bins, n_modes
	return ps, bins


def power_spectrum_mu(input_array, los_axis = 0, mubins=20, kbins=10, box_dims = None, weights=None,
					exclude_zero_modes = True):
	'''
	Calculate the power spectrum and bin it in mu=cos(theta) and k
	input_array is the array to calculate the power spectrum from
	
	Parameters: 
		* input_array (numpy array): the data array
		* los_axis = 0 (integer): the line-of-sight axis
		* mubins = 20 (integer): the number of mu bins
		* kbins = 10 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* exlude_zero_modes = True (bool): if true, modes with any components
			of k equal to zero will be excluded.
			
	Returns: 
		A tuple with (Pk, mubins, kbins), where Pk is an array with the 
		power spectrum of dimensions (n_mubins x n_kbins), 
		mubins is an array with the mu bin centers and
		kbins is an array with the k bin centers.
	
	'''

	box_dims = _get_dims(box_dims, input_array.shape)

	#Calculate the power spectrum
	powerspectrum = power_spectrum_nd(input_array, box_dims=box_dims)	

	return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes)


def cross_power_spectrum_mu(input_array1, input_array2, los_axis = 0, mubins=20, kbins=10, 
						box_dims = None, weights=None, exclude_zero_modes = True):
	'''
	Calculate the cross power spectrum and bin it in mu=cos(theta) and k
	input_array is the array to calculate the power spectrum from
	
	Parameters: 
		* input_array1 (numpy array): the first data array
		* input_array2 (numpy array): the second data array
		* los_axis = 0 (integer): the line-of-sight axis
		* mubins = 20 (integer): the number of mu bins
		* kbins = 10 (integer or array-like): The number of bins,
			or a list containing the bin edges. If an integer is given, the bins
			are logarithmically spaced.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		* exlude_zero_modes = True (bool): if true, modes with any components
			of k equal to zero will be excluded.
		
	Returns: 
		A tuple with (Pk, mubins, kbins), where Pk is an array with the 
		cross power spectrum of dimensions (n_mubins x n_kbins), 
		mubins is an array with the mu bin centers and
		kbins is an array with the k bin centers.
		
	TODO:
		Add support for (non-numpy) lists for the bins
	'''

	box_dims = _get_dims(box_dims, input_array1.shape)
	
	#Calculate the power spectrum
	powerspectrum = cross_power_spectrum_nd(input_array1, input_array2, box_dims=box_dims)	
	
	return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes)


def mu_binning(powerspectrum, los_axis = 0, mubins=20, kbins=10, box_dims=None, weights=None,
			exclude_zero_modes=True, binning='log'):
	'''
	This function is for internal use only.
	'''
	
	if weights != None:
		powerspectrum *= weights

	assert(len(powerspectrum.shape)==3)

	k_comp, k = _get_k(powerspectrum, box_dims)

	mu = _get_mu(k_comp, k, los_axis)

	#Calculate k values, and make k bins
	kbins = _get_kbins(kbins, box_dims, k, binning=binning)
	dk = (kbins[1:]-kbins[:-1])/2.
	n_kbins = len(kbins)-1
		
	#Exclude k_perp = 0 modes
	if exclude_zero_modes:
		good_idx = _get_nonzero_idx(powerspectrum.shape, los_axis)
	else:
		good_idx = np.ones_like(powerspectrum)

	#Make mu bins
	if isinstance(mubins,int):
		mubins = np.linspace(-1., 1., mubins+1)
	dmu = (mubins[1:]-mubins[:-1])/2.
	n_mubins = len(mubins)-1

	#Remove the zero component from the power spectrum. mu is undefined here
	powerspectrum[tuple(np.array(powerspectrum.shape)/2)] = 0.

	#Bin the data
	print_msg('Binning data...')
	outdata = np.zeros((n_mubins,n_kbins))
	for ki in range(n_kbins):
		print_msg('Bin %d of %d' % (ki, n_kbins))
		kmin = kbins[ki]
		kmax = kbins[ki+1]
		kidx = get_eval()('(k >= kmin) & (k < kmax)')
		kidx *= good_idx
		for i in range(n_mubins):
			mu_min = mubins[i]
			mu_max = mubins[i+1]
			idx = get_eval()('(mu >= mu_min) & (mu < mu_max) & kidx')
			outdata[i,ki] = np.mean(powerspectrum[idx])

			if weights != None:
				outdata[i,ki] /= weights[idx].mean()

	return outdata, mubins[:-1]+dmu, kbins[:-1]+dk


#Some methods for internal use

def _get_k(input_array, box_dims):
	'''
	Get the k values for input array with given dimensions.
	Return k components and magnitudes.
	For internal use.
	'''
	dim = len(input_array.shape)
	if dim == 1:
		x = np.arange(len(input_array))
		center = x.max()/2.
		kx = 2.*np.pi*(x-center)/box_dims[0]
		return [kx], kx
	elif dim == 2:
		x,y = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		k = np.sqrt(kx**2 + ky**2)
		return [kx, ky], k
	elif dim == 3:
		x,y,z = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2, \
						(z.max()-z.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		kz = 2.*np.pi * (z-center[2])/box_dims[2]

		k = get_eval()('(kx**2 + ky**2 + kz**2 )**(1./2.)') 		
		return [kx,ky,kz], k


def _get_mu(k_comp, k, los_axis):
	'''
	Get the mu values for given k values and 
	a line-of-sight axis.
	For internal use
	'''
		
	#Line-of-sight distance from center 
	if los_axis == 0:
		los_dist = k_comp[0]
	elif los_axis == 1:
		los_dist = k_comp[1]
	elif los_axis == 2:
		los_dist = k_comp[2]
	else:
		raise Exception('Your space is not %d-dimensional!' % los_axis)

	#mu=cos(theta) = k_par/k
	mu = los_dist/np.abs(k)
	mu[np.where(k < 0.001)] = np.nan
	
	return mu


def _get_kbins(kbins, box_dims, k, binning='log', breakpoint=0.1):
	'''
	Make a list of bin edges if kbins is an integer,
	otherwise return it as it is.
	'''
	if isinstance(kbins,int):
		kmin = 2.*np.pi/min(box_dims)
		if binning=='linear': kbins = np.linspace(kmin, k.max(), kbins+1)
		elif binning=='log': kbins = 10**np.linspace(np.log10(kmin), np.log10(k.max()), kbins+1)
		else:
			kbins_low  = np.linspace(kmin, k.max(), kbins+1)
			kbins_high = 10**np.linspace(np.log10(kmin), np.log10(k.max()), kbins+1)
			kbins = np.hstack((kbins_low[kbins_low<breakpoint],kbins_high[kbins_high>breakpoint]))		
	return kbins


def _get_dims(box_dims, ashape):
	'''
	If box dims is a scalar, assume that dimensions
	are cubic and make a list
	If it's not given, assume it's the default value of the box
	size
	Otherwise, return as it is
	'''
	if box_dims == None:
		return [conv.LB]*len(ashape)
	if not hasattr(box_dims, '__iter__'):
		return [box_dims]*len(ashape)
	return box_dims

def dimensionless_ps(data, kbins=100, box_dims=None, binning='log', factor=10):
	'''
	Parameters
	----------
	data    : The numpy data whose power spectrum is to be determined.
	kbins   : Number of bins for in the k-space (Default: 100).
	box_dims: The size of the box in Mpc (Default: Takes the value from the set_sim_constants).
	binning : The type of binning to be used for the k-space (Default: 'log').
	factor  : The factor multiplied to the given kbins to smooth the spectrum from (Default: 10).
	Return
	----------
	(\Delta^2, ks)
	'''
        Pk, ks = power_spectrum_1d(data, kbins=kbins*factor, box_dims=box_dims, binning=binning)
        ks_new = np.array([ks[factor*(i+0.5)] for i in xrange(kbins)])
        k3Pk   = ks**3*Pk
        k3Pk_  = np.array([k3Pk[factor*i:factor*(i+1)].mean() for i in xrange(kbins)])
        return k3Pk_/2/np.pi**2, ks_new

def _get_nonzero_idx(ps_shape, los_axis):
	'''
	Get the indices where k_perp != 0
	'''
	x,y,z = np.indices(ps_shape)
	if los_axis == 0:
		zero_idx = (y == ps_shape[1]/2)*(z == ps_shape[2]/2)
	elif los_axis == 1:
		zero_idx = (x == ps_shape[0]/2)*(z == ps_shape[2]/2)
	else:
		zero_idx = (x == ps_shape[0]/2)*(y == ps_shape[1]/2)
	good_idx = np.invert(zero_idx)
	return good_idx

