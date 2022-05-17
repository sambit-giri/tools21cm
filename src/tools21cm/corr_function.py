import numpy as np 
from scipy.interpolate import splrep,splev
from .power_spectrum import power_spectrum_1d

def correlation_function(input_array_nd, rbins=10, kbins=10, box_dims=None, binning='log'):
	'''
	Function estimates the spherically averaged power spectrum and 
	then calulates the correlation function using the following equation:
	.. math:: \\xi(r) = (2\\pi^2)^{-1} \\int k^2 P(k) \\mathrm{sinc}(kr)dk

	Parameters: 
				input_array_nd (numpy array): the data array
				rbins = 10 (integer or array-like): The number of bins,
				    or a list containing the bin edges. If an integer is given, the bins
				    are logarithmically spaced.
				kbins = 100 (integer): The number of bins,
				    used to estimate the power spectrum. If an integer is given, the bins
				    are logarithmically spaced.
				box_dims = None (float or array-like): the dimensions of the 
				    box in Mpc. If this is None, the current box volume is used along all
				    dimensions. If it is a float, this is taken as the box length
				    along all dimensions. If it is an array-like, the elements are
				    taken as the box length along each axis.
				binning = 'log' : It defines the type of binning in k-space. The other option is 
				                'linear' or 'mixed'.
	Returns: 
				A tuple with (Pk, bins), where Pk is an array with the 
				power spectrum and bins is an array with the k bin centers.
	'''

	ps, ks, n_modes = power_spectrum_1d(input_array_nd, 
											kbins=rbins if kbins is None else kbins, 
											box_dims=None, 
											return_n_modes=True, 
											binning='log', 
											)
	if type(rbins)==int: 
		rbins = 10**np.linspace(np.log10(2*box_dims/max(input_array_nd.shape)), np.log10(box_dims/2), rbins) if binning=='log' \
					else np.linspace(2*box_dims/max(input_array_nd.shape), box_dims/2, rbins)

	ps_tck = splrep(ks[np.isfinite(ps)], ps[np.isfinite(ps)])
	knew  = np.linspace(ks[0],ks[-1],100)
	integ = knew**2*splev(knew,ps_tck)*np.sinc(knew[None,:]*rbins[:,None])
	corr  = np.trapz(integ, knew, axis=1) 

	return corr/2/np.pi**2, rbins

