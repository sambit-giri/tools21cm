import numpy as np 
from scipy.interpolate import splrep,splev
from scipy.spatial import cKDTree
from tqdm import tqdm
from .power_spectrum import power_spectrum_1d, cross_power_spectrum_1d
from . import conv

def ps_to_corr(ps, ks, rbins=10, box_dims=None, n_grid=None, binning='log'):
	'''
	Function that converts the spherically averaged power spectrum into 
	the correlation function using the following equation:
	.. math:: \\xi(r) = \\int 4\\pi k^2 P(k) \\mathrm{sinc}(kr)dk
	'''
	if isinstance(rbins, (int)): 
		rbins = 10**np.linspace(np.log10(2*box_dims/n_grid), np.log10(box_dims/2), rbins) if binning=='log' \
					else np.linspace(2*box_dims/n_grid, box_dims/2, rbins)

	# if binning=='log':
	# 	ps_tck = splrep(np.log10(ks[np.isfinite(ps)]), ps[np.isfinite(ps)], k=1)
	# 	ps_fun = lambda k: splev(np.log10(k),ps_tck)
	# else:
	# 	ps_tck = splrep(ks[np.isfinite(ps)], ps[np.isfinite(ps)], k=1)
	# 	ps_fun = lambda k: splev(k,ps_tck)
	# intergral_bins = kbins*10
	# knew = np.linspace(ks[0],ks[-1],intergral_bins)
	# pnew = ps_fun(knew)
	knew, pnew = ks[np.isfinite(ps)], ps[np.isfinite(ps)]

	integ = knew**2*pnew*np.sinc(knew[None,:]*rbins[:,None])
	corr  = np.trapz(integ, knew, axis=1) 

	return corr*4*np.pi, rbins

def correlation_function(input_array_nd, rbins=10, kbins=10, box_dims=None, binning='log', k_binning='log'):
	'''
	Function estimates the spherically averaged power spectrum and 
	then calulates the correlation function using the following equation:
	.. math:: \\xi(r) = \\int 4\\pi k^2 P(k) \\mathrm{sinc}(kr)dk

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
				binning = 'log' (str): It defines the type of binning in k-space. The other option is 
				                'linear' or 'mixed'.
				intergral_bins = 100 (int): The number bins to use for numerically solving the intergral.
	Returns: 
				A tuple with (Pk, bins), where Pk is an array with the 
				power spectrum and bins is an array with the k bin centers.
	'''

	ps, ks, n_modes = power_spectrum_1d(input_array_nd, 
											kbins=rbins*10 if kbins is None else kbins, 
											box_dims=box_dims, 
											return_n_modes=True, 
											binning=k_binning, 
											)
	return ps_to_corr(ps, ks, rbins=rbins, box_dims=box_dims, n_grid=input_array_nd.shape[0], binning=binning)

def cross_correlation_function(input_array1_nd, input_array2_nd, rbins=10, kbins=100, box_dims=None, binning='log', k_binning='log'):
	'''
	Function estimates the spherically averaged cross power spectrum and 
	then calulates the cross correlation function using the following equation:
	.. math:: \\xi(r) = \\int 4\\pi k^2 P(k) \\mathrm{sinc}(kr)dk

	Parameters: 
				input_array1_nd (numpy array): the data array 1
				input_array2_nd (numpy array): the data array 2
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
				binning = 'log' (str): It defines the type of binning in k-space. The other option is 
				                'linear' or 'mixed'.
				intergral_bins = 100 (int): The number bins to use for numerically solving the intergral.
	Returns: 
				A tuple with (Pk, bins), where Pk is an array with the 
				power spectrum and bins is an array with the k bin centers.
	'''

	ps, ks, n_modes = cross_power_spectrum_1d(input_array1_nd, input_array2_nd,
											kbins=rbins*10 if kbins is None else kbins, 
											box_dims=box_dims, 
											return_n_modes=True, 
											binning=k_binning, 
											)

	return ps_to_corr(ps, ks, rbins=rbins, box_dims=box_dims, n_grid=input_array1_nd.shape[0], binning=binning)

def landy_szalay_estimator(data, randoms=None, rbins=10, box_dims=None, binning='log', **kwargs):
	'''
	Function to estimate the two-point correlation function using the Landy-Szalay estimator.

	Parameters:
		data (numpy array): array of positions of data points (e.g., galaxies).
		randoms (numpy array): array of positions of random points.
		rbins (integer or array-like): The number of radial bins or bin edges.
			If an integer is provided, bins are logarithmically or linearly spaced.
		box_dims (float): Length of the box in each direction in Mpc.
		binning (string): 'log' for logarithmic binning, 'linear' for linear binning.
		
	Returns:
		xi (numpy array): The estimated correlation function.
		rbin_centers (numpy array): The centers of the radial bins.
	'''
	if box_dims is None:
		box_dims = conv.LB 
		print(f'Setting box_dims to ({box_dims},{box_dims},{box_dims}) Mpc.')
		
	if data.ndim==3:
		data = np.argwhere(data>0)*box_dims/data.shape[0]
		
	if randoms is None:
		randoms = np.random.uniform(data.min(), data.max(), data.shape)
		
	# Determine the radial bins
	if isinstance(rbins, int):
		rmin = kwargs.get('rmin')
		rmax = kwargs.get('rmax')
		if rmin is None:
			rmin = 0.01  
		if rmax is None:
			rmax = box_dims # 50.0  
		if binning == 'log':
			rbins = np.logspace(np.log10(rmin), np.log10(rmax), rbins)
		else:
			rbins = np.linspace(rmin, rmax, rbins)

	rbin_centers = (rbins[:-1] + rbins[1:]) / 2  # Midpoints of the bins

	# Step 1: Pair Counting

	# Create k-d trees for fast nearest neighbor searching
	data_tree = cKDTree(data)
	random_tree = cKDTree(randoms)

	# Count DD pairs (data-data)
	DD, _ = np.histogram(data_tree.query_pairs(rmax, output_type='ndarray'), bins=rbins)

	# Count RR pairs (random-random)
	RR, _ = np.histogram(random_tree.query_pairs(rmax, output_type='ndarray'), bins=rbins)

	# Count DR pairs (data-random)
	DR = np.zeros(len(rbins)-1)
	for point in tqdm(data):
		distances, _ = random_tree.query(point, k=len(randoms), distance_upper_bound=rmax)
		distances = distances[distances <= rmax]  # Exclude infinite distances (points outside search range)
		DR += np.histogram(distances, bins=rbins)[0]

	# Normalize the pair counts
	num_data = len(data)
	num_random = len(randoms)

	DD = DD / (num_data * (num_data - 1) / 2.0)
	RR = RR / (num_random * (num_random - 1) / 2.0)
	DR = DR / (num_data * num_random)

	# Step 2: Calculate the Landy-Szalay estimator
	xi = (DD - 2 * DR + RR) / RR

	return xi, rbin_centers

# Example usage:
# data = np.random.uniform(0, 100, (64, 64, 64))  # 1000 random data points in a 100x100x100 box
# randoms = np.random.uniform(0, 100, (1000, 3))  # 1000 random points for the random catalog
# xi, r_bins = landy_szalay_estimator(data, randoms)
