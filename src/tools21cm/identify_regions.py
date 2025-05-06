'''
Methods to identify regions of interest in images.
'''

import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from . import superpixels
from sklearn.cluster import KMeans

def bubbles_from_slic(data, n_segments=5000, bins='knuth', verbose=True):
	"""
	@ Giri at al. (2018b)
	It is a method to identify regions of interest in noisy images.
	The method is an implementation of the superpixel based approach, called SLIC (Achanta et al. 2010),
	used in the field of computer vision.

	Parameters
	----------
	data      : ndarray
		The brightness temperature cube.
	n_segments: int
		Number of superpixels (Default: 2000).
	bins      : int or str
		Number of bins for the PDF used for stitching.
		'blocks', 'knuth', 'scotts', 'freedman' rules to determine bins automatically 
		can also be choosen. (Default: 'knuth').

	Returns
	-------
	Binary cube where pixels identified as region of interest are the True.
	"""
	labels  = superpixels.slic_cube(data, n_segments=n_segments, verbose=verbose)
	bin_sl  = superpixels.stitch_superpixels(data, labels, bins=bins, binary=True, verbose=verbose)
	return bin_sl

def bubbles_from_kmeans(data, upper_lim=True, n_jobs=1, n_clusters=3):
	"""
	@ Giri at al. (2018a)

	It is a method to identify regions of interest in noisy images.
	The method finds the optimal threshold using the 1D PDF of the image. It gives similar results compared
	to the Otsu's method.

	Parameters
	----------
	data      : ndarray
		The brightness temperature/xfrac cube.
	upper_lim : bool
		This decides which mode in the PDF is to be identified.
		'True' identifies ionized regions in brightness temperature, while
		'False' identifies in the xfrac data (Default: True).
	n_jobs    : int
		Number of cores to use (Default: 1).
	n_cluster : int
		Number of clusters found in the PDF (Default: 3).

	Returns
	-------
	Binary cube where pixels identified as region of interest are the True.
	"""
	if np.unique(data).size<2:
		print('The data array is single valued and thus the entire array is one region.')
		return np.ones_like(data)
	if np.unique(data).size==2: n_clusters=2
	if n_clusters==2: array, t_th = threshold_kmeans(data, upper_lim=upper_lim, n_jobs=n_jobs)
	else: array = threshold_kmeans_3cluster(data, upper_lim=upper_lim, n_jobs=n_jobs)
	return array

def bubbles_from_fixed_threshold(data, threshold=0, upper_lim=True):
	"""
	@ Giri at al. (2018a)

	It is a method to identify regions of interest in noisy images.
	The method uses a fixed threshold.

	Parameters
	----------
	data      : ndarray
		The brightness temperature or ionization fraction cube.
	threshold : float
		The fixed threshold value (Default: 0). 
	upper_lim : bool
		This decides which mode in the PDF is to be identified.
		'True' identifies ionized regions in brightness temperature, while
		'False' identifies in the xfrac data (Default: True).

	Returns
	-------
	Binary cube where pixels identified as region of interest are the True.
	"""
	if upper_lim: return (data<=threshold)
	else: return  (data>=threshold)

def bubbles_from_triangle3D(data, upper_lim=True, nbins=256):
	"""
	Gazagnes et al. (2021) https://arxiv.org/abs/2011.08260
	Giri at al. (2018b)    https://arxiv.org/abs/1801.06550
	Zack et al. (1977)     https://journals.sagepub.com/doi/10.1177/25.7.70454

	It is a method to identify regions of interest in noisy images.
	The method uses a triangle threshold proposed in Zack et al. (1977) for 2D images.
	Here we use the same method for 3D data as proposed in Giri at al. (2018b) and Gazagnes et al. (2021).

	Parameters
	----------
	data      : ndarray
		The brightness temperature or ionization fraction cube. 
	upper_lim : bool
		This decides which mode in the PDF is to be identified.
		'True' identifies ionized regions in brightness temperature, while
		'False' identifies in the xfrac data (Default: True).
	nbins : int
		The number of bins used for histogram (Default: 256).

	Returns
	-------
	Binary cube where pixels identified as region of interest are the True.
	"""
	threshold = threshold_triangle(data, nbins=nbins)
	if upper_lim: return (data<=threshold)
	else: return  (data>=threshold)

def bubbles_from_triangle2D(data, upper_lim=True, nbins=256, loc_axis=2):
	"""
	Gazagnes et al. (2021) https://arxiv.org/abs/2011.08260
	Giri at al. (2018b)    https://arxiv.org/abs/1801.06550
	Zack et al. (1977)     https://journals.sagepub.com/doi/10.1177/25.7.70454

	It is a method to identify regions of interest in noisy images.
	The method uses a triangle threshold proposed in Zack et al. (1977) for 2D images.
	Here we use the technique proposed in Gazagnes et al. (2021) and determine threshold from 2D slices and determine take the median value.

	Parameters
	----------
	data      : ndarray
		The brightness temperature or ionization fraction cube. 
	upper_lim : bool
		This decides which mode in the PDF is to be identified.
		'True' identifies ionized regions in brightness temperature, while
		'False' identifies in the xfrac data (Default: True).
	nbins : int
		The number of bins used for histogram (Default: 256).

	Returns
	-------
	Binary cube where pixels identified as region of interest are the True.
	"""
	if loc_axis in [0,-3]: threshold_s = np.array([threshold_triangle(data[i,:,:], nbins=nbins) for i in range(data.shape[0])])
	elif loc_axis in [1,-2]: threshold_s = np.array([threshold_triangle(data[:,i,:], nbins=nbins) for i in range(data.shape[1])])
	else: threshold_s = np.array([threshold_triangle(data[:,:,i], nbins=nbins) for i in range(data.shape[2])])
	threshold = np.median(threshold_s) 
	if upper_lim: return (data<=threshold)
	else: return  (data>=threshold)


def threshold_kmeans(cube, upper_lim=False, mean_remove=True, n_jobs=1):
	#The input is the brightness temperature cube.
	
	array = np.zeros(cube.shape)
	X  = cube.reshape(-1,1)
	y = KMeans(n_clusters=2, n_init='auto').fit_predict(X)
	t_th = X[y==0].max()/2.+X[y==1].max()/2.
	if upper_lim: array[cube<=t_th] = 1
	else: array[cube>t_th] = 1
	print("The output contains a tuple with binary-cube and determined-threshold.")
	return array, t_th
	
def threshold_kmeans_3cluster(cube, upper_lim=False, n_jobs=1):
	#The input is the brightness temperature cube.
	
	km = KMeans(n_clusters=3, n_init='auto')
	X  = cube.reshape(-1,1)
	array = np.zeros(X.shape)
	km.fit(X)
	y = km.labels_
	centers = km.cluster_centers_
	if upper_lim: true_label = centers.argmin()
	else: true_label = centers.argmax()
	array[y==true_label] = 1
	array = array.reshape(cube.shape)
	return array
