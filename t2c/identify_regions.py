'''
Methods to identify regions of interest in images.
'''

from skimage.filters import threshold_otsu
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from . import superpixels
from sklearn.cluster import KMeans
import numpy as np

def bubbles_from_slic(data, n_segments=5000, bins='knuth'):
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
	labels  = superpixels.slic_cube(data, n_segments=n_segments)
	bin_sl  = superpixels.stitch_superpixels(data, labels, bins=bins, binary=True)
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

def threshold_kmeans(cube, upper_lim=False, mean_remove=True, n_jobs=1):
	#The input is the brightness temperature cube.
	
	array = np.zeros(cube.shape)
	#km = KMeans(n_clusters=2)
	# if mean_remove:
	# 	if upper_lim: X = cube[cube<=cube.mean()].reshape(-1,1)
	# 	else: X = cube[cube>=cube.mean()].reshape(-1,1)
	# else:
	#  	X  = cube.reshape(-1,1)
	X  = cube.reshape(-1,1)
	y = KMeans(n_clusters=2, n_jobs=n_jobs).fit_predict(X)
	t_th = X[y==0].max()/2.+X[y==1].max()/2.
	if upper_lim: array[cube<=t_th] = 1
	else: array[cube>t_th] = 1
	print("The output contains a tuple with binary-cube and determined-threshold.")
	return array, t_th
	
def threshold_kmeans_3cluster(cube, upper_lim=False, n_jobs=1):
	#The input is the brightness temperature cube.
	
	km = KMeans(n_clusters=3, n_jobs=n_jobs)
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
