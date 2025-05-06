import numpy as np
import sys, scipy
from time import time
from sklearn.neighbors import BallTree, KDTree
from sklearn.neighbors import NearestNeighbors
from .bubble_stats import fof
from skimage.measure import label

def EulerCharacteristic(data, thres=0.5, neighbors=6, speed_up='numba', verbose=True):
	"""
	Parameters
	----------
	data     : ndarray
		The data cube containing the structure.
	thres    : float
		The threshold to create the binary field from the data (Default: 0.5)
		Ignore this parameter if data is already a binary field.
	neighbors: int
		Define the connectivity to the neighbors (Default: 6).
	speed_up: str
		Method used to speed up calculation.
	verbose: bool
		If True, verbose is printed.

	Returns
	-------
	Euler characteristics value.
	"""
	tstart = time()
	A = (data>thres).astype(int)
	if speed_up is None: speed_up = 'numpy'
	if speed_up.lower()=='cython': 
		try:
			from . import ViteBetti_cython as VB
		except:
			print('Cython module not found. Using the python module that might be slow...')
			from . import ViteBetti as VB
	elif speed_up.lower()=='numba': 
		try:
			from . import ViteBetti_numba as VB
		except:
			print('Numba module not found. Using the python module that might be slow...')
			from . import ViteBetti as VB
	else: 
		from . import ViteBetti as VB
	if verbose: print(f'Creating CubeMap...')
	if neighbors==6 or neighbors==4: C = VB.CubeMap(A)
	else: C = VB.CubeMap(1-A)
	if verbose: print(f'...done in {(time()-tstart)/60:.2f} mins')
	elem, count = np.unique(C, return_counts=1)
	V = count[elem==1] if len(count[elem==1])!=0 else 0
	E = count[elem==2] if len(count[elem==2])!=0 else 0
	F = count[elem==3] if len(count[elem==3])!=0 else 0
	C = count[elem==4] if len(count[elem==4])!=0 else 0
	return float(V-E+F-C)

def betti0(data, thres=0.5, neighbors=6):
	"""
	Parameters
	----------
	data     : ndarray
		The data cube containing the structure.
	thres    : float
		The threshold to create the binary field from the data (Default: 0.5)
		Ignore this parameter if data is already a binary field.
	neighbors: int
		Define the connectivity to the neighbors (Default: 6).

	Returns
	-------
	Betti 0
	"""
	A = (data>thres)*1
	if neighbors==6 or neighbors==4: b0 = label(A, return_num=1, connectivity=1)[1]#np.unique(fof(B, use_skimage=1)[0]).size-1
	else: b0 = label(A, return_num=1, connectivity=2)[1]
	return b0

def betti2(data, thres=0.5, neighbors=6):
	"""
	Parameters
	----------
	data     : ndarray
		The data cube containing the structure.
	thres    : float
		The threshold to create the binary field from the data (Default: 0.5)
		Ignore this parameter if data is already a binary field.
	neighbors: int
		Define the connectivity to the neighbors (Default: 6).

	Returns
	-------
	Betti 2
	"""
	A = (data>thres)*1
	if neighbors==6 or neighbors==4: b2 = label(1-A, return_num=1, connectivity=1)[1]#b2 = np.unique(fof(1-B, use_skimage=1)[0]).size-1
	else: b2 = label(1-A, return_num=1, connectivity=2)[1]
	return b2

def betti1(data, thres=0.5, neighbors=6, b0=None, b2=None, chi=None, speed_up='numba', verbose=True):
	"""
	Parameters
	----------
	data     : ndarray
		The data cube containing the structure.
	thres    : float
		The threshold to create the binary field from the data (Default: 0.5)
		Ignore this parameter if data is already a binary field.
	neighbors: int
		Define the connectivity to the neighbors (Default: 6).
	use_numba: bool
		If True, numba package is used to speed up calculation.

	Returns
	-------
	Betti 1
	"""
	if chi is None: chi = EulerCharacteristic(data, thres=thres, neighbors=neighbors, speed_up=speed_up, verbose=verbose)
	if b0 is None: b0  = betti0(data, thres=thres, neighbors=neighbors)
	if b2 is None: b2  = betti2(data, thres=thres, neighbors=neighbors)
	return b0 + b2 - chi

def genus(data, xth=0.5):
	"""
	@ Chen & Rong (2010) 

	It is an implementation of Gauss-Bonnet theorem in digital spaces.

	Parameters
	----------
	data : ndarray
		The input data set which contains the structures.
	xth  : float
		The threshold to put on the data set (Default: 0.5).

	Returns
	-------
	Genus
	"""
	data = data>=xth
	stela = np.zeros((3,3,3))
	stela[:2,:2,:2] = 1
	ma    = count_neighbors(data, stela)+count_neighbors(1-data, stela)
	steld = np.zeros((3,3,3))
	steld[0,0,0], steld[0,0,1], steld[0,0,2] = 1,1,1
	steld[0,1,0], steld[0,1,1], steld[0,1,2] = 1,1,1
	steld[1,1,0], steld[1,1,1], steld[1,1,2] = 1,1,1
	steld[1,0,0], steld[1,0,1], steld[1,0,2] = 1,1,1
	steld[2,0,1], steld[2,0,2], steld[2,1,1], steld[2,1,2] = 1,1,1,1
	md    = count_neighbors(data, steld)+count_neighbors(1-data, steld)
	stele = np.zeros((3,3,3))
	stele[0,1,0], stele[0,1,1], stele[0,1,2], stele[1,1,0] = 1,1,1,1
	stele[1,1,2], stele[1,2,1], stele[1,2,2], stele[2,2,1] = 1,1,1,1
	stele[2,1,1], stele[1,0,0], stele[1,0,1], stele[2,0,1] = 1,1,1,1
	stele[0,0,0], stele[0,0,1], stele[0,0,2], stele[1,0,2] = 1,1,1,1
	stele[2,0,2], stele[2,1,2], stele[2,2,2], stele[1,1,1] = 1,1,1,1
	me    = count_neighbors(data, stele)+count_neighbors(1-data, stele)
	stelf = np.zeros((3,3,3))
	stelf[0,1,0], stelf[0,1,1], stelf[0,2,1], stelf[1,2,1] = 1,1,1,1
	stelf[1,2,2], stelf[1,1,2], stelf[2,1,2], stelf[2,1,1] = 1,1,1,1
	stelf[2,0,1], stelf[1,0,1], stelf[1,0,0], stelf[1,1,0] = 1,1,1,1
	stelf[0,0,0], stelf[0,0,1], stelf[0,0,2], stelf[0,2,2] = 1,1,1,1
	stelf[1,0,2], stelf[2,0,2], stelf[1,1,1]               = 1,1,1
	mf    = count_neighbors(data, stelf)+count_neighbors(1-data, stelf)
	g = 1 + (md + 2*(me+mf) - ma)/8.				   ## Gauss-Bonnet theorem
	return g


def count_neighbors(data, stela):
	count  = count_neighbors_once(data, stela)
	stela1 = np.rot90(stela,  1, (0,1)); count  += count_neighbors_once(data, stela1)
	stela2 = np.rot90(stela,  2, (0,1)); count  += count_neighbors_once(data, stela2)
	stela3 = np.rot90(stela,  3, (0,1)); count  += count_neighbors_once(data, stela3)
	stela4 = np.rot90(stela,  2, (0,2)); count  += count_neighbors_once(data, stela4)
	stela5 = np.rot90(stela4, 1, (0,1)); count  += count_neighbors_once(data, stela5)
	stela6 = np.rot90(stela4, 2, (0,1)); count  += count_neighbors_once(data, stela6)
	stela7 = np.rot90(stela4, 3, (0,1)); count  += count_neighbors_once(data, stela7)
	return count

def count_neighbors_once(data, stela):
	stel26 = np.ones((3,3,3))
	neb26  = scipy.ndimage.filters.convolve(data, stel26, mode='wrap')*data
	neba   = scipy.ndimage.filters.convolve(data, stela, mode='wrap')*data
	elem, numb = np.unique(neba[neba==neb26], return_counts=1)
	return 0 if numb[elem==stela.sum()].size==0 else numb[elem==stela.sum()][0]



