import numpy as np
from sklearn.neighbors import BallTree, KDTree
from sklearn.neighbors import NearestNeighbors
import scipy
import ViteBetti
from bubble_stats import fof

def EulerCharacteristic(data, thres=0.5):
	"""
	Parameters
	----------
	data : The data cube containing the structure.
	thres: The threshold to create the binary field from the data (Default: 0.5)
	       Ignore this parameter if data is already a binary field.
	"""
        A = data>thres
	B = (A*1)
	C = ViteBetti.CubeMap(B)
	D = ViteBetti.CubeMap(1-B)
        E = ViteBetti.EulerCharacteristic_seq(C)/2. + ViteBetti.EulerCharacteristic_seq(D)/2
	return E

def beta0(data, thres=0.5):
	"""
	Parameters
	----------
	data : The data cube containing the structure.
	thres: The threshold to create the binary field from the data (Default: 0.5)
	       Ignore this parameter if data is already a binary field.
	"""
	A = data>thres
	B = (A*1)
	b0 = np.unique(fof(B, use_skimage=1)[0]).size-1
	return b0

def beta2(data, thres=0.5):
	"""
	Parameters
	----------
	data : The data cube containing the structure.
	thres: The threshold to create the binary field from the data (Default: 0.5)
	       Ignore this parameter if data is already a binary field.
	"""
	A = data>thres
	B = (A*1)
	b2 = np.unique(fof(1-B, use_skimage=1)[0]).size-1
	return b2

def beta1(data, thres=0.5):
	"""
	Parameters
	----------
	data : The data cube containing the structure.
	thres: The threshold to create the binary field from the data (Default: 0.5)
	       Ignore this parameter if data is already a binary field.
	"""
	chi = EulerCharacteristic(data, thres=thres)
	b0  = beta0(data, thres=thres)
	b2  = beta2(data, thres=thres)
	return b0 + b2 - chi

def genus(data, xth=0.5):
	"""
	It is an implementation of Gauss-Bonnet theorem in digital spaces.
	@ Chen & Rong (2010) 
	Parameters
	----------
	data : The input data set which contains the structures.
	xth  : The threshold to put on the data set (Default: 0.5).
	Return
	----------
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



