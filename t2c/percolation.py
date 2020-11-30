'''
Methods to study EoR/CD through the lens of percolation theory.
'''

import numpy as np
import os,sys
import datetime, time
from . import bubble_stats
from tqdm import tqdm

def find_percolation_cluster(data, xth=0.5, connectivity=1):
	"""
	Determines the sizes using the friends-of-friends approach.
	It assumes the length of the grid as the linking length.
	
	Parameters
	----------
	data: ndarray 
		The array containing the input data
	xth: float 
		The threshold value (Default: 0.5)

	Returns
	-------
	Size of the cluster if percolation cluster is present else -1.
	"""

	out_map, size_list = bubble_stats.fof(data, xth=xth, connectivity=connectivity)
	out_uniq  = np.unique(out_map, return_counts=1)
	if out_uniq[0].size<2:
		if out_uniq[0][0]==1: return out_map.size
		print('There is no percolation cluster in the data.') 
		return 0
	out_label = np.zeros_like(out_map)
	out_label[out_map==out_uniq[0][1]] = 1
	isPerc = False
	if np.any(out_label+np.roll(out_label, 1, axis=0)>1): isPerc = True
	if np.any(out_label+np.roll(out_label, 1, axis=1)>1): isPerc = True
	if np.any(out_label+np.roll(out_label, 1, axis=2)>1): isPerc = True
	if isPerc:
		return out_label.sum()
	print('There is no percolation cluster in the data.')
	return 0




