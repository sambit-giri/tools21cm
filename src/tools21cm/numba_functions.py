import numpy as np
from numba import jit, prange
from .usefuls import *

print("Using numba module")

@jit
def superpixel_map_numba(data, labels, mns=None):
	from superpixels import get_superpixel_means
	print("Using numba module")
	if mns is None: mns = get_superpixel_means(data, labels=labels)
	sp_map = np.zeros(data.shape)
	for i in prange(mns.size): sp_map[labels==i] = mns[i]
	return sp_map

@jit
def apply_operator_labelled_data_numba(data, labels, operator=np.mean):
	print("Using numba module")
	X   = data.flatten()
	y   = labels.flatten()
	elems, num = np.unique(y, return_counts=1)
	X1  = X[y.argsort()]
	out = []
	idx_low = 0
	for i in elems:
		idx_high = idx_low + num[i]
		out.append(operator(X1[idx_low:idx_high]))
		idx_low  = idx_high
	return out
