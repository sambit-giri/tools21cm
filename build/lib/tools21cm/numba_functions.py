import numpy as np
from numba import autojit, prange
from usefuls import *

print("Using numba module")

@autojit
def superpixel_map_numba(data, labels, mns=None):
	from superpixels import get_superpixel_means
	print("Using numba module")
	if mns is None: mns = get_superpixel_means(data, labels=labels)
	sp_map = np.zeros(data.shape)
	for i in prange(mns.size): sp_map[labels==i] = mns[i]
	return sp_map

@autojit
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

@autojit
def get_uv_daily_observation_numba(ncells, z, filename=None, total_int_time=4., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
	The radio telescopes observe the sky for 'total_int_time' hours each day. The signal is recorded 
	every 'int_time' seconds. 
	Parameters
	----------
	z              : Redhsift of the slice observed.
	ncells         : The number of cell used to make the image.
	filename       : Name of the file containing the antenna configurations (text file).
	total_int_time : Total hours of observation per day (in hours).
	int_time       : Integration time of the telescope observation (in seconds).
	boxsize        : The comoving size of the sky observed. Default: It is determined from the 
			 simulation constants set.
	declination    : The declination angle of the SKA (in degree). Default: 30. 
	"""
	from telescope_functions import from_antenna_config, get_uv_coverage, earth_rotation_effect
	Nbase, N_ant = from_antenna_config(filename, z)
	uv_map0      = get_uv_coverage(Nbase, z, ncells, boxsize=boxsize)
	uv_map       = np.zeros(uv_map0.shape)
	tot_num_obs  = int(3600.*total_int_time/int_time)
	print("Making uv map from daily observations.")
	print("Using numba module")
	for i in prange(tot_num_obs-1):
		new_Nbase = earth_rotation_effect(Nbase, i+1, int_time, declination=declination)
		uv_map1   = get_uv_coverage(new_Nbase, z, ncells, boxsize=boxsize)
		uv_map   += uv_map1
		if verbose:
			perc = (i+2)*100/tot_num_obs
			msg = str(perc) + '%'
			loading_verbose(msg)
	uv_map = (uv_map+uv_map0)/tot_num_obs
	return uv_map, N_ant
