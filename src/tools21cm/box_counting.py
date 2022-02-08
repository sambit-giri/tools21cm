import numpy as np
from scipy.signal import fftconvolve
from scipy import interpolate

def box_count_sliding(data, size, q, window='ball'):
	assert window in ['ball', 'cube']
	kernel = get_kernel(size, window)
	smooth = fftconvolve(data, kernel, mode='same')
	return np.sum(smooth**q)


def get_kernel(size, window):
	if window=='cube': kern = np.ones((size, size, size))
	elif window=='ball':
		kern = np.zeros((size, size, size))
		if size % 2 == 0:
			size = int(size/2)
			x,y = np.mgrid[-size:size, -size:size]
		else:
			size = int(size/2)
			x,y = np.mgrid[-size:size+1, -size:size+1]
		kern[x**2+y**2<=size**2] = 1
	return kern/kern.sum()

def Renyi_diemsnions(data, q):
	rmin, rmax = 2, min(data.shape)/4
	rr  = np.round(np.exp(np.linspace(np.log(rmin), np.log(rmax), 10)))
	Z_r = np.array([box_count_sliding(data, size, q) for size in rr])
	dd  = np.log(Z_r)/np.log(rr)
	fdd = interpolate.interp1d(rr, dd, fill_value='extrapolate', kind='cubic')
	return fdd(0)
