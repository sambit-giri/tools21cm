'''
Methods to calcluate the sizes of the regions of interest 
and estimate the size distributions.
'''

import numpy as np
from .Friends_of_Friends import FoF_search
from scipy import ndimage
import os,sys
import datetime, time
from . import mfp_np, spa_np, conv, morph 
from .scipy_func import *
from tqdm import tqdm
from joblib import Parallel, delayed
from .usefuls import loading_msg

def fof(data, xth=0.5, connectivity=1):
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
	map: ndarray
		array with each regions of interest label
	sizes: list
		all the sizes
	"""
	use_skimage=True
	t1 = datetime.datetime.now()
	data = (data>=xth)
	if 'skimage' in sys.modules and use_skimage:
		from skimage import morphology
		out_map = morphology.label(data, connectivity=connectivity)
		elements, size_list = np.unique(out_map, return_counts=True)
		size_list = size_list[1:]
	else: out_map, size_list = FoF_search(data, xth)
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("Program runtime: %f minutes." %runtime)
	print("The output is a tuple containing output-map and volume-list array respectively.")

	return out_map, size_list

def spa(data, xth=0.95, boxsize=None, nscales=20, upper_lim=False, binning='log'):
	"""
	Determines the sizes using the Spherical-Averege (SPA) approach.
	
	Parameters
	----------
	input     : ndarray
		3D array of ionization fraction.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	nscales   : int
		The number of different radii to consider (Default: 20).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: True).

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	if boxsize is None: boxsize = conv.LB
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	rs, ni = spa_np.spa_np(data, xth=xth, binning=binning, nscales=nscales)
	rs_ = rs*boxsize/data.shape[0]
	ni_ = ni/np.sum(ni)
	return rs_, ni_


def mfp(data, xth=0.5, boxsize=None, iterations=10000000, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None):
	"""
	Determines the sizes using the Mean-Free-Path (MFP) approach.
	
	Parameters
	----------
	input     : ndarray
		2D/3D array of ionization fraction/brightness temperature.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	iterations: int
		Number of iterations (Default: 10_000_000).
	verbose   : bool
		It prints the progress of the program (Default: True).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: False).
	bins      : int
		Give number of bins or an array of sizes to re-bin into (Default: None).
	r_min     : float
		Minimum size after rebinning (Default: None).
	r_max     : float
		Maximum size after rebinning (Default: None).

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	iterations = int(iterations)
	if boxsize is None:
		boxsize = conv.LB
		print('Boxsize is set to %.2f Mpc.'%boxsize) 
	dim = len(data.shape)
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	check_box = (data>=xth).sum()
	if verbose:
		print(f'{check_box}/{data.size} cells are marked as region of interest (ROI).')
	if check_box==0:
		data = np.ones(data.shape)
		iterations = 3
	if dim == 2:
		if verbose: print("MFP method applied on 2D data.")
		out = mfp_np.mfp2d(data, xth, iterations=iterations, verbose=verbose)
	elif dim == 3:
		if verbose: print("MFP method applied on 3D data.")
		out = mfp_np.mfp3d(data, xth, iterations=iterations, verbose=verbose)
	else:
		if verbose: print("The data doesn't have the correct dimension")
		return 0
	nn = out[0]/iterations
	rr = out[1]
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	if verbose: print("\nProgram runtime: %.2f minutes." %runtime)
	if check_box==0:
		if verbose: print("There is no ROI in the data. Therefore, the number density of all the sizes are zero.")
		# return rr*boxsize/data.shape[0], np.zeros(rr.shape)
		nn = np.zeros(rr.shape)
	if verbose: print("The output contains a tuple with three values: r, rdP/dr")
	if verbose: print("The curve has been normalized.")

	r0,p0 = rr*boxsize/data.shape[0], rr*nn #rr[nn.argmax()]*boxsize/data.shape[0]
	if bins is not None: r0,p0 = rebin_bsd(r0, p0, bins=bins, r_min=r_min, r_max=r_max)
	return r0, p0

def averaged_mfp(data, xth=0.5, boxsize=None, iterations=10000000, rays_per_point=10, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None, n_jobs=-1):
	"""
	Determines the sizes using the averaged Mean-Free-Path (MFP) approach.
	
	Parameters
	----------
	input     : ndarray
		2D/3D array of ionization fraction/brightness temperature.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	iterations: int
		Number of iterations (Default: 10_000_000).
	iterations: int
		Number of iterations (Default: 10).
	verbose   : bool
		It prints the progress of the program (Default: True).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: False).
	bins      : int
		Give number of bins or an array of sizes to re-bin into (Default: None).
	r_min     : float
		Minimum size after rebinning (Default: None).
	r_max     : float
		Maximum size after rebinning (Default: None).

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	iterations = int(iterations)
	rays_per_point = int(rays_per_point)
	if boxsize is None:
		boxsize = conv.LB
		print('Boxsize is set to %.2f Mpc.'%boxsize) 
	dim = len(data.shape)
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	check_box = (data>=xth).sum()
	if check_box==0:
		data = np.ones(data.shape)
		iterations = 3
	if dim == 3:
		ar = np.zeros_like(data)
		ar[data >= xth] = 1
		loc = np.argwhere(ar == 1)
		rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
		xs, ys, zs = loc[rand_loc, 0], loc[rand_loc, 1], loc[rand_loc, 2]

		print("MFP method applied on 3D data")
		def compute_ray_length(j):
			point = [xs[j], ys[j], zs[j]]
			num_szj, size_pxj = mfp_np.mfp3d(data, xth, iterations=rays_per_point, verbose=False, point=point)
			return np.sum(num_szj * size_pxj) / np.sum(num_szj)
		ray_list = Parallel(n_jobs=n_jobs)(delayed(compute_ray_length)(j) for j in tqdm(range(iterations), disable=not verbose))
		ray_list = np.array(ray_list)

		size_px, num_sz = np.unique(ray_list, return_counts=1)
		out = [num_sz, size_px]
	else:
		print("The data doesn't have the correct dimension")
		return 0
	if out is None: return None

	# nn = out[0]/iterations
	# rr = out[1]
	ht = np.histogram(np.log(ray_list[ray_list>0]), density=True, bins=50)
	nn, rr = ht[0], np.exp(ht[1][1:]/2+ht[1][:-1]/2)
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("Program runtime: %f minutes." %runtime)
	if check_box==0:
		print("There is no ROI in the data. Therefore, the number density of all the sizes are zero.")
		# return rr*boxsize/data.shape[0], np.zeros(rr.shape)
		nn = np.zeros(rr.shape)
	print("The output contains a tuple with three values: r, rdP/dr")
	print("The curve has been normalized.")

	# Return radii (in physical units) and fraction of side lines which
	# have this side line. This is different from a bubble size distribution!
	r0,p0 = rr*boxsize/data.shape[0], nn #rr[nn.argmax()]*boxsize/data.shape[0]
	if bins is not None: r0,p0 = rebin_bsd(r0, p0, bins=bins, r_min=r_min, r_max=r_max)
	return r0, p0

def mfp_from_point(data, point, xth=0.5, boxsize=None, iterations=10000000, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None):
	"""
	Determines the sizes using the Mean-Free-Path (MFP) approach.
	
	Parameters
	----------
	input     : ndarray
		2D/3D array of ionization fraction/brightness temperature.
	point     : ndarray or list
		[x, y, z] of the point.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	iterations: float
		Number of iterations (Default: 1e7).
	verbose   : bool
		It prints the progress of the program (Default: True).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: False).
	bins      : int
		Give number of bins or an array of sizes to re-bin into (Default: None).
	r_min     : float
		Minimum size after rebinning (Default: None).
	r_max     : float
		Maximum size after rebinning (Default: None).

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	if boxsize is None:
		boxsize = conv.LB
		print('Boxsize is set to %.2f Mpc.'%boxsize) 
	dim = len(data.shape)
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	check_box = (data>=xth).sum()
	if check_box==0:
		data = np.ones(data.shape)
		iterations = 3
	if dim == 2:
		print("MFP method applied on 2D data")
		out = mfp_np.mfp2d(data, xth, iterations=iterations, verbose=verbose, point=point)
	elif dim == 3:
		print("MFP method applied on 3D data")
		out = mfp_np.mfp3d(data, xth, iterations=iterations, verbose=verbose, point=point)
	else:
		print("The data doesn't have the correct dimension")
		return 0
	if out is None: return None

	nn = out[0]/iterations
	rr = out[1]
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("Program runtime: %f minutes." %runtime)
	if check_box==0:
		print("There is no ROI in the data. Therefore, the number density of all the sizes are zero.")
		# return rr*boxsize/data.shape[0], np.zeros(rr.shape)
		nn = np.zeros(rr.shape)
	print("The output contains a tuple with three values: r, rdP/dr")
	print("The curve has been normalized.")

	# Return radii (in physical units) and fraction of side lines which
	# have this side line. This is different from a bubble size distribution!
	r0,p0 = rr*boxsize/data.shape[0], nn #rr[nn.argmax()]*boxsize/data.shape[0]
	if bins is not None: r0,p0 = rebin_bsd(r0, p0, bins=bins, r_min=r_min, r_max=r_max)
	return r0, p0


def rebin_bsd(rr, pp, bins=10, r_min=None, r_max=None):
	fp = interp1d(rr, pp, kind='cubic')
	if np.array(bins).size == 1:
		if r_min is None: r_min = rr.min()+1
		if r_max is None: r_max = rr.max()-10
		rs = 10**np.linspace(np.log10(r_min), np.log10(r_max), bins)
	else: rs = np.array(bins)
	return rs, fp(rs)


def dist_from_volumes(sizes, resolution=1., bins=100, null_factor=None):
	"""
	Volume distribution and effective radius distribution.
	
	Parameters
	----------
	sizes      : list
		List of volumes in pixel**3 units.
	resolution : float
		Distance between two pixels in cMpc (Default: 1).
	bins       : int
		Number of bins of volumes in log space.

	Returns
	-------
	v  : ndarray
		volumes of the regions
	Pv : ndarray
		probability of finding the corresponding volume 
	r  : ndarray
		effective radii of the regions
	Pr : ndarray
		probability of finding the corresponding effective radius
	"""
	vols  = np.array(sizes)
	radii = (vols*3./4./np.pi)**(1./3.)
	ht_v  = np.histogram(np.log10(vols), bins=bins)
	ht_r  = np.histogram(np.log10(radii), bins=bins)
	vs, d_v  = np.zeros(len(ht_v[0])+1), np.zeros(len(ht_v[0])+1)
	vs       = 10.**ht_v[1]*resolution**3
	d_v[:-1] = 1.*ht_v[0]/np.sum(ht_v[0])
	if null_factor is None: dummy = d_v[d_v!=0].min()/1000.
	else: dummy = null_factor
	d_v[d_v==0] = dummy
	rs, d_r  = np.zeros(len(ht_r[0])+1), np.zeros(len(ht_r[0])+1)
	rs       = 10.**ht_r[1]*resolution
	d_r[1:]  = 1.*ht_r[0]/np.sum(ht_r[0])
	if null_factor is None: d_r[0]   = d_r[d_r!=0].min()/1000.
	else: d_r[0]  = null_factor
	print("The output is a tuple conatining 4 numpy array: V, VdP/dV, r, rdp/dr.")
	return vs, d_v, rs, d_r
	

def get_distribution(array, resolution=1., bins=100, sizes=False):
	if sizes:
		sizes = array
	else:
		mn, mx = array.min(), array.max()
		sizes  = np.arange(mx)+1.
		for i in range(int(mx)):
			label = i+1
			sizes[i] = len(array[array==label])
			print(label,)
	ht   = np.histogram(np.log(sizes), bins=bins)
	vols, dist = np.zeros(len(ht[0])+1), np.zeros(len(ht[0])+1)
	vols      = np.exp(ht[1])*resolution
	dist[:-1] = ht[0]

	return sizes, dist/np.sum(dist), vols

def plot_fof_sizes(sizes, bins=100, boxsize=None, normalize='box'):
	lg = np.log10(np.array(sizes))
	ht = np.histogram(lg, bins=bins)
	xx = 10**ht[1]
	yy = ht[0]*xx[:-1]
	if boxsize is None: boxsize = conv.LB
	if normalize=='ionized': zz = yy/np.sum(yy)
	else: zz = yy/boxsize**3
	dummy = zz[zz!=0].min()/10.
	zz[zz==0] = dummy
	zz = np.hstack((zz,dummy))
	print("The output is Size, Size**2 dP/d(Size), lowest value")
	return xx, zz, dummy

def disk_structure(n):
    struct  = np.zeros((2*n+1, 2*n+1, 2*n+1))
    x, y, z = np.indices((2*n+1, 2*n+1, 2*n+1))
    mask = (x - n)**2 + (y - n)**2 + (z - n)**2 <= n**2
    struct[mask] = 1
    return struct.astype(np.bool)


def granulometry_CDF(data, sizes=None, verbose=True, n_jobs=1):
	s = max(data.shape)
	if sizes is None: sizes = np.arange(1, s/2, 2)
	sizes = sizes.astype(int)
	# granulo = np.zeros((len(sizes)))
	np.random.shuffle(sizes)
	verbose = True
	if verbose:
		# func = lambda n: ndimage.binary_opening(data, structure=disk_structure(sizes[n])).sum()
		func = lambda n: morph.binary_opening(data, structure=disk_structure(sizes[n])).sum()
		granulo = np.array(Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(range(len(sizes))) ))
		# for n in tqdm(range(len(sizes))): granulo[n] = ndimage.binary_opening(data, structure=disk_structure(sizes[n])).sum()
		print("Completed.")
	arg_sort = np.argsort(sizes)
	return granulo[arg_sort]

def granulometry_bsd(data, xth=0.5, boxsize=None, verbose=True, upper_lim=False, sampling=2, log_bins=None, n_jobs=1):
	"""
	Determined the sizes using the Granulometry (Gran) approach.
	It is based on Kakiichi et al. (2017)

	Parameters
	----------
	input     : ndarray
		2D/3D array of ionization fraction/brightness temperature.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	verbose   : bool
		It prints the progress of the program (Default: True).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: False).
	sampling  : int
		Give the resolution of the radii in the pixel units (Default: 2).
	n_jobs    : int
	    Give the number of CPUs.

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	t1 = datetime.datetime.now()
	if boxsize is None: boxsize = conv.LB
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	mask = data > xth
	# if log_bins is not None: sz = np.unique((10**np.linspace(0, np.log10(data.shape[0]/4), log_bins)).astype(int))
	# else: sz   = np.arange(1, data.shape[0]/4, sampling)
	# granulo = granulometry_CDF(mask, sizes=sz, verbose=verbose, n_jobs=n_jobs)
	# rr = (sz*boxsize/data.shape[0])[:-1]
	# nn = np.array([(granulo[i]-granulo[i+1])/np.abs(sz[i]-sz[i+1]) for i in range(len(granulo)-1)])
	if n_jobs>1: print('Parallelization not implemented yet.')
	area, dFdR, R = _granulometry(mask, verbose=verbose)
	Rs = (R*boxsize/data.shape[0])
	dFdlnR = R*dFdR
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	if verbose:
		# print("\nProgram runtime: %f minutes." %runtime)
		print("The output contains a tuple with three values: r, rdP/dr, dP/dr")
		print("The curve has been normalized.")
	return Rs, dFdlnR, dFdR


def _granulometry(data, n_jobs=1, verbose=True):  

    def disk(n):
        struct = np.zeros((2 * n + 1, 2 * n + 1))
        x, y = np.indices((2 * n + 1, 2 * n + 1))
        mask = (x - n)**2 + (y - n)**2 <= n**2
        struct[mask] = 1
        return struct.astype(np.bool)
    def ball(n):
        struct = np.zeros((2*n+1, 2*n+1, 2*n+1))
        x, y, z = np.indices((2*n+1, 2*n+1, 2*n+1))
        mask = (x - n)**2 + (y - n)**2 + (z - n)**2 <= n**2
        struct[mask] = 1
        return struct.astype(np.bool)

    s = max(data.shape)
    dim   = data.ndim
    pixel = range(np.int(s/2))
    area0 = np.float(data.sum())
    area  = np.zeros_like(pixel)

    def func(n):
        if dim == 2:
            opened_data = morph.binary_opening(data,structure=disk(n))
        if dim == 3:
            opened_data = morph.binary_opening(data,structure=ball(n))
        return float(opened_data.sum())

    if n_jobs>1:
        area = np.array(Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in tqdm(pixel) ))
    else:
        t1 = time.time()
        for n in pixel:
            # print 'binary opening the data with radius =',n,' [pixels]'
            if verbose: loading_msg('R = {} pixels | time elapsed {:.1f} s'.format(n, (time.time()-t1)))
            area[n] = func(n)
            if area[n] == 0:
                break

    area = area.astype(float)
    # print(area)
    pattern_spectrum = np.append((area[:-1]-area[1:]).astype(float)/area[0], 0)
    # print(pattern_spectrum)

    return area, pattern_spectrum, np.array(pixel)

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import reconstruction

def h_transform(image, h, verbose=True):
	if verbose: print('Applying h-transform...')
	seed = np.copy(image)
	if image.ndim==3: seed[1:-1, 1:-1, 1:-1] = image.min()
	else: seed[1:-1, 1:-1] = image.min()
	mask = image
	seed = image - h
	dilated = reconstruction(seed, mask, method='dilation')
	image = image - dilated
	if verbose: print('...done')
	return image

def _watershed(image, h=None, threshold_rel=None, footprint=None, verbose=True):
	distance = ndimage.distance_transform_edt(image)
	if h is not None: distance1 = h_transform(distance, h, verbose=verbose)
	if footprint is None: footprint = np.ones((3, 3)) if image.ndim==2 else np.ones((3,3,3))
	if verbose: print('Finding watersheds...')
	coords = peak_local_max(distance if h is None else distance1, footprint=footprint, labels=image, threshold_rel=threshold_rel)
	mask = np.zeros(distance.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndimage.label(mask)
	labels = watershed(-distance if h is None else -distance1, markers, mask=image)
	if verbose: print('...done')
	return labels

def watershed_bsd(data, xth=0.5, boxsize=None, verbose=True, upper_lim=False, h=None, n_jobs=1):
	"""
	Still needs testing.

	Determined the sizes using the Watershed approach.
	It is based on Lin et al. (2016)

	Parameters
	----------
	input     : ndarray
		2D/3D array of ionization fraction/brightness temperature.
	xth       : float
		The threshold value (Default: 0.5).
	boxsize   : float
		The boxsize in cMpc can be given (Default: conv.LB).
	verbose   : bool
		It prints the progress of the program (Default: True).
	upper_lim : bool
		It decides if the threshold is the upper limit or the lower limit (Default: False).
	h         : float
	    Parameter for performing h-transform to smooth out local peaks.
	n_jobs    : int
	    Give the number of CPUs.

	Returns
	-------
	r  : ndarray
		sizes of the regions
	dn : ndarray
		probability of finding the corresponding size 
	"""
	t1 = datetime.datetime.now()
	if boxsize is None: boxsize = conv.LB
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	mask = data > xth
	# if log_bins is not None: sz = np.unique((10**np.linspace(0, np.log10(data.shape[0]/4), log_bins)).astype(int))
	# else: sz   = np.arange(1, data.shape[0]/4, sampling)
	# granulo = granulometry_CDF(mask, sizes=sz, verbose=verbose, n_jobs=n_jobs)
	# rr = (sz*boxsize/data.shape[0])[:-1]
	# nn = np.array([(granulo[i]-granulo[i+1])/np.abs(sz[i]-sz[i+1]) for i in range(len(granulo)-1)])
	if n_jobs>1: print('Parallelization not implemented yet.')
	labels = _watershed(mask, h=h)
	uniq   = np.unique(labels[labels>0], return_counts=1)
	r_list = np.cbrt(3*uniq[0]/4/np.pi)
	r_pixl = np.arange(0, np.int(r_list.max()+2), 1)-0.5 
	
	ht = np.histogram(r_list, bins=r_pixl)
	dFdR, R = ht[0]/ht[0].sum(), ht[1][1:]/2+ht[1][:-1]/2

	Rs = (R*boxsize/data.shape[0])
	dFdlnR = R*dFdR
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	# print("\nProgram runtime: %f minutes." %runtime)
	print("The output contains a tuple with three values: r, rdP/dr, dP/dr")
	print("The curve has been normalized.")
	return Rs, dFdlnR, dFdR

