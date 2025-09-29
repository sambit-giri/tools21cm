import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import spearmanr
from skimage.segmentation import slic, mark_boundaries
from skimage.filters import threshold_otsu
from scipy.signal import argrelextrema
import sys, time
from tqdm import tqdm

def slic_cube(cube, n_segments=5000, compactness=0.1, 
			  max_iter=20, sigma=0, min_size_factor=0.5, max_size_factor=3, 
			  cmap=None, verbose=True):
	if cmap is not None: 
		color   = plt.get_cmap(cmap)
		multichannel = True
		cube = color(cube)
		cube = np.delete(cube, 3, -1)
	else:
		multichannel = False
	if verbose: print('Estimating superpixel labels using SLIC...')
	try: 
		labels = slic(cube, n_segments=n_segments, compactness=compactness, 
					  max_num_iter=max_iter, sigma=sigma, max_size_factor=max_size_factor, 
					  slic_zero=True, multichannel=multichannel, start_label=0)
	except: 
		labels = slic(cube, n_segments=n_segments, compactness=compactness, 
					  max_num_iter=max_iter, sigma=sigma, max_size_factor=max_size_factor, 
					  slic_zero=True, start_label=0)
	if verbose: print("The output contains the labels with %d segments"%(labels.max()+1))
	return labels

def see_label(out_map, label):
	binary = np.zeros(out_map.shape)
	if out_map[out_map==label].size: binary[out_map==label] = 1
	else: print("The entered label in not present in the map.")
	return binary

def binary_stitch(data, labels, stitch='mean', thres=None):
	X1 = data.reshape(-1,1)
	y  = labels.reshape(-1,1)
	y1 = [X1[y==i].mean() for i in np.unique(y)]
	if not thres:
		if stitch=='otsu': thres = threshold_otsu(np.array(y1))
		else: thres = X1.mean()
	y2 = np.zeros(y.shape)
	for i in np.unique(y): y2[y==i] = y1[i]
	y2 = y2 < thres
	return y2.reshape(data.shape)

def boundaries(sl, lab, mark=True, factor=1.01):
	assert sl.ndim == 2 and lab.ndim == 2
	bd  = mark_boundaries(np.zeros(sl.shape), lab)
	out = sl.copy()
	if mark: out[bd[:,:,0]==1] = sl.max()*factor
	else: out[0,0] = sl.max()*factor
	return out

def under_segmentation_error(labels, truths, b=0.25, verbose=True):
	assert labels.shape == truths.shape
	nx,ny,nz = labels.shape
	uu = 0
	for i in np.unique(truths):
		for j in np.unique(labels):
			gg, ss  = np.zeros((nx,ny,nz)), np.zeros((nx,ny,nz))
			gg[truths==i] = 1
			ss[labels==j] = 1
			inter = gg+ss
			if inter[inter==2].size > b*ss.sum(): uu += ss.sum()
			if verbose: print(i, j)
	U = (uu-labels.size)/labels.size
	return U

def stitch_using_histogram(data, mns, labels, bins='knuth', binary=True, on_superpixel_map=True, verbose=True):
	if bins in ['knuth', 'scotts', 'freedman', 'blocks']:
		if 'astroML' in sys.modules: from astroML.density_estimation import histogram
		else: 
			bins = 'auto'
			from numpy import histogram
	else:
		from numpy import histogram
	ht  = histogram(mns, bins=bins)
	if ht[0].argmax()==0: peaks = argrelextrema(np.hstack((0,ht[0])), np.greater)[0]
	else: peaks = argrelextrema(ht[0], np.greater)[0]
	if peaks.size==1:
		x1,y1 = ht[1][0], ht[0][0]
		y0 = ht[0].max()
		bla = ht[0].argmax()
		x0 = ht[1][bla]
		m0 = 1.*(y0-y1)/(x0-x1)
		d2 = np.array([np.abs((ht[0][i]-y1)-m0*(ht[1][i]-x1)) for i in range(bla)])
		thres = ht[1][d2.argmax()]/2. + ht[1][d2.argmax()+1]/2.
		if binary:
			#y = np.zeros(Ls.shape)
			#for i in np.unique(Ls): y[Ls==i] = mns[i]
			if on_superpixel_map: 
				out = superpixel_map(data, labels, mns=mns, verbose=verbose)
				out = out<thres
			else: out = data<thres
			return out
		else: return thres
	else:
		if len(np.unique(mns))==1: return np.zeros_like(data)
		thres = threshold_otsu(mns)
		if binary: 
			if on_superpixel_map: 
				out = superpixel_map(data, labels, mns=mns, verbose=verbose)
				out = out<thres
			else: out = data<thres
			return out
		else: return thres

def stitch_superpixels(data, labels, bins='knuth', binary=True, on_superpixel_map=True, verbose=True):
	mns = get_superpixel_means(data, labels=labels, verbose=verbose)
	stitched = stitch_using_histogram(data, mns, labels, bins=bins, binary=binary, on_superpixel_map=on_superpixel_map, verbose=verbose)
	return stitched

def apply_operator_labelled_data(data, labels, operator=np.mean, verbose=True):
	#if 'numba' in sys.modules: 
	#	from .numba_functions import apply_operator_labelled_data_numba
	#	out = apply_operator_labelled_data_numba(data, labels, operator=np.mean)
	#	return out
	X   = data.flatten()
	y   = labels.flatten()
	elems, num = np.unique(y, return_counts=1)
	X1  = X[y.argsort()]
	out = []
	idx_low = 0
	if verbose:
		time.sleep(1)
		for i in tqdm(elems, disable=False if verbose else True):
			idx_high = idx_low + num[i]
			out.append(operator(X1[idx_low:idx_high]))
			idx_low  = idx_high
	else:
		for i in elems:
			idx_high = idx_low + num[i]
			out.append(operator(X1[idx_low:idx_high]))
			idx_low  = idx_high
	return out

def get_superpixel_means(data, labels=None, slic_segments=3000, verbose=True):
	if labels is None: 
		print('Superpixel labels not provided.')
		labels = slic_cube(data, n_segments=slic_segments, verbose=verbose)
	if verbose:
		print('Estimating the superpixel mean map...')
	mns = apply_operator_labelled_data(data, labels, operator=np.mean, verbose=verbose)
	if verbose: print('...done')
	return np.array(mns)

def get_superpixel_sigmas(data, labels=None, slic_segments=5000, verbose=True):
	if labels is None: 
		print('Superpixel labels not provided.')
		labels = slic_cube(data, n_segments=slic_segments, verbose=verbose)
	if verbose:
		print('Estimating the superpixel sigma map...')
	sigs = apply_operator_labelled_data(data, labels, operator=np.std, verbose=verbose)
	if verbose: print('...done')
	return np.array(sigs)

def get_superpixel_n_pixels(data, labels=None, slic_segments=5000, verbose=True):
	if labels is None: 
		print('Superpixel labels not provided.')
		labels = slic_cube(data, n_segments=slic_segments, verbose=verbose)
	X    = data.flatten()
	y    = labels.flatten()
	elems, n_pixels = np.unique(y, return_counts=1)
	return n_pixels

def get_superpixel_SNRs(means=None, sigmas=None, n_pix=None, data=None, labels=None, slic_segments=5000, pixels=False):
	if means is None: means = get_superpixel_means(data, labels=labels, slic_segments=slic_segments)
	if sigmas is None: sigmas = get_superpixel_sigmas(data, labels=labels, slic_segments=slic_segments)
	if n_pix is None: n_pix = get_superpixel_n_pixels(data, labels=labels, slic_segments=slic_segments)
	sorts  = np.argsort(means)
	sigmas = sigmas[sorts]
	n_pix  = n_pix[sorts]
	means  = means[sorts]
	if pixels:
		pxls = get_superpixel_pixels(data=data, labels=labels, slic_segments=slic_segments)
		pxls = pxls[sorts]
		return means/sigmas, means, sigmas, n_pix, pxls
	return means/sigmas, means, sigmas, n_pix

def get_superpixel_pixels(data=None, labels=None, slic_segments=5000, verbose=True):
	if labels is None: 
		print('Superpixel labels not provided.')
		labels = slic_cube(data, n_segments=slic_segments, verbose=verbose)
	if verbose: print('Constructing a list of pixels in each superpixel.')
	pxl = apply_operator_labelled_data(data, labels, operator=np.array, verbose=verbose)
	if verbose: print('...done')
	return pxl

def mean_estimate(data, means=None, sigmas=None, n_pix=None, labels=None, slic_segments=5000, SNR_thres=5):
	#if means is None: means  = get_superpixel_means(data, labels=labels, slic_segments=slic_segments)
	#if sigmas is None: sigmas = get_superpixel_sigmas(data, labels=labels, slic_segments=slic_segments)
	#if n_pix is None: n_pix = get_superpixel_n_pixels(data, labels=labels, slic_segments=slic_segments)
	SNRs, means, sigmas, n_pix = get_superpixel_SNRs(means=means, sigmas=sigmas, n_pix=n_pix)
	if SNR_thres<=0: locs = SNRs<SNR_thres
	else: locs = SNRs<-SNR_thres
	global_means  = np.abs(means[locs])
	if global_means.size==0: return 0, 0
	global_sigmas = sigmas[locs]
	global_n_pix  = n_pix[locs]
	global_comb   = get_combined_mean_std(global_means, global_sigmas, global_n_pix)
	return global_comb[0], global_comb[1] #/np.sum(global_n_pix)

def get_combined_mean_std(means, sigmas, ns):
	assert means.size==sigmas.size and means.size==ns.size
	mn = np.sum(means*ns)/np.sum(ns)
	s_ = np.sum((ns-1)*sigmas**2 + ns*(means-mn)**2)/(np.sum(ns)-1)  #Bessel correction done
	st = np.sqrt(s_)
	return mn, st

def xfrac_mass_estimate(dT, z):
	return 1. - dT/27./np.sqrt((1+z)/10)

def xfrac_volume_estimate(binary):
	return binary.mean()

def superpixel_map(data, labels, mns=None, verbose=True):
	#if 'numba' in sys.modules: 
	#	from .numba_functions import superpixel_map_numba
	#	sp_map = superpixel_map_numba(data, labels, mns=mns)
	#	return sp_map
	if mns is None: mns = get_superpixel_means(data, labels=labels, verbose=verbose)
	sp_map = np.zeros(data.shape)
	if verbose:
		print('Constructing the superpixel map...')
		time.sleep(1)
		for i in tqdm(range(mns.size)): sp_map[labels==i] = mns[i]
	else:
		for i in range(mns.size): sp_map[labels==i] = mns[i]
	return sp_map


