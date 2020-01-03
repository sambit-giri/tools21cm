import numpy as np
from .Friends_of_Friends import FoF_search
from scipy import ndimage
import os,sys
import datetime, time
from . import mfp_np, spa_np, conv
from scipy.interpolate import interp1d

def fof(data, xth=0.5, neighbors=4, use_skimage=False):
	"""
	FOF bubble
	
	Parameter
	---------
	input  : 3D array of ionization fraction.
	xth    : The threshold value (Default: 0.5).

	Output
	------
	The output is a tuple containing output-map and volume-list array.
	"""
	t1 = datetime.datetime.now()
	if 'skimage' in sys.modules and use_skimage:
		from skimage import morphology
		out_map = morphology.label(data, neighbors=neighbors)
		elements, size_list = np.unique(out_map, return_counts=True)
		size_list = size_list[1:]
	else: out_map, size_list = FoF_search(data, xth)
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("Program runtime: %f minutes." %runtime)
	print("The output is a tuple containing output-map and volume-list array respectively.")

	return out_map, size_list


#def zahn(data, xth=0.5, boxsize=100, nscales=20, upper_lim=False):
#	"""
#	ZAHN bubble
#	
#	Parameter
#	---------
#	input     : 3D array of ionization fraction.
#	xth       : The threshold value (Default: 0.5).
#	boxsize   : The boxsize in cMpc can be given (Default: 100).
#	nscales   : The number of different radii to consider (Default: 20).
#	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).
#
#	Output
#	------
#	The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut).
#	"""
#	t1 = datetime.datetime.now()
#	if (upper_lim): 
#		data = -1.*data
#		xth  = -1.*xth
#
#	zahnbubble.zahn(data, xth, boxsize*0.7, nscales)
#
#	f1 = open('dummy_output_of_zahn.dat')
#	names1 = [l1.strip() for l1 in f1.readlines()]
#	f1.close()
#	os.remove('dummy_output_of_zahn.dat')
#
#	f2 = open('center_zahn.dat')
#	names2 = [l2.strip() for l2 in f2.readlines()]
#	f2.close()
#	os.remove('center_zahn.dat')
#
#	os.remove('rhoHI.asci')
#	os.remove('sizes.asci')
#
#	radius    = []
#	num_ion   = []
#	num_neut  = []
#	inbin     = []
#	filternum = []
#
#	radius2   = []
#	avg_ion   = []
#
#	for i in xrange(len(names1)):
#		row = np.array(names1[i].split())
#		row.astype(np.float)
#		radius.append(row[0])
#		num_ion.append(row[1])
#		num_neut.append(row[2])
#		inbin.append(row[3])
#		filternum.append(row[4])
#
#	for i in xrange(len(names2)):
#		row = np.array(names2[i].split())
#		row.astype(np.float)
#		radius2.append(row[0])
#		avg_ion.append(row[1])
#
#	t2 = datetime.datetime.now()
#	runtime = (t2-t1).total_seconds()/60
#
#	print "Program runtime: %f minutes." %runtime
#	print "The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut)."
#	print "The curve has been normalized."
#
#	return np.array(radius).astype(float), np.array(num_ion).astype(float), np.array(num_neut).astype(float)
#
#def spa(data, xth=0.95, boxsize=100, nscales=30, upper_lim=False):
#	"""
#	Spherical-Averege (SPA) bubble
#	
#	Parameter
#	---------
#	input     : 3D array of ionization fraction.
#	xth       : The threshold value (Default: 0.5).
#	boxsize   : The boxsize in cMpc can be given (Default: 100).
#	nscales   : The number of different radii to consider (Default: 20).
#	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).
#
#	Output
#	------
#	The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut).
#	"""
#	rr,ni,nn = zahn(data, xth=xth, boxsize=boxsize, nscales=nscales, upper_lim=upper_lim)
#	r_min = boxsize/data.shape[0]
#	rr_   = rr[rr>=r_min]
#	ni_   = ni[rr>=r_min]
#	nn_   = nn[rr>=r_min]
#	return rr_, ni_*ni.sum()/ni_.sum(), nn_*nn.sum()/nn_.sum()

def spa(data, xth=0.95, boxsize=None, nscales=20, upper_lim=False, binning='log'):
	"""
	Spherical-Averege (SPA) bubble
	
	Parameter
	---------
	input     : 3D array of ionization fraction.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: conv.LB).
	nscales   : The number of different radii to consider (Default: 20).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).

	Output
	------
	The output is a tuple containing three values: r, rdp/dr(ion).
	"""
	if boxsize is None: boxsize = conv.LB
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	rs, ni = spa_np.spa_np(data, xth=xth, binning=binning, nscales=nscales)
	rs_ = rs*boxsize/data.shape[0]
	ni_ = ni/np.sum(ni)
	return rs_, ni_


def mfp(data, xth=0.5, boxsize=None, iterations = 10000000, verbose=True, upper_lim=False, bins=None, r_min=None, r_max=None):
	"""
	Mean-Free-Path (MFP) bubble
	
	Parameter
	---------
	input     : 2D/3D array of ionization fraction/brightness temperature.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: conv.LB).
	iterations: Number of iterations (Default: 1e7).
	verbose   : It prints the progress of the program (Default: True).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: False).
	bins      : Give number of bins or an array of sizes to re-bin into (Default: None).
	r_min     : Minimum size after rebinning (Default: None).
	r_max     : Maximum size after rebinning (Default: None).

	Output
	------
	The output contains a tuple with three values: r, rdP/dr.
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
		print("MFP method applied on 2D data (ver 1.0)")
		#out = mfp2d(data, xth, iterations=iterations, verbose=verbose)
		out = mfp_np.mfp2d(data, xth, iterations=iterations, verbose=verbose)
	elif dim == 3:
		print("MFP method applied on 3D data (ver 1.0)")
		#out = mfp3d(data, xth, iterations=iterations, verbose=verbose)
		out = mfp_np.mfp3d(data, xth, iterations=iterations, verbose=verbose)
	else:
		print("The data doesn't have the correct dimension")
		return 0
	nn = out[0]/iterations
	rr = out[1]
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("\nProgram runtime: %f minutes." %runtime)
	if check_box==0:
		print("There is no ROI in the data. Therefore, the BSD is zero everywhere.")
		return rr*boxsize/data.shape[0], np.zeros(rr.shape)
	print("The output contains a tuple with three values: r, rdP/dr")
	print("The curve has been normalized.")

	r0,p0 = rr*boxsize/data.shape[0], rr*nn #rr[nn.argmax()]*boxsize/data.shape[0]
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
	
	Parameter
	---------
	sizes      : List of volumes in pixel**3 units.
	resolution : Distance between two pixels in cMpc (Default: 1).
	bins       : Number of bins of volumes in log space.

	Output
	------
	The output is a tuple conatining 4 numpy arrays: V, VdP/dV, r, rdp/dr.
	The radius calculated here is the effective radius.
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
		for i in xrange(int(mx)):
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


def granulometry_CDF(data, sizes=None, verbose=True):
	s = max(data.shape)
	if sizes is None: sizes = np.arange(1, s/2, 2)
	granulo = np.zeros((len(sizes)))
	for n in xrange(len(sizes)): granulo[n] = ndimage.binary_opening(data, structure=disk_structure(sizes[n])).sum()
	#if verbose: print "Completed:", 100*(n+1)/len(sizes), "%"
	print("Completed.")
	return granulo

def granulometry_bsd(data, xth=0.5, boxsize=None, verbose=True, upper_lim=False, sampling=2):
	"""
	Granulometry (Gran) bubble
	@ based on Kakiichi et al. (2017)
	Parameter
	---------
	input     : 2D/3D array of ionization fraction/brightness temperature.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: conv.LB).
	verbose   : It prints the progress of the program (Default: True).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: False).
	sampling  : Give the resolution of the radii in the pixel units (Default: 2).

	Output
	------
	The output contains a tuple with three values: r, rdP/dr.
	"""
	t1 = datetime.datetime.now()
	if boxsize is None: boxsize = conv.LB
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	mask = data > xth
	sz   = np.arange(1, data.shape[0]/4, sampling)
	granulo = granulometry_CDF(mask, sizes=sz, verbose=verbose)

	rr = (sz*boxsize/data.shape[0])[:-1]
	nn = np.array([granulo[i]-granulo[i+1] for i in xrange(len(granulo)-1)])

	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print("\nProgram runtime: %f minutes." %runtime)
	print("The output contains a tuple with three values: r, rdP/dr")
	print("The curve has been normalized.")
	return rr, nn/nn.sum()



