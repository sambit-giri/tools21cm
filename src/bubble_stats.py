import numpy as np
from Friends_of_Friends import FoF_search
import zahnbubble
import os
import datetime, time
from mfp import mfp2d, mfp3d
import mfp_np

def fof(data, xth=0.5):
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
	out_map, size_list = FoF_search(data, xth)
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "Program runtime: %f minutes." %runtime
	print "The output is a tuple containing output-map and volume-list array respectively."

	return out_map, size_list


def zahn(data, xth=0.5, boxsize=100, nscales=20, upper_lim=False):
	"""
	ZAHN bubble
	
	Parameter
	---------
	input     : 3D array of ionization fraction.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: 100).
	nscales   : The number of different radii to consider (Default: 20).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).

	Output
	------
	The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut).
	"""
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth

	zahnbubble.zahn(data, xth, boxsize*0.7, nscales)

	f1 = open('dummy_output_of_zahn.dat')
	names1 = [l1.strip() for l1 in f1.readlines()]
	f1.close()
	os.remove('dummy_output_of_zahn.dat')

	f2 = open('center_zahn.dat')
	names2 = [l2.strip() for l2 in f2.readlines()]
	f2.close()
	os.remove('center_zahn.dat')

	os.remove('rhoHI.asci')
	os.remove('sizes.asci')

	radius    = []
	num_ion   = []
	num_neut  = []
	inbin     = []
	filternum = []

	radius2   = []
	avg_ion   = []

	for i in xrange(len(names1)):
		row = np.array(names1[i].split())
		row.astype(np.float)
		radius.append(row[0])
		num_ion.append(row[1])
		num_neut.append(row[2])
		inbin.append(row[3])
		filternum.append(row[4])

	for i in xrange(len(names2)):
		row = np.array(names2[i].split())
		row.astype(np.float)
		radius2.append(row[0])
		avg_ion.append(row[1])

	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "Program runtime: %f minutes." %runtime
	print "The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut)."
	print "The curve has been normalized."

	return np.array(radius).astype(float), np.array(num_ion).astype(float), np.array(num_neut).astype(float)

def spa(data, xth=0.95, boxsize=100, nscales=30, upper_lim=False):
	"""
	Spherical-Averege (SPA) bubble
	
	Parameter
	---------
	input     : 3D array of ionization fraction.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: 100).
	nscales   : The number of different radii to consider (Default: 20).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).

	Output
	------
	The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut).
	"""
	rr,ni,nn = zahn(data, xth=xth, boxsize=boxsize, nscales=nscales, upper_lim=upper_lim)
	r_min = boxsize/data.shape[0]
	rr_   = rr[rr>=r_min]
	ni_   = ni[rr>=r_min]
	nn_   = nn[rr>=r_min]
	return rr_, ni_*ni.sum()/ni_.sum(), nn_*nn.sum()/nn_.sum()

def mfp(data, xth=0.5, boxsize=100, iterations = 10000000, verbose=True, upper_lim=False):
	"""
	Mean-Free-Path (MFP) bubble
	
	Parameter
	---------
	input     : 2D/3D array of ionization fraction/brightness temperature.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: 100).
	iterations: Number of iterations (Default: 1e7).
	verbose   : It prints the progress of the program (Default: True).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: False).

	Output
	------
	The output contains a tuple with three values: r, rdP/dr, max(r).
	"""
	dim = len(data.shape)
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	if dim == 2:
		print "MFP method applied on 2D data (ver 0.2)"
		#out = mfp2d(data, xth, iterations=iterations, verbose=verbose)
		out = mfp_np.mfp2d(data, xth, iterations=iterations, verbose=verbose)
	elif dim == 3:
		print "MFP method applied on 3D data (ver 0.2)"
		#out = mfp3d(data, xth, iterations=iterations, verbose=verbose)
		out = mfp_np.mfp3d(data, xth, iterations=iterations, verbose=verbose)
	else:
		print "The data doesn't have the correct dimension"
		return 0
	nn = out[0]/iterations
	rr = out[1]
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "\nProgram runtime: %f minutes." %runtime
	print "The output contains a tuple with three values: r, rdP/dr, Most Probable r"
	print "The curve has been normalized."

	return rr*boxsize/data.shape[0], rr*nn, rr[nn.argmax()]*boxsize/data.shape[0]




def dist_from_volumes(sizes, resolution=1., bins=100):
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
	dummy = d_v[d_v!=0].min()/1000.
	d_v[d_v==0] = dummy
	rs, d_r  = np.zeros(len(ht_r[0])+1), np.zeros(len(ht_r[0])+1)
	rs       = 10.**ht_r[1]*resolution
	d_r[1:]  = 1.*ht_r[0]/np.sum(ht_r[0])
	d_r[0]   = d_r[d_r!=0].min()/1000.
	print "The output is a tuple conatining 4 numpy array: V, VdP/dV, r, rdp/dr."
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
			print label,
	ht   = np.histogram(np.log(sizes), bins=bins)
	vols, dist = np.zeros(len(ht[0])+1), np.zeros(len(ht[0])+1)
	vols      = np.exp(ht[1])*resolution
	dist[:-1] = ht[0]

	return sizes, dist/np.sum(dist), vols

def plot_fof_sizes(sizes, bins=100):
	lg = np.log10(np.array(sizes))
	ht = np.histogram(lg, bins=bins)
	xx = 10**ht[1]
	yy = ht[0]*xx[:-1]
	zz = yy/np.sum(yy)
	dummy = zz[zz!=0].min()/10.
	zz[zz==0] = dummy
	zz = np.hstack((zz,dummy))
	print "The output is Size, Size**2 dP/d(Size), lowest value"
	return xx, zz, dummy

