import numpy as np
import fofbubble
import zahnbubble
import os
import datetime, time
from mfp import mfp2d, mfp3d
import mfp_np

def fof(data, xth=0.5, boxsize=100):
	"""
	FOF bubble
	
	Parameter
	---------
	input  : 3D array of ionization fraction.
	xth    : The threshold value (Default: 0.5).
	boxsize: The boxsize in cMpc can be given (Default: 100).

	Output
	------
	The output is a tuple containing radius and volume array.
	"""
	t1 = datetime.datetime.now()
	fofbubble.fof(data, xth, boxsize)
	os.remove('dummy_output_of_fof.dat')
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "Program runtime: %f minutes." %runtime
	print "The output is a numpy array containing radius and volume array respectively."

	return f[:,1:]

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

	#zahn_out = np.array([radius, num_ion, num_neut, inbin, filternum])
	#center   = np.array([radius2, avg_ion])

	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "Program runtime: %f minutes." %runtime
	print "The output is a tuple containing three values: r, rdp/dr(ion), rdp/dr(neut)."
	print "The curve has been normalized."

	return np.array(radius).astype(float), np.array(num_ion).astype(float), np.array(num_neut).astype(float)

def spa(data, xth=0.95, boxsize=100, nscales=30, upper_lim=False):
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

def mfp_MPI(data, xth=0.5, boxsize=100, iterations = 10000000, verbose=True, upper_lim=False, nodes=5):
	"""
	Mean-Free-Path (MFP) bubble with MPI
	(Not fixed)
	
	Parameter
	---------
	input     : 2D/3D array of ionization fraction/brightness temperature.
	xth       : The threshold value (Default: 0.5).
	boxsize   : The boxsize in cMpc can be given (Default: 100).
	iterations: Number of iterations (Default: 1e7).
	verbose   : It prints the progress of the program (Default: True).
	upper_lim : It decides if the threshold is the upper limit or the lower limit (Default: True).
	nodes	  : Number of nodes for the code to run on parralel (Default: 5). 
                    For better efficiency, the 'iterations' should be divisible by 'nodes'.

	Output
	------
	The output contains a tuple with three values: r, rdP/dr, Sizes.
	"""
	dim = len(data.shape)
	h  = 0.7
	t1 = datetime.datetime.now()
	if (upper_lim): 
		data = -1.*data
		xth  = -1.*xth
	file_stamp = str(time.time())
	file_data = 'data_'+file_stamp
	file_size = 'size_'+file_stamp+'.txt'
	np.save(file_data, data)
	if dim == 2:
		print "MFP method applied on 2D data"
		run_prog = 'mpiexec -n '+str(nodes)+' python ~/lib/python/mfp2d_MPI.py '+file_data+' '+file_size+' '+str(xth)+' '+str(iterations)+' '+str(verbose)
	elif dim == 3:
		print "MFP method applied on 3D data"
		run_prog = 'mpiexec -n '+str(nodes)+' python ~/lib/python/mfp3d_MPI.py '+file_data+' '+file_size+' '+str(xth)+' '+str(iterations)+' '+str(verbose)
	else:
		print "The data doesn't have the correct dimension"
		return 0
	os.system(run_prog)
	sizes = np.loadtxt(file_size)
	sizes = sizes.astype(float)*boxsize/data.shape[0]/h
	os.remove(file_data+'.npy')
	os.remove(file_size)
	ht = np.histogram(sizes, bins=1000)
	t2 = datetime.datetime.now()
	runtime = (t2-t1).total_seconds()/60

	print "\nProgram runtime: %f minutes." %runtime
	print "The output contains a tuple with three values: r, rdP/dr, Sizes"
	print "The curve has been normalized."

	return ht[1][:-1], ht[0]*ht[1][:-1]/iterations, sizes


