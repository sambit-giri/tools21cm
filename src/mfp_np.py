import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys

def mfp3d(arr, xth=0.5, iterations=10000000, verbose=True):
	#3D interpolation is required
	#RegularGridInterpolator in scipy(>0.14) is used to do the interpolation
	
	info    = arr.shape
	longy	= max([info[0], info[1], info[2]])
	longest = int(np.sqrt(3)*longy)
	num_sz  = np.zeros(longest)

	ar  = np.zeros(arr.shape)
	ar[arr >= xth] = 1

	loc = np.argwhere(ar == 1)
	rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
	thetas   = np.random.randint(0, 360, size=iterations)
	phis     = np.random.randint(0, 360, size=iterations)
	ls       = np.sin(thetas*np.pi/180)*np.cos(phis*np.pi/180)
	ms       = np.sin(thetas*np.pi/180)*np.sin(phis*np.pi/180)
	ns       = np.cos(thetas*np.pi/180)
	xs,ys,zs = loc[rand_loc,0],loc[rand_loc,1],loc[rand_loc,2]
	
	interp_func = RegularGridInterpolator((np.arange(info[0]), np.arange(info[1]), np.arange(info[2])), ar, bounds_error=False, fill_value=0)

	for rr in xrange(longest):
		xs,ys,zs = xs+ls,ys+ms,zs+ns
		pts    = np.vstack((xs,ys,zs)).T
		vals   = interp_func(pts)
		check  = np.argwhere(vals<=0.5)
		num_sz[rr] = check.shape[0]
		xs,ys,zs = np.delete(xs, check),np.delete(ys, check),np.delete(zs, check)
		ls,ms,ns = np.delete(ls, check),np.delete(ms, check),np.delete(ns, check)
		if verbose:
			perc = (rr+1)*100/longest
			msg  = str(perc) + '%'
			loading_verbose(msg)
		if not xs.size: break
	msg  = str(100) + '%'
	loading_verbose(msg)
	size_px = np.arange(longest)
	return num_sz, size_px

def mfp2d(arr, xth=0.5, iterations=1000000, verbose=True):
	#2D interpolation is required
	#RegularGridInterpolator in scipy(>0.14) is used to do the interpolation
	
	info    = arr.shape
	longy	= max([info[0], info[1]])
	longest = int(np.sqrt(2)*longy)
	num_sz  = np.zeros(longest)

	ar  = np.zeros(arr.shape)
	ar[arr >= xth] = 1

	loc = np.argwhere(ar == 1)
	rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
	thetas   = np.random.randint(0, 360, size=iterations)
	ls       = np.sin(thetas*np.pi/180)
	ms       = np.cos(thetas*np.pi/180)

	xs,ys    = loc[rand_loc,0],loc[rand_loc,1]
	
	interp_func = RegularGridInterpolator((np.arange(info[0]), np.arange(info[1])), ar, bounds_error=False, fill_value=0)

	for rr in xrange(longest):
		xs,ys  = xs+ls,ys+ms
		pts    = np.vstack((xs,ys)).T
		vals   = interp_func(pts)
		check  = np.argwhere(vals<=0.5)
		num_sz[rr] = check.shape[0]
		xs,ys  = np.delete(xs, check),np.delete(ys, check)
		ls,ms  = np.delete(ls, check),np.delete(ms, check)
		if verbose:
			perc = (rr+1)*100/longest
			msg  = str(perc) + '%'
			loading_verbose(msg)
		if not xs.size: break
	msg  = str(100) + '%'
	loading_verbose(msg)
	size_px = np.arange(longest)
	return num_sz, size_px


def loading_verbose(string):
	msg = ("Completed: " + string )
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()


