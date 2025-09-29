import numpy as np
from numba import jit, prange
from .usefuls import *
from . import cosmo as cm
from . import conv
import itertools

print("Using numba module")

KB_SI   = 1.38e-23
c_light = 2.99792458e+10  #in cm/s
janskytowatt = 1e-26

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

@jit
def get_uv_daily_observation_numba(ncells, z, filename=None, total_int_time=4., int_time=10., boxsize=None, declination=-30., verbose=True):
	"""
	The radio telescopes observe the sky for 'total_int_time' hours each day. The signal is recorded 
	every 'int_time' seconds. 
	Parameters
	----------
	ncells         : The number of cell used to make the image.
	z              : Redhsift of the slice observed.
	filename       : Name of the file containing the antenna configurations (text file).
	total_int_time : Total hours of observation per day (in hours).
	int_time       : Integration time of the telescope observation (in seconds).
	boxsize        : The comoving size of the sky observed. Default: It is determined from the 
			 simulation constants set.
	declination    : The declination angle of the SKA (in degree). Default: 30. 
	"""
	#from .telescope_functions import from_antenna_config, get_uv_coverage, earth_rotation_effect
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
			msg = '%.1f %%'%(perc)
			loading_verbose(msg)
	uv_map = (uv_map+uv_map0)/tot_num_obs
	return uv_map, N_ant

@jit
def from_antenna_config(filename, z, nu=None):
	"""
	The function reads the antenna positions (N_ant antennas) from the file given.
	Parameters
	----------
	filename: Name of the file containing the antenna configurations (text file).
	z       : Redhsift of the slice observed.
	nu      : The frequency observed by the telescope.
	Returns
	----------
	Nbase   : Numpy array (N_ant(N_ant-1)/2 x 3) containing the ux,uy,uz values derived 
	          from the antenna positions.
	N_ant   : Number of antennas.
	"""
	z = float(z)
	if filename is None:
		from .radio_telescope_sensitivity import SKA1_LowConfig_Sept2016
		antll  = SKA1_LowConfig_Sept2016()
	else: antll  = np.loadtxt(filename, dtype=str)
	antll  = antll[:,-2:].astype(float)
	Re     = 6.371e6                                            # in m
	pp     = np.pi/180
	if not nu: nu = cm.z_to_nu(z)                              # MHz
	antxyz = np.zeros((antll.shape[0],3))		            # in m
	antxyz[:,0] = Re*np.cos(antll[:,1]*pp)*np.cos(antll[:,0]*pp)
	antxyz[:,1] = Re*np.cos(antll[:,1]*pp)*np.sin(antll[:,0]*pp)
	antxyz[:,2] = Re*np.sin(antll[:,1]*pp)	
	#del pp, antll
	N_ant = antxyz.shape[0]
	pair_comb = itertools.combinations(range(N_ant), 2)
	pair_comb = list(pair_comb)	
	lam = c_light/(nu*1e6)/1e2 			            # in m
	Nbase = []
	for ii,jj in list(pair_comb):
		ux = (antxyz[ii,0]-antxyz[jj,0])/lam
		uy = (antxyz[ii,1]-antxyz[jj,1])/lam
		uz = (antxyz[ii,2]-antxyz[jj,2])/lam
		if ux==0: print(ii,jj)
		Nbase.append([ux,uy,uz])
	Nbase = np.array(Nbase)	
	return Nbase, N_ant

@jit
def earth_rotation_effect(Nbase, slice_num, int_time, declination=-30.):
	"""
	The rotation of the earth over the observation times makes changes the part of the 
	sky measured by each antenna.
	Parameter:
	---------
	Nbase       : The array containing all the ux,uy,uz values of the antenna configuration.
	slice_num   : The number of the observed slice after each of the integration time.
	int_time    : The integration time is the time after which the signal is recorded (in seconds).
	declination : The angle of declination refers to the lattitute where telescope is located 
		      (in degres). Default: -30
	Return
	----------
	new_Nbase   : It is the new Nbase calculated for the rotated antenna configurations.
	"""

	p     = np.pi/180.
	delta = p*declination
	k     = slice_num
	HA    =-15.0*p*(k-1)*int_time/(3600.0) - np.pi/180.0*90.0 + np.pi/180.0*360.0
	
	new_Nbase = np.zeros(Nbase.shape)
	new_Nbase[:,0] = np.sin(HA)*Nbase[:,0] + np.cos(HA)*Nbase[:,1]
	new_Nbase[:,1] = -1.0*np.sin(delta)*np.cos(HA)*Nbase[:,0] + np.sin(delta)*np.sin(HA)*Nbase[:,1] + np.cos(delta)*Nbase[:,2]
	new_Nbase[:,2] = np.cos(delta)*np.cos(HA)*Nbase[:,0] - np.cos(delta)*np.sin(HA)*Nbase[:,1] + np.sin(delta)*Nbase[:,2]
	return new_Nbase

@jit
def get_uv_coverage(Nbase, z, ncells, boxsize=None):
	"""
	It calculated the uv_map for the uv-coverage.
	Parameters
	----------
	Nbase   : The array containing all the ux,uy,uz values of the antenna configuration.
	z       : Redhsift of the slice observed.
	ncells  : The number of cell used to make the image.
	boxsize : The comoving size of the sky observed. Default: It is determined from the 
	          simulation constants set.
	Returns
	----------
	uv_map  : ncells x ncells numpy array containing the number of baselines observing each pixel.
	"""
	z = float(z)
	if not boxsize: boxsize = conv.LB
	uv_map = np.zeros((ncells,ncells))
	theta_max = boxsize/cm.z_to_cdist(z)
	Nb  = np.round(Nbase*theta_max/2)
	Nb  = Nb[(Nb[:,0]<ncells/2)]
	Nb  = Nb[(Nb[:,1]<ncells/2)]
	Nb  = Nb[(Nb[:,2]<ncells/2)]
	Nb  = Nb[(Nb[:,0]>=-ncells/2)]
	Nb  = Nb[(Nb[:,1]>=-ncells/2)]
	Nb  = Nb[(Nb[:,2]>=-ncells/2)]
	xx,yy,zz = Nb[:,0], Nb[:,1], Nb[:,2]
	for p in range(xx.shape[0]): uv_map[int(xx[p]),int(yy[p])] += 1
	return uv_map

