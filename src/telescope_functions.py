import numpy as np
import itertools
#import c2raytools as c2t
import sys
from usefuls import *
import cosmology as cm
import conv

KB_SI   = 1.38e-23
c_light = 2.99792458e+10  #in cm/s
janskytowatt = 1e-26

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
	antll  = np.loadtxt(filename, dtype=str)
	antll  = antll[:,-2:].astype(float)
	Re     = 6.371e6                                            # in m
	pp     = np.pi/180
	if not nu: nu = cm.z_to_nu(z)                              # MHz
	antxyz = np.zeros((antll.shape[0],3))		            # in m
	antxyz[:,0] = Re*np.cos(antll[:,1]*pp)*np.cos(antll[:,0]*pp)
	antxyz[:,1] = Re*np.cos(antll[:,1]*pp)*np.sin(antll[:,0]*pp)
	antxyz[:,2] = Re*np.sin(antll[:,1]*pp)	
	del pp, antll
	N_ant = antxyz.shape[0]
	Nbase = np.zeros((N_ant*(N_ant-1)/2,3))
	pair_comb = itertools.combinations(xrange(N_ant), 2)
	pair_comb = list(pair_comb)	
	lam = c_light/(nu*1e6)/1e2 			            # in m
	for i in xrange(Nbase.shape[0]):
		ii,jj = pair_comb[i]
		ux = (antxyz[ii,0]-antxyz[jj,0])/lam
		uy = (antxyz[ii,1]-antxyz[jj,1])/lam
		uz = (antxyz[ii,2]-antxyz[jj,2])/lam
		Nbase[i,:] = ux,uy,uz 
	return Nbase, N_ant

def earth_rotation_effect(Nbase, slice_num, int_time, declination=30.):
	"""
	The rotation of the earth over the observation times makes changes the part of the 
	sky measured by each antenna.
	Parameter:
	---------
	Nbase       : The array containing all the ux,uy,uz values of the antenna configuration.
	slice_num   : The number of the observed slice after each of the integration time.
	int_time    : The integration time is the time after which the signal is recorded (in seconds).
	declination : The angle of declination refers to the lattitute where telescope is located 
		      (in degres). Default: 30
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

def get_uv_daily_observation(ncells, z, filename, total_int_time=4., int_time=10., boxsize=None, declination=30., verbose=True):
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
	Nbase, N_ant = from_antenna_config(filename, z)
	uv_map0      = get_uv_coverage(Nbase, z, ncells, boxsize=boxsize)
	uv_map       = np.zeros(uv_map0.shape)
	tot_num_obs  = int(3600.*total_int_time/int_time)
	print "Making uv map from daily observations."
	for i in xrange(tot_num_obs-1):
		new_Nbase = earth_rotation_effect(Nbase, i+1, int_time, declination=declination)
		uv_map1   = get_uv_coverage(new_Nbase, z, ncells, boxsize=boxsize)
		uv_map   += uv_map1
		if verbose:
			perc = (i+2)*100/tot_num_obs
			msg = str(perc) + '%'
			loading_verbose(msg)
	uv_map = (uv_map+uv_map0)/tot_num_obs
	return uv_map, N_ant
	

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
	if not boxsize: boxsize = conv.LB
	uv_map = np.zeros((ncells,ncells))
	theta_max = boxsize/cm.z_to_cdist(z)
	Nb  = (Nbase*theta_max).astype(int)
	xx,yy,zz = Nb[:,0], Nb[:,1], Nb[:,2]
	xx = xx[xx<ncells]
	yy = yy[yy<ncells]
	xx = xx[xx>-ncells]
	yy = yy[yy>-ncells]
	xx = xx[xx<ncells]
	yy = yy[yy<ncells]
	xx = xx[xx>-ncells]
	yy = yy[yy>-ncells]
	for p in xrange(xx.shape[0]):
		uv_map[xx[p],yy[p]] += 1
	return uv_map


def kanan_noise_image_ska(z, uv_map, depth_mhz, obs_time, int_time, N_ant_ska=564., verbose=True):
	"""
	It calculates the rms of the noise added by the interferrometers of ska. 
	Parameters
	----------
	z         : Redhsift of the slice observed.
	uv_map    : ncells x ncells numpy array containing the number of baselines observing each pixel.
	depth_mhz : The bandwidth of the observation (in MHz).
	obs_time  : The total hours of observations time.
	N_ant_ska : Number of anntennas in SKA. Default: 564.
	Returns
	----------
	sigma     : The rms of the noise in the image produced by SKA for uniformly distributed antennas.
	rms_noise : The rms of the noise due to the antenna positions in uv field.
	"""
	nuso  = 1420.0/(1.0 + z)
	delnu = depth_mhz*1e3	                                            # in kHz
	effective_baseline = np.sum(uv_map)
	T_sys_atnu300MHz= 60.0  					    #K
	T_sys = T_sys_atnu300MHz*(300.0/nuso)**2.55
	ant_radius_ska  = 35./2. 	                                    #in m
	nu_crit = 1.1e5 						    # in kHz
	if nuso>nu_crit: ep = (nu_crit/nuso)**2
	else: ep = 1. 	
	A_ant_ska = ep*np.pi*ant_radius_ska*ant_radius_ska
	sigma     = np.sqrt(2.0)*KB_SI*(T_sys/A_ant_ska)/np.sqrt((depth_mhz*1e6)*(obs_time*3600.0))/janskytowatt*1e3/np.sqrt(N_ant_ska*N_ant_ska/2.0) ## in mJy
	rms_noi  = 1e6*np.sqrt(2)*KB_SI*T_sys/A_ant_ska/np.sqrt(depth_mhz*1e6*int_time)/janskytowatt #in muJy
	sigma    = rms_noi/np.sqrt(N_ant_ska*(N_ant_ska-1)/2.0)/np.sqrt(3600*obs_time/int_time)      #in muJy
	if verbose:
		print '\nExpected: rms in image in muJy per beam for full =', sigma
		print 'Effective baseline =', sigma*np.sqrt(N_ant_ska*N_ant_ska/2.0)/np.sqrt(effective_baseline), 'm'
		print 'Calculated: rms in the visibility =', rms_noi, 'muJy'
	return sigma, rms_noi

def apply_uv_response(array, uv_map):
	"""
	Parameters
	----------
	array     : A complex 2d array of signal in the uv field.
	uv_map    : Numpy array containing the number of baselines observing each pixel.
	Returns 
	----------
	new_array : It is the 'array' after degrading the resoltion with the baseline configuration.
	"""
	noise_real = np.real(array)
	noise_img  = np.imag(array)
	noise_four = np.zeros(noise_real.shape)+1.j*np.zeros(noise_real.shape)
	ncells     = noise_real.shape[0]
	for i in xrange(ncells):
		for j in xrange(ncells):
			if uv_map[i,j] == 0: noise_four[i,j] = 0
			else: noise_four[i,j] = noise_real[i,j]/np.sqrt(uv_map[i,j]) + 1.j*noise_img[i,j]/np.sqrt(uv_map[i,j])
	return noise_four


def kelvin_jansky_conversion(ncells, z, boxsize=None):
	"""
	Parameters
	----------
	ncells  : Number of cells/pixels in the image.
	z       : Redshift
	boxsize : The comoving size of the sky observed. Default: It is determined from the 
	          simulation constants set.	
	Returns
	----------
	The conversion factor multiplied to values in kelvin to get values in jansky.
	"""
	if not boxsize: boxsize = conv.LB
	KB_SI       = 1.38e-23
	janskytowatt= 1e-26
	dist_z      = cm.z_to_cdist(z)
	boxsize_pp  = boxsize/dist_z				 #in rad	
	omega_pixel = boxsize_pp**2/ncells**2
	omega_total = boxsize_pp**2.0
	c_light_SI  = 2.99792458e+8                              #in m
	mktomujy_nuc= 2.0*KB_SI/c_light_SI/c_light_SI/janskytowatt*((cm.z_to_nu(z)*1e6)**2.0)*1e3
	con_sol     = mktomujy_nuc*omega_pixel
	return con_sol

def jansky_2_kelvin(array, z, boxsize=None, ncells=None):
	"""
	Parameters
	----------
	array   : Numpy array containing the values in jansky.
	z       : Redshift
	boxsize : The comoving size of the sky observed. Default: It is determined from the 
	          simulation constants set.
	ncells  : The number of grid cells. Default: None
	
	Returns
	----------
	A numpy array with values in mK.
	"""
	if not ncells: ncells  = array.shape[0]
	con_sol = kelvin_jansky_conversion(ncells, z, boxsize=boxsize)	
	return  array/con_sol

def kelvin_2_jansky(array, z, boxsize=None, ncells=None):
	"""
	Parameters
	----------
	array   : Numpy array containing the values in mK.
	z       : Redshift
	boxsize : The comoving size of the sky observed. Default: It is determined from the 
	          simulation constants set.
	Returns
	----------
	A numpy array with values in muJy.
	"""
	if not ncells: ncells  = array.shape[0]
	con_sol = kelvin_jansky_conversion(ncells, z, boxsize=boxsize)	
	return  array*con_sol
