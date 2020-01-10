'''
Various values and functions to deal with unit conversions regarding file types
'''

from . import const
import numpy as np
from . import helper_functions as hf

#Conversion factors and other stuff relating to C2Ray simulations
boxsize=244.0
LB=boxsize/const.h
nbox_fine=8000

M_box      = const.rho_matter*(LB*const.Mpc)**3 # mass in box (g, not M0)
M_grid     = M_box/(float(nbox_fine)**3)

#conversion factor for the velocity (sim units to km/s)
lscale = (LB)/float(nbox_fine)*const.Mpc # size of a cell in cm, comoving
tscale = 2.0/(3.0*np.sqrt(const.Omega0)*const.H0/const.Mpc*1.e5) # time scale, when divided by (1+z)2
velconvert = lambda z: lscale/tscale*(1.0+z)/1.e5

def set_sim_constants(boxsize_cMpc):
	'''This method will set the values of relevant constants depending on the 
	simulation
	
	Parameters:
		boxsize_cMpc (float): the box size in cMpc/h. Valid values are 37, 47, 64, 114, 200, 244, 425 or 500
		
	Returns:
		Nothing.
	'''
	global boxsize, LB, nbox_fine, M_box, M_grid, lscale, tscale, velconvert

	boxsize = boxsize_cMpc
	LB = boxsize/const.h	
	if hf.flt_comp(boxsize, 425.):
		hf.print_msg('Setting conversion factors for 425/h Mpc box')
		nbox_fine = 10976
	elif hf.flt_comp(boxsize, 114.):
		hf.print_msg('Setting conversion factors for 114/h Mpc box')
		nbox_fine = 6144
	elif hf.flt_comp(boxsize, 64.):
		hf.print_msg('Setting conversion factors for 64/h Mpc box')
		nbox_fine = 3456
	elif hf.flt_comp(boxsize, 37.):
		hf.print_msg('Setting conversion factors for 37/h Mpc box')
		nbox_fine = 2048
	elif hf.flt_comp(boxsize, 500.):
		hf.print_msg('Setting_conversion_factors for 500/h Mpc box')
		nbox_fine = 13824
	elif hf.flt_comp(boxsize, 244.):
		hf.print_msg('Setting_conversion_factors for 244/h Mpc box')
		nbox_fine = 8000
	elif hf.flt_comp(boxsize, 47.):
		hf.print_msg('Setting_conversion_factors for 47/h Mpc box')
		nbox_fine = 3456
	elif hf.flt_comp(boxsize, 200.):
		hf.print_msg('Setting_conversion_factors for 200/h Mpc box')
		nbox_fine = 3456
	else:
		raise Exception('Invalid boxsize (%.3f cMpc)' % boxsize_cMpc)

	M_box      = const.rho_matter*(LB*const.Mpc)**3 # mass in box (g, not M0)
	M_grid     = M_box/(float(nbox_fine)**3)
	lscale = (LB)/float(nbox_fine)*const.Mpc # size of a cell in cm, comoving
	tscale = 2.0/(3.0*np.sqrt(const.Omega0)*const.H0/const.Mpc*1.e5) # time scale, when divided by (1+z)2
	velconvert = lambda z: lscale/tscale*(1.0+z)/1.e5


def gridpos_to_mpc(gridpos):
	'''
	Convert a position or length in simulation grid units to Mpc
	
	Parameters:
		gridpos (float or numpy array): the position in simulation grid units
		
	Returns:
		The position converted to Mpc with the origin unchanged
	'''
	return gridpos*lscale/const.Mpc


def gridvel_to_kms(gridvel, z):
	'''
	Convert a velocity in simulation grid units to km/s
	
	Parameters:
		gridvel (float or numpy array): the velocity in simulation grid units
		z (float or numpy array): the redshift
		
	Returns:
		The velocity in km/s
	'''
	return gridvel*velconvert(z)


def gridmass_to_msol(grid_mass):
	'''
	Convert a halo mass from simulation grid units to solar masses
	
	Parameters:
		gridmass (float or numpy array): the mass in grid mass units
		
	Returns:
		The mass in log10(Msol) 
	'''
	return np.log10(grid_mass*M_grid*const.solar_masses_per_gram)
