'''
Methods to estimate the brightness temperature.
'''

import numpy as np
from . import const, conv
from . import cosmo
from .helper_functions import print_msg, read_cbin, \
	get_data_and_type, determine_redshift_from_filename
from astropy import constants, units
from astropy.cosmology import Planck18

def calc_dt_halo(mhalo, box_dim, z, Y_p=0.249, cosmo=None):
	'''
	Calculate the differential brightness temperature assuming T_s >> T_CMB

	Parameters:
		mhalo (numpy array): the ionization fraction
		box_dim (float): Length of the simulation box in each direction.
		z (float): The redshift.
		Y_p = 0.249 (float): Helium abundance mass factor.
		cosmo : astropy.cosmology.Cosmology, optional
        Cosmology object. If None, assumes Planck18 cosmology.
		
	Returns:
		The differential brightness temperature (in mK) as a numpy array with
		the same dimensions as xfrac.
	'''
	if cosmo is None:
		print('Assuming Planck18 cosmology.')
		cosmo = Planck18
		
	if isinstance(mhalo, units.quantity.Quantity):
		mhalo = mhalo.to('Msun')
	else:
		print('The provided halo mass is assumed to be in Msun units.')
		mhalo = mhalo*units.Msun

	if not isinstance(box_dim, units.quantity.Quantity):
		box_dim = box_dim*units.Mpc
	
	mu = 1/(1-Y_p)
	Vcell = (box_dim/mhalo.shape[0])**3
	nHI = (mhalo/Vcell/mu/constants.m_p).to('1/cm^3')
	nH  = (cosmo.Ob0*cosmo.critical_density0/mu/constants.m_p).to('1/cm^3')
	xHI = (nHI/nH).to('')
	Cdt = mean_dt(z)
	dt = Cdt*xHI
	return dt

def calc_dt(xfrac, dens, z = -1):
	'''
	Calculate the differential brightness temperature assuming T_s >> T_CMB
	
	Parameters:
		xfrac (XfracFile object, string or numpy array): the ionization fraction
		dens (DensityFile object, string or numpy array): density in cgs units
		z = -1 (float): The redshift (if < 0 this will be figured out from the files)
		
	Returns:
		The differential brightness temperature (in mK) as a numpy array with
		the same dimensions as xfrac.
	'''

	xi, xi_type = get_data_and_type(xfrac)
	rho, rho_type = get_data_and_type(dens)
	xi = xi.astype('float64')
	rho = rho.astype('float64')
	
	if z < 0:
		z = determine_redshift_from_filename(xfrac)
		if z < 0:
			z = determine_redshift_from_filename(dens)
		if z < 0:
			raise Exception('No redshift specified. Could not determine from file.')
	
	print_msg('Making dT box for z=%f' % z)
	
	#Calculate dT
	return _dt(rho, xi, z)

def calc_dt_full(xfrac, dens, temp, z = -1, correct=True):
	'''
	Calculate the differential brightness temperature assuming only that Lyman alpha is fully coupled so T_s = T_k 

	(NOT T_s >> T_CMB)
	
	Parameters:
		xfrac (XfracFile object, string or numpy array): the ionization fraction
		dens (DensityFile object, string or numpy array): density in cgs units
        	temp (TemperFile object, string or numpy array): the temperature in K
		z = -1 (float): The redshift (if < 0 this will be figured out from the files)
		correct = True (bool): if true include a correction for partially ionized cells.

	Returns:
		The differential brightness temperature (in mK) as a numpy array with
		the same dimensions as xfrac.
	'''

	xi, xi_type   = get_data_and_type(xfrac)
	Ts, Ts_type   = get_data_and_type(temp)
	rho, rho_type = get_data_and_type(dens)
	xi  = xi.astype('float64')
	Ts  = Ts.astype('float64')
	rho = rho.astype('float64')
	
	if z < 0:
		z = determine_redshift_from_filename(xfrac)
		if z < 0: z = determine_redshift_from_filename(dens)
		if z < 0: z = determine_redshift_from_filename(temp)
		if z < 0: raise Exception('No redshift specified. Could not determine from file.')
	
	print_msg('Making full dT box for z=%f' % z)
	
	print("Calculating corrected dbt")
	return _dt_full(rho, xi, Ts, z, correct)

def calc_dt_lightcone(xfrac, dens, lowest_z, los_axis = 2):
	'''
	Calculate the differential brightness temperature assuming T_s >> T_CMB
	for lightcone data.
	
	Parameters:
		xfrac (string or numpy array): the name of the ionization 
			fraction file (must be cbin), or the xfrac lightcone data
		dens (string or numpy array): the name of the density 
			file (must be cbin), or the density data
		lowest_z (float): the lowest redshift of the lightcone volume
		los_axis = 2 (int): the line-of-sight axis
		
	Returns:
		The differential brightness temperature (in mK) as a numpy array with
		the same dimensions as xfrac.
	'''
	
	try:
		xfrac = read_cbin(xfrac)
	except Exception:
		pass
	try:
		dens = read_cbin(dens)
	except:
		pass
	dens = dens.astype('float64')
		
	cell_size = conv.LB/xfrac.shape[(los_axis+1)%3]
	cdist_low = cosmo.z_to_cdist(lowest_z)
	cdist = np.arange(xfrac.shape[los_axis])*cell_size + cdist_low
	z = cosmo.cdist_to_z(cdist)
	return _dt(dens, xfrac, z)

def calc_dt_full_lightcone(xfrac, temp, dens, lowest_z, los_axis = 2, correct=True):
	'''
	Calculate the differential brightness temperature assuming only that Lyman alpha is fully coupled so T_s = T_k

	(NOT T_s >> T_CMB) for lightcone data. UNTESTED
	
	Parameters:
		xfrac (string or numpy array): the name of the ionization 
			fraction file (must be cbin), or the xfrac lightcone data
                temp (string or numpy array): the name of the temperature
                       file (must be cbin), or the temp lightcone data
		dens (string or numpy array): the name of the density 
			file (must be cbin), or the density data
		lowest_z (float): the lowest redshift of the lightcone volume
		los_axis = 2 (int): the line-of-sight axis
		correct = True (bool): if true include a correction for 
                        partially ionized cells.

	Returns:
		The differential brightness temperature (in mK) as a numpy array with
		the same dimensions as xfrac.
	'''
	
	try:
		xfrac = read_cbin(xfrac)
	except Exception:
		pass
	try:
		temp = read_cbin(temp)
	except Exception:
		pass
	try:
		dens = read_cbin(dens)
	except:
		pass
	dens = dens.astype('float64')
		
	cell_size = conv.LB/xfrac.shape[(los_axis+1)%3]
	cdist_low = cosmo.z_to_cdist(lowest_z)
	cdist = np.arange(xfrac.shape[los_axis])*cell_size + cdist_low
	z = cosmo.cdist_to_z(cdist)
	print("Redshift: ", str(z))
	return _dt_full(dens, xfrac,temp, z, correct)

def mean_dt(z):
	'''
	Get the mean dT at redshift z
	
	Parameters:
		z (float or numpy array): the redshift
		
	Returns:
		dT (float or numpy array) the mean brightness temperature
		in mK
	'''
	Ez = np.sqrt(const.Omega0*(1.0+z)**3+const.lam+\
				(1.0-const.Omega0-const.lam)*(1.0+z)**2)

	Cdt = const.meandt/const.h*(1.0+z)**2/Ez
	
	return Cdt
	

def _dt(rho, xi, z):
		
	rho_mean = const.rho_crit_0*const.OmegaB if rho.min()>=0 else 1

	Cdt = mean_dt(z)
	dt = Cdt*(1.0-xi)*rho/rho_mean
	
	return dt

def _dt_full(rho, xi, Ts, z, correct):
        z = np.mean(z)
        print("Redshift:" , str(z))
        rho_mean = const.rho_crit_0*const.OmegaB
        Tcmb 	 = const.Tcmb0*(1+z) 
        Cdt      = mean_dt(z)

        if correct:
                # Correct the kinetic temperature in partially ionized cells.
                
                # Find the temperature at a reference redshift zi and assume
                # adiabatic cooling since then to find the minimum temperature
                # at the redshift of interest.
                # Original code use the recombination redshift (z=1089) but
                # since Compton scattering couples kinetic and CMB temperature
                # up to z=200 (approximately), it is better to use z=200 as a
                # reference.
                # To do: check this against CMBFAST or similar code.
                zi = 200 #1089
                Ti = const.Tcmb0*(1+zi)
                T_min = Ti*(1+z)**2/(1+zi)**2
                # Assume the kinetic temperature in ionized regions to be
                # some fixed value 
                T_HII = 2.0e4
                # Calculate the temperature of the neutral medium in a cell by
                # assuming that the ionized part of the cell is fully ionized
                # (so a fraction xi of the cell's volume is ionized) and that
                # the temperature of the ionized part is T_HII
                Ts_new = (Ts-xi*T_HII)/(1.-xi)
                Ts_new[Ts_new < T_min] = T_min
                if np.any(Ts_new < 0): print("WARNING: negative temperatures")
        else:
                Ts_new=Ts
        # Calculate the differential temperature brightness
        dt = ((Ts_new-Tcmb)/Ts_new)*Cdt*(1.0-xi)*rho/rho_mean
        return dt

# def subtract_mean_channelwise(dt, axis=-1):
# 	"""
# 	Parameters:
# 		dt  (ndarray): Brightness temperature whose channel-wise should be subtracted.
# 		axis (int): Frequency axis (Defualt:-1).

# 	Returns:
# 		numpy array
# 	"""
# 	assert dt.ndim == 3
# 	if axis != -1 or axis != 2: dt = np.swapaxes(dt, axis, 2)
# 	for i in range(dt.shape[2]): dt[:,:,i] -= dt[:,:,i].mean()
# 	if axis != -1 or axis != 2: dt = np.swapaxes(dt, axis, 2)
# 	return dt
