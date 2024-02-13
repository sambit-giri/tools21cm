from . import const, conv
import numpy as np

def tau(ionfractions, redshifts, num_points = 50):
	'''
	Calculate the optical depth to Thomson scattering.
	
	Parameters:
		* ionfractions (numpy array): an array containing the ionized fraction
			in various points along the line-of-sight.
		* redshifts (numpy array): an array containing the redshifts of
			the same points as ionfractions. Must be the same length as
			ionfractions
		* num_points = 50 (integer): the number of points used for the integration
		
	Returns:
		Tuple containing (tau_0, tau_z)
		
		tau_0 is the optical depth at each redshift
		
		tau_z is the corresponding redshift
		
	Notes:
		* The Universe is assumed to be fully ionized at the lowest redshift supplied.
		* To get the total optical depth, look at the last value in tau_0
		
	Example:
		To calculate the optical depth for a scenario where the Universe is instantaneously
		reionized:
		
		>>> z_reion = 11.
		>>> redshifts = np.linspace(z_reion, 1100., 50)
		>>> ionfracs = np.zeros(len(redshifts))
		>>> tau0, tau_z = tau(ionfracs, redshifts)
		>>> print 'Total tau: ', tau0[-1]
		0.0884755058758
		
	'''
	
	if len(ionfractions) != len(redshifts):
		print('Incorrect length of ionfractions')
		raise Exception()
	
	ionfractions, redshifts = ionfractions[np.argsort(redshifts)], redshifts[np.argsort(redshifts)]

	sigma_T = 6.65e-25
	chi1 = 1.0+const.abu_he
	coeff = 2.0*(const.c*1e5)*sigma_T*const.OmegaB/const.Omega0*const.rho_crit_0*\
	chi1/const.mean_molecular/const.m_p/(3.*const.H0cgs)

	tau_z = np.hstack((np.arange(1,num_points+1)/float(num_points)*redshifts[0], redshifts))

	tau0 = np.zeros(len(redshifts)+num_points)

	#Optical depth for a completely ionized Universe
	tau0[0:num_points] = coeff*(np.sqrt(const.Omega0*(1+tau_z[0:num_points])**3+const.lam) - 1)

	for i in range(num_points, len(redshifts)+num_points):
		tau0[i] =tau0[i-1]+1.5*coeff*const.Omega0 * \
		(ionfractions[i-1-num_points]*(1+tau_z[i-1])**2/np.sqrt(const.Omega0*(1+tau_z[i-1])**3+const.lam) \
		+ ionfractions[i-num_points]*(1+tau_z[i])**2/np.sqrt(const.Omega0*(1+tau_z[i])**3+const.lam) ) * \
		(tau_z[i]-tau_z[i-1])/2


	return tau0, tau_z

