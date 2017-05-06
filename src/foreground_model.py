import numpy as np
import c2raytools as c2t
from telescope_functions import jansky_2_kelvin, from_antenna_config

def galactic_synch_fg(z, ncells, boxsize, max_baseline=2.):
	"""
	@ Ghara et al. (2017)
	Parameter
	---------
	z           : Redshift observed with 21-cm.
	ncells      : Number of cells on each axis.
	boxsize     : Size of the FOV in Mpc.
	max_baseline: Maximum baseline of the radio telescope in km (Default: 2).
	Return
	------
	A 2D numpy array of brightness temperature in mK.
	"""
	X  = np.random.normal(size=(ncells, ncells))
	Y  = np.random.normal(size=(ncells, ncells))
	nu = c2t.z_to_nu(z)
	nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1
	c_light, m2Mpc = 3e8, 3.24e-23
	lam   = c_light/(nu*1e6)/1000.
	bb    = np.mgrid[1:ncells+1,1:ncells+1]*max_baseline/ncells/lam
	l_cb  = 2*np.pi*np.sqrt(bb[0,:,:]**2+bb[1,:,:]**2)
	C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
	solid_angle = boxsize**2/c2t.z_to_cdist(z)**2
	AA = np.sqrt(solid_angle*C_syn/2)
	T_four = AA*(X+Y*1j)
	T_real = np.real(np.fft.ifft2(T_four))
	return T_real

def extragalactic_pointsource_fg(z, ncells, boxsize, S_max=100):
	"""
	@ Ghara et al. (2017)
	Parameter
	---------
	z           : Redshift observed with 21-cm.
	ncells      : Number of cells on each axis.
	boxsize     : Size of the FOV in Mpc.
	S_max       : Maximum flux of the point source to model in muJy (Default: 100).
	Return
	------
	A 2D numpy array of brightness temperature in mK.
	"""
	nu = c2t.z_to_nu(z)
	fg = np.zeros((ncells,ncells))
	dS = 0.01
	Ss = np.arange(0.1, S_max, dS)
	solid_angle = boxsize**2/c2t.z_to_cdist(z)**2
	N  = int(10**3.75*np.trapz(Ss**(-1.6), x=Ss, dx=dS)*solid_angle)
	x,y = np.random.random_integers(0, high=ncells, size=(2,N))
	alpha_ps = 0.7+0.1*np.random.random(size=N)
	nu_s, S_s = 150, S_max
	S_nu = S_s*(nu/nu_s)**(-alpha_ps)
	for p in xrange(S_nu.size): fg[x[p],y[p]] = S_nu[p]
	return jansky_2_kelvin(fg, z, boxsize=boxsize, ncells=ncells)
	
	
	
	 
	
