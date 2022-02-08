'''
Methods to simulate and analyse the foreground signal for 21 cm signal.
'''

import numpy as np
from .telescope_functions import jansky_2_kelvin, from_antenna_config
from . import cosmology as cm
from . import conv

def galactic_synch_fg(z, ncells, boxsize, max_baseline=2.):
	"""
	@ Ghara et al. (2017)

	Parameters
	----------
	z           : float
		Redshift observed with 21-cm.
	ncells      : int
		Number of cells on each axis.
	boxsize     : float
		Size of the FOV in Mpc.
	max_baseline: float
		Maximum baseline of the radio telescope in km (Default: 2).
	Returns
	-------
	A 2D numpy array of brightness temperature in mK.
	"""
	X  = np.random.normal(size=(ncells, ncells))
	Y  = np.random.normal(size=(ncells, ncells))
	nu = cm.z_to_nu(z)
	nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1
	#c_light, m2Mpc = 3e8, 3.24e-23
	#lam   = c_light/(nu*1e6)/1000.
	U_cb  = (np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]+0.5)*cm.z_to_cdist(z)/boxsize
	l_cb  = 2*np.pi*np.sqrt(U_cb[0,:,:]**2+U_cb[1,:,:]**2)
	C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
	solid_angle = boxsize**2/cm.z_to_cdist(z)**2
	AA = np.sqrt(solid_angle*C_syn/2)
	T_four = AA*(X+Y*1j)
	T_real = np.abs(np.fft.ifft2(T_four))   #in Jansky
	return jansky_2_kelvin(T_real*1e6, z, boxsize=boxsize, ncells=ncells)

def extragalactic_pointsource_fg(z, ncells, boxsize, S_max=100):
	"""
	@ Ghara et al. (2017)

	Parameters
	----------
	z           : float
		Redshift observed with 21-cm.
	ncells      : int
		Number of cells on each axis.
	boxsize     : float
		Size of the FOV in Mpc.
	S_max       : float
		Maximum flux of the point source to model in muJy (Default: 100).

	Returns
	-------
	A 2D numpy array of brightness temperature in mK.
	"""
	nu = cm.z_to_nu(z)
	fg = np.zeros((ncells,ncells))
	dS = 0.01
	Ss = np.arange(0.1, S_max, dS)
	solid_angle = boxsize**2/cm.z_to_cdist(z)**2
	N  = int(10**3.75*np.trapz(Ss**(-1.6), x=Ss, dx=dS)*solid_angle)
	x,y = np.random.random_integers(0, high=ncells-1, size=(2,N))
	alpha_ps = 0.7+0.1*np.random.random(size=N)
	S_s  = np.random.choice(Ss, N)
	nu_s = 150
	S_nu = S_s*(nu/nu_s)**(-alpha_ps)
	for p in xrange(S_nu.size): fg[x[p],y[p]] = S_nu[p]
	return jansky_2_kelvin(fg, z, boxsize=boxsize, ncells=ncells)

def diabolo_filter(z, ncells=None, array=None, boxsize=None, mu=0.5, funct='step', small_base=40):
	assert funct in ['step', 'sigmoid', 'gaussian']
	assert ncells or array is not None
	if boxsize is None: boxsize = conv.LB
	if array is not None: ncells = max(array.shape)
	filt = np.zeros((ncells, ncells, ncells))
	k0   = np.linspace(-ncells*np.pi/boxsize, ncells*np.pi/boxsize, ncells)
	a = k0.reshape(-1,1)
	for i in xrange(k0.size-1): a = np.hstack((a,k0.reshape(-1,1)))
	b = k0.reshape(1,-1)
	for i in xrange(k0.size-1): b = np.vstack((b,k0.reshape(1,-1)))
	k2 = np.sqrt(a**2+b**2)
	#mu1 = np.sqrt(1 - mu**2) 
	kmin = np.pi/(cm.z_to_cdist(z)*(21./small_base/1e2))
	for i in xrange(ncells):
		kpp = k0[i]
		kpr = np.abs(kpp)*(1-mu**2)/mu
		if funct == 'sigmoid': 
			ss = 1-1./(1+np.exp(10*(k2-kpr)))
			#ss = 1./(1+np.exp(10*(-kpr+kmin)))*1./(1+np.exp(10*(k2-kpr)))
		else: 
			ss = np.ones(k2.shape)
			ss[k2<=kpr]  = 0
		ss[k2<=kmin] = 0
		filt[:,:,i] = ss
	if array.shape[2]<ncells: filt = filt[:,:,ncells/2-array.shape[2]/2:ncells/2+array.shape[2]/2]
	print("A diabolo filter made with %s function."%funct)
	return filt

def barrel_filter(z, ncells=None, array=None, boxsize=None, k_par_min=None, small_base=40):
	assert ncells or array is not None
	if boxsize is None: boxsize = conv.LB
	if array is not None: ncells = max(array.shape)
	k0   = np.linspace(-ncells*np.pi/boxsize, ncells*np.pi/boxsize, ncells)
	a = k0.reshape(-1,1)
	for i in xrange(k0.size-1): a = np.hstack((a,k0.reshape(-1,1)))
	b = k0.reshape(1,-1)
	for i in xrange(k0.size-1): b = np.vstack((b,k0.reshape(1,-1)))
	k2 = np.sqrt(a**2+b**2)
	if k_par_min is None: k_par_min = np.pi/(cm.z_to_cdist(z)*(21./small_base/1e2))
	ss = np.ones(k2.shape)
	ss[k2<=k_par_min] = 0
	filt = np.zeros((ncells,ncells,ncells))
	for i in xrange(ncells): filt[:,:,i] = filt
	if array.shape[2]<ncells: filt = filt[:,:,ncells/2-array.shape[2]/2:ncells/2+array.shape[2]/2]
	print("A barrel filter made with step function.")
	return filt	

def remove_wedge_image(dt, z, mu=0.5, funct='step', boxsize=None, filt=None):
	if filt is None: filt = diabolo_filter(z, array=dt, mu=mu, funct=funct, boxsize=boxsize)
	fft_dt = np.fft.fftn(dt)
	apply_filt = np.fft.fftshift(np.fft.fftshift(fft_dt)*filt)
	dt_new = np.real(np.fft.ifftn(apply_filt))
	return dt_new
	
	
	
	 
	
