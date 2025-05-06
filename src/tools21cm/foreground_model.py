'''
Methods to simulate and analyse the foreground signal for 21 cm signal.
'''

import numpy as np
from .scipy_func import *
from .radio_telescope_sensitivity import jansky_2_kelvin, from_antenna_config
from . import cosmo as cm
from . import const
from . import conv

def galactic_synch_fg(z, ncells, boxsize, rseed=False):
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
	rseed: int
		random seed to have the same realisation (Default: False).
	Returns
	-------
	A 2D numpy array of brightness temperature in mK.
	"""
	if(isinstance(z, float)):
		z = np.array([z])
	else:
		z = np.array(z, copy=False)
	gf_data = np.zeros((ncells, ncells, z.size))

	if(rseed): np.random.seed(rseed)
	X  = np.random.normal(size=(ncells, ncells))
	Y  = np.random.normal(size=(ncells, ncells))
	nu_s,A150,beta_,a_syn,Da_syn = 150,513,2.34,2.8,0.1

	for i in range(0, z.size):
		nu = cm.z_to_nu(z[i])
		U_cb  = (np.mgrid[-ncells/2:ncells/2,-ncells/2:ncells/2]+0.5)*cm.z_to_cdist(z[i])/boxsize
		l_cb  = 2*np.pi*np.sqrt(U_cb[0,:,:]**2+U_cb[1,:,:]**2)
		C_syn = A150*(1000/l_cb)**beta_*(nu/nu_s)**(-2*a_syn-2*Da_syn*np.log(nu/nu_s))
		solid_angle = boxsize**2/cm.z_to_cdist(z[i])**2
		AA = np.sqrt(solid_angle*C_syn/2)
		T_four = AA*(X+Y*1j)
		T_real = np.abs(np.fft.ifft2(T_four))   #in Jansky
		gf_data[..., i] = jansky_2_kelvin(T_real*1e6, z[i], boxsize=boxsize, ncells=ncells)
	return gf_data.squeeze()

def extragalactic_pointsource_fg(z, ncells, boxsize, rseed=False, S_max=100):
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
	if(isinstance(z, float)):
		z = np.array([z])
	else:
		z = np.array(z, copy=False)
	exgf_data = np.zeros((ncells, ncells, z.size))

	dS, nu_s = 0.01, 150
	solid_angle = boxsize**2/cm.z_to_cdist(z.min())**2
	
	Ss = np.arange(0.1, S_max, dS)
	N  = int(10**3.75*np.trapz(Ss**(-1.6), x=Ss, dx=dS)*solid_angle)
	if(rseed): np.random.seed(rseed)
	x, y = np.random.random_integers(0, high=ncells-1, size=(2,N))
	alpha_ps = 0.7+0.1*np.random.random(size=N)
	S_s  = np.random.choice(Ss, N)

	for i in range(0, z.size):
		nu = cm.z_to_nu(z[i])
		S_nu = S_s*(nu/nu_s)**(-alpha_ps)
		for p in range(S_nu.size):
			exgf_data[x[p], y[p], i] = jansky_2_kelvin(S_nu[p], z[i], boxsize=boxsize, ncells=ncells)
	return exgf_data.squeeze()


def diabolo_filter(z, ncells=None, array=None, boxsize=None, mu=0.5, funct='step', small_base=40):
	assert funct in ['step', 'sigmoid', 'gaussian']
	assert ncells or array is not None
	if boxsize is None: boxsize = conv.LB
	if array is not None: ncells = max(array.shape)
	filt = np.zeros((ncells, ncells, ncells))
	k0   = np.linspace(-ncells*np.pi/boxsize, ncells*np.pi/boxsize, ncells)
	a = k0.reshape(-1,1)
	for i in range(k0.size-1): a = np.hstack((a,k0.reshape(-1,1)))
	b = k0.reshape(1,-1)
	for i in range(k0.size-1): b = np.vstack((b,k0.reshape(1,-1)))
	k2 = np.sqrt(a**2+b**2)
	#mu1 = np.sqrt(1 - mu**2) 
	kmin = np.pi/(cm.z_to_cdist(z)*(21./small_base/1e2))
	for i in range(ncells):
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
	for i in range(k0.size-1): a = np.hstack((a,k0.reshape(-1,1)))
	b = k0.reshape(1,-1)
	for i in range(k0.size-1): b = np.vstack((b,k0.reshape(1,-1)))
	k2 = np.sqrt(a**2+b**2)
	if k_par_min is None: k_par_min = np.pi/(cm.z_to_cdist(z)*(21./small_base/1e2))
	ss = np.ones(k2.shape)
	ss[k2<=k_par_min] = 0
	filt = np.zeros((ncells,ncells,ncells))
	for i in range(ncells): filt[:,:,i] = filt
	if array.shape[2]<ncells: filt = filt[:,:,ncells/2-array.shape[2]/2:ncells/2+array.shape[2]/2]
	print("A barrel filter made with step function.")
	return filt	

def remove_wedge_image(dt, z, mu=0.5, funct='step', boxsize=None, filt=None):
	if filt is None: filt = diabolo_filter(z, array=dt, mu=mu, funct=funct, boxsize=boxsize)
	fft_dt = np.fft.fftn(dt)
	apply_filt = np.fft.fftshift(np.fft.fftshift(fft_dt)*filt)
	dt_new = np.real(np.fft.ifftn(apply_filt))
	return dt_new


def rolling_wedge_removal_lightcone(lightcone, redshifts, cell_size=None, chunk_length=None, OMm=None, buffer_threshold = 1e-10):
	"""Rolling over the lightcone and removing the wedge for every frequency channel.

	Parameters:
		lightcone (array): The lightcone, of shape `(n_cells, n_cells, n_redshfts)`.
		redshifts (array): The redshifts of every frequency channel.
		cell_size (float): Resolution of the lightcone voxels.
			It is assumed that the lightcone is made of cubical voxels of volume `cell_size^3`.
		chunk_length (int): Length of the box in z-direction used for wedge removal.
			Defaults to `n_cells`.
		OMm (float): Omega matter.
		buffer_threshold (float): Threshold which defines a wedge buffer.
			Buffer is then calculated as k-value for which Blackman taper power is below threshold.
	Returns:
		Cleaned lightcone.
	"""
	def one_over_E(z, OMm):
		return 1 / np.sqrt(OMm*(1.+z)**3 + (1 - OMm))
	def multiplicative_factor(z, OMm):
		return 1 / one_over_E(z, OMm) / (1+z) * quad(lambda x: one_over_E(x, OMm), 0, z)[0]

	if cell_size is None:
		cell_size = np.min(lightcone.shape)
	if chunk_length is None:
		chunk_length = np.min(lightcone.shape)
	if OMm is None:
		OMm = const.Omega0

	Box_uv = np.fft.fft2(lightcone.astype(np.float32), axes=(0, 1))
	redshifts = redshifts.astype(np.float32)
	MF = np.array([multiplicative_factor(z, OMm) for z in redshifts], dtype = np.float32)

	k = np.fft.fftfreq(len(lightcone), d=cell_size)
	k_parallel = np.fft.fftfreq(chunk_length, d=cell_size)
	delta_k = k_parallel[1] - k_parallel[0]
	k_cube = np.meshgrid(k, k, k_parallel)
	k_perp, k_par = np.sqrt(k_cube[0]**2 + k_cube[1]**2).astype(np.float32), k_cube[2].astype(np.float32)

	bm = np.abs(np.fft.fft(np.blackman(chunk_length)))**2
	buffer = delta_k * (np.where(bm / np.amax(bm) <= buffer_threshold)[0][0] - 1)
	buffer = buffer.astype(np.float32)
	BM = np.blackman(chunk_length).astype(np.float32)[np.newaxis, np.newaxis, :]

	box_shape = Box_uv.shape
	Box_final = np.empty(box_shape, dtype = np.float32)
	empty_box = np.zeros(k_cube[0].shape, dtype = np.float32)
	Box_uv = np.concatenate((empty_box, Box_uv, empty_box), axis = 2)
    
	for i in range(chunk_length, box_shape[-1] + chunk_length):
		t_box = np.copy(Box_uv[..., i-chunk_length//2: i+chunk_length//2])
		t_box *= BM
		W = k_par / (k_perp * MF[min(i - chunk_length // 2 - 1, box_shape[-1] - 1)] + buffer)
		w = np.logical_or(W < -1., W > 1.)
		Box_final[..., i - chunk_length] = np.real(np.fft.ifftn(np.fft.fft(t_box, axis = -1) * w))[..., chunk_length // 2]
        
	return Box_final.astype(np.float32)
