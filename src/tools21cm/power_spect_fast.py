import numpy as np, gc
from scipy import interpolate, stats

#from numba import jit, prange
#from astropy.stats import histogram 
from tqdm import tqdm

from . import cosmo, smoothing, const 
from .scipy_func import numpy_product

def power_spect_nd(input_array, box_dims, verbose=True):
	''' 
	Calculate the power spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used
	
	Parameters:
		* input_array (numpy array): the array to calculate the 
			power spectrum of. Can be of any dimensions.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
	
	Returns:
		The power spectrum in the same dimensions as the input array.		
	'''

	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	if verbose: print( 'Calculating power spectrum...')
	ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
	power_spectrum = np.abs(ft)**2
	if verbose: print( '...done')

	# scale
	#print(box_dims)
	boxvol = numpy_product(box_dims)
	#print(boxvol)
	pixelsize = boxvol/(numpy_product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol
	
	return power_spectrum

def phase_spect_nd(input_array, box_dims, verbose=True):
	''' 
	Calculate the phase spectrum of input_array and return it as an n-dimensional array,
	where n is the number of dimensions in input_array
	box_side is the size of the box in comoving Mpc. If this is set to None (default),
	the internal box size is used
	
	Parameters:
		* input_array (numpy array): the array to calculate the 
			power spectrum of. Can be of any dimensions.
		* box_dims = None (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
	
	Returns:
		The power spectrum in the same dimensions as the input array.		
	'''

	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	if verbose: print( 'Calculating power spectrum...')
	ft = np.fft.fftshift(np.fft.fftn(input_array.astype('float64')))
	phase_spectrum = np.arctan(np.imag(ft)/np.real(ft)) #np.abs(ft)**2
	if verbose: print( '...done')

	# scale
	#print(box_dims)
	boxvol = np.product(box_dims)
	#print(boxvol)
	pixelsize = boxvol/(np.product(input_array.shape))
	power_spectrum *= pixelsize**2/boxvol
	
	return power_spectrum

def _get_k(input_array, box_dims):
	'''
	Get the k values for input array with given dimensions.
	Return k components and magnitudes.
	For internal use.
	'''
	if np.array(box_dims).size!=3: box_dims = np.array([box_dims,box_dims,box_dims])
	dim = len(input_array.shape)
	if dim == 1:
		x = np.arange(len(input_array))
		center = x.max()/2.
		kx = 2.*np.pi*(x-center)/box_dims[0]
		return [kx], kx
	elif dim == 2:
		x,y = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		k = np.sqrt(kx**2 + ky**2)
		return [kx, ky], k
	elif dim == 3:
		x,y,z = np.indices(input_array.shape, dtype='int32')
		center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2, \
						(z.max()-z.min())/2])
		kx = 2.*np.pi * (x-center[0])/box_dims[0]
		ky = 2.*np.pi * (y-center[1])/box_dims[1]
		kz = 2.*np.pi * (z-center[2])/box_dims[2]

		k = np.sqrt(kx**2 + ky**2 + kz**2 )
		return [kx,ky,kz], k


def power_spect_1d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False, kmin=None, kmax=None):
	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	if kmin is None: kmin = 2*np.pi/np.array(box_dims).max()
	if kmax is None: kmax = k.max()
	if binning=='log': 
		ks = np.linspace(np.log10(kmin), np.log10(kmax), kbins+1)
		k  = np.log10(k)
	elif binning=='linear': ks = np.linspace(kmin, kmax, kbins+1)
	elif binning=='auto': 
		#kht = histogram(k.flatten(), bins='knuth')
		n_bin  = 1.*k.size/kbins
		k_sort = np.sort(k.flatten())
		ks = np.array([k_sort[int(i)] for i in np.arange(n_bin/2., k.size, n_bin)])
	if binning!='auto': ks = (ks[:-1]+ks[1:])/2.
	kkk = np.hstack((k.min(),(ks[1:]+ks[:-1])/2, k.max()))
	k_width = kkk[1:]-kkk[:-1]
	ps = np.zeros(kbins)
	n_modes = np.zeros(kbins)
	k, power = k.flatten(), power.flatten()
	for i,a in enumerate(ks):
		arg = np.argwhere(np.abs(k-a)<=k_width[i]/2.)
		ps[i] = power[arg].sum()
		n_modes[i] = arg.size

	ps = ps/n_modes
	if binning=='log': ks = 10**ks
	if return_modes: return ps, ks, n_modes
	return ps, ks

def power_spect_2d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False, nu_axis=2, window=None):
	'''
	Calculate the power spectrum and bin it in kper and kpar
	input_array is the array to calculate the power spectrum from
	
	Parameters: 
		input_array (numpy array): the data array
		nu_axis = 2 (integer): the line-of-sight axis
		kbins = 10 (integer or array-like): The number of bins,
			If you want different bins for kper and kpar, then provide a list [n_kper, n_par]
		box_dims = 244/.7 (float or array-like): the dimensions of the 
			box. If this is None, the current box volume is used along all
			dimensions. If it is a float, this is taken as the box length
			along all dimensions. If it is an array-like, the elements are
			taken as the box length along each axis.
		return_n_modes = False (bool): if true, also return the
			number of modes in each bin
		binning = 'log' : It defines the type of binning in k-space. The other options are
				    'linear' or 'mixed'.
		window = None : It tappers the data in the frequency direction to control shape change at the boundary slices. 
					The other options are 'blackmanharris' and 'tukey'. If the data has sharp change in the angular/spatial 
					direction, please provide a 3D window as a numpy array.
			
	Returns: 
		A tuple with (Pk, kper_bins, kpar_bins) if return_modes is False else (Pk, kper_bins, kpar_bins, n_modes), 
		where Pk is an array with the power spectrum of dimensions (n_kper x n_kpar), 
		mubins is an array with the mu bin centers,
		kbins is an array with the k bin centers and 
		n_modes is the number of modes.
	
	'''
	if window is not None:
		from scipy.signal import windows
		if window.lower()=='blackmanharris':
				input_array *= windows.blackmanharris(input_array.shape[-1])[None,None,:]
		elif window.lower()=='tukey':
				input_array *= windows.tukey(input_array.shape[-1])[None,None,:]
		else:
				input_array *= window

	if np.array(kbins).size==1: 
		kbins = [kbins, kbins]
	if not isinstance(kbins[0], int): 
		binning = None

	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	kdict = {}
	kdict['0'], kdict['1'], kdict['2'] = kx, ky, kz
	del kx, ky, kz
	kz = kdict[str(nu_axis)]
	kp = np.sqrt(kdict[str(np.setdiff1d([0,1,2],nu_axis)[0])]**2+kdict[str(np.setdiff1d([0,1,2],nu_axis)[1])]**2)
	kz = np.abs(kz)
	# print(np.abs(kp[kp!=0]).min(),np.abs(kz[kz!=0]).min())
	if binning is None:
		kper = np.array(kbins[0])
		kpar = np.array(kbins[1])
		kbins = [len(kper),len(kpar)]
	elif binning=='log':
		kper = np.linspace(np.log10(np.abs(kp[kp!=0]).min()), np.log10(kp.max()), kbins[0]+1)
		kpar = np.linspace(np.log10(np.abs(kz[kz!=0]).min()), np.log10(kz.max()), kbins[1]+1)
		kp, kz  = np.log10(kp), np.log10(kz)
	elif binning=='linear':
		kper = np.linspace(np.abs(kp[kp!=0]).min(), kp.max(), kbins[0]+1)
		kpar = np.linspace(np.abs(kz[kz!=0]).min(), kz.max(), kbins[1]+1)
	k_width = kper[1]-kper[0], kpar[1]-kpar[0]
	# print(10**kper,10**kpar)
	kper = (kper[:-1]+kper[1:])/2.
	kpar = (kpar[:-1]+kpar[1:])/2.
	# print(10**kper,10**kpar)
	ps = np.zeros((kbins[0],kbins[1]))
	n_modes = np.zeros((kbins[0],kbins[1]))
	kp, kz, power = kp.flatten(), kz.flatten(), power.flatten()
	for i,a in tqdm(enumerate(kper)):
		for j,b in enumerate(kpar):
			arg = np.intersect1d(np.argwhere(np.abs(kp-a)<=k_width[0]/2.), np.argwhere(np.abs(kz-b)<=k_width[1]/2.))
			ps[i,j] = power[arg].sum()
			n_modes[i,j] = arg.size

	ps = ps/n_modes
	if binning=='log': kper, kpar = 10**kper, 10**kpar
	if return_modes: return ps, kper, kpar, n_modes
	# print(kper,kpar)
	return ps, kper, kpar
	ps = ps/n_modes
	if binning=='log': kper, kpar = 10**kper, 10**kpar
	if return_modes: return ps, kper, kpar, n_modes
	# print(kper,kpar)
	return ps, kper, kpar


def dimensional_power(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False):
	ps, ks, n_modes = power_spect_1d(input_array, kbins=kbins, binning=binning, box_dims=box_dims, return_modes=1)
	if return_modes: return ps*ks**3/2/np.pi**2, ks, n_modes
	return ps*ks**3/2/np.pi**2, ks


def power_spect_mu(input_array, kbins=10, box_dims=244/.7, return_modes=False, mubins=10, binning='log', nu_axis=2):
	if type(binning)==str: binning = [binning, binning]
	power = power_spect_nd(input_array, box_dims, verbose=0)
	[kx,ky,kz], k = _get_k(input_array, box_dims)
	kdict = {}
	kdict['0'], kdict['1'], kdict['2'] = kx, ky, kz
	del kx, ky, kz
	kpar = kdict[str(nu_axis)]
	kper = np.sqrt(kdict[str(np.setdiff1d([0,1,2],nu_axis)[0])]**2+kdict[str(np.setdiff1d([0,1,2],nu_axis)[1])]**2)
	m = np.abs(kpar/k)

	if binning[0]=='log': 
		ks = np.linspace(np.log10(np.abs(k[k!=0]).min()), np.log10(k.max()), kbins+1)
		k  = np.log10(k)
	else: ks = np.linspace(np.abs(k[k!=0]).min(), k.max(), kbins+1)
	ks = (ks[:-1]+ks[1:])/2.
	k_width = ks[1]-ks[0]

	if binning[1]=='log': 
		m1 = m[np.isfinite(m)]
		mu = np.linspace(np.log10(np.abs(m1[m1!=0]).min()), np.log10(m1.max()), mubins+1)
		m  = np.log10(m)
	else: mu = np.linspace(0,1,mubins+1); 
	mu = (mu[:-1]+mu[1:])/2.
	m_width = mu[1]-mu[0]

	ps = np.zeros((kbins, mubins))
	n_modes = np.zeros((kbins, mubins))

	k, m, power = k.flatten(), m.flatten(), power.flatten()
	for i,a in enumerate(ks):
		for j,b in enumerate(mu):
			arg = np.intersect1d(np.argwhere(np.abs(k-a)<=k_width/2.), np.argwhere(np.abs(m[np.isfinite(m)]-b)<=m_width/2.))
			ps[i,j] = power[arg].sum()
			n_modes[i,j] = arg.size


	ps = ps/n_modes
	if binning[0]=='log': ks = 10**ks
	if binning[1]=='log': mu = 10**mu
	if return_modes: return ps, ks, mu, n_modes
	return ps, ks, mu

def plot_2d_power(ps, xlabel='$k_\perp$', ylabel='$k_\parallel$', ps_label='$P(k_\perp,k_\parallel)$',
				  fig=None, ax=None, plotting_scale={'x': 'log', 'y': 'log', 'z': 'log'}, 
				  draw_wedge={'z': 9.0, 'fov_deg': 90.0, 'ls':'--', 'color': 'k'}, **kwargs):
	'''
	Plotting the 2D or cylindrical power spectrum.
	
	Parameters: 
		ps (tuple or dict): The data in the form of tuple (Pk, kper_bins, kpar_bins) or
							dictionary {'Pk': Pk, 'kper': kper_bins, 'kpar': kpar_bins}
		xlabel = '$k_\perp$' : Label to use for the x-axis. 
		ylabel = '$k_\parallel$' : Label to use for the y-axis. 
		fig = None : Provide the matplotlib figure for plotting.
		plotting_scale = {'x': 'log', 'y': 'log', 'z': 'log'} : Provide the plotting scales.
		draw_wedge={'z': 9.0, 'fov_deg': 90.0, 'ls':'--', 'color': 'k'} : If not None, then the wedge is drawn.

	Returns: 
		The matplotlib figure where the power spectrum is plotted.
	'''
	import matplotlib.pyplot as plt
	import matplotlib.colors as colors

	try: pp, kper, kpar = (ps[ke] for ke in ['Pk','kper','kpar'])
	except: pp, kper, kpar = ps
	fp = interpolate.interp2d(kper, kpar, pp.T, kind='linear')

	if (fig == None) and (ax == None): 
		fig, ax = plt.subplots(1,1,figsize=(7,5))
	elif (fig == None) and (ax != None):
		pass
	else: 
		ax = fig.axes[0] 
	
	#X, Y = kper[:-1]/2+kper[1:]/2, kpar[:-1]/2+kpar[1:]/2
	X, Y = kper, kpar

	C = fp(X,Y)
	norm = colors.LogNorm(vmin=C[np.isfinite(C)].min(), vmax=C[np.isfinite(C)].max()) if plotting_scale['z']=='log' else None 
	pcm = ax.pcolormesh(X, Y, C, norm=norm, **kwargs)
	if draw_wedge is not None:
		f_kpar = horizon_wedge_equation(draw_wedge['z'], fov_deg=draw_wedge['fov_deg'])
		ax.plot(X, f_kpar(X), ls=draw_wedge['ls'], color=draw_wedge['color'])
		ax.axis([X.min(),X.max(),Y.min(),Y.max()])

	plt.colorbar(pcm, ax=ax, label=ps_label, pad=0.01)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xscale(plotting_scale['x'])
	ax.set_yscale(plotting_scale['y'])
	#plt.show()
	return 1

def horizon_wedge_equation(z, fov_deg=90.0):
	'''
	Equation 14 in arXiv:2201.10798

	Parameters: 
		z (float) : Redshift.
		fov_deg = 90 : The field of view (FoV) of the horizon wedge equation in degrees.	
	Returns: 
		A lambda function k_parallel(k_perpendicular).
	'''
	f_kpar = lambda kper: kper*np.sin(fov_deg*np.pi/180)/(1+z)*\
							smoothing.hubble_parameter(z)/const.c*cosmo.z_to_cdist(z)
	return f_kpar

# def plot_2d_power(ps, xticks, yticks, xlabel, ylabel):
# 	import matplotlib.pyplot as plt
# 	xticks, yticks = np.round(xticks, decimals=2), np.round(yticks, decimals=2)
# 	plt.imshow(ps, origin='lower')
# 	locs, labels = plt.yticks()
# 	new_labels = yticks[locs.astype(int)[1:-1]]
# 	plt.yticks(locs[1:-1], new_labels)
# 	plt.ylabel(ylabel)
# 	locs, labels = plt.xticks()
# 	new_labels = xticks[locs.astype(int)[1:-1]]
# 	plt.xticks(locs[1:-1], new_labels)
# 	plt.xlabel(xlabel)
# 	plt.colorbar()
# 	plt.show()



