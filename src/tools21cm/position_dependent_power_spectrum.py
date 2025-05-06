import numpy as np
from tqdm import tqdm 
from .power_spectrum import *

def power_spectrum_response(cube, cube2=None, Ncuts=4, kbins=15, box_dims=244/.7, binning='log', verbose=True):
	"""
	Calculate the response of a field to large-scale fluctuations (Giri et al. 2019, arXiv:1811.09633).

	Parameters:
	- cube1 (numpy.ndarray): The first input 3D data cube.
	- cube2 (numpy.ndarray): The second input 3D data cube. Default: None, which is replaced with the first cube.
	- Ncuts (int): Number of cuts along each dimension for the box division. Default: 4.
	- kbins (int): Number of bins for the power spectrum calculation. Default: 15.
	- box_dims (float): Size of the box in comoving Mpc. Default: 244/h.
	- binning (str): Binning method for the power spectrum ('log' or 'linear'). Default: 'log'.
	- verbose (bool): If True, display progress. Default: True.

	Returns:
	- numpy.ndarray: Integrated bispectrum cross-power spectrum.
	- numpy.ndarray: Wavenumbers corresponding to the power spectrum.
	"""
	return integrated_bispectrum_cross(cube, cube if cube2 is None else cube2, Ncuts=Ncuts, kbins=kbins, box_dims=box_dims, binning=binning, normalize=True, verbose=verbose)


def integrated_bispectrum(cube, Ncuts=4, kbins=15, box_dims=244/.7, binning='log', normalize=False, verbose=True):
	"""
	Calculate the integrated bispectrum auto-power spectrum (Giri et al. 2019, arXiv:1811.09633).

	Parameters:
	- cube1 (numpy.ndarray): The first input 3D data cube.
	- cube2 (numpy.ndarray): The second input 3D data cube. Default: None, which is replaced with the first cube.
	- Ncuts (int): Number of cuts along each dimension for the box division. Default: 4.
	- kbins (int): Number of bins for the power spectrum calculation. Default: 15.
	- box_dims (float): Size of the box in comoving Mpc. Default: 244/h.
	- binning (str): Binning method for the power spectrum ('log' or 'linear'). Default: 'log'.
	- normalize (bool): If True, normalize the result. Default: False.
	- verbose (bool): If True, display progress. Default: True.

	Returns:
	- numpy.ndarray: Integrated bispectrum cross-power spectrum.
	- numpy.ndarray: Wavenumbers corresponding to the power spectrum.
	"""
	return integrated_bispectrum_cross(cube, cube, Ncuts=Ncuts, kbins=kbins, box_dims=box_dims, binning=binning, normalize=normalize, verbose=verbose)

def integrated_bispectrum_cross(cube1, cube2, Ncuts=4, kbins=15, box_dims=244/.7, binning='log', normalize=False, verbose=True):
	"""
	Calculate the integrated bispectrum cross-power spectrum (Giri et al. 2019, arXiv:1811.09633).

	Parameters:
	- cube1 (numpy.ndarray): The first input 3D data cube.
	- cube2 (numpy.ndarray): The second input 3D data cube.
	- Ncuts (int): Number of cuts along each dimension for the box division.
	- kbins (int): Number of bins for the power spectrum calculation.
	- box_dims (float): Size of the box in comoving Mpc.
	- binning (str): Binning method for the power spectrum ('log' or 'linear').
	- normalize (bool): If True, normalize the result.
	- verbose (bool): If True, display progress.

	Returns:
	- numpy.ndarray: Integrated bispectrum cross-power spectrum.
	- numpy.ndarray: Wavenumbers corresponding to the power spectrum.
	"""
	assert cube1.shape == cube2.shape
	assert cube1.shape[0]%Ncuts==0 and cube1.shape[1]%Ncuts==0 and cube1.shape[2]%Ncuts==0
	Lx,Ly,Lz = cube1.shape[0]/Ncuts,cube1.shape[1]/Ncuts,cube1.shape[2]/Ncuts
	rLs = [[Lx/2.+i*Lx,Ly/2.+j*Ly,Lz/2.+k*Lz] for i in range(Ncuts) for j in range(Ncuts) for k in range(Ncuts)]
	B_k   = np.zeros(kbins, dtype=np.float64)
	P_k   = np.zeros(kbins, dtype=np.float64)
	sig2  = 0
	n_box = Ncuts**3
	V_L   = (Lx*Ly*Lz)
	for i in tqdm(range(n_box), disable=not verbose):
		w1 = _W_L(cube1, rLs[i], [Lx,Ly,Lz])
		w2 = _W_L(cube2, rLs[i], [Lx,Ly,Lz])
		c1 = cube1 * w1
		c2 = cube2 * w2
		pk, ks = power_spectrum_1d(c1, kbins=kbins, box_dims=box_dims, binning=binning)
		d_mean = c2.sum(dtype=np.float64)/V_L
		B_k   += pk*d_mean
		P_k   += pk
		sig2  += (d_mean)**2   #c2.var(dtype=np.float64)
		# if verbose: print("%.2f %%"%(100*(i+1)/n_box))
	B_k  = B_k/n_box
	P_k  = P_k/n_box
	sig2 = sig2/n_box
	if verbose: print('The long wavelength mode is %.3f/cMpc'%(2*np.pi/(box_dims/Ncuts)))
	if normalize: return B_k/P_k/sig2, ks
	return B_k, ks

def _W_L(array, rL, L):
	'''
	Cubical heaviside filter.
	'''
	assert array.ndim == np.array(rL).size
	out = np.zeros(array.shape)
	if np.array(L).size==1: L = [L for i in range(array.ndim)]
	xl = [int(rL[0]-L[0]/2),int(rL[0]+L[0]/2)]
	yl = [int(rL[1]-L[1]/2),int(rL[1]+L[1]/2)]
	zl = [int(rL[2]-L[2]/2),int(rL[2]+L[2]/2)]
	out[xl[0]:xl[1], yl[0]:yl[1], zl[0]:zl[1]] = 1
	return out

def integrated_bispectrum_cross_slide(cube1, cube2, L_subbox=400, kbins=15, box_dims=244/.7, binning='log', normalize=False, verbose=True, slide_overlap=0.5):
	"""
	Calculate the integrated bispectrum cross-power spectrum (Giri et al. 2019, arXiv:1811.09633).

	Parameters:
	- cube1 (numpy.ndarray): The first input 3D data cube.
	- cube2 (numpy.ndarray): The second input 3D data cube.
	- L_subbox (float): Box length of the sub-boxes (in number of cells/grids).
	- kbins (int): Number of bins for the power spectrum calculation.
	- box_dims (float): Size of the box in comoving Mpc.
	- binning (str): Binning method for the power spectrum ('log' or 'linear').
	- normalize (bool): If True, normalize the result.
	- verbose (bool): If True, display progress.

	Returns:
	- numpy.ndarray: Integrated bispectrum cross-power spectrum.
	- numpy.ndarray: Wavenumbers corresponding to the power spectrum.
	"""
	print('IMPLEMENTATION NOT OVER.')
	assert cube1.shape == cube2.shape
	L = cube1.shape[0]
	Lx,Ly,Lz = L_subbox, L_subbox, L_subbox
	L_slide  = slide_overlap*L_subbox
	sn = int(np.round((L-L_subbox)/(L_subbox-L_slide)+1))
	L_slide = int(np.round(L_subbox-(L-L_subbox)/(sn-1)))
	rLs = [[L_subbox*0.5+(L_subbox-L_slide)*i,L_subbox*0.5+(L_subbox-L_slide)*j,L_subbox*0.5+(L_subbox-L_slide)*k] for i in range(sn) for j in range(sn) for k in range(sn)]
	B_k   = np.zeros(kbins, dtype=np.float64)
	P_k   = np.zeros(kbins, dtype=np.float64)
	sig2  = 0
	n_box = len(rLs)
	V_L   = (Lx*Ly*Lz)
	for i in tqdm(range(n_box), disable=not verbose):
		w1 = _W_L(cube1, rLs[i], [Lx,Ly,Lz])
		w2 = _W_L(cube2, rLs[i], [Lx,Ly,Lz])
		c1 = cube1 * w1
		c2 = cube2 * w2
		pk, ks = power_spectrum_1d(c1, kbins=kbins, box_dims=box_dims, binning=binning)
		d_mean = c2.sum(dtype=np.float64)/V_L
		B_k   += pk*d_mean
		P_k   += pk
		sig2  += (d_mean)**2   #c2.var(dtype=np.float64)
		# if verbose: print("%.2f %%"%(100*(i+1)/n_box))
	B_k  = B_k/n_box
	P_k  = P_k/n_box
	sig2 = sig2/n_box
	if verbose: print('The long wavelength mode is %.3f/cMpc'%(2*np.pi/(box_dims*L_subbox/Lx)))
	if normalize: return B_k/P_k/sig2, ks
	return B_k, ks