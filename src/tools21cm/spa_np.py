import numpy as np
from . import usefuls
from scipy.signal import fftconvolve
from skimage import morphology
from tqdm import tqdm

def spa_np(data, xth=0.95, nscales=30, binning='log', verbose=True):
	"""
	@Zahn et al. (2007)
	"""
	Rmx = data.shape[0]
	if binning=='linear': Rs_ = np.linspace(1,Rmx/2.,nscales)
	else: Rs_ = np.exp(np.linspace(np.log(2.),np.log(Rmx/2.),nscales))
	ins = np.zeros(nscales)
	#nns = np.zeros(nscales)
	rad = np.zeros(data.shape)
	if verbose:
		for i in tqdm(range(nscales)):
			ra = Rs_[i]
			#kernel = put_sphere(np.zeros((Rmx,Rmx,Rmx)), [Rmx/2.,Rmx/2.,Rmx/2.], ra, label=1.)
			kernel = morphology.ball(ra)
			smooth = fftconvolve(data, kernel/kernel.sum(), mode='same')
			rad[smooth>=xth] = ra
			#print("Comepleted {0:.1f} %".format(100*(i+1)/nscales))
	else:
		for i in range(nscales):
			ra = Rs_[i]
			#kernel = put_sphere(np.zeros((Rmx,Rmx,Rmx)), [Rmx/2.,Rmx/2.,Rmx/2.], ra, label=1.)
			kernel = morphology.ball(ra)
			smooth = fftconvolve(data, kernel/kernel.sum(), mode='same')
			rad[smooth>=xth] = ra
	
	for i in range(nscales): ins[i] = rad[rad==Rs_[i]].size
	return Rs_, ins


def put_sphere(array, centre, radius, label=1, periodic=True, verbose=False):
	assert array.ndim == 3
	nx, ny, nz = array.shape
	aw  = np.argwhere(np.isfinite(array))
	RR  = ((aw[:,0]-centre[0])**2 + (aw[:,1]-centre[1])**2 + (aw[:,2]-centre[2])**2).reshape(array.shape)
	array[RR<=radius**2] = label
	if periodic: 
		RR2 = ((aw[:,0]+nx-centre[0])**2 + (aw[:,1]+ny-centre[1])**2 + (aw[:,2]+nz-centre[2])**2).reshape(array.shape)
		array[RR2<=radius**2] = label
		if verbose: print("Periodic circle of radius %d made at (%d,%d,%d)"%(radius, centre[0], centre[1], centre[2]))
	else: 
		if verbose: print("Non-periodic circle of radius %d made at (%d,%d,%d)"%(radius, centre[0], centre[1], centre[2]))
	return array

