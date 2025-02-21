'''
Contains functions to estimate various three point statistics.
'''

import numpy as np, gc
from time import time, sleep
from tqdm import tqdm
from scipy import fftpack, stats
from joblib import Parallel, delayed

from . import const
from . import conv
from .helper_functions import print_msg, get_eval
from .scipy_func import numpy_product
from .power_spectrum import _get_dims, _get_k, _get_kbins, apply_window

def fft_nd(input_array, box_dims=None, verbose=False):
    ''' 
    Calculate the power spectrum of input_array and return it as an n-dimensional array.
    
    Parameters:
            input_array (numpy array): the array to calculate the 
                    power spectrum of. Can be of any dimensions.
            box_dims = None (float or array-like): the dimensions of the 
                    box in Mpc. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
    
    Returns:
            The Fourier transform in the same dimensions as the input array.           
    '''
    box_dims  = _get_dims(box_dims, input_array.shape)
    k_comp, k = _get_k(input_array, box_dims)
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
    #ft = np.abs(ft)
    return ft, k_comp, k

class Bispectrum:
    def __init__(self, input_array, box_dims=None, verbose=True):
        box_dims, box_dims_input  = _get_dims(box_dims, input_array.shape), box_dims
        if verbose and box_dims_input is None: print(f'box_dims set to {box_dims} Mpc')
        self.input_array = input_array
        self.box_dims = box_dims
        self.verbose = verbose
         
    def spherical_shell_mask(self, array_shape, k_center, k_width, k_comp=None, k_mag=None):
        assert k_comp is not None or k_mag is not None
        if k_mag is None: k_mag = np.sum(np.array(k_comp)**2, axis=0)
        try: out = np.zeros(array_shape)
        except: out = np.zeros((array_shape[0],array_shape[1],array_shape[2]))
        out[np.abs(k_mag-k_center)<k_width/2] = 1
        return out

    def cosine_rule(self, k1, k2, alpha):
        k3 = np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(alpha))
        return k3

    def bispectrum_fast(self, input_array_fft,
                        k1, k2, k3,
                        dk1, dk2, dk3,
                        box_dims=None,
                        k_mag=None,
                        return_n_modes=False,
                        binning='log',
                    ):
        box_vol = numpy_product(box_dims)
        n_pixel = (numpy_product(input_array_fft.shape)).astype(int)
        
        shell1 = self.spherical_shell_mask(input_array_fft.shape, k1, dk1, k_mag=k_mag)
        dfft1  = input_array_fft*shell1
        s1     = np.fft.ifftn(np.fft.fftshift(shell1))
        d1     = np.fft.ifftn(np.fft.fftshift(dfft1))
        
        if k2==k1: 
            s2, d2 = s1, d1
        else: 
            shell2 = self.spherical_shell_mask(input_array_fft.shape, k2, dk2, k_mag=k_mag)
            dfft2  = input_array_fft*shell2
            s2     = np.fft.ifftn(np.fft.fftshift(shell2))
            d2     = np.fft.ifftn(np.fft.fftshift(dfft2))
            
        if k3==k1: 
            s3, d3 = s1, d1
        elif k3==k2: 
            s3, d3 = s2, d2
        else: 
            shell3 = self.spherical_shell_mask(input_array_fft.shape, k3, dk3, k_mag=k_mag)
            dfft3  = input_array_fft*shell3
            s3     = np.fft.ifftn(np.fft.fftshift(shell3))
            d3     = np.fft.ifftn(np.fft.fftshift(dfft3))
            
        d123 = np.real(d1*d2*d3)
        s123 = np.real(s1*s2*s3)
        b123 = np.sum(d123)/np.sum(s123) * box_vol.astype(np.float64)**2/n_pixel.astype(np.float64)**3
        
        return b123

    def bispectrum_k1k2(self,
                        k1, k2,
                        dk=0.2,
                        n_bins=10, #kbins=10,
                        box_dims=None,
                        return_n_modes=False,
                        binning='linear',
                        verbose=True,
                        window=None,
                        n_jobs=1,
                    ):
        tstart = time()
        input_array_nd = self.input_array.astype(np.float64)
        input_array = apply_window(input_array, window)

        box_dims, box_dims_input  = _get_dims(box_dims, input_array_nd.shape), box_dims
        if verbose and box_dims_input is None: print(f'box_dims set to {box_dims} Mpc')
        
        if verbose: print(f'Computing bispectrum with k1,k2={k1:.2f},{k2:.2f} 1/Mpc...')
        input_array_fft, k_comp, k_mag = fft_nd(input_array_nd, box_dims=box_dims, verbose=verbose)
        if verbose: print('FFT of data done')
            
        if binning=='linear': alphas  = np.linspace(0, np.pi, n_bins)
        else: alphas  = 10**np.linspace(-2, np.log10(np.pi), n_bins)
        k3_list = self.cosine_rule(k1, k2, alphas) #np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(alphas))
        
        def run_loop(i):
            k3 = k3_list[i]
            b123 = self.bispectrum_fast(input_array_fft,
                            k1, k2, k3,
                            dk, dk, dk,
                            k_mag=k_mag,
                            box_dims=box_dims,
                            return_n_modes=return_n_modes,
                            binning=binning,
                        )
            # if verbose: print(f'{i+1}/{theta_bins} | k3, dk3 = {k3:.5f}, {dk3:.5f}')
            return b123
        
        if n_jobs in [0,1]: 
            B_list = np.array([run_loop(i) for i in tqdm(range(n_bins))])   
        else:
            B_list = np.array(Parallel(n_jobs=n_jobs)(delayed(run_loop)(i) for i in tqdm(range(n_bins))))
        
        out_dict = {
            'B': B_list,
            'alpha': alphas, 
            'theta': np.pi - alphas,
            'k1': k1,
            'k2': k2,
            'k3': k3_list,
            }
        
        if verbose: print(f'...done | Runtime: {time()-tstart:.3f} s')
        return out_dict


    def bispectrum_k(self,
                    dk=0.2,
                    n_bins=10, #kbins=10,
                    box_dims=None,
                    return_n_modes=False,
                    binning='log',
                    verbose=True,
                    window=None,
                    n_jobs=1,
                    ):
        tstart = time()
        input_array_nd = self.input_array.astype(np.float64)
        input_array = apply_window(input_array, window)

        box_dims, box_dims_input  = _get_dims(box_dims, input_array_nd.shape), box_dims
        if verbose and box_dims_input is None: print(f'box_dims set to {box_dims} Mpc')
        
        if verbose: print(f'Computing bispectrum with k1=k2=k3...')
        input_array_fft, k_comp, k_mag = fft_nd(input_array_nd, box_dims=box_dims, verbose=verbose)
        if verbose: print('FFT of data done')

        kbins = _get_kbins(n_bins, box_dims, k_mag, binning=binning, kmax=k_mag.max()/2, kmin=None)
        dk = dk*np.ones_like(kbins) if isinstance(dk,float) else dk
        
        def run_loop(i):
            k1 = kbins[i]
            dk1 = dk[i]
            b123 = self.bispectrum_fast(input_array_fft,
                            k1, k1, k1,
                            dk1, dk1, dk1,
                            k_mag=k_mag,
                            box_dims=box_dims,
                            return_n_modes=return_n_modes,
                            binning=binning,
                        )
            # if verbose: print(f'{i+1}/{theta_bins} | k3, dk3 = {k3:.5f}, {dk3:.5f}')
            return b123
        
        if n_jobs in [0,1]:
            B_list = np.array([run_loop(i) for i in tqdm(range(n_bins))])       
        else:
            B_list = np.array(Parallel(n_jobs=n_jobs)(delayed(run_loop)(i) for i in tqdm(range(n_bins))))
        
        out_dict = {
            'B': B_list,
            'k': kbins[1:]/2+kbins[:-1]/2, 
            }
        
        if verbose: print(f'...done | Runtime: {time()-tstart:.3f} s')
        return out_dict
    

class BispectrumPylians:
    def __init__(self, input_array, box_dims=None, verbose=True, n_jobs=1, MAS='CIC'):
        try:
            import Pk_library as PKL
            self.PKL = PKL
        except:
            print('Install the Pylians package to use this class:')
            print('https://pylians3.readthedocs.io/en/master/installation.html')

        box_dims, box_dims_input  = _get_dims(box_dims, input_array.shape)[0], box_dims
        if verbose and box_dims_input is None: print(f'box_dims set to {box_dims} Mpc')
        
        self.input_array = input_array
        self.box_dims = box_dims
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.MAS = MAS

    def bispectrum_k1k2(self, k1, k2, n_bins=15):
        threads = self.n_jobs
        MAS = self.MAS
        BoxSize = self.box_dims #Size of the density field in Mpc
        if isinstance(n_bins,(int,float)):
            theta = np.linspace(0, np.pi, n_bins) #array with the angles between k1 and k2
        else:
            theta = n_bins 
        if self.verbose:
            print(f'k1, k2 = {k1:.2f}/Mpc, {k2:.2f}/Mpc')
            print(f'\\theta bins = {theta}')
        delta = self.input_array
        
        BBk = self.PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)
        Bk  = BBk.B     #bispectrum
        Qk  = BBk.Q     #reduced bispectrum
        k3  = BBk.k     #k-bins for power spectrum
        Pk  = BBk.Pk    #power spectrum
        return {
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'theta': theta,
            'Bk': Bk,
            'Qk': Qk,
            'Pk': Pk,
            }

def bispectrum_k1k2(input_array, k1, k2, n_bins=15, box_dims=None, window=None, n_jobs=1, **kwargs):
    """
    Calculate the bispectrum for given k1 and k2 values.

    Parameters:
    -----------
    input_array : numpy.ndarray
        The input array representing the data.
    k1 : float
        The first wavenumber.
    k2 : float
        The second wavenumber.
    n_bins : int, optional
        The number of bins for the bispectrum calculation. Default is 15.
    box_dims : tuple, optional
        The dimensions of the box. Default is None.
    window : numpy.ndarray, optional
        The window function to apply to the input array. Default is None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    --------
    outp : dict
        A dictionary containing the bispectrum results.
    """
    input_array = apply_window(input_array, window).astype(np.float32)
    bisp = BispectrumPylians(input_array, box_dims=box_dims, verbose=kwargs.get('verbose', True), n_jobs=n_jobs)
    outp = bisp.bispectrum_k1k2(k1, k2, n_bins=n_bins)
    return outp

def bispectrum_equilateral(input_array, n_bins=15, box_dims=None, window=None, n_jobs=1, **kwargs):
    """
    Calculate the equilateral bispectrum.

    Parameters:
    -----------
    input_array : numpy.ndarray
        The input array representing the data.
    n_bins : int, optional
        The number of bins for the bispectrum calculation. Default is 15.
    box_dims : tuple, optional
        The dimensions of the box. Default is None.
    window : numpy.ndarray, optional
        The window function to apply to the input array. Default is None.
    n_jobs : int, optional
        The number of jobs to run in parallel. Default is 1.
    **kwargs : dict
        Additional keyword arguments.

    Returns:
    --------
    outp : dict
        A dictionary containing the equilateral bispectrum results.
    """
    tstart = time()
    input_array = apply_window(input_array, window).astype(np.float32)
    binning = kwargs.get('binning', 'log')
    kmin = 2 * np.pi / box_dims
    kmax = kmin * np.min(input_array.shape) / 2
    if binning.lower() in ['linear', 'lin']:
        kbins = np.linspace(kmin, kmax, n_bins + 1)
        kbins = (kbins[1:] + kbins[:-1]) / 2
    elif binning.lower() in ['logarithmic', 'log']:
        kbins = np.linspace(np.log10(kmin), np.log10(kmax), n_bins + 1)
        kbins = 10 ** ((kbins[1:] + kbins[:-1]) / 2)
    verbose = kwargs.get('verbose', True)
    bisp = BispectrumPylians(input_array, box_dims=box_dims, verbose=False, n_jobs=n_jobs)
    outp = {ke: np.array([]) for ke in ['k1', 'k2', 'k3', 'theta', 'Bk', 'Qk', 'Pk']}
    for ii, ki in enumerate(kbins):
        if verbose:
            print(f'{ii + 1}/{len(kbins)} | k={ki:.2f}')
        outi = bisp.bispectrum_k1k2(ki, ki, n_bins=np.array([np.pi / 3]))
        for ke in ['k1', 'k2', 'k3', 'theta', 'Bk', 'Qk', 'Pk']:
            outp[ke] = np.append(outp[ke], outi[ke])
    if verbose:
        print(f'Total Runtime: {time() - tstart:.3f} s')
    return outp

