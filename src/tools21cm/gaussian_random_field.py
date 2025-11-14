'''
Created on Apr 23, 2015

@author: Hannes Jensen (Original Author), Sambit K. Giri (Current Maintainer)
'''

import numpy as np
from scipy import fftpack
from .power_spectrum import _get_dims, _get_k, power_spectrum_1d
from .scipy_func import *

def make_gaussian_random_field(dims, box_dims, power_spectrum, random_seed=None):
    '''
    Generate a Gaussian random field with the specified
    power spectrum.
    
    Parameters:
        * dims (tuple): the dimensions of the field in number
            of cells. Can be 2D or 3D.
        * box_dims (float or tuple): the dimensions of the field
            in cMpc.
        * power_spectrum (callable, one parameter): the desired 
            spherically-averaged power spectrum of the output.
            Given as a function of k
        * random_seed (int): the seed for the random number generation
            
    Returns:
        The Gaussian random field as a numpy array
    '''
    #Verify input
    assert len(dims) == 2 or len(dims) == 3
    
    #Generate FT map
    if random_seed != None:
        np.random.seed(random_seed)
    map_ft_real = np.random.normal(loc=0., scale=1., size=dims)
    map_ft_imag = np.random.normal(loc=0., scale=1., size=dims)
    map_ft = map_ft_real + 1j*map_ft_imag
    
    #Get k modes
    box_dims = _get_dims(box_dims, map_ft_real.shape)
    assert len(box_dims) == len(dims)
    k_comp, k = _get_k(map_ft_real, box_dims)
    k[k==0] = np.abs(k)[k!=0].min()/10
    
    #Scale factor
    # Updated for python3: map() no longer returns a list, but an iterable
    # instead, which breaks the code. 200601/GM
    #boxvol = np.product(map(float,box_dims))
    boxvol = numpy_product([float(i) for i in box_dims])
    pixelsize = boxvol/(numpy_product(map_ft_real.shape))
    scale_factor = pixelsize**2/boxvol
    
    #Scale to power spectrum
    map_ft *= np.sqrt(power_spectrum(k)/scale_factor)
    
    #Inverse FT
    map_ift = fftpack.ifftn(fftpack.fftshift(map_ft))
    
    #Return real part
    map_real = np.real(map_ift)
    return map_real


def make_gaussian_random_field_like_field(input_field, box_dims, kbins=10, random_seed=None):
    '''
    Generate a Gaussian random field with the same power spectrum as the
    input field.
    
    Parameters: 
        * input_field (numpy array): The field to take the power spectrum
            from
        * box_dims (float or tuple): The dimensions of the input_field in cMpc
        * kbins (int): The number of bins in k-space
        * random_seed (int): the seed for the random number generation
        
    Returns:
        A Gaussian random field with the same dimensions and power spectrum
        as the input field
        
    .. note::
        This function is experimental. There are often interpolation issues
        at low k. Use with care.
    '''
    
    ps_k = _get_ps_func_for_field(input_field, box_dims, kbins=kbins)
    random_field = make_gaussian_random_field(input_field.shape, box_dims, \
                                        power_spectrum=ps_k, random_seed=random_seed)
    return random_field
    
    
def _get_ps_func_for_field(input_field, box_dims, kbins=10):
    '''
    Return ps(k) for the specified field. For internal use.
    '''
    ps_input, k_input, n_modes = power_spectrum_1d(input_field, \
                            box_dims=box_dims, kbins=kbins, return_n_modes=True)
    ps_k = interp1d(k_input[n_modes>0], ps_input[n_modes>0], kind='linear', \
                    bounds_error=False, fill_value=0.)
    # tckp = splrep(np.log10(k_input[n_modes>0]), np.log10(ps_input[n_modes>0]), k=1)
    # ps_k = lambda k: 10**splev(np.log10(k), tckp)
    return ps_k
        
def fourier_phase_shuffled(input_array, **kwargs):
    ''' 
    Reconstruct the input_array by removing the amplitude information.
    
    Parameters:
            input_array (numpy array): the array to calculate the 
                    power spectrum of. Can be of any dimensions.
    
    Returns:
            The amplitude and phase fields.           
    '''
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
    ft_abs = np.abs(ft)
    ft_phi = np.angle(ft)
    np.random.shuffle(ft_phi)
    ft_new = ft_abs*np.exp(1j * ft_phi)
    output_array = np.real(fftpack.ifftn(fftpack.fftshift(ft_new)))
    return output_array