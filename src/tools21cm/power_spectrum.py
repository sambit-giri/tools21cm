'''
Contains functions to estimate various two point statistics.
'''

import numpy as np, gc
from . import const
from . import conv
from .helper_functions import print_msg, get_eval
from .scipy_func import numpy_product
from scipy import fftpack, stats

def apply_window(input_array, window):
    if window is None: return input_array
    from scipy.signal import windows
    if window.lower()=='blackmanharris':
            input_array *= windows.blackmanharris(input_array.shape[-1])[None,None,:]
    elif window.lower()=='tukey':
            input_array *= windows.tukey(input_array.shape[-1])[None,None,:]
    else:
            input_array *= window
    return input_array


def power_spectrum_nd(input_array, box_dims=None, verbose=False, **kwargs):
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
                The power spectrum in the same dimensions as the input array.           
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)

        if(verbose): print( 'Calculating power spectrum...')
        ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
        power_spectrum = np.abs(ft)**2
        if(verbose): print( '...done')

        # scale
        boxvol = numpy_product(box_dims)
        pixelsize = boxvol/(numpy_product(input_array.shape))
        power_spectrum *= pixelsize**2/boxvol
        
        return power_spectrum


def cross_power_spectrum_nd(input_array1, input_array2, box_dims, **kwargs):
        ''' 
        Calculate the cross power spectrum two arrays and return it as an n-dimensional array.
        
        Parameters:
                input_array1 (numpy array): the first array to calculate the 
                        power spectrum of. Can be of any dimensions.
                input_array2 (numpy array): the second array. Must have same 
                        dimensions as input_array1.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
        
        Returns:
                The cross power spectrum in the same dimensions as the input arrays.
                
        TODO:
                Also return k values.
        '''

        assert(input_array1.shape == input_array2.shape)

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1.shape)

        print_msg( 'Calculating power spectrum...')
        ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
        ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
        power_spectrum = np.real(ft1)*np.real(ft2)+np.imag(ft1)*np.imag(ft2)
        print_msg( '...done')

        # scale
        boxvol = numpy_product(box_dims)
        pixelsize = boxvol/(numpy_product(input_array1.shape))
        power_spectrum *= pixelsize**2/boxvol

        return power_spectrum


def radial_average(input_array, box_dims, kbins=10, binning='log', breakpoint=0.1, **kwargs):
        '''
        Radially average data. Mostly for internal use.
        
        Parameters: 
                input_array (numpy array): the data array
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                        
        Returns:
                A tuple with (data, bins, n_modes), where data is an array with the 
                averaged data, bins is an array with the bin centers and n_modes is the 
                number of modes in each bin

        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        k_comp, k = _get_k(input_array, box_dims)
        kbins = _get_kbins(kbins, box_dims, k, binning=binning, breakpoint=breakpoint)
        
        #Bin the data
        print_msg('Binning data...')
        dk = (kbins[1:]-kbins[:-1])/2.
        #Total power in each bin
        outdata = np.histogram(k.flatten(), bins=kbins,
                                                weights = input_array.flatten())[0]
        #Number of modes in each bin
        n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
        outdata /= n_modes
        
        return outdata, kbins[:-1]+dk, n_modes
        

def power_spectrum_1d(input_array_nd, kbins=100, box_dims=None, return_n_modes=False, binning='log', breakpoint=0.1, window=None, **kwargs):
        ''' Calculate the spherically averaged power spectrum of an array 
        and return it as a one-dimensional array.
        
        Parameters: 
                input_array_nd (numpy array): the data array
                kbins = 100 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
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
                A tuple with (Pk, bins), where Pk is an array with the 
                power spectrum and bins is an array with the k bin centers.
        '''

        input_array_nd = apply_window(input_array_nd, window)

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array_nd.shape)

        input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)      

        ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims=box_dims, binning=binning, breakpoint=breakpoint)
        if return_n_modes:
                return ps, bins, n_modes
        return ps, bins


def power_spectrum_2d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False, nu_axis=2, window=None, **kwargs):
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
        input_array = apply_window(input_array, window)
        
        if type(kbins) == list:
                binning = None
        elif np.array(kbins).size==1: 
                kbins = [kbins, kbins]
        elif not isinstance(kbins[0], int): 
                binning = None

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)
        power = power_spectrum_nd(input_array, box_dims)
        k_xyz, k = _get_k(input_array, box_dims)
        xy_axis = [0, 1, 2]
        xy_axis.remove(nu_axis)
        kz = np.abs(k_xyz[nu_axis])
        kp = np.sqrt(k_xyz[xy_axis[0]]**2 + k_xyz[xy_axis[1]]**2)
        del k_xyz, k, xy_axis
        gc.collect()

        if binning is None:
                kper = np.array(kbins[0])
                kpar = np.array(kbins[1])
        else:
                if binning=='log':
                        kper = np.linspace(np.log10(kp[kp!=0].min()), np.log10(kp.max()), kbins[0]+1)
                        kpar = np.linspace(np.log10(kz[kz!=0].min()), np.log10(kz.max()), kbins[1]+1)
                        kp, kz  = np.log10(kp), np.log10(kz)
                elif binning=='linear':
                        kper = np.linspace(kp[kp!=0].min(), kp.max(), kbins[0]+1)
                        kpar = np.linspace(kz[kz!=0].min(), kz.max(), kbins[1]+1)
        
        kp, kz, power = kp.flatten(), kz.flatten(), power.flatten()
        ps = stats.binned_statistic_2d(x=kp, y=kz, values=power, statistic='mean', bins=[kper, kpar])

        if binning=='log':
                kper_mid = np.power(10, 0.5*(kper[:-1]+kper[1:]))
                kpar_mid = np.power(10, 0.5*(kpar[:-1]+kpar[1:]))
        else:
                kper_mid = (kper[:-1]+kper[1:])/2.
                kpar_mid = (kpar[:-1]+kpar[1:])/2.

        if return_modes: 
                n_modes = stats.binned_statistic_2d(x=kp, y=kz, values=None, statistic='count', bins=[kper, kpar])
                return ps.statistic, kper_mid, kpar_mid, n_modes.statistic
        else:
                return ps.statistic, kper_mid, kpar_mid


def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, kbins=100, box_dims=None, return_n_modes=False, binning='log', breakpoint=0.1, **kwargs):
        ''' Calculate the spherically averaged cross power spectrum of two arrays 
        and return it as a one-dimensional array.
        
        Parameters: 
                input_array1_nd (numpy array): the first data array
                input_array2_nd (numpy array): the second data array
                kbins = 100 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                return_n_modes = False (bool): if true, also return the
                        number of modes in each bin
                binning = 'log' : It defines the type of binning in k-space. The other option is 
                                    'linear' or 'mixed'.
                        
        Returns: 
                A tuple with (Pk, bins), where Pk is an array with the 
                cross power spectrum and bins is an array with the k bin centers.
        '''
        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1_nd.shape)

        input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd, box_dims=box_dims)      

        ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims = box_dims, binning=binning, breakpoint=breakpoint)
        if return_n_modes:
                return ps, bins, n_modes
        return ps, bins


def power_spectrum_mu(input_array, los_axis=0, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, **kwargs):
        '''
        Calculate the power spectrum and bin it in mu=cos(theta) and k.
        
        Parameters: 
                input_array (numpy array): the data array
                los_axis = 0 (integer): the line-of-sight axis
                mubins = 20 (integer): the number of mu bins
                kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                return_n_modes = False (bool): if true, also return the
                        number of modes in each bin
                exlude_zero_modes = True (bool): if true, modes with any components
                        of k equal to zero will be excluded.
                absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1] 
                        
        Returns: 
                A tuple with (Pk, mubins, kbins), where Pk is an array with the 
                power spectrum of dimensions (n_mubins x n_kbins), 
                mubins is an array with the mu bin centers and
                kbins is an array with the k bin centers.
        
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)

        #Calculate the power spectrum
        powerspectrum = power_spectrum_nd(input_array, box_dims=box_dims)       

        ps, mu_bins, k_bins, n_modes = mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes, absolute_mus=absolute_mus)

        if return_n_modes:
                return ps, mu_bins, k_bins, n_modes
        return ps, mu_bins, k_bins


def cross_power_spectrum_mu(input_array1, input_array2, los_axis=0, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, **kwargs):
        '''
        Calculate the cross power spectrum and bin it in mu=cos(theta) and k.
        
        Parameters: 
                input_array1 (numpy array): the first data array
                input_array2 (numpy array): the second data array
                los_axis = 0 (integer): the line-of-sight axis
                mubins = 20 (integer): the number of mu bins
                kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                return_n_modes = False (bool): if true, also return the
                        number of modes in each bin
                exlude_zero_modes = True (bool): if true, modes with any components
                        of k equal to zero will be excluded.
                absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1] 
                
        Returns: 
                A tuple with (Pk, mubins, kbins), where Pk is an array with the 
                cross power spectrum of dimensions (n_mubins x n_kbins), 
                mubins is an array with the mu bin centers and
                kbins is an array with the k bin centers.
                
        TODO:
                Add support for (non-numpy) lists for the bins
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1.shape)
        
        #Calculate the power spectrum
        powerspectrum = cross_power_spectrum_nd(input_array1, input_array2, box_dims=box_dims)  
        
        ps, mu_bins, k_bins, n_modes = mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes, absolute_mus=absolute_mus)
        if return_n_modes:
                return ps, mu_bins, k_bins, n_modes
        return ps, mu_bins, k_bins


def mu_binning(powerspectrum, los_axis = 0, mubins=20, kbins=10, box_dims=None, weights=None,
                        exclude_zero_modes=True, binning='log', absolute_mus=True):
        '''
        This function is for internal use only.
        '''
        
        if weights != None:
                powerspectrum *= weights

        assert(len(powerspectrum.shape)==3)

        k_comp, k = _get_k(powerspectrum, box_dims)

        mu = _get_mu(k_comp, k, los_axis, absolute_mus)

        #Calculate k values, and make k bins
        kbins = _get_kbins(kbins, box_dims, k, binning=binning)
        dk = (kbins[1:]-kbins[:-1])/2.
        n_kbins = len(kbins)-1
                
        #Exclude k_perp = 0 modes
        if exclude_zero_modes:
                good_idx = _get_nonzero_idx(powerspectrum.shape, los_axis)
        else:
                good_idx = np.ones_like(powerspectrum)

        #Make mu bins
        min_mu=0.0 if absolute_mus else -1.0
        if isinstance(mubins,int):
                mubins = np.linspace(min_mu, 1.0 , mubins+1)
        dmu = (mubins[1:]-mubins[:-1])/2.
        n_mubins = len(mubins)-1

        #Remove the zero component from the power spectrum. mu is undefined here
        powerspectrum[tuple((np.array(powerspectrum.shape)/2).astype(int))] = 0.

        #Bin the data
        print_msg('Binning data...')
        outdata = np.zeros((n_mubins,n_kbins))
        n_modes = np.zeros((n_mubins,n_kbins))
        for ki in range(n_kbins):
                print_msg('Bin %d of %d' % (ki, n_kbins))
                kmin = kbins[ki]
                kmax = kbins[ki+1]
                kidx = (k >= kmin) & (k < kmax)
                kidx = kidx*good_idx
                for i in range(n_mubins):
                        mu_min = mubins[i]
                        mu_max = mubins[i+1]
                        idx = (mu >= mu_min) & (mu < mu_max) & kidx.astype(bool)
                        outdata[i,ki] = np.mean(powerspectrum[idx])
                        n_modes[i,ki] = np.size(powerspectrum[idx])
                        if weights != None:
                                outdata[i,ki] /= weights[idx].mean()

        return outdata, mubins[:-1]+dmu, kbins[:-1]+dk, n_modes

def get_k(input_array, box_dims):
        box_dims = _get_dims(box_dims, input_array.shape)
        return _get_k(input_array, box_dims)
        
def get_kbins(kbins, box_dims, k=None, array=None, binning='log'):
        assert k is not None or array is not None
        box_dims = _get_dims(box_dims, array.shape)
        if k is None: k_comp, k = _get_k(array, box_dims)
        return _get_kbins(kbins, box_dims, k, binning=binning)

#Some methods for internal use

def _get_k(input_array, box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
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
        nx,ny,nz = input_array.shape
        x,y,z  = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2, \
                            nz/2 if nz%2==0 else (nz-1)/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        kz = 2.*np.pi * (z-center[2])/box_dims[2]

        k = np.sqrt(kx**2 + ky**2 + kz**2 )     
        return [kx,ky,kz], k


def _get_mu(k_comp, k, los_axis, absolute_mus):
        '''
        Get the mu values for given k values and 
        a line-of-sight axis.
        For internal use
        '''
                
        #Line-of-sight distance from center 
        if los_axis == 0:
                los_dist = k_comp[0]
        elif los_axis == 1:
                los_dist = k_comp[1]
        elif los_axis == 2:
                los_dist = k_comp[2]
        else:
                raise Exception('Your space is not %d-dimensional!' % los_axis)

        #mu=cos(theta) = k_par/k
        mu = np.abs(los_dist/k) if absolute_mus else los_dist/np.abs(k)
        mu[np.where(k < 0.001)] = np.nan
        
        return mu


def _get_kbins(kbins, box_dims, k, binning='log', breakpoint=0.1, kmin=None, kmax=None):
        '''
        Make a list of bin edges if kbins is an integer,
        otherwise return it as it is.
        '''
        if isinstance(kbins,int):
                kmin = 2.*np.pi/min(box_dims) if kmin is None else kmin
                kmax = k.max() if kmax is None else kmax
                if binning=='linear': kbins = np.linspace(kmin, kmax, kbins+1)
                elif binning=='log': kbins = 10**np.linspace(np.log10(kmin), np.log10(kmax), kbins+1)
                else:
                        kbins_low  = np.linspace(kmin, kmax, kbins+1)
                        kbins_high = 10**np.linspace(np.log10(kmin), np.log10(kmax), kbins+1)
                        kbins = np.hstack((kbins_low[kbins_low<breakpoint],kbins_high[kbins_high>breakpoint]))          
        return kbins


def _get_dims(box_dims, ashape):
        '''
        If box dims is a scalar, assume that dimensions
        are cubic and make a list
        If it's not given, assume it's the default value of the box
        size
        Otherwise, return as it is
        '''
        if box_dims == None:
                return [conv.LB]*len(ashape)
        if not hasattr(box_dims, '__iter__'):
                return [box_dims]*len(ashape)
        return box_dims

def dimensionless_ps(data, kbins=100, box_dims=None, binning='log', factor=10):
        r'''
        Dimensionless power spectrum is P(k)*k^3/(2pi^2)

        Parameters
        ----------
        data    : ndarray
                The numpy data whose power spectrum is to be determined.
        kbins   : int
                Number of bins for in the k-space (Default: 100).
        box_dims: float
                The size of the box in Mpc (Default: Takes the value from the set_sim_constants).
        binning : str
                The type of binning to be used for the k-space (Default: 'log').
        factor  : int
                The factor multiplied to the given kbins to smooth the spectrum from (Default: 10).

        Returns
        -------
        (\Delta^2, ks)
        '''
        from scipy.interpolate import interp1d
        Pk, ks = power_spectrum_1d(data, kbins=kbins*factor, box_dims=box_dims, binning=binning)
        f_Dlta = interp1d(ks, Pk*ks**3/2/np.pi**2)
        knew   = 10**np.linspace(np.log10(ks[1]),np.log10(ks[-1]), kbins) if binning=='log' else np.linspace(ks[1],ks[-1], kbins)
        return f_Dlta(knew), knew

def _get_nonzero_idx(ps_shape, los_axis):
        '''
        Get the indices where k_perp != 0
        '''
        x,y,z = np.indices(ps_shape)
        if los_axis == 0:
                zero_idx = (y == ps_shape[1]/2)*(z == ps_shape[2]/2)
        elif los_axis == 1:
                zero_idx = (x == ps_shape[0]/2)*(z == ps_shape[2]/2)
        else:
                zero_idx = (x == ps_shape[0]/2)*(y == ps_shape[1]/2)
        good_idx = np.invert(zero_idx)
        return good_idx


def anisotropy_ratio_r_mu(input_array, los_axis=0, mu_cut=0.5, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, **kwargs):
    '''
    Calculate the anisotropy ratio r_mu(k).

    Parameters: 
            input_array (numpy array): the data array
            los_axis = 0 (integer): the line-of-sight axis
            mu_cut = 0.5 (float): the value of mu to split the P(k,mu) data.
            mubins = 20 (integer): the number of mu bins
            kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            box_dims = None (float or array-like): the dimensions of the 
                    box in Mpc. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            return_n_modes = False (bool): if true, also return the
                    number of modes in each bin
            exlude_zero_modes = True (bool): if true, modes with any components
                    of k equal to zero will be excluded.
            absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1] 

    Returns: 
            A tuple with (Pk, mubins, kbins), where Pk is an array with the 
            power spectrum of dimensions (n_mubins x n_kbins), 
            mubins is an array with the mu bin centers and
            kbins is an array with the k bin centers.

    '''
    Pk, mubins, kbins = power_spectrum_mu(input_array, los_axis=los_axis, mubins=mubins, kbins=kbins, box_dims=box_dims, weights=weights, exclude_zero_modes=exclude_zero_modes, return_n_modes=return_n_modes, absolute_mus=absolute_mus, **kwargs)
    Pup = Pk[mubins>=mu_cut,:]
    Pdn = Pk[mubins<mu_cut,:]
    rup = np.array([pp[np.isfinite(pp)].mean() for pp in Pup.T])
    rdn = np.array([pp[np.isfinite(pp)].mean() for pp in Pdn.T])     
    rmu = rup/rdn-1.0
    return rmu, kbins