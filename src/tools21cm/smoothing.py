'''
Methods to smooth or reduce resolution of the data to reduce noise.
'''

import numpy as np
from . import const, conv
from . import cosmo as cm
import scipy.ndimage as ndimage
import scipy.interpolate
from scipy import signal
from scipy.fftpack import fft, ifft, fftn, ifftn
from numpy.fft import rfftn, irfftn
from math import ceil, floor
from numpy import array, asarray, roll
from .helper_functions import fftconvolve, find_idx
from tqdm import tqdm

def gauss_kernel(size, sigma=1.0, fwhm=None):
        ''' 
        Generate a normalized gaussian kernel, defined as
        exp(-(x^2 + y^2)/(2sigma^2)).
        
        
        Parameters:
                size (int): Width of output array in pixels.
                sigma = 1.0 (float): The sigma parameter for the Gaussian.
                fwhm = None (float or None): The full width at half maximum.
                                If this parameter is given, it overrides sigma.
                
        Returns:
                numpy array with the Gaussian. The dimensions will be
                size x size or size x sizey depending on whether
                sizey is set. The Gaussian is normalized so that its
                integral is 1.  
        '''
        
        if fwhm != None:
                sigma = fwhm/(2.*np.sqrt(2.*np.log(2)))

        if size % 2 == 0:
                size = int(size/2)
                x,y = np.mgrid[-size:size, -size:size]
        else:
                size = int(size/2)
                x,y = np.mgrid[-size:size+1, -size:size+1]
        
        g = np.exp(-(x*x + y*y)/(2.*sigma*sigma))

        return g/g.sum()


def gauss_kernel_3d(size, sigma=1.0, fwhm=None):
        ''' 
        Generate a normalized gaussian kernel, defined as
        exp(-(x^2 + y^2 + z^2)/(2sigma^2)).
        
        
        Parameters:
                size (int): Width of output array in pixels.
                sigma = 1.0 (float): The sigma parameter for the Gaussian.
                fwhm = None (float or None): The full width at half maximum.
                                If this parameter is given, it overrides sigma.
                
        Returns:
                numpy array with the Gaussian. The dimensions will be
                size x size x size. The Gaussian is normalized so that its
                integral is 1.  
        '''
        
        if fwhm != None:
                sigma = fwhm/(2.*np.sqrt(2.*np.log(2)))

        if size % 2 == 0:
                size = int(size/2)
                x,y,z = np.mgrid[-size:size, -size:size, -size:size]
        else:
                size = int(size/2)
                x,y,z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]
        
        g = np.exp(-(x*x + y*y + z*z)/(2.*sigma*sigma))

        return g/g.sum()


def tophat_kernel(size, tophat_width):
        '''
        Generate a square tophat kernel
        
        Parameters:
                size (int): the size of the array
                tophat_width (int): the size of the tophat kernel
                
        Returns:
                The kernel as a (size,size) array
        '''
        kernel   = np.zeros((size,size))
        center   = kernel.shape[0]/2
        idx_low  = int(center-np.floor(tophat_width/2.))
        idx_high = int(center+np.ceil(tophat_width/2.))
        kernel[idx_low:idx_high, idx_low:idx_high] = 1.
        kernel /= np.sum(kernel)
        return kernel


def tophat_kernel_3d(size, tophat_width, shape="cube"):
        '''
        Generate a 3-dimensional tophat kernel with
        the specified size
        
        Parameters:
                size (integer or list-like): the size of
                        the tophat kernel along each dimension.
                tophat_width (int): the size of the tophat kernel
                shape (string): "cube": cubic tophat; "sphere": spherical tophat
        
        Returns:
                The normalized kernel
        '''
        kernel = np.zeros((size, size, size))
        if shape == "cube":
                center   = kernel.shape[0]/2 
                idx_low  = int(center-np.floor(tophat_width/2.))
                idx_high = int(center+np.ceil(tophat_width/2.))
                kernel[idx_low:idx_high, idx_low:idx_high, idx_low:idx_high ] = 1.
        else:
                if size % 2 == 0:
                        size = int(size/2)
                        x,y,z = np.mgrid[-size:size, -size:size, -size:size]
                else:
                        size = int(size/2)
                        x,y,z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]
                radius=np.sqrt(x*x+y*y+z*z)
                kernel[np.nonzero(radius <= 0.5*tophat_width)]=1.
                
        kernel /= np.sum(kernel)
        return kernel


def lanczos_kernel(size, kernel_width):
        '''
        Generate a 2D Lanczos kernel.
        
        Parameters:
                size (int): the size of the array
                kernel_width (int): the width of the kernel
                
        Returns:
                The kernel as a (size,size) array

        '''
        #x,y = np.mgrid[-size*0.5:size*0.5, -size*0.5:size*0.5]
        xi = np.linspace(-size*0.5, size*0.5, size)
        yi = np.linspace(-size*0.5, size*0.5, size)
        x, y = np.meshgrid(xi, yi)
        a = kernel_width
        kernel = np.sinc(x)*np.sinc(x/a)*np.sinc(y)*np.sinc(y/a)
        kernel[np.abs(x) > a] = 0.
        kernel[np.abs(y) > a] = 0.
        kernel /= kernel.sum()
        
        return kernel


def smooth_gauss(input_array, sigma=1.0, fwhm=None):
        ''' 
        Smooth the input array with a Gaussian kernel specified either by
        sigma (standard deviation of the Gaussian function) or FWHM (Full 
        Width Half Maximum). The latter is more appropriate when considering
        the resolution of a telescope.
        
        Parameters:
                input_array (numpy array): the array to smooth
                sigma=1.0 (float): the width of the kernel (variance)
                fwhm = None (float or None): The full width at half maximum.
                                If this parameter is given, it overrides sigma.

        Returns:
                The smoothed array. A numpy array with the same
                dimensions as the input.
        '''
        kernel = gauss_kernel(input_array.shape[0], sigma=sigma, fwhm=fwhm)
        return smooth_with_kernel(input_array, kernel)


def smooth_tophat(input_array, tophat_width):
        ''' 
        Smooth the input array with a square tophat kernel.
        
        Parameters:
                input_array (numpy array): the array to smooth
                tophat_width (int): the width of the kernel in cells

        Returns:
                The smoothed array. A numpy array with the same
                dimensions as the input.
        '''
        #For some reason fftconvolve works produces edge effects with
        #an even number of cells, so we pad the array with an extra pixel
        #if this is the case
        if input_array.shape[0] % 2 == 0:
                from .angular_coordinates import _get_padded_slice
                padded = _get_padded_slice(input_array, input_array.shape[0]+1)
                out = smooth_tophat(padded, tophat_width)
                return out[:-1,:-1]
        
        kernel = tophat_kernel(input_array.shape[0], tophat_width)
        return smooth_with_kernel(input_array, kernel)


def smooth_lanczos(input_array, kernel_width):
        ''' 
        Smooth the input array with a Lanczos kernel.
        
        Parameters:
                input_array (numpy array): the array to smooth
                kernel_width (int): the width of the kernel in cells

        Returns:
                The smoothed array. A numpy array with the same
                dimensions as the input.
        '''

        kernel = lanczos_kernel(input_array.shape[0], kernel_width)
        return smooth_with_kernel(input_array, kernel)


def smooth_with_kernel(input_array, kernel):
        ''' 
        Smooth the input array with an arbitrary kernel.
        
        Parameters:
                input_array (numpy array): the array to smooth
                kernel (numpy array): the smoothing kernel. Must
                        be the same size as the input array

        Returns:
                The smoothed array. A numpy array with the same
                dimensions as the input.
        '''
        assert len(input_array.shape) == len(kernel.shape)
        
        out = fftconvolve(input_array, kernel)
        # out = signal.fftconvolve(input_array, kernel, mode='full')
        
        return out


def get_beam_w(baseline, z):
        '''
        Calculate the width of the beam for an
        interferometer with a given maximum baseline.
        It is assumed that observations are done at
        lambda = 21*(1+z) cm
        
        Parameters:
                baseline (float): the maximum baseline in meters
                z (float): the redshift
                
        Returns:
                The beam width in arcminutes
        '''
        
        fr = const.nu0 / (1.0+z) #21 cm frequency at z
        lw = const.c/fr/1.e6*1.e3 # wavelength in m
        beam_w = lw/baseline/np.pi*180.*60.
        return beam_w


def interpolate3d(input_array, x, y, z, order=0):
        '''
        This function is a recreation of IDL's interpolate
        routine. It takes an input array, and interpolates it
        to a new size, which can be irregularly spaced.
        
        Parameters:
                input_array (numpy array): the array to interpolate
                x (numpy array): the output coordinates along the x axis
                        expressed as (fractional) indices 
                y (numpy array): the output coordinates along the y axis
                        expressed as (fractional) indices 
                z (numpy array): the output coordinates along the z axis
                        expressed as (fractional) indices
                order (int): the order of the spline interpolation. Default
                        is 0 (linear interpolation). Setting order=1 gives the same
                        behaviour as IDL's interpolate function with default parameters.

        Returns:
                Interpolated array with shape (nx, ny, nz), where nx, ny and nz
                are the lengths of the arrays x, y and z respectively.
        '''
        
        
        inds = np.zeros((3, len(x), len(y), len(z)))
        inds[0,:,:] = x[:,np.newaxis,np.newaxis]
        inds[1,:,:] = y[np.newaxis,:,np.newaxis]
        inds[2,:,:] = z[np.newaxis,np.newaxis,:]
        new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', \
                                                                        order=order)
        
        return new_array


def interpolate2d(input_array, x, y, order=0):
        '''
        Same as interpolate2d but for 2D data
        
        Parameters:
                input_array (numpy array): the array to interpolate
                x (numpy array): the output coordinates along the x axis
                        expressed as (fractional) indices 
                y (numpy array): the output coordinates along the y axis
                        expressed as (fractional) indices 
                order (int): the order of the spline interpolation. Default
                        is 0 (linear interpolation). Setting order=1 gives the same
                        results as IDL's interpolate function

        Returns:
                Interpolated array with shape (nx, ny), where nx and ny
                are the lengths of the arrays x and y respectively.
        '''

        inds = np.zeros((2, len(x), len(y)))
        inds[0,:] = x[:,np.newaxis]
        inds[1,:] = y[np.newaxis,:]
        new_array = ndimage.map_coordinates(input_array, inds, mode='wrap', \
                                                                        order=order, prefilter=True)
        
        return new_array

def smooth_lightcone(lightcone, z_array, box_size_mpc=False, max_baseline=2., ratio=1.):
        """
        This smooths in both angular and frequency direction assuming both to be smoothed by same scale.

        Parameters:
                lightcone (numpy array): The lightcone that is to be smoothed.
                z_array (float)        : The lowest value of the redshift in the lightcone or the whole redshift array.
                box_size_mpc (float)   : The box size in Mpc. Default value is determined from 
                                           the box size set for the simulation (set_sim_constants)
                max_baseline (float)   : The maximun baseline of the telescope in km. Default value 
                                           is set as 2 km (SKA core).
                ratio (int)            : It is the ratio of smoothing scale in frequency direction and 
                                           the angular direction (Default value: 1).

        Returns:
                (Smoothed_lightcone, redshifts) 
        """
        if (not box_size_mpc): box_size_mpc=conv.LB
        if(z_array.shape[0] == lightcone.shape[2]):
                input_redshifts = z_array.copy()
        else:
                z_low = z_array
                cell_size = 1.0*box_size_mpc/lightcone.shape[0]
                distances = cm.z_to_cdist(z_low) + np.arange(lightcone.shape[2])*cell_size
                input_redshifts = cm.cdist_to_z(distances)

        output_dtheta  = (1+input_redshifts)*21e-5/max_baseline
        output_ang_res = output_dtheta*cm.z_to_cdist(input_redshifts)
        output_dz      = ratio*output_ang_res/const.c
        for i in range(len(output_dz)):
                output_dz[i] = output_dz[i] * hubble_parameter(input_redshifts[i])
        output_lightcone = smooth_lightcone_tophat(lightcone, input_redshifts, output_dz)
        output_lightcone = smooth_lightcone_gauss(output_lightcone, output_ang_res*lightcone.shape[0]/box_size_mpc)
        return output_lightcone, input_redshifts

def smooth_coeval(cube, z, box_size_mpc=False, max_baseline=2., ratio=1., nu_axis=2, verbose=True):
        """
        This smooths the coeval cube by Gaussian in angular direction and by tophat along the third axis.

        Parameters:
                coeval_cube (numpy array): The data cube that is to be smoothed.
                z (float)                : The redshift of the coeval cube.
                box_size_mpc (float)     : The box size in Mpc. Default value is determined from 
                                             the box size set for the simulation (set_sim_constants)
                max_baseline (float)     : The maximun baseline of the telescope in km. Default value 
                                             is set as 2 km (SKA core).
                ratio (int)              : It is the ratio of smoothing scale in frequency direction and 
                                             the angular direction (Default value: 1).
                nu_axis (int)            : Frequency axis

        Returns:
                Smoothed_coeval_cube
        """
        if (not box_size_mpc): box_size_mpc=conv.LB     
        output_dtheta  = (1+z)*21e-5/max_baseline
        output_ang_res = output_dtheta*cm.z_to_cdist(z) * cube.shape[0]/box_size_mpc
        output_cube = smooth_coeval_tophat(cube, output_ang_res*ratio, nu_axis=nu_axis, verbose=verbose)
        output_cube = smooth_coeval_gauss(output_cube, output_ang_res, nu_axis=nu_axis)
        return output_cube

def smooth_coeval_tophat(cube, width, nu_axis, verbose=True):
        """
        This smooths the slices perpendicular to the given axis of the cube by tophat filter.

        Parameters:
                cube (numpy array)  : The data cube that is to be smoothed.
                width (float)       : The width of the tophat filter.
                nu_axis (int)       : Frequency axis

        Returns:
                Smoothed_cube
        """
        kernel = tophat_kernel(cube.shape[nu_axis], width)
        output_cube = np.zeros(cube.shape)
        if nu_axis==0:
                for i in tqdm(range(cube.shape[1]), disable=False if verbose else True):
                        output_cube[:,i,:] = smooth_with_kernel(cube[:,i,:], kernel)
        else:
                for i in tqdm(range(cube.shape[0]), disable=False if verbose else True):
                        output_cube[i,:,:] = smooth_with_kernel(cube[i,:,:], kernel)
        return output_cube

def smooth_coeval_gauss(cube, fwhm, nu_axis):
        """
        This smooths the slices parallel to the given axis of the cube by Gaussian filter.

        Parameters:
                cube (numpy array)  : The data cube that is to be smoothed.
                fwhm (float)        : The fwhm of the Gaussian filter.
                nu_axis (int)       : Frequency axis

        Returns:
                Smoothed_cube
        """
        one = np.ones(cube.shape[nu_axis])
        output_cube = smooth_lightcone_gauss(cube, fwhm*one, nu_axis=nu_axis)
        return output_cube

def smooth_lightcone_tophat(lightcone, redshifts, dz, verbose=True):
        """
        This smooths the slices perpendicular to the third axis of the lightcone by tophat filter.

        Parameters:
                lightcone (numpy array) : The lightcone that is to be smoothed.
                redshifts (numpy array) : The redshift of each slice along the third axis.
                dz (float)              : redshift width 

        Returns:
                Smoothed_lightcone
        """
        output_lightcone = np.zeros(lightcone.shape)
        for i in tqdm(range(output_lightcone.shape[2]), disable=False if verbose else True):
                z_out_low  = redshifts[i]-dz[i]/2
                z_out_high = redshifts[i]+dz[i]/2
                idx_low  = int(np.ceil(find_idx(redshifts, z_out_low)))
                idx_high = int(np.ceil(find_idx(redshifts, z_out_high)))
                output_lightcone[:,:,i] = np.mean(lightcone[:,:,idx_low:idx_high+1], axis=2)
        return output_lightcone

def smooth_lightcone_gauss(lightcone,fwhm,nu_axis=2):
        """
        This smooths the slices perpendicular to the third axis of the lightcone by tophat filter.

        Parameters:
                lightcone (numpy array) : The lightcone that is to be smoothed.
                fwhm (numpy array)      : The fwhm values of the Gaussian filter at each slice along frequency axis.
                nu_axis (int)           : frequency axis 

        Returns:
                Smoothed_lightcone
        """
        assert lightcone.shape[nu_axis] == len(fwhm)
        output_lightcone = np.zeros(lightcone.shape)
        for i in range(output_lightcone.shape[nu_axis]):
                if nu_axis==0: output_lightcone[i,:,:] = smooth_gauss(lightcone[i,:,:], fwhm=fwhm[i])
                elif nu_axis==1: output_lightcone[:,i,:] = smooth_gauss(lightcone[:,i,:], fwhm=fwhm[i])
                else: output_lightcone[:,:,i] = smooth_gauss(lightcone[:,:,i], fwhm=fwhm[i])
        return output_lightcone

def smooth_lightcone_uv_threshold(lightcone, uv_box, threshold=0.0):
        """Smoothing (de-noising) the lightcone by removing noisy UV cells below `threshold`.

        Beware, if maximal baseline is to be set, one should clean the `uv_box` beforehand.

        Parameters:
                lightcone (numpy array) : The lightcone to be smoothed.
                uv_box (numpy array)    : The UV box.
                threshold (float)       : The threshold below which UV cells are removed.

        Returns:
                Smoothed lightcone.
        """
        if not isinstance(threshold, float) or threshold < 0.0:
                return ValueError("Threshold value should be a positive float.")
        
        lightcone = np.fft.fft2(lightcone, axes = (0, 1))
        uv_bool = uv_box < threshold
        lightcone[uv_bool] = 0.0

        return np.real(np.fft.ifft2(lightcone, axes = (0, 1)))

def hubble_parameter(z):
        """
        It calculates the Hubble parameter at any redshift.
        """
        part = np.sqrt(const.Omega0*(1.+z)**3+const.lam)
        return const.H0 * part




def remove_baselines_from_uvmap(uv_map, z, max_baseline=2, box_size_mpc=False):
    if (not box_size_mpc): box_size_mpc=conv.LB  
    output_dtheta = (1+z)*21e-5/max_baseline
    output_dx_Mpc = output_dtheta*cm.z_to_cdist(z)
    output_dx_res = output_dx_Mpc * uv_map.shape[0]/box_size_mpc
    fft_dk_res_invMpc = box_size_mpc/output_dx_Mpc
    filt = np.zeros_like(uv_map)
    xx, yy = np.meshgrid(np.arange(uv_map.shape[0]), np.arange(uv_map.shape[1]), sparse=True)
    rr1 = (xx**2 + yy**2)
    rr2 = ((uv_map.shape[0]-xx)**2 + yy**2)
    rr3 = (xx**2 + (uv_map.shape[1]-yy)**2)
    rr4 = ((uv_map.shape[0]-xx)**2 + (uv_map.shape[1]-yy)**2)
    filt[rr1<=fft_dk_res_invMpc**2] = 1
    filt[rr2<=fft_dk_res_invMpc**2] = 1
    filt[rr3<=fft_dk_res_invMpc**2] = 1
    filt[rr4<=fft_dk_res_invMpc**2] = 1
    filt[0,0] = 0
    return filt*uv_map

def convolve_uvmap(array, z=None, uv_map=None, max_baseline=None, box_size_mpc=False, 
        filename=None, total_int_time=6.0, int_time=10.0, declination=-30.0, verbose=True):
    if (not box_size_mpc): box_size_mpc=conv.LB  
    if uv_map is None: 
        uv_map, N_ant  = get_uv_map(array.shape[0],
                                    z,
                                    filename=filename,
                                    total_int_time=total_int_time,
                                    int_time=int_time,
                                    boxsize=box_size_mpc,
                                    declination=declination,
                                    verbose=verbose,
                                )
    if max_baseline is not None: uv_map = remove_baselines_from_uvmap(uv_map, z, max_baseline=max_baseline, box_size_mpc=box_size_mpc)
    img_arr  = np.fft.fft2(array)
    kernel2d = uv_map #np.ones_like(uv_map); kernel2d[uv_map==0] = 0
    img_arr *= kernel2d/kernel2d.max()
    img_map  = np.fft.ifft2(img_arr)
    return np.real(img_map)



def convolve_uvmap_coeval(cube, z, box_size_mpc=False, max_baseline=2., ratio=1., nu_axis=2, verbose=True,
                filename=None, total_int_time=6.0, int_time=10.0, declination=-30.0, uv_map=None):
        """
        This smooths the coeval cube by Gaussian in angular direction and by tophat along the third axis.

        Parameters:
                coeval_cube (numpy array): The data cube that is to be smoothed.
                z (float)                : The redshift of the coeval cube.
                box_size_mpc (float)     : The box size in Mpc. Default value is determined from 
                                             the box size set for the simulation (set_sim_constants)
                max_baseline (float)     : The maximun baseline of the telescope in km. Default value 
                                             is set as 2 km (SKA core).
                ratio (int)              : It is the ratio of smoothing scale in frequency direction and 
                                             the angular direction (Default value: 1).
                nu_axis (int)            : Frequency axis

        Returns:
                Smoothed_coeval_cube
        """
        if (not box_size_mpc): box_size_mpc=conv.LB  
        if uv_map is None: 
                uv_map, N_ant  = get_uv_map(array.shape[0],
                                            z,
                                            filename=filename,
                                            total_int_time=total_int_time,
                                            int_time=int_time,
                                            boxsize=box_size_mpc,
                                            declination=declination,
                                            verbose=verbose,
                                        )
        if max_baseline is not None: uv_map = remove_baselines_from_uvmap(uv_map, z, max_baseline=max_baseline, box_size_mpc=box_size_mpc)
    
        output_dtheta  = (1+z)*21e-5/max_baseline
        output_ang_res = output_dtheta*cm.z_to_cdist(z) * cube.shape[0]/box_size_mpc
        output_cube = smooth_coeval_tophat(cube, output_ang_res*ratio, nu_axis=nu_axis, verbose=verbose)
        if nu_axis not in [2,-1]: output_cube = np.swapaxes(output_cube,nu_axis,2)
        output_cube = np.array([convolve_uvmap(output_cube[:,:,i], uv_map=uv_map, verbose=verbose, box_size_mpc=box_size_mpc) for i in tqdm(range(output_cube.shape[2]),disable=not verbose)])
        if nu_axis not in [2,-1]: output_cube = np.swapaxes(output_cube,nu_axis,2)
        return output_cube

def smooth_line(y, window=3, kind='tophat'):
    """
    Smooths a 1D array using a specified window type.

    Parameters:
    -----------
    y : array-like
        The input data to smooth, typically a 1D array.
    window : int or array-like
        The size of the smoothing window (for predefined types) or a custom window (as an array).
        If an integer is provided, it defines the width of the window.
    kind : str, optional
        The type of window to use. Options are:
        - 'tophat' or 'boxcar': A uniform window of equal weights.
        - 'gaussian' or 'normal': A Gaussian window.
        - 'triangular': A triangular window.
        - 'hamming': A Hamming window.
        - 'hanning': A Hanning window.
        Defaults to 'tophat'.

    Returns:
    --------
    y_smooth : ndarray
        The smoothed version of the input array `y`.

    Raises:
    -------
    ValueError
        If the provided `kind` is not recognized.

    Examples:
    ---------
    >>> y = [1, 2, 3, 4, 5, 6, 7]
    >>> smooth_line(y, window=3, kind='gaussian')
    array([...])  # Smoothed values
    """
    if isinstance(window, int):
        if kind.lower() in ['tophat', 'boxcar']:
            window = np.ones(window) / window
        elif kind.lower() in ['normal', 'gaussian']:
            x = np.linspace(-3 * window, 3 * window, 6 * window + 1)
            window = np.exp(-x**2 / (2 * (window**2)))
            window /= np.sum(window)  # Normalize the Gaussian
        elif kind.lower() == 'triangular':
            window = np.arange(1, window + 1)
            window = np.concatenate([window, window[::-1][1:]])
            window = window / window.sum()
        elif kind.lower() == 'hamming':
            window = np.hamming(window)
        elif kind.lower() == 'hanning':
            window = np.hanning(window)
        else:
            raise ValueError(f"Unsupported window kind: '{kind}'. Choose from 'tophat', 'gaussian', 'triangular', 'hamming', or 'hanning'.")
    elif isinstance(window, (list, np.ndarray)):
        window = np.array(window)
        window /= np.sum(window)  # Ensure the custom window is normalized
    else:
        raise ValueError("Window must be an integer or an array-like object.")

    y_smooth = np.convolve(y, window, mode='same')
    return y_smooth
