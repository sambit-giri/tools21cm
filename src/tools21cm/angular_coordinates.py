'''
Methods to convert data between physical (cMpc) coordinates
and observational (angular-frequency) coordinates.
'''

import numpy as np
from .lightcone import redshifts_at_equal_comoving_distance
from . import cosmo as cm
from . import conv
from . import helper_functions as hf
from . import smoothing
from . import const
from scipy.signal import fftconvolve
from tqdm import tqdm


def physical_lightcone_to_observational(physical_lightcone, input_z_low, output_dnu, output_dtheta, input_box_size_mpc=None, verbose=True, order=2, mode='pad'):
    '''
    Interpolate a lightcone volume from physical (length) units to observational (angle/frequency) units.
    
    Parameters
    ----------
    physical_lightcone : numpy.ndarray
        The input lightcone volume in physical coordinates with shape 
        (N, N, M) where N is the number of cells per side in angular 
        direction and M is the number of cells along the line-of-sight
    input_z_low : float or array-like
        The lowest redshift of the input lightcone. If array-like, should 
        contain redshifts for each slice along the line-of-sight
    output_dnu : float
        The frequency resolution of the output volume in MHz
    output_dtheta : float
        The angular resolution of the output in arcmin
    input_box_size_mpc : float, optional
        The size of the input field of view in Mpc. If None (default), 
        uses the value from conv.LB
    verbose : bool, optional
        Whether to show progress messages (default: True)
    order : int, optional
        The order of the spline interpolation (0-5). Default is 2.
        Use order=0 for ionization fraction data.
    mode : str, optional
        How to handle the field of view at different redshifts:
        
        - 'pad' (default): Fix FoV at lowest redshift (largest angular size)
          and pad higher redshift slices
        - 'crop': Fix FoV at highest redshift (smallest angular size) 
          and crop lower redshift slices
        - 'full', 'extend': Same as 'pad'
        - 'valid': Same as 'crop'

    Returns
    -------
    tuple
        output_volume : numpy.ndarray
            The transformed lightcone in observational coordinates with shape 
            (N_theta, N_theta, N_nu)
        output_freqs : numpy.ndarray
            Array of output frequencies in MHz corresponding to the slices 
            along the line-of-sight
    '''
    assert mode.lower() in ['pad', 'full', 'extend', 'crop', 'valid'], "Accepted input for mode: 'pad', 'full', 'extend', 'crop', 'valid'."

    if input_box_size_mpc == None:
        input_box_size_mpc = conv.LB

    if isinstance(input_z_low,(float,int)):
        cell_size = input_box_size_mpc/physical_lightcone.shape[0]
        distances = cm.z_to_cdist(input_z_low) + np.arange(physical_lightcone.shape[2])*cell_size
        input_z_high = cm.cdist_to_z(distances).max()
    else:
        input_z_low, input_z_high = input_z_low.min(), input_z_low.max()

    #Calculate the FoV in degrees at lowest z (largest one)
    fov_deg_low  = cm.angular_size_comoving(input_box_size_mpc, input_z_low)
    fov_deg_high = cm.angular_size_comoving(input_box_size_mpc, input_z_high)

    #For each output redshift: average the corresponding slices
    if verbose:
        print(f'At the lowest redshift (z={input_z_low:.3f}), the angluar scale is {fov_deg_low:.3f} deg')
        print(f'At the highest redshift (z={input_z_high:.3f}), the angluar scale is {fov_deg_high:.3f} deg')
        print('Making observational lightcone...')
        print('Binning in frequency...')

    lightcone_freq, output_freqs = bin_lightcone_in_frequency(physical_lightcone,\
                                                            input_z_low, input_box_size_mpc, output_dnu)

    #Calculate dimensions of output volume
    n_cells_theta  = int(fov_deg_low*60./output_dtheta)
    n_cells_nu = len(output_freqs)

    #Go through each slice and make angular slices for each one
    if verbose:
        print('Binning in angle...')
    output_volume = np.zeros((n_cells_theta, n_cells_theta, n_cells_nu))
    for i in tqdm(range(n_cells_nu), disable=not verbose):
        z = cm.nu_to_z(output_freqs[i])
        output_volume[:,:,i] = physical_slice_to_angular(lightcone_freq[:,:,i], z, \
                                        slice_size_mpc=input_box_size_mpc, fov_deg=fov_deg_low,\
                                        dtheta=output_dtheta, order=order)
        
    if mode.lower() in ['pad', 'full', 'extend']:
        pass
    elif mode.lower() in ['crop', 'valid']:
        n_cells_theta_out = int(fov_deg_high*60./output_dtheta)
        output_volume = output_volume[(n_cells_theta-n_cells_theta_out)//2:(n_cells_theta+n_cells_theta_out)//2,(n_cells_theta-n_cells_theta_out)//2:(n_cells_theta+n_cells_theta_out)//2,:]
    else:
        print(f"mode={mode} is not implemented")
        
    return output_volume, output_freqs


def observational_lightcone_to_physical(observational_lightcone, input_freqs, input_dtheta, verbose=True, order=2):
    '''
    Interpolate a lightcone volume measured in observational (angle/frequency)
    units into  physical (length) units. The output resolution will be set
    to the coarest one, as determined either by the angular or the frequency
    resolution. The lightcone must have the LoS as the last index, with 
    frequencies decreasing along the LoS.
    
    Parameters:
        observational_lightcone (numpy array): the input lightcone volume
        input_freqs (numpy array): the frequency in MHz of each slice along the 
            line of sight of the input
        input_dheta (float): the angular size of a cell in arcmin
        verbose (bool): show progress bar
        order (int): The order of the spline interpolation, default is 2. 
            The order has to be in the range 0-5.
            Use order=0 for ionization fraction data.
        
    Returns:
        * The output volume
        * The redshifts along the LoS of the output
        * The output cell size in Mpc
    '''
    assert input_freqs[0] > input_freqs[-1]
    assert observational_lightcone.shape[0] == observational_lightcone.shape[1]
    
    #Determine new cell size - set either by frequency or angle.
    #The FoV size in Mpc is set by the lowest redshift
    dnu = input_freqs[0]-input_freqs[1]
    z_low = cm.nu_to_z(input_freqs[0])
    fov_deg = observational_lightcone.shape[0]*input_dtheta/60.
    fov_mpc = fov_deg/cm.angular_size_comoving(1., z_low)
    cell_size_perp = fov_mpc/observational_lightcone.shape[0]
    cell_size_par = cm.nu_to_cdist(input_freqs[-1])-cm.nu_to_cdist(input_freqs[-2])
    output_cell_size = max([cell_size_par, cell_size_perp])
    hf.print_msg('Making physical lightcone with cell size %.2f Mpc' % output_cell_size)
    #Go through each slice along frequency axis. Cut off excess and 
    #interpolate down to correct resolution
    n_cells_perp = int(fov_mpc/output_cell_size)
    output_volume_par = np.zeros((n_cells_perp, n_cells_perp, observational_lightcone.shape[2]))
    for i in tqdm(range(output_volume_par.shape[2]), disable=not verbose):
        z = cm.nu_to_z(input_freqs[i])
        output_volume_par[:,:,i] = angular_slice_to_physical(observational_lightcone[:,:,i],\
                                                    z, slice_size_deg=fov_deg, output_cell_size=output_cell_size,\
                                                    output_size_mpc=fov_mpc, order=order)
    #Bin along frequency axis
    output_volume, output_redshifts = bin_lightcone_in_mpc(output_volume_par, \
                                                input_freqs, output_cell_size)
    
    return output_volume, output_redshifts, output_cell_size


def physical_slice_to_angular(input_slice, z, slice_size_mpc, fov_deg, dtheta, order=0):
    '''
    Interpolate a slice in physical coordinates to angular coordinates.
    
    Parameters:
        input_slice (numpy array): the 2D slice in physical coordinates
        z (float): the redshift of the input slice
        slice_size_Mpc (float): the size of the input slice in cMpc
        fov_deg (float): the field-of-view in degrees. The output will be
            padded to match this size
        dtheta (float): the target resolution in arcmin
        
    Returns:
        (angular_slice, size_deg)
    
    '''
    #Resample
    fov_mpc = cm.deg_to_cdist(fov_deg, z)
    cell_size_mpc = fov_mpc/(fov_deg*60./dtheta)
    n_cells_resampled = int(slice_size_mpc/cell_size_mpc)
    #Avoid edge effects with even number of cells
    if n_cells_resampled % 2 == 0: 
        n_cells_resampled -= 1
    resampled_slice = resample_slice(input_slice, n_cells_resampled, order)
    
    #Pad the array
    slice_n = resampled_slice.shape[0]
    padded_n = int(fov_deg*60./dtheta)# np.round(slice_n*(fov_mpc/slice_size_mpc))
    if padded_n < slice_n:
        if slice_n - padded_n > 2:
            print('Warning! Padded slice is significantly smaller than original!')
            print('This should not happen...')
        padded_n = slice_n
    padded_slice = _get_padded_slice(resampled_slice, padded_n)
    
    return padded_slice
    

def angular_slice_to_physical(input_slice, z, slice_size_deg, output_cell_size, output_size_mpc, order=0, prefilter=True):
    '''
    Interpolate a slice in angular coordinates to physical
    
    Parameters:
        input_slice (numpy array): the 2D slice in observational coordinates
        z (float): the redshift of the input slice
        slice_size_deg (float): the size of the input slice in deg
        output_cell_size (float): the output cell size in cMpc
        output_size_mpc (float): the output size in mpc

    Returns:
        (physical_slice, size_mpc)
    '''
    #Resample
    slice_size_mpc = cm.deg_to_cdist(slice_size_deg, z)
    n_cells_resampled = int(slice_size_mpc/output_cell_size)
    #Avoid edge effects with even number of cells
    if n_cells_resampled % 2 == 0: 
        n_cells_resampled += 1
    resampled_slice = resample_slice(input_slice, n_cells_resampled, order, prefilter)
    
    #Remove cells to get correct size
    n_cutout_cells = int(output_size_mpc/output_cell_size)# np.round(resampled_slice.shape[0]*output_size_mpc/slice_size_mpc)
    if n_cutout_cells > input_slice.shape[0]:
        if input_slice.shape[0] - n_cutout_cells > 2:
            print('Warning! Cutout slice is larger than original.')
            print('This should not happen')
        n_cutout_cells = input_slice.shape[0]
    slice_cutout = resampled_slice[:n_cutout_cells, :n_cutout_cells]
        
    return slice_cutout


def resample_slice(input_slice, n_output_cells, order=0, prefilter=True):
    '''
    Resample a 2D slice to new dimensions.
    
    Parameters:
        input_slice (numpy array): the input slice
        n_output_cells (int) : the number of output cells

    Returns:
        output slice
    '''
    tophat_width = np.round(input_slice.shape[0]/n_output_cells)
    if tophat_width < 1 or (not prefilter):
        tophat_width = 1
    slice_smoothed = smoothing.smooth_tophat(input_slice, tophat_width)

    idx = np.linspace(0, slice_smoothed.shape[0]-1, n_output_cells)
    output_slice = smoothing.interpolate2d(slice_smoothed, idx, idx, order=order)

    return output_slice


def bin_lightcone_in_frequency(lightcone, z_low, box_size_mpc, dnu):
    '''
    Bin a lightcone in frequency bins.
    
    Parameters:
        lightcone (numpy array): the lightcone in length units
        z_low (float): the lowest redshift of the lightcone
        box_size_mpc (float): the side of the lightcone in Mpc
        dnu (float): the width of the frequency bins in MHz
        
    Returns:
        * The lightcone, binned in frequencies with high frequencies first
        * The frequencies along the line of sight in MHz
    '''
    #Figure out dimensions and make output volume
    cell_size = box_size_mpc/lightcone.shape[0]
    distances = cm.z_to_cdist(z_low) + np.arange(lightcone.shape[2])*cell_size
    input_redshifts = cm.cdist_to_z(distances)
    input_frequencies = cm.z_to_nu(input_redshifts)
    nu1 = input_frequencies[0]
    nu2 = input_frequencies[-1]
    output_frequencies = np.arange(nu1, nu2, -dnu)
    output_lightcone = np.zeros((lightcone.shape[0], lightcone.shape[1], \
                                 len(output_frequencies)))
    
    #Bin in frequencies by smoothing and indexing
    max_cell_size = cm.nu_to_cdist(output_frequencies[-1])-cm.nu_to_cdist(output_frequencies[-2])
    smooth_scale = np.round(max_cell_size/cell_size)
    if smooth_scale < 1:
        smooth_scale = 1

    hf.print_msg('Smooth along LoS with scale %f' % smooth_scale)
    tophat3d = np.ones((1,1,int(smooth_scale)))
    tophat3d /= np.sum(tophat3d)
    lightcone_smoothed = fftconvolve(lightcone, tophat3d)
    
    for i in range(output_lightcone.shape[2]):
        nu = output_frequencies[i]
        idx = int(hf.find_idx(input_frequencies, nu))
        output_lightcone[:,:,i] = lightcone_smoothed[:,:,idx]

    return output_lightcone, output_frequencies


def bin_lightcone_in_mpc(lightcone, frequencies, cell_size_mpc):
    '''
    Bin a lightcone in Mpc slices along the LoS
    '''
    distances = cm.nu_to_cdist(frequencies)
    n_output_cells = (distances[-1]-distances[0])/cell_size_mpc
    output_distances = np.arange(distances[0], distances[-1], cell_size_mpc)
    output_lightcone = np.zeros((lightcone.shape[0], lightcone.shape[1], int(n_output_cells)))
    
    #Bin in Mpc by smoothing and indexing
    smooth_scale = np.round(len(frequencies)/n_output_cells)

    tophat3d = np.ones((1,1,int(smooth_scale)))
    tophat3d /= np.sum(tophat3d)
    lightcone_smoothed = fftconvolve(lightcone, tophat3d, mode='same')
    
    for i in range(output_lightcone.shape[2]):
        idx = int(hf.find_idx(distances, output_distances[i]))
        output_lightcone[:,:,i] = lightcone_smoothed[:,:,int(idx)]
    
    output_redshifts = cm.cdist_to_z(output_distances)
        
    return output_lightcone, output_redshifts


def _get_padded_slice(input_slice, padded_n):
    slice_n = input_slice.shape[0]
    padded_slice = np.zeros((padded_n, padded_n))
    padded_slice[:slice_n, :slice_n] = input_slice
    padded_slice[slice_n:, :slice_n] = input_slice[:(padded_n-slice_n),:slice_n]
    padded_slice[:slice_n, slice_n:] = input_slice[:,:(padded_n-slice_n)]
    padded_slice[slice_n:, slice_n:] = input_slice[:(padded_n-slice_n), :(padded_n-slice_n)]
    return padded_slice

def padding_lightcone(lc, padded_n=None, mode='wrap', verbose=True):
    '''
    Pad lightcone in the field of view direction.
    
    Parameters:
        lc (numpy array): the light-cone in physical coordinates with third axis as the light-of-sight.
        padded_n (int): number of cells to pad. The default value is half of the number of cells.
        mode (str): mode used for padding (see the documentation of numpy.pad). 
        verbose (bool): show progress bar.

    Returns:
        padded lightcone
    '''
    if padded_n is None: padded_n = int(lc.shape[1]/2)
    out_lc = np.zeros((lc.shape[0]+2*padded_n,lc.shape[1]+2*padded_n,lc.shape[2]))
    for i in tqdm(range(lc.shape[2]), disable=not verbose):
        out_lc[:,:,i] = np.pad(lc[:,:,i], padded_n, mode=mode)
    return out_lc
