'''
Methods to construct lightcones.
'''

from . import const, conv
import numpy as np
import os, glob, time
from .helper_functions import get_mesh_size, \
    determine_redshift_from_filename, get_data_and_type, print_msg, find_idx
from .density_file import DensityFile
from .vel_file import VelocityFile
from . import cosmo as cm
from . import statistics as st
from . import smoothing
import scipy.interpolate
from tqdm import tqdm

def make_lightcone(filenames, z_low = None, z_high = None, depth_mhz = None, file_redshifts = None, cbin_bits = 32, cbin_order = 'c', los_axis = 0, raw_density = False, interpolation='linear', reading_function=None, box_length_mpc=None):
    '''
    Generate a lightcone from xfrac, density, or dT data cubes.
    
    Parameters:
    -----------
    filenames : str, array-like, or dict
        The coeval cubes can be specified in one of the following formats:
            - An array of file names.
            - A text file containing the file names.
            - A dictionary with redshift values as keys and data as values.
    
    z_low : float, optional
        The lowest redshift for the lightcone. If not provided, the redshift of 
        the lowest-z coeval cube is used.
    
    z_high : float, optional
        The highest redshift for the lightcone. If not provided, the redshift of 
        the highest-z coeval cube is used.
    
    depth_mhz : float, optional
        The frequency depth of the lightcone in MHz. Used in determining the 
        output redshifts.
    
    file_redshifts : str or array-like, optional
        The redshifts corresponding to the coeval cubes. This can be:
            - None: redshifts are inferred from file names.
            - Array: an array containing the redshift of each coeval cube.
            - Filename: a file from which redshifts will be read.
    
    cbin_bits : int, default=32
        The number of bits in the cbin format data files, if applicable.
    
    cbin_order : char, default='c'
        The data order in the cbin format files ('c' for C-order, 'f' for Fortran-order).
    
    los_axis : int, default=0
        The axis to use as the line-of-sight for the coeval cubes.
    
    raw_density : bool, default=False
        If True and the data is from a density file, the raw simulation units will be 
        returned instead of the density in cgs units.
    
    interpolation : str, default='linear'
        The interpolation method to use for slices between output redshifts.
        Options: 'linear', 'step', 'sigmoid', 'step_cell'.
    
    reading_function : callable, optional
        A custom function to read the data cubes. If not provided, the function will 
        use a default method to read the files.
    
    box_length_mpc : float, optional
        The length of the simulation box in comoving megaparsecs (Mpc). Used to 
        calculate the output redshifts.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - lightcone (ndarray): The lightcone volume, where the first two axes 
          have the same size as the input cubes.
        - z (ndarray): An array containing the redshifts along the line-of-sight.
        
    Notes:
    ------
    - If `z_low` is provided, that redshift will be the lowest included, even if 
      there is no coeval cube at exactly that redshift. This may result in subtle 
      differences from results calculated with the old `freq_box` routine.
    - If `filenames` is a dictionary, it should have redshift values as keys and 
      corresponding file names as values.
    '''
    
    if not interpolation in ['linear', 'step', 'sigmoid', 'step_cell']:
        raise ValueError('Unknown interpolation type: %s' % interpolation)
    
    if isinstance(filenames,dict):
        if reading_function is None: 
            reading_function = lambda x: x
        file_redshifts = np.sort(list(filenames.keys()))
        filenames = {ii: filenames[zi] for ii,zi in enumerate(file_redshifts)}
    
    if reading_function is None:
        #Figure out output redshifts, file names and size of output
        filenames = _get_filenames(filenames)
        file_redshifts = _get_file_redshifts(file_redshifts, filenames)
        mesh_size = get_mesh_size(filenames[0])
    else:
        assert file_redshifts is not None
        mesh_size = reading_function(filenames[0]).shape

    assert len(file_redshifts) == len(filenames)
    
    output_z = _get_output_z(file_redshifts, z_low, z_high, mesh_size[0], box_length_mpc=box_length_mpc, dnu=depth_mhz)

    #Make the output 32-bit to save memory 
    lightcone = np.zeros((mesh_size[0], mesh_size[1], len(output_z)), dtype='float32')
    
    comoving_pos_idx = 0
    z_bracket_low = None; z_bracket_high = None
    data_low = None; data_high = None
    
    # Make the lightcone, one slice at a time
    # print_msg('Making lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    print('Making lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    time.sleep(1)
    for ii in tqdm(range(len(output_z))):
        z = output_z[ii]
        z_bracket_low_new = file_redshifts[file_redshifts <= z].max()
        z_bracket_high_new = file_redshifts[file_redshifts > z].min()
        
        #Do we need a new file for the low z?
        if z_bracket_low_new != z_bracket_low:
            z_bracket_low = z_bracket_low_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_low))
            if data_high is None:
                if reading_function is None: data_low, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order, raw_density)
                else: data_low = reading_function(filenames[file_idx])#; print('yes')
            else: #No need to read the file again
                data_low = data_high
            
        #Do we need a new file for the high z?
        if z_bracket_high_new != z_bracket_high:
            z_bracket_high = z_bracket_high_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_high))
            if reading_function is None: data_high, datatype = get_data_and_type(filenames[file_idx], cbin_bits, cbin_order, raw_density)
            else: data_high = reading_function(filenames[file_idx])
        
        #Make the slice by interpolating, then move to next index
        data_interp = _get_interp_slice(data_high, data_low, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx, los_axis, interpolation)
        lightcone[:,:,comoving_pos_idx] = data_interp
        # print('%.2f %% completed.'%(100*(ii+1)/output_z.size))
        comoving_pos_idx += 1
    print('...done')
    return lightcone, output_z


def make_velocity_lightcone(vel_filenames, dens_filenames, z_low = None, \
                            z_high = None, file_redshifts = None, los_axis = 0,
                            vel_reading_function=None, dens_reading_function=None):
    '''
    Generate a lightcone from velocity data. Since velocity files contain momentum
    rather than actual velocity, filenames for both velocity and density must be specified.
    
    Parameters:
    -----------
    vel_filenames : str, array-like, or dict
        The coeval velocity cubes. Can be specified in one of the following formats:
            - An array of file names.
            - A text file containing the file names.
            - A dictionary with redshift values as keys and data as values.
    
    dens_filenames : str, array-like, or dict
        The coeval density cubes in the same way as `vel_filenames`.
    
    z_low : float, optional
        The lowest redshift for the lightcone. If not provided, the redshift of 
        the lowest-z coeval cube is used.
    
    z_high : float, optional
        The highest redshift for the lightcone. If not provided, the redshift of 
        the highest-z coeval cube is used.
    
    file_redshifts : str or array-like, optional
        The redshifts corresponding to the coeval cubes. This can be:
            - None: redshifts are inferred from file names.
            - Array: an array containing the redshift of each coeval cube.
            - Filename: a file from which redshifts will be read.
    
    los_axis : int, default=0
        The axis to use as the line-of-sight for the coeval cubes.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - lightcone (ndarray): The lightcone volume where the first two axes 
          have the same size as the input cubes.
        - z (ndarray): An array containing the redshifts along the line-of-sight.
    
    Notes:
    ------
    If `z_low` is provided, that redshift will be the lowest included, even if 
    there is no coeval cube at exactly that redshift. This may result in subtle 
    differences from results calculated with the old `freq_box` routine.
    
    If `vel_filenames` or `dens_filenames` is provided as a dictionary, it should 
    have redshift values as keys and corresponding file names as values.
    '''
    
    if isinstance(vel_filenames, dict):
        file_redshifts = np.sort(list(vel_filenames.keys()))
        vel_filenames = {ii: vel_filenames[zi] for ii, zi in enumerate(file_redshifts)}
        dens_filenames = {ii: dens_filenames[zi] for ii, zi in enumerate(file_redshifts)}
        if vel_reading_function is None:
            vel_reading_function = lambda x: x
        if dens_reading_function is None:
            dens_reading_function = lambda x: x
        mesh_size = dens_reading_function(dens_filenames[0]).shape
        get_kms = lambda vel_file, dfile: vel_file
    else:
        dens_filenames = _get_filenames(dens_filenames)
        file_redshifts = _get_file_redshifts(file_redshifts, dens_filenames)
        vel_filenames = _get_filenames(vel_filenames)
        mesh_size = get_mesh_size(dens_filenames[0])
        if vel_reading_function is None:
            vel_reading_function = lambda x: VelocityFile(x)
        if dens_reading_function is None:
            dens_reading_function = lambda x: DensityFile(x)
        get_kms = lambda vel_file, dfile: vel_file.get_kms_from_density(dfile)


    assert(len(file_redshifts) == len(vel_filenames))
    assert(len(vel_filenames) == len(dens_filenames))
    
    output_z = _get_output_z(file_redshifts, z_low, z_high, mesh_size[0])

    lightcone = np.zeros((3, mesh_size[0], mesh_size[1], len(output_z)), dtype='float32')
    
    comoving_pos_idx = 0
    z_bracket_low = None; z_bracket_high = None
    
    print('Making velocity lightcone between %f < z < %f' % (output_z.min(), output_z.max()))
    time.sleep(1)
    for ii in tqdm(range(len(output_z))):
        z = output_z[ii]
        z_bracket_low_new = file_redshifts[file_redshifts <= z].max()
        z_bracket_high_new = file_redshifts[file_redshifts > z].min()
        
        if z_bracket_low_new != z_bracket_low:
            z_bracket_low = z_bracket_low_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_low))
            dfile = dens_reading_function(dens_filenames[file_idx])
            vel_file = vel_reading_function(vel_filenames[file_idx])
            data_low = get_kms(vel_file,dfile)
            del dfile
            del vel_file
            
        if z_bracket_high_new != z_bracket_high:
            z_bracket_high = z_bracket_high_new
            file_idx = np.argmin(np.abs(file_redshifts - z_bracket_high))
            dfile = dens_reading_function(dens_filenames[file_idx])
            vel_file = vel_reading_function(vel_filenames[file_idx])
            data_high = get_kms(vel_file,dfile)
            del dfile
            del vel_file
        
        data_interp = _get_interp_slice(data_high, data_low, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx, los_axis)
        lightcone[:,:,:,comoving_pos_idx] = data_interp
        
        comoving_pos_idx += 1
    print('...done')
    return lightcone, output_z


def _get_output_z(file_redshifts, z_low, z_high, box_grid_n, box_length_mpc=None, dnu=None):
    '''
    Determine the output redshifts. For internal use.
    '''
    if z_low is None:
        z_low = file_redshifts.min()
    if z_high is None:
        z_high = file_redshifts.max()
    
    if(dnu):
        output_z = redshifts_at_equal_frequncy_width(z_low, z_high, dnu)
    else:
        output_z = redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n, box_length_mpc=box_length_mpc)

    if min(output_z) < min(file_redshifts) or max(output_z) > max(file_redshifts):
        print('Warning! You have specified a redshift range of %.3f < z < %.3f' % (min(output_z), max(output_z)))
        print('but you only have files for the range %.3f < z < %.3f.' % (min(file_redshifts), max(file_redshifts)))
        print('The redshift range will be truncated.')
        output_z = output_z[output_z >= min(file_redshifts)]
        output_z = output_z[output_z <= max(file_redshifts)]
    if len(output_z) < 1:
        raise Exception('No valid redshifts in range!')

    return output_z


def redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=256, box_length_mpc=None):
    ''' 
    Make a frequency axis vector with equal spacing in co-moving LOS coordinates. 
    The comoving distance between each frequency will be the same as the cell
    size of the box.
    
    Parameters:
        z_low (float): The lower redshift
        z_high (float): The upper redhisft 
        box_grid_n = 256 (int): the number of slices in an input box
        box_length_mpc (float): the size of the box in cMpc. If None,
        set to conv.LB
             
    Returns:
        numpy array containing the redshifts
        
    '''
    if box_length_mpc is None:
        box_length_mpc = conv.LB
    assert(z_high > z_low)

    z = z_low
    z_array = []

    while z < z_high:
        z_array.append(z)
        nu = const.nu0/(1.0+z)
        dnu = const.nu0*const.Hz(z)*box_length_mpc/(1.0 + z)**2/const.c/float(box_grid_n)
        z = const.nu0/(nu - dnu) - 1.0

    return np.array(z_array)


def redshifts_at_equal_frequncy_width(z_low, z_high, dnu):
    ''' 
    Make a redshift vector with equally space in the frequency domain. 
    
    Parameters:
        z_low (float): The lower redshift
        z_high (float): The upper redhisft 
        dnu (float): The frequency channel width in MHz
             
    Returns:
        numpy array containing the redshifts
        
    '''
    assert(z_high > z_low)

    z = z_high
    z_array = []

    while z > z_low:
        z_array.append(z)
        z = (z - dnu/const.nu0 * (1+z))/(1 + dnu/const.nu0 * (1+z))

    return np.array(z_array)[::-1]


def get_lightcone_subvolume(lightcone, redshifts, central_z, \
                            depth_mhz=None, depth_mpc=None, odd_num_cells=True, \
                            subtract_mean=True, fov_Mpc=None, verbose=True):
    '''
    Extract a subvolume from a lightcone, at a given central redshift,
    and with a given depth. The depth can be specified in Mpc or MHz.
    You must give exactly one of these parameters.
    
    Parameters:
        ligthcone (numpy array): the lightcone
        redshifts (numpy array): the redshifts along the LOS
        central_z (float): the central redshift of the subvolume
        depth_mhz (float): the depth of the subvolume in MHz
        depth_mpc (float): the depth of the subvolume in Mpc
        odd_num_cells (bool): if true, the depth of the box will always 
                be an odd number of cells. This avoids problems with 
                power spectrum calculations.
        subtract_mean (bool): if true, subtract the mean of the signal (Default: True)
        fov_Mpc (float): the FoV size in Mpc
        verbose (bool) : turning the verbosity. 
        
    Returns:
        Tuple with (subvolume, dims, sub_redshifts) where dims is a tuple
        with the dimensions of the subvolume in Mpc
    '''
    
    assert len(np.nonzero([depth_mhz, depth_mpc])) == 1
    
    if fov_Mpc is None:
        fov_Mpc = conv.LB
        
    central_nu = cm.z_to_nu(central_z)
    if depth_mpc != None: #Depth is given in Mpc
        central_dist = cm.nu_to_cdist(central_nu)
        min_cdist, max_cdist = central_dist-depth_mpc/2., central_dist+depth_mpc/2.
        low_z = cm.cdist_to_z(min_cdist)
        high_z = cm.cdist_to_z(max_cdist)
        min_freq, max_freq = cm.z_to_nu(high_z), cm.z_to_nu(low_z)
    else: #Depth is given in MHz
        min_freq, max_freq = central_nu-depth_mhz/2., central_nu+depth_mhz/2.
        min_cdist, max_cdist = cm.z_to_cdist(cm.nu_to_z(max_freq)), cm.z_to_cdist(cm.nu_to_z(min_freq))
        low_z = cm.nu_to_z(max_freq)
        high_z = cm.nu_to_z(min_freq)
        
    if low_z < redshifts.min():
        raise Exception('Lowest z is outside range')
    if high_z > redshifts.max():
        raise Exception('Highest z is outside range')
        
    #low_n = int(find_idx(redshifts, low_z))
    #high_n = int(find_idx(redshifts, high_z))
    low_n = np.argmin(np.abs(redshifts - low_z))
    high_n = np.argmin(np.abs(redshifts - high_z))

    if (high_n-low_n) % 2 == 0 and odd_num_cells:
        high_n += 1
    
    subbox = lightcone[:,:,low_n:high_n]
    if subtract_mean:
        subbox = st.subtract_mean_signal(subbox, los_axis=2)
    #box_depth = float(subbox.shape[2])/lightcone.shape[1]*fov_Mpc
    box_depth = max_cdist-min_cdist
    box_dims = (fov_Mpc, fov_Mpc, box_depth)
    if(verbose): 
        print('Slice redshift: %.3f %.3f' %(redshifts[low_n], redshifts[high_n]))
        print('Slice frequencies (MHz): %.1f %.1f' %(min_freq, max_freq))
        print('Slice comoving distances (cMpc): %.2f %.2f' %(min_cdist, max_cdist))
        print('Slice indices:', low_n, high_n)
        print('Lightcone Subvolume : %s, [%.2f, %.2f, %.2f] cMpc' %(subbox.shape, fov_Mpc, fov_Mpc, box_depth))
    
    return subbox, box_dims, redshifts[low_n:high_n]


def _get_interp_slice(data_high, data_low, z_bracket_high, z_bracket_low, z, \
                      comoving_pos_idx, los_axis, interpolation='linear'):
    '''
    Interpolate between two data slices. For internal use.
    '''
    slice_ind = comoving_pos_idx % data_low.shape[1]
    slice_low = _get_slice(data_low, slice_ind, los_axis)
    slice_high = _get_slice(data_high, slice_ind, los_axis)
    if interpolation == 'linear':
        slice_interp = ((z-z_bracket_low)*slice_high + \
                    (z_bracket_high - z)*slice_low)/(z_bracket_high-z_bracket_low)
    elif interpolation == 'step':
        transition_z = (z_bracket_high-z_bracket_low)/2.
        if z < transition_z:
            slice_interp = slice_low.copy()
        else:
            slice_interp = slice_high.copy()
    elif interpolation == 'sigmoid':
        zp = -10. + 20.*(z-z_bracket_low)/(z_bracket_high-z_bracket_low)
        beta = 2.
        g = 1./(1.+np.exp(-beta*zp))
        slice_interp = slice_low*(1.-g) + slice_high*g
    elif interpolation == 'step_cell':
        slice_interp = _get_step_weighted_slice(slice_low, slice_high, z_bracket_high, z_bracket_low, z)
    else:
        raise Exception('Unknown interpolation method: %s' % interpolation)
    
    return slice_interp


def _get_step_weighted_slice(slice_low, slice_high, z_bracket_high, z_bracket_low, z):
    '''
    Interpolate using a step function where the step position is based
    on the proximity to an ionized region
    '''
    smoothed = smoothing.smooth_gauss(slice_high, sigma=4.) 
    diff = np.abs(slice_high-slice_low)
    
    #smoothed=1 means early transition. smoothed=0 means late
    #smoothed -= changed_cells.min()
    #smoothed /= (changed_cells.max()-changed_cells.min())
    step_transitions = _get_step_transitions(smoothed, diff>1.e-3)
    step_transitions = z_bracket_low + step_transitions*(z_bracket_high-z_bracket_low)
    
    interp_slice = slice_high.copy()
    interp_slice[z < step_transitions] = slice_low[z < step_transitions]
    return interp_slice


def _get_step_transitions(cell_dist, diff_idx):
    #Replace each value in cell_dist with the number of cells 
    #with a lower value. Then normalize to be between 0 and 1
    values_flat = cell_dist[diff_idx].flatten()
    values_sorted = sorted(values_flat+np.random.random(len(values_flat))*1.e-9)
    values_uniform = np.linspace(values_sorted[0], values_sorted[-1], len(values_sorted))
    f = scipy.interpolate.interp1d(values_sorted, values_uniform, bounds_error=False, fill_value=0.)
    output_values = f(cell_dist)
    output_values -= output_values.min()
    output_values /= output_values.max()
    return output_values


def _get_slice(data, idx, los_axis, slice_depth=1):
    '''
    Slice a data cube along a given axis. For internal use.
    '''
    assert len(data.shape) == 3 or len(data.shape) == 4
    assert los_axis >= 0 and los_axis < 3
    
    idx1 = idx
    idx2 = idx1+slice_depth

    if len(data.shape) == 3: #scalar field
        if los_axis == 0:
            return np.squeeze(data[idx1:idx2,:,:])
        elif los_axis == 1:
            return np.squeeze(data[:,idx1:idx2,:])
        return np.squeeze(data[:,:,idx1:idx2])
    else: #Vector field
        if los_axis == 0:
            return np.squeeze(data[:,idx1:idx2,:,:])
        elif los_axis == 1:
            return np.squeeze(data[:,:,idx1:idx2,:])
        return np.squeeze(data[:,:,:,idx1:idx2])
    
    
def _get_filenames(filenames_in):
    '''
    If filenames_in is a list of files, return as it is
    If it is a directory, make sure it only contains data files,
    then return the list of files in the directory
    If it is a text file, read the list of files from the file
    '''
    
    if hasattr(filenames_in, '__iter__'):
        filenames_out = filenames_in
    elif os.path.isdir(filenames_in):
        files_in_dir = glob.glob(filenames_in + '/*')
        extensions = [os.path.splitext(f)[-1] for f in files_in_dir]
        if not _all_same(extensions):
            raise Exception('The directory may only contain one file type.')
        filenames_out = files_in_dir
    elif os.path.isfile(filenames_in):
        f = open(filenames_in)
        names = [l.strip() for l in f.readlines()]
        f.close()
        filenames_out = names
    else:
        raise Exception('Invalid filenames input')
        
    return np.array(filenames_out)

    
def _get_file_redshifts(redshifts_in, filenames):
    '''
    If redshifts_in is None, try to determine from file names
    If it's a directory, read the redshifts
    Else, return as is
    '''
    
    if hasattr(redshifts_in, '__iter__'):
        redshifts_out = redshifts_in
    elif redshifts_in is None:
        redshifts_out = [determine_redshift_from_filename(f) for f in filenames]
        redshifts_out = np.array(redshifts_out)
    elif os.path.exists(redshifts_in):
        redshifts_out = np.loadtxt(redshifts_in)
    else:
        raise Exception('Invalid data for file redshifts.')
    
    return redshifts_out



def _all_same(items):
    return all(x == items[0] for x in items)

    
