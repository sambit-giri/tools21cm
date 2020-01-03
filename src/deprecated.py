'''
Created on Oct 9, 2013

@author: Hannes Jensen

This module contains old or renames methods that are kept
for compatibility reasons
'''

import numpy as np
import os
from .lightcone import redshifts_at_equal_comoving_distance, _get_interp_slice
from .xfrac_file import XfracFile
from .density_file import DensityFile
from .helper_functions import print_msg, get_dens_redshifts, get_mesh_size, \
    determine_redshift_from_filename, get_data_and_type, read_cbin, save_cbin
from .temperature import calc_dt
from . import smoothing

def freq_box(xfrac_dir, dens_dir, z_low, z_high):
    ''' 
    Make frequency (lightcone) boxes of density, ionized fractions, 
    and brightness temperature. The function reads xfrac and density
    files from the specified directories and combines them into a 
    lighcone box going from z_low to z_high.
    
    This routine is more or less a direct translation of Garrelt's 
    IDL routine.
    
    Parameters: 
        * xfrac_dir (string): directory containing xfrac files
        * dens_dir (string): directory containing density files
        * z_low (float): lowest redshift to include
        * z_high (float): highest redshift to include.

    Returns: 
        Tuple with (density box, xfrac box, dt box, redshifts), where
        density box, xfrac box and dt box are numpy arrays containing
        the lightcone quantities. redshifts is an array containing the 
        redshift for each slice.
        
    .. note::
        Since this function relies on filenames to get redshifts,
        all the data files must follow the common naming convenstions.
        Ionization files must be named xfrac3d_z.bin and densityfiles
        zn_all.dat
        
    .. note::
        The make_lightcone method is meant to replace this method. It
        is more general and easier to use.
    
    Example:
        Make a lightcone cube ranging from z = 7 to z = 8:
    
        >>> xfrac_dir = '/path/to/data/xfracs/'
        >>> dens_dir = '/path/to/data/density/'
        >>> xcube, dcube, dtcube, z = c2t.freq_box(xfrac_dir, density_dir, z_low=7.0, z_high=8.)
        
    '''

    #Get the list of redshifts where we have simulation output files
    dens_redshifts = get_dens_redshifts(dens_dir, z_low )
    mesh_size = get_mesh_size(os.path.join(dens_dir, '%.3fn_all.dat' % dens_redshifts[0]))

    #Get the list of redhifts and frequencies that we want for the observational box
    output_z = redshifts_at_equal_comoving_distance(z_low, z_high, box_grid_n=mesh_size[0])
    output_z = output_z[output_z > dens_redshifts[0]]
    output_z = output_z[output_z < dens_redshifts[-1]]
    if len(output_z) < 1:
        raise Exception('No valid redshifts in range!')

    #Keep track of output simulation files to use
    xfrac_file_low = XfracFile(); xfrac_file_high = XfracFile()
    dens_file_low = DensityFile(); dens_file_high = DensityFile()
    z_bracket_low = None; z_bracket_high = None

    #The current position in comoving coordinates
    comoving_pos_idx = 0

    #Build the cube
    xfrac_lightcone = np.zeros((mesh_size[0], mesh_size[1], len(output_z)))
    dens_lightcone = np.zeros_like(xfrac_lightcone)
    dt_lightcone = np.zeros_like(xfrac_lightcone)
    
    for z in output_z:
        #Find the output files that bracket the redshift
        z_bracket_low_new = dens_redshifts[dens_redshifts <= z][0]
        z_bracket_high_new = dens_redshifts[dens_redshifts >= z][0]

        if z_bracket_low_new != z_bracket_low:
            z_bracket_low = z_bracket_low_new
            xfrac_file_low = XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_low))
            dens_file_low = DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_low))
            dt_cube_low = calc_dt(xfrac_file_low, dens_file_low)

        if z_bracket_high_new != z_bracket_high:
            z_bracket_high = z_bracket_high_new
            xfrac_file_high = XfracFile(os.path.join(xfrac_dir, 'xfrac3d_%.3f.bin' % z_bracket_high))
            dens_file_high = DensityFile(os.path.join(dens_dir, '%.3fn_all.dat' % z_bracket_high))
            dt_cube_high = calc_dt(xfrac_file_high, dens_file_high)
            
        slice_ind = comoving_pos_idx % xfrac_file_high.mesh_x
        
        #Ionized fraction
        xi_interp = _get_interp_slice(xfrac_file_high.xi, xfrac_file_low.xi, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx)
        xfrac_lightcone[:,:,comoving_pos_idx] = xi_interp

        #Density
        rho_interp = _get_interp_slice(dens_file_high.cgs_density, dens_file_low.cgs_density, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx)
        dens_lightcone[:,:,comoving_pos_idx] = rho_interp

        #Brightness temperature
        dt_interp = _get_interp_slice(dt_cube_high, dt_cube_low, z_bracket_high, \
                                    z_bracket_low, z, comoving_pos_idx)
        dt_lightcone[:,:,comoving_pos_idx] = dt_interp

        print_msg( 'Slice %d of %d' % (comoving_pos_idx, len(output_z)) )
        comoving_pos_idx += 1

    return xfrac_lightcone, dens_lightcone, dt_lightcone, output_z


def read_binary_with_meshinfo(filename, bits=32, order='C'):
    return read_cbin(filename, bits, order)


def save_binary_with_meshinfo(filename, data, bits=32, order='C'):
    save_cbin(filename, data, bits, order)
    
def smooth(input_array, sigma):
    return smoothing.smooth_gauss(input_array, sigma) 
