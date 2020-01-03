import numpy as np
from . import const
from . import conv
from .helper_functions import print_msg, get_interpolated_array, read_cbin
from . import vel_file
from . import density_file

def get_distorted_dt(dT, kms, redsh, los_axis=0, velocity_axis = 0, num_particles=10, periodic=True):
    ''' 
    Apply peculiar velocity distortions to a differential
    temperature box, using the Mesh-Particle-Mesh method,
    as described in http://arxiv.org/abs/1303.5627
    
    Parameters:
        * dT (numpy array): the differential temperature box
        * kms (numpy array): velocity in km/s, array of dimensions 
            (3,mx,my,mz) where (mx,my,mz) is dimensions of dT
        * redsh (float): the redshift
        * los_axis = 0 (int): the line-of-sight axis of the output volume
            (must be 0, 1 or 2)
        * velocity_axis = 0 (int): the index that indicates los velocity
        * num_particles = 10 (int): the number of particles to use per cell
            A higher number gives better accuracy, but worse performance.
        * periodic = True (bool): whether or not to apply periodic boundary
            conditions along the line-of-sight. If you are making a lightcone
            volume, this should be False.
        
    Returns:
        The redshift space box as a numpy array with same dimensions as dT.
        
    Example:
        Read a density file, a velocity file and an xfrac file, calculate the 
        brightness temperature, and convert it to redshift space.
        
        >>> vfile = c2t.VelocityFile('/path/to/data/8.515v_all.dat')
        >>> dfile = c2t.DensityFile('/path/to/data/8.515n_all.dat')
        >>> xfile = c2t.XfracFile('/path/to/data/xfrac3d_8.515.bin')
        >>> dT = c2t.calc_dt(xfile, dfile)
        >>> kms = vfile.get_kms_from_density(dfile)
        >>> dT_zspace = get_distorted_dt(dT, kms, dfile.z, los_axis = 0)
        
    .. note::
        At the moment, it is a requirement that dimensions perpendicular to
        the line-of-sight are equal. For example, if the box dimensions are
        (mx, my, mz) and the line-of-sight is along the z axis, then mx
        has to be equal to my.
        
    .. note::
        If dT is a lightcone volume, los_axis is not necessarily the
        same as velocity_axis. The lightcone volume methods in c2raytools
        all give output volumes that have the line-of-sight as the last index,
        regardless of the line-of-sight axis. For these volumes, you should
        always use los_axis=2 and set velocity_axis equal to whatever was
        used when producing the real-space lightcones.
    
    '''
    #Volume dimensions
    mx,my,mz = dT.shape
    assert(mx == my or my == mz or mx == mz) #TODO: this should not be a requirement 
    grid_depth = dT.shape[los_axis]
    grid_width = dT.shape[(los_axis+1)%3]
    box_depth = grid_depth * (conv.LB/float(grid_width))

    #Take care of different LOS axes
    assert(los_axis == 0 or los_axis == 1 or los_axis == 2)
    if los_axis == 0:
        get_skewer = lambda data, i, j : data[:,i,j]
    elif los_axis == 1:
        get_skewer = lambda data, i, j : data[i,:,j]
    else:
        get_skewer = lambda data, i, j : data[i,j,:]

    #Input redshift can be a float or an array, but we need an array
    redsh = np.atleast_1d(redsh)

    print_msg('Making velocity-distorted box...')
    print_msg('The (min) redshift is %.3f' % redsh[0])
    print_msg('The box size is %.3f cMpc' % conv.LB)
    
    #Figure out the apparent position shift 
    vpar = kms[velocity_axis,:,:,:]
    z_obs = (1+redsh)/(1.-vpar/const.c)-1.
    dr = (1.+z_obs)*vpar/const.Hz(z_obs)

    #Make the distorted box
    distbox = np.zeros_like(dT)
    particle_dT = np.zeros(grid_depth*num_particles)

    last_percent = 0
    for i in range(grid_width):
        percent_done = int(float(i)/float(grid_width)*100)
        if percent_done%10 == 0 and percent_done != last_percent:
            print_msg('%d %%' % percent_done)
            last_percent = percent_done
        for j in range(grid_width):

            #Take a 1D skewer from the dT box
            dT_skewer = get_skewer(dT,i,j)

            #Create particles along the skewer and assign dT to the particles
            particle_pos = np.linspace(0, box_depth, grid_depth*num_particles)
            for n in range(num_particles): 
                particle_dT[n::num_particles] = dT_skewer/float(num_particles)

            #Calculate LOS velocity for each particle
            dr_skewer_pad = get_skewer(dr,i,j)
            np.insert(dr_skewer_pad, 0, dr_skewer_pad[-1])
            dr_skewer = get_interpolated_array(dr_skewer_pad, len(particle_pos), 'linear')
            
            #Apply velocity shift
            particle_pos += dr_skewer

            #Periodic boundary conditions
            if periodic:
                particle_pos[np.where(particle_pos < 0)] += box_depth
                particle_pos[np.where(particle_pos > box_depth)] -= box_depth

            #Regrid particles to original resolution
            dist_skewer = np.histogram(particle_pos, \
                                       bins=np.linspace(0, box_depth, grid_depth+1), \
                                       weights=particle_dT)[0]
            if los_axis == 0:
                distbox[:,i,j] = dist_skewer
            elif los_axis == 1:
                distbox[i,:,j] = dist_skewer
            else:
                distbox[i,j,:] = dist_skewer

    print_msg('Old dT (mean,var): %3f, %.3f' % ( dT.astype('float64').mean(), dT.var()) )
    print_msg('New dT (mean,var): %.3f, %.3f' % (distbox.mean(), distbox.var()) )
    return distbox


def make_pv_box(dT_filename, vel_filename, dens_filename, z, los = 0, num_particles=10):
    '''
    Convenience method to read files and make a distorted box.
    
    Parameters:
        * dT_filename (string): the name of the dT file
        * vel_filename (string): the name of the velocity file
        * dens_filename (string): the name of the density file
        * z (float): the redshift
        * los (integer): the line-of-sight axis
        * num_particles (integer): the number of particles to pass
            to get_distorted_dt
        
    Returns:
        The redshift space box
    '''

    dT = read_cbin(dT_filename, bits=32, order='c')
    vfile = vel_file.VelocityFile(vel_filename)
    dfile = density_file.DensityFile(dens_filename)
    kms = vfile.get_kms_from_density(dfile)
    dT_pv = get_distorted_dt(dT, kms, redsh = z, los_axis = los, \
                            num_particles = num_particles)
    return dT_pv

