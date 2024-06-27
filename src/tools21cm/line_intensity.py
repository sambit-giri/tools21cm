import numpy as np
from glob import glob
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18

from .nbody_file import halo_list_to_grid

def Mhi_on_grid(z, mass, pos_xyz, box_dim, n_grid, cosmo=None, 
                recipe='Padmanabhan2017', **kwargs):
    """
    Assign HI line intensity to dark matter halos and distribute them on a grid.

    Parameters
    ----------
    z : float
        Redshift.
    mass : ndarray
        Mass of halos.
    pos_xyz : ndarray
        Positions of the halos.
    box_dim : float
        Length of the simulation box in each direction.
    n_grid : int
        Number of grid cells along each direction.
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology object. If None, assumes Planck18 cosmology.
    recipe : str, optional
        The recipe to use for mapping the halo masses to the HI signal. Implemented recipes:
        - 'Padmanabhan2017' (1611.06235). Default is 'Padmanabhan2017'.
    **kwargs : dict, optional
        Additional parameters for the specific recipe. 

    Returns
    ----------
    lim_map : ndarray
        The HI line intensity distributed on a grid of shape (n_grid, n_grid, n_grid).
    """
    assert recipe.lower() in ['padmanabhan2017']
    if cosmo is None:
        print('Assuming Planck18 cosmology.')
        cosmo = Planck18
    Ob = cosmo.Ob0
    Om = cosmo.Om0 
    hlittle = cosmo.h

    if recipe.lower() in ['padmanabhan2017']:
        Y_p = kwargs.get('Y_p', 0.249)
        alpha = kwargs.get('alpha', 0.9)
        v_c0 = kwargs.get('v_c0', 10**1.56*units.km/units.s)
        beta = kwargs.get('beta', -0.58)
        
        if isinstance(mass, units.quantity.Quantity):
            mhalo = mass.to('Msun')
        else:
            print('The provided halo mass is assumed to be in Msun units.')
            mhalo = mass*units.Msun
        if isinstance(mass, units.quantity.Quantity):
            srcpos = pos_xyz.to('Mpc')
        else:
            print('The provided halo mass is assumed to be in Mpc units.')
            srcpos = pos_xyz*units.Mpc
        fHc = (1-Y_p)*Ob/Om
        Delc = 178
        v_c = lambda Mh,z: 96.6*units.km/units.s*(Delc*Om/54.4)**(1/6)*((1+z)/3.3)**(1/2)*(Mh/(1e11*units.Msun))**(1/3)
        v_c_Mh = v_c(mhalo,z)
        M_hi = lambda Mh: alpha*fHc*Mh*(Mh/hlittle/units.Msun)**beta*np.exp(-(v_c0/v_c_Mh)**3)
        mhi_msun = M_hi(mhalo)
        binned_mhi, bin_edges, bin_num = halo_list_to_grid(mhi_msun, srcpos, box_dim, n_grid)
        return binned_mhi