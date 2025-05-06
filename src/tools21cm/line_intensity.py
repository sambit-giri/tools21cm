import numpy as np
from glob import glob
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18

from .nbody_file import halo_list_to_grid

def M_HI_on_grid(z, mass, pos_xyz, box_dim, n_grid, cosmo=None, 
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

    if not isinstance(box_dim, units.quantity.Quantity):
        box_dim = box_dim*units.Mpc

    if recipe.lower() in ['padmanabhan2017']:
        Y_p = kwargs.get('Y_p', 0.249)
        alpha = kwargs.get('alpha', 0.9)
        v_c0 = kwargs.get('v_c0', 10**1.56*units.km/units.s)
        beta = kwargs.get('beta', -0.58)
        
        fHc = (1-Y_p)*Ob/Om
        Delc = 178
        v_c = lambda Mh,z: 96.6*units.km/units.s*(Delc*Om/54.4)**(1/6)*((1+z)/3.3)**(1/2)*(Mh/(1e11*units.Msun))**(1/3)
        v_c_Mh = v_c(mhalo,z)
        M_hi = lambda Mh: alpha*fHc*Mh*(Mh.to('Msun').value/hlittle/1e11)**beta*np.exp(-(v_c0/v_c_Mh)**3)
        mhi_msun = M_hi(mhalo).to('Msun').value
        binned_mhi, bin_edges, bin_num = halo_list_to_grid(mhi_msun, srcpos, box_dim, n_grid)
        return binned_mhi*units.Msun
    
def L_CII_on_grid(z, mass, pos_xyz, box_dim, n_grid, SFR=None, cosmo=None, 
                recipe='Silva2015m1', **kwargs):
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
    SFR : function or dict
        A function taking halo mass and redshift to give the star formation rate 
        or a dictionary containing parameters values for the provided recipe.
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology object. If None, assumes Planck18 cosmology.
    recipe : str, optional
        The recipe to use for mapping the halo masses to the CII signal. Implemented recipes:
        - 'Silva2015m1' (1410.4808). Default is 'Silva2015m1'.
    **kwargs : dict, optional
        Additional parameters for the specific recipe. 

    Returns
    ----------
    lim_map : ndarray
        The HI line intensity distributed on a grid of shape (n_grid, n_grid, n_grid).
    """
    assert recipe.lower() in ['silva2015m1', 'silva2015m2', 'silva2015m3', 'silva2015m4']
    if cosmo is None:
        print('Assuming Planck18 cosmology.')
        cosmo = Planck18
    Ob = cosmo.Ob0
    Om = cosmo.Om0 
    hlittle = cosmo.h

    if isinstance(mass, units.quantity.Quantity):
        mhalo = mass.to('Msun')
    else:
        print('The provided halo mass is assumed to be in Msun units.')
        mhalo = mass*units.Msun
    if isinstance(pos_xyz, units.quantity.Quantity):
        srcpos = pos_xyz.to('Mpc')
    else:
        print('The provided halo mass is assumed to be in Mpc units.')
        srcpos = pos_xyz*units.Mpc
    
    if not isinstance(box_dim, units.quantity.Quantity):
        box_dim = box_dim*units.Mpc

    if SFR is None: SFR = 'silva2013'
    if isinstance(SFR, str):
        if SFR.lower() in ['silva2013', 'silva2015']:
            SFR0 = kwargs.get('SFR0', 2.25e-26*units.Msun/units.yr)
            a_SFR = kwargs.get('a_SFR', 2.59)
            b_SFR = kwargs.get('b_SFR', -0.62)
            d_SFR = kwargs.get('d_SFR', 0.40)
            e_SFR = kwargs.get('e_SFR', -2.25)
            c1_SFR = kwargs.get('c1_SFR', 8e8*units.Msun)
            c2_SFR = kwargs.get('c2_SFR', 7e9*units.Msun)
            c3_SFR = kwargs.get('c3_SFR', 1e11*units.Msun)
            SFR = lambda M,z: SFR0*(1+(z-7)*7.5e-2)*M.to('Msun').value**a_SFR*(1+M/c1_SFR)**b_SFR*(1+M/c2_SFR)**d_SFR*(1+M/c3_SFR)**e_SFR

    if recipe.lower() in ['silva2015m1']:
        a_LCII = kwargs.get('a_LCII', 0.8475)
        b_LCII = kwargs.get('b_LCII', 7.2203)
    elif recipe.lower() in ['silva2015m2']:
        a_LCII = kwargs.get('a_LCII', 1.0000)
        b_LCII = kwargs.get('b_LCII', 6.9647)
    elif recipe.lower() in ['silva2015m3']:
        a_LCII = kwargs.get('a_LCII', 0.8727)
        b_LCII = kwargs.get('b_LCII', 6.7250)
    elif recipe.lower() in ['silva2015m4']:
        a_LCII = kwargs.get('a_LCII', 0.9231)
        b_LCII = kwargs.get('b_LCII', 6.5234)
    else:
        print(f'{recipe} is an unknown recipe for assigning L_CII signal.')
        return None 

    if recipe.lower() in ['silva2015m1', 'silva2015m2', 'silva2015m3', 'silva2015m4']:
        LCII_lsun = 10**(a_LCII*np.log10(SFR(mhalo,z).to('Msun/yr').value)+b_LCII)
        binned_Lcii, bin_edges, bin_num = halo_list_to_grid(LCII_lsun, srcpos, box_dim, n_grid)
        return binned_Lcii*units.Lsun