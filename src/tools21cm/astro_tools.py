import numpy as np
from glob import glob
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18

def FoV_cMpc2(FoV, redshift, cosmo=None):
    '''
    Convert field of view from square degrees to comoving megaparsecs squared (cMpc^2).

    Parameters:
    -----------
    FoV : astropy.units.Quantity
        Field of view in square degrees (deg^2).
    redshift : float
        Redshift of the object.
    cosmo : astropy.cosmology instance, optional
        Cosmology model to use. If None, the Planck 2018 cosmology (Planck18) will be used.

    Returns:
    --------
    FoV_cMpc2 : astropy.units.Quantity
        Field of view in comoving megaparsecs squared (cMpc^2).

    Notes:
    ------
    This function assumes that the field of view is small, and approximates the comoving 
    distance at the given redshift as uniform across the field of view.
    '''
    if cosmo is None:
        cosmo = Planck18
        print('Assuming Planck 2018 cosmology.')
    if isinstance(FoV,(int,float)): FoV *= units.rad**2
    D = cosmo.comoving_distance(redshift)
    FoV_cMpc2 = FoV.to('rad2').value * D**2
    return FoV_cMpc2


def mag_to_flux(mag):
    '''
    Convert apparent magnitude to flux density in CGS units (erg/s/cm^2/Hz).

    Parameters:
    -----------
    mag : astropy.units.Quantity
        Apparent magnitude in the AB system.

    Returns:
    --------
    flux : astropy.units.Quantity
        Flux density in units of erg/s/cm^2/Hz.

    Notes:
    ------
    The conversion assumes the AB magnitude system, where the zero point corresponds to a flux
    of 3631 Jy (Jansky).
    '''
    if isinstance(mag, (int,float)): mag *= units.mag
    flux = 10**(-(mag.to('mag').value + 48.60)/2.5) * units.erg/units.s/units.cm**2/units.Hz
    return flux


def Muv_to_muv(Muv, z, cosmo=None):
    '''
    Convert absolute magnitude to apparent magnitude in the UV band.

    Parameters:
    -----------
    Muv : astropy.units.Quantity
        Absolute magnitude in the UV band.
    z : float
        Redshift of the object.
    cosmo : astropy.cosmology instance, optional
        Cosmology model to use. If None, the Planck 2018 cosmology (Planck18) will be used.

    Returns:
    --------
    m_uv : float
        Apparent magnitude in the UV band.

    Notes:
    ------
    This function includes a K-correction term that accounts for the redshift of the object. The
    K-correction assumes a simple scaling with (1 + z), but more complex models could be applied 
    depending on the spectral energy distribution of the object.
    '''
    if cosmo is None:
        cosmo = Planck18
        print('Assuming Planck 2018 cosmology.')
    if isinstance(Muv, (int,float)): Muv *= units.mag
    Kcorrection = 2.5 * np.log10(1 + z) 
    m_uv = Muv + cosmo.distmod(z).value - Kcorrection
    return m_uv


def muv_to_Muv(muv, z, cosmo=None):
    '''
    Convert apparent magnitude to absolute magnitude in the UV band.

    Parameters:
    -----------
    muv : astropy.units.Quantity
        Apparent magnitude in the UV band.
    z : float
        Redshift of the object.
    cosmo : astropy.cosmology instance, optional
        Cosmology model to use. If None, the Planck 2018 cosmology (Planck18) will be used.

    Returns:
    --------
    Muv : float
        Absolute magnitude in the UV band.

    Notes:
    ------
    The K-correction term is applied to adjust for the effects of redshift. It assumes a simple
    scaling with (1 + z), but could be refined depending on the specific emission properties of 
    the object.
    '''
    if cosmo is None:
        cosmo = Planck18
        print('Assuming Planck 2018 cosmology.')
    if isinstance(muv, (int,float)): muv *= units.mag
    Kcorrection = 2.5 * np.log10(1 + z) * units.mag
    Muv = muv - cosmo.distmod(z) + Kcorrection
    return Muv


def Muv_to_L1500(M_AB):
    '''
    Convert absolute UV magnitude to UV luminosity at 1500 Å (erg/s/Hz).

    Parameters:
    -----------
    M_AB : astropy.units.Quantity
        Absolute magnitude in the AB system.

    Returns:
    --------
    L1500 : astropy.units.Quantity
        Luminosity at 1500 Å in units of erg/s/Hz.

    Notes:
    ------
    This conversion is based on the AB magnitude system, where a magnitude of M_AB = 34.1 corresponds 
    to a luminosity of 10^34.1 erg/s/Hz.
    '''
    if isinstance(Muv, (int,float)): Muv *= units.mag
    SI2cgs = 1e7  # 1 W/Hz = 1e7 erg/s/Hz
    L1500 = SI2cgs * 10.**((34.1 - M_AB.to('mag').value)/2.5) * units.erg/units.s/units.Hz
    return L1500
