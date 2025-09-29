'''
Created on Feb 27, 2015

@author: Hannes Jensen
'''
from . import const
import numpy as np
from . import cosmo as cm

def noise_error_ps(nu_c, k, t, **kwargs):
    '''
    Calculate the system noise error on the
    power spectrum, using the analytical expression in 
    Mellema et al 2013 (equation 11). If no arguments are given,
    the noise is calculated for LOFAR-like parameters
    with delta k = k, and a bandwidth of 10 MHz.
    
    Parameters:
        * nu_c (float): the central observing frequency
        * k (float or array-like): the k mode(s)
        * t (float): the observing time in hours
        
    Valid kwargs:
        * Rmax (float): the radius of the array in meters
        * Aeff (float or function): the effective area. Can be a 
            function of nu.
        * Nstat (int): the number of stations
        * Tsys (float or function): the system temperature. Can be a
            function of nu.
        * B (float): the bandwidth in MHz
        * epsilon (float): the width of the k bins in terms of k
        * multipole (int): if this is zero (default), the 
            noise is calculated for the monopole (spherially-averaged).
            Otherwise it is calculated for the given multipole moment
            of the power spectrum.
    
    Returns:
        The system noise error in mK^2

    '''
    wavel = const.c/nu_c*1.e-3
    t = t*60.*60. #s
    Rmax = kwargs.get('Rmax', 1500.)
    Acore = Rmax**2*np.pi #m^2
    Aeff = kwargs.get('Aeff', lambda nu: 526.*(nu/150.)**(-2))
    if hasattr(Aeff, '__call__'):
        Aeff_val = Aeff(nu_c)
    else:
        Aeff_val = Aeff
    Nstat = kwargs.get('Nstat', 48)
    Acoll = Nstat*Aeff_val
    B = kwargs.get('B', 10.)
    Dc = cm.nu_to_cdist(nu_c)
    deltaDc = np.abs(Dc - cm.nu_to_cdist(nu_c+B))
    OmegaFoV = wavel**2/Aeff_val
    Tsys = kwargs.get('Tsys', lambda nu: (140. + 60.*(nu/300.)**(-2.55))*1000.)
    if hasattr(Tsys, '__call__'):
        Tsys_val = Tsys(nu_c)
    else:
        Tsys_val = Tsys
    epsilon = kwargs.get('epsilon', 1)
    multipole = kwargs.get('multipole', 0)
    multipole_factor = np.sqrt(2*multipole+1)
    Delta_noise = (2./np.pi)*k**(3./2.)*np.sqrt(Dc**2*deltaDc*OmegaFoV)*(Tsys_val/np.sqrt(B*1.e6*t))**2*(Acore*Aeff_val/(Acoll**2))/np.sqrt(epsilon)
    
    return Delta_noise*multipole_factor


