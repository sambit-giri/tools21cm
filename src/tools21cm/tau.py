from . import const, conv
from .lightcone import make_lightcone
import numpy as np
from tqdm import tqdm

def tau(ionfractions, redshifts, num_points = 50):
    '''
    Calculate the optical depth to Thomson scattering.

    Parameters
    ----------
    ionfractions : ndarray
        An array containing the ionized fraction at various points along the line-of-sight.

    redshifts : ndarray
        An array containing the redshifts corresponding to the same points as `ionfractions`. 
        Must be the same length as `ionfractions`.

    num_points : int, optional
        The number of initial points used for the integration to account for high-redshift 
        contributions where ionization is assumed to be negligible. Default is 50.

    Returns
    -------
    tuple
        A tuple containing:
        
        tau_0 : ndarray
            Optical depth at each redshift. The length is equal to the length of `redshifts` + `num_points`.
        
        tau_z : ndarray
            The corresponding redshift array. The length is equal to the length of `redshifts` + `num_points`.

    Notes
    -----
    - The Universe is assumed to be fully ionized at the lowest redshift supplied.
    - To get the total optical depth, refer to the last value in `tau_0`.
    '''

    if len(ionfractions) != len(redshifts):
        print('Incorrect length of ionfractions')
        raise Exception()

    ionfractions, redshifts = ionfractions[np.argsort(redshifts)], redshifts[np.argsort(redshifts)]

    sigma_T = 6.65e-25
    chi1 = 1.0+const.abu_he
    coeff = 2.0*(const.c*1e5)*sigma_T*const.OmegaB/const.Omega0*const.rho_crit_0*\
            chi1/const.mean_molecular/const.m_p/(3.*const.H0cgs)

    tau_z = np.hstack((np.arange(1,num_points+1)/float(num_points)*redshifts[0], redshifts))

    tau0 = np.zeros(len(redshifts)+num_points)

    #Optical depth for a completely ionized Universe
    tau0[0:num_points] = coeff*(np.sqrt(const.Omega0*(1+tau_z[0:num_points])**3+const.lam) - 1)

    for i in range(num_points, len(redshifts)+num_points):
        tau0[i] = tau0[i-1]+1.5*coeff*const.Omega0 * \
        (ionfractions[i-1-num_points]*(1+tau_z[i-1])**2/np.sqrt(const.Omega0*(1+tau_z[i-1])**3+const.lam) \
        + ionfractions[i-num_points]*(1+tau_z[i])**2/np.sqrt(const.Omega0*(1+tau_z[i])**3+const.lam) ) * \
        (tau_z[i]-tau_z[i-1])/2

    return tau0, tau_z

def tau_map(ionfractions, redshifts=None, num_points=50, reading_function=None):
    '''
    Calculate the optical depth to Thomson scattering for lightcones or maps.

    Parameters
    ----------
    ionfractions : ndarray or dict
        Ionized fraction data across various points along the line-of-sight.

    redshifts : ndarray, optional
        Array of redshift values corresponding to the ionfractions data. Must be the same length 
        as ionfractions if `ionfractions` is an ndarray. If `ionfractions` is a dict and redshifts 
        is None, redshifts are inferred from the keys of the dictionary.

    num_points : int, optional
        Number of initial points used for the integration to account for high-redshift contributions 
        where ionization is assumed to be negligible. Default is 50.

    reading_function : callable, optional
        Custom function to read and process ionization fraction data when `ionfractions` is provided 
        in a format that requires specialized reading. The function should take a filename or data 
        identifier as input and return an ndarray of ionization fractions.

    Returns
    -------
    tuple
        A tuple containing:
        
        tau_0 : ndarray
            Optical depth values at each spatial position and redshift. The shape is 
            (N_x, N_y, N_z + num_points), where N_x and N_y are spatial dimensions, 
            and N_z is the number of redshift slices in `output_z`.
        
        tau_z : ndarray
            Array of redshift values corresponding to each slice in `tau_0`. 
            The length is `N_z + num_points`.

    Notes
    -----
    - The Universe is assumed to be fully ionized at the lowest redshift supplied.
    - The `reading_function` is useful when working with custom data formats.
    '''
    if redshifts is None:
        assert isinstance(ionfractions, dict), "redshifts must be provided if ionfractions is not a dict."
        redshifts = np.sort(np.array(list(ionfractions.keys())))

    if isinstance(ionfractions, np.ndarray):
        lightcone, output_z = ionfractions, redshifts
    else:
        lightcone, output_z = make_lightcone(ionfractions, file_redshifts=redshifts, reading_function=reading_function)

    sigma_T = 6.65e-25  # Thomson scattering cross-section in cm^2
    chi1 = 1.0 + const.abu_he  # Accounting for Helium abundance
    coeff = (2.0 * const.c * 1e5 * sigma_T * const.OmegaB / const.Omega0 * const.rho_crit_0 *
                chi1 / const.mean_molecular / const.m_p / (3.0 * const.H0cgs))

    tau_z = np.hstack((np.linspace(output_z[0] / num_points, output_z[0], num_points), output_z))
    tau_0 = np.zeros((lightcone.shape[0], lightcone.shape[1], len(tau_z)))

    # Optical depth for a completely ionized Universe at high redshifts
    tau_0[:, :, :num_points] = coeff * (np.sqrt(const.Omega0 * (1 + tau_z[:num_points])**3 + const.lam) - 1)

    for i in tqdm(range(num_points, len(tau_z))):
        z_prev = tau_z[i - 1]
        z_curr = tau_z[i]
        x_e_prev = lightcone[:, :, i - num_points - 1]
        x_e_curr = lightcone[:, :, i - num_points]
        
        integrand_prev = x_e_prev * (1 + z_prev)**2 / np.sqrt(const.Omega0 * (1 + z_prev)**3 + const.lam)
        integrand_curr = x_e_curr * (1 + z_curr)**2 / np.sqrt(const.Omega0 * (1 + z_curr)**3 + const.lam)
        
        delta_tau = 1.5 * coeff * const.Omega0 * (integrand_prev + integrand_curr) * (z_curr - z_prev) / 2
        tau_0[:, :, i] = tau_0[:, :, i - 1] + delta_tau

    return tau_0, tau_z