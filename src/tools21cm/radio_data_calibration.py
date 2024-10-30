'''
Methods:

* simulate the radio telescope observation strategy
* simulate telescope gains
'''

import numpy as np
import sys
from .telescope_functions import *
from .usefuls import *
from . import conv
from . import cosmo as cm
from . import smoothing as sm
import scipy
from glob import glob
from time import time, sleep
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm

def get_gain_map(ncells, z, gain_model={'name': 'random_gaussian', 'mu': 1, 'std': 0.1}, filename=None, 
                 total_int_time=6., int_time=10., boxsize=None, declination=-30., include_mirror_baselines=False, verbose=True):
    """
    Create the gain and uv maps.

    Parameters
    ----------
    ncells: int
        Number of cells
    z: float
        Redshift.
    total_int_time: float
        Total observation per day time in hours
    int_time: float
        Intergration time in seconds
    declination: float
        Declination angle in deg
    filename: str
        The path to the file containing the telescope configuration.

            - As a default, it takes the SKA-Low configuration from Sept 2016
            - It is not used if uv_map and N_ant is provided
    boxsize: float
        Boxsize in Mpc	
    verbose: bool
        If True, verbose is shown

    Returns
    -------
    uv_map: ndarray
        array of gridded uv coverage.
    N_ant: int
        Number of antennae
    """
    if isinstance(gain_model, dict):
        if gain_model['name'].lower()=='random_gaussian':
            mu, std = gain_model['mu'], gain_model['std']
            gain_model = lambda N: np.random.normal(mu, std, N)
            
    z = float(z)
    Nbase, N_ant = from_antenna_config(filename, z)
    uv_map       = np.zeros((ncells,ncells))
    tot_num_obs  = int(3600.*total_int_time/int_time)
    if verbose: 
        print("Making uv map from daily observations.")
        time.sleep(1)
    for i in tqdm(range(tot_num_obs)):
        new_Nbase = earth_rotation_effect(Nbase, i, int_time, declination=declination)
        uv_map1   = grid_uv_tracks(new_Nbase, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
        uv_map   += uv_map1
    uv_map = uv_map/tot_num_obs
    print('...done')
    return uv_map, N_ant

    if not filename: N_ant = SKA1_LowConfig_Sept2016().shape[0]
    if isinstance(gain_model, dict):
        if gain_model['name'].lower()=='random_gaussian':
            mu, std = gain_model['mu'], gain_model['std']
            gain_model = lambda N: np.random.normal(mu, std, N)

    full_uv_data, N_ant  = get_uv_daily_observation(ncells, z, filename, total_int_time=total_int_time, int_time=int_time, boxsize=boxsize, declination=declination, verbose=verbose, full_data=True)
    
    uv_map = np.zeros((ncells,ncells))
    n_step = len(full_uv_data.keys())
    gain_maps = {}
    for i in tqdm(range(n_step)):
        uv_map1 = full_uv_data[i]
        uv_map += uv_map1 
        if gain_model.lower()=='random_gaussian':
            gain_map1 = gain_model(ncells*ncells).reshape(ncells,ncells)
            gain_map1[gain_map1<0] = 0
            gain_map1[uv_map1==0]  = 0
            gain_maps[i] = gain_map1
    return {
            'uv_map': uv_map, 
            'N_ant': N_ant,
            'gain_maps': gain_maps,
            }