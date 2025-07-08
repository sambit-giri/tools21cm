import numpy as np
import os, pickle
import itertools
from tqdm import tqdm
from . import cosmo as cm
from . import conv
from .const import KB_SI, c_light_cgs, c_light_SI, janskytowatt
from .read_files import get_package_resource_path
import astropy.units as u
from astropy.coordinates import EarthLocation

def subarray_type_to_antxyz(subarray_type, verbose=True):
    """
    Resolves antenna configuration from various input types.

    Parameters
    ----------
    subarray_type : str or np.ndarray
        If string, loads a pre-defined SKA-Low layout (e.g., "AA4").
        If numpy array, it's used directly as the antenna positions.
        Otherwise, loads a default SKA1-Low configuration.
    verbose : bool, optional
        If True, enables verbose output during layout loading.

    Returns
    -------
    antxyz : astropy.units.Quantity
        A 2D array of antenna (x, y, z) positions with units of meters.
    N_ant : int
        The total number of antennas in the configuration.
    """
    if isinstance(subarray_type, str):
        antxyz = get_SKA_Low_layout(subarray_type=subarray_type, unit='xyz', verbose=verbose)
    elif isinstance(subarray_type, np.ndarray): 
        antxyz = subarray_type
    else:
        antll  = SKA1_LowConfig_Sept2016()
        antxyz = geographic_to_cartesian_coordinate_system(antll)*u.m
    N_ant = antxyz.shape[0]
    return antxyz, N_ant

def from_antenna_config(antxyz, z, nu=None):
    """
    Calculates baseline vectors (u,v,w) from antenna XYZ positions.

    The function converts physical antenna separations into baseline coordinates
    in units of wavelength.

    Parameters
    ----------
    antxyz: astropy.units.Quantity or np.ndarray
        The radio telescope antenna configuration in meters.
    z : float
        Redhsift of the slice observed, used to calculate frequency.
    nu : float, optional
        The frequency of observation in MHz. If None, it is calculated from z.

    Returns
    -------
    Nbase : np.ndarray
        Numpy array of shape (N_baselines, 3) containing the (u,v,w) values.
    N_ant : int
        The total number of antennas.
    """
    z = float(z)
    if antxyz is None: 
        antll  = SKA1_LowConfig_Sept2016()
        antll  = antll[:,-2:].astype(float)
        antxyz = geographic_to_cartesian_coordinate_system(antll)
    else:
        antxyz = antxyz.to('m').value
    if not nu: 
        nu = cm.z_to_nu(z)                           # MHz

    N_ant = antxyz.shape[0]
    pair_comb = itertools.combinations(range(N_ant), 2)
    pair_comb = list(pair_comb)	
    lam = c_light_cgs/(nu*1e6)/1e2 			            # in m
    Nbase = []
    for ii,jj in list(pair_comb):
        ux = (antxyz[ii,0]-antxyz[jj,0])/lam
        uy = (antxyz[ii,1]-antxyz[jj,1])/lam
        uz = (antxyz[ii,2]-antxyz[jj,2])/lam
        if ux==0: print(ii,jj)
        Nbase.append([ux,uy,uz])
    Nbase = np.array(Nbase)	
    return Nbase, N_ant

def earth_rotation_effect(Nbase, slice_nums, int_time, total_int_time, declination=-30.):
    """
    Applies Earth's rotation to baseline vectors for given time slices.

    This function simulates the change in projected baseline coordinates (u,v,w)
    over the course of an observation due to the Earth's rotation.

    Parameters
    ----------
    Nbase : np.ndarray
        Array of initial (u,v,w) baseline coordinates.
    slice_nums : int or np.ndarray
        The index or indices of the time slice(s) to compute.
    int_time : float
        Integration time per slice in seconds.
    total_int_time : float
        Total duration of the observation in hours.
    declination : float, optional
        The declination of the pointing center in degrees. Default is -30.

    Returns
    -------
    new_Nbase : np.ndarray
        The rotated (u,v,w) coordinates. The shape will be (N_baselines, 3)
        if slice_nums is a single integer, or (N_slices, N_baselines, 3)
        if slice_nums is an array.
    """
    # Convert to radians
    delta = np.deg2rad(declination)
    
    # Ensure slice_nums is an array for vectorized handling of multiple slices
    slice_nums = np.atleast_1d(slice_nums)

    # # Total number of integration steps
    # n_steps = int((total_int_time * 3600) / int_time)

    # Center observation around meridian: HA ranges from -ΔHA to +ΔHA
    HA0 = -0.5 * total_int_time*3600 + slice_nums * int_time
    HA  = np.deg2rad(15.0 * HA0 / 3600.)  #- np.pi / 2 + 2 * np.pi # Convert to radians (15 deg/hour)

    # # Compute hour angles (HA) for each slice_num in vectorized manner
    # HA = -15.0 * p * (slice_nums - 1) * int_time / 3600. - np.pi / 2 + 2 * np.pi

    # Rotation matrices for each HA and declination delta
    cos_HA, sin_HA = np.cos(HA), np.sin(HA)
    cos_delta, sin_delta = np.cos(delta), np.sin(delta)

    # Pre-allocate new_Nbase for each time slice
    new_Nbase = np.empty((len(slice_nums), *Nbase.shape))
    
    # Apply rotation transformations
    new_Nbase[:, :, 0] = sin_HA[:, None] * Nbase[:, 0] + cos_HA[:, None] * Nbase[:, 1]
    new_Nbase[:, :, 1] = (-sin_delta * cos_HA[:, None] * Nbase[:, 0] +
                          sin_delta * sin_HA[:, None] * Nbase[:, 1] + cos_delta * Nbase[:, 2])
    new_Nbase[:, :, 2] = (cos_delta * cos_HA[:, None] * Nbase[:, 0] -
                          cos_delta * sin_HA[:, None] * Nbase[:, 1] + sin_delta * Nbase[:, 2])
    
    return new_Nbase.squeeze() if len(slice_nums) == 1 else new_Nbase

def get_uv_daily_observation(ncells, z, antxyz=None, total_int_time=4., int_time=10., boxsize=None, declination=-30., include_mirror_baselines=False, verbose=True):
    """
    Simulates a full observation to generate a UV coverage map.

    This function combines baseline calculation, Earth rotation simulation, and
    UV track gridding to produce a final 2D UV coverage map, representing the
    density of measurements in the Fourier plane.

    Parameters
    ----------
    ncells : int
        Number of cells in each dimension of the output grid.
    z : float
        Redshift of the observed slice.
    antxyz : np.ndarray, optional
        The radio telescope antenna configuration. If None, a default is loaded.
    total_int_time : float
        Total observation time per day in hours.
    int_time : float
        Integration time per observation in seconds.
    boxsize : float, optional
        Comoving size of the observed sky area in Mpc.
    declination : float
        Declination of the telescope pointing in degrees.
    include_mirror_baselines : bool, optional
        If True, includes mirrored baselines for hermitian conjugation.
    verbose : bool, optional
        If True, prints progress information.

    Returns
    -------
    uv_map : np.ndarray
        2D array `(ncells, ncells)` of baseline counts per pixel.
    N_ant : int
        The total number of antennas.
    """
    # Load antenna configurations and initialize parameters
    Nbase, N_ant = from_antenna_config(antxyz, z)
    uv_map = np.zeros((ncells, ncells), dtype=np.float32)
    total_observations = int((total_int_time * 3600) / int_time)
    
    if verbose: 
        print("Generating uv map from daily observations...")
    
    # Vectorize the observation loop by calculating all rotations at once
    time_indices = np.arange(total_observations) + 1
    all_rotated_Nbase = (earth_rotation_effect(Nbase, i, int_time, total_int_time, declination) for i in time_indices)
    
    # Grid uv tracks for each observation without individual loops
    for rotated_Nbase in tqdm(all_rotated_Nbase, total=total_observations, disable=not verbose, desc="Gridding uv tracks"):
        uv_map += grid_uv_tracks(rotated_Nbase, z, ncells, boxsize=boxsize, include_mirror_baselines=include_mirror_baselines)
    
    uv_map /= total_observations  # Normalize by the total number of observations
    
    if verbose:
        print("...done")

    return uv_map, N_ant

def grid_uv_tracks(Nbase, z, ncells, boxsize=None, include_mirror_baselines=False):
    """
    Grids baseline (u,v) tracks onto a 2D map for a single time snapshot.

    This function takes a set of (u,v,w) coordinates, projects them onto a 2D
    grid corresponding to the field of view, and counts how many baselines
    fall into each grid cell.

    Parameters
    ----------
    Nbase : np.ndarray
        Array containing (u,v,w) values for the antenna baselines.
    z : float
        Redshift of the slice observed.
    ncells : int
        Number of cells in each dimension of the output grid.
    boxsize : float, optional
        Comoving size of the sky observed. Uses a default if None.
    include_mirror_baselines : bool, optional
        If True, also grids the hermitian conjugate baselines at (-u, -v).

    Returns
    -------
    uv_map : np.ndarray
        A 2D array `(ncells, ncells)` of baseline counts per pixel, shifted
        so the zero-frequency is at the center.
    """
    z = float(z)
    if not boxsize: 
        boxsize = conv.LB  # Assuming conv.LB is defined elsewhere in your code

    uv_map = np.zeros((ncells, ncells), dtype=np.int32)  # Using int32 for faster summing operations
    theta_max = boxsize / cm.z_to_cdist(z)
    
    # Include mirrored baselines if requested
    if include_mirror_baselines:
        Nbase = np.vstack([Nbase, -Nbase])

    # Scale and round the coordinates in a single operation for efficiency
    Nb = np.round(Nbase * theta_max).astype(int)

    # Vectorized conditionals to select points within grid bounds, skipping for-loops
    mask = (
        (Nb[:, 0] < ncells / 2) & (Nb[:, 0] >= -ncells / 2) &
        (Nb[:, 1] < ncells / 2) & (Nb[:, 1] >= -ncells / 2)
    )
    Nb = Nb[mask]
    
    # Convert from negative to positive indexing within the grid and use broadcasting
    xx = (Nb[:, 0] + ncells // 2).astype(int)
    yy = (Nb[:, 1] + ncells // 2).astype(int)
    
    # Fast counting using numpy's bincount and reshaping
    uv_map_flat = np.bincount(xx * ncells + yy, minlength=ncells * ncells)
    uv_map = uv_map_flat.reshape(ncells, ncells)

    # Include mirrored baselines if requested
    if include_mirror_baselines:
        # uv_map += np.flip(np.flip(uv_map, axis=0), axis=1)  # Flip along both axes for mirror effect
        uv_map = uv_map/2.  # Average with mirrored baselines
    
    return np.fft.fftshift(uv_map)

def geographic_to_cartesian_coordinate_system(antll):
    """
    Converts geographic coordinates (longitude, latitude) into Cartesian coordinates (x, y, z)
    assuming a spherical Earth with a fixed radius, using Astropy for unit handling.

    Parameters
    ----------
    antll : np.ndarray
        A 2D NumPy array of shape (N, 2), where:
        - Column 0 is longitude in degrees (λ).
        - Column 1 is latitude in degrees (φ).

    Returns
    -------
    antxyz : np.ndarray
        A 2D NumPy array of shape (N, 3) containing the Cartesian coordinates:
        - Column 0: x-coordinate in meters.
        - Column 1: y-coordinate in meters.
        - Column 2: z-coordinate in meters.

    Notes
    -----
    - The Earth's radius (mean radius) is assumed to be 6,371 km.
    - Astropy is used to ensure unit consistency.
    """
    Re = 6371. * u.km  # Earth's radius
    if not isinstance(antll, u.quantity.Quantity):
        antll *= u.deg
    lon = antll[:, 0] 
    lat = antll[:, 1] 

    # Calculate Cartesian coordinates
    x = (Re * np.cos(lat) * np.cos(lon)).to(u.m)
    y = (Re * np.cos(lat) * np.sin(lon)).to(u.m)
    z = (Re * np.sin(lat)).to(u.m)

    # Combine into a single array
    antxyz = np.vstack((x.value, y.value, z.value)).T
    return antxyz


def cartesian_to_geographic_coordinate_system(antxyz):
    """
    Converts Cartesian coordinates (x, y, z) back to geographic coordinates 
    (longitude, latitude, height above Earth's surface), using Astropy for unit handling.

    Parameters
    ----------
    antxyz : np.ndarray
        A 2D NumPy array of shape (N, 3), where:
        - Column 0 is the x-coordinate in meters.
        - Column 1 is the y-coordinate in meters.
        - Column 2 is the z-coordinate in meters.

    Returns
    -------
    antll : np.ndarray
        A 2D NumPy array of shape (N, 3) containing the geographic coordinates:
        - Column 0: Longitude (λ) in degrees.
        - Column 1: Latitude (φ) in degrees.
        - Column 2: Height (h) in meters above the Earth's surface.

    Notes
    -----
    - The Earth's radius (mean radius) is assumed to be 6,371 km.
    - Astropy is used to ensure unit consistency.
    """
    Re = 6371. * u.km  # Earth's radius

    # Extract Cartesian coordinates and attach units
    if isinstance(antxyz, u.quantity.Quantity):
        antxyz *= u.m
    x = antxyz[:, 0]
    y = antxyz[:, 1]
    z = antxyz[:, 2]

    # Compute radial distance (r)
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute geographic coordinates
    lon = np.arctan2(y, x).to(u.deg)
    lat = np.arcsin((z / r)).to(u.deg)
    height = (r - Re).to(u.m)

    # Combine into a single array
    antll = np.vstack((lon.value, lat.value, height.value)).T
    return antll

def antenna_positions_to_baselines(antxyz):
    """
    Calculates all possible baseline vectors from a set of antenna positions.

    Parameters
    ----------
    antxyz : np.ndarray or astropy.units.Quantity
        An array `(N_ant, 3)` of antenna positions.

    Returns
    -------
    np.ndarray or astropy.units.Quantity
        An array `(N_ant * (N_ant - 1), 3)` of baseline vectors.
    """
    # from itertools import combinations
    # baselines = [(a1-a2) for a1,a2 in tqdm(combinations(antxyz, 2))]
    from itertools import permutations
    baselines = np.array([(a1-a2) for a1,a2 in tqdm(permutations(antxyz, 2))])
    try: 
        return baselines*antxyz.unit
    except:
        return baselines
    
def enu_to_ecef(enu, lat_deg, lon_deg, origin_ecef):
    """
    Converts local East-North-Up (ENU) coordinates to ECEF.

    Parameters
    ----------
    enu : np.ndarray
        Positions `(N, 3)` in the local ENU frame (meters).
    lat_deg : float
        Latitude of the local frame's origin in degrees.
    lon_deg : float
        Longitude of the local frame's origin in degrees.
    origin_ecef : np.ndarray
        ECEF position `(3,)` of the local frame's origin (meters).

    Returns
    -------
    np.ndarray
        Positions `(N, 3)` in the ECEF frame (meters).
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    R = np.array([
        [-np.sin(lon), -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
        [ np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
        [          0,              np.cos(lat),              np.sin(lat)]
    ])
    
    ecef_local = enu @ R.T
    return ecef_local + origin_ecef

def ecef_to_xyz(ecef, lon_deg, origin_ecef):
    """
    Converts ECEF coordinates to a local interferometric XYZ frame.

    The XYZ frame is defined with X pointing to local East, Y to local North,
    and Z aligned with the ECEF Z-axis at the array center.
    
    Parameters
    ----------
    ecef : np.ndarray
        Antenna positions `(N, 3)` in ECEF coordinates (meters).
    lon_deg : float
        Longitude of the array center in degrees.
    origin_ecef : np.ndarray
        ECEF position `(3,)` of the array center (meters).
        
    Returns
    -------
    np.ndarray
        Antenna positions `(N, 3)` in the local XYZ frame (meters).
    """
    lon_rad = np.deg2rad(lon_deg)
    R = np.array([
        [ np.cos(lon_rad),  np.sin(lon_rad), 0],
        [-np.sin(lon_rad),  np.cos(lon_rad), 0],
        [0,                 0,               1]
    ])
    
    local = ecef - origin_ecef
    return local @ R.T + origin_ecef

def get_SKA_Low_layout(subarray_type="AA4", unit='ecef', verbose=True):
    """
    Loads SKA Low antenna layout data from pre-defined configuration files.

    Parameters
    ----------
    subarray_type : str, optional
        The name of the SKA Low subarray, e.g., "AA4", "AA2", "AASTAR".
    unit : str, optional
        The desired output coordinate system: 'ecef', 'enu', or 'xyz'.
    verbose : bool, optional
        If True, prints information about the loaded layout.

    Returns
    -------
    astropy.units.Quantity
        An `(N_ant, 3)` array of antenna positions in the specified `unit`.
    """
    if subarray_type.upper() == "AA4":
        filename = 'input_data/skalow_AA4_layout.txt'
    elif subarray_type.upper() in ["AASTAR", "AA*"]:
        filename = 'input_data/skalow_AAstar_layout.txt'
    elif subarray_type.upper() == "AA2":
        filename = 'input_data/skalow_AA2_layout.txt'
    elif subarray_type.upper() == "AA1":
        filename = 'input_data/skalow_AA1_layout.txt'
    elif subarray_type.upper() == "AA0.5":
        filename = 'input_data/skalow_AA0.5_layout.txt'
    else:
        filename = 'input_data/skalow1_layout.txt'

    # Load lat, lon, height from file
    lon, lat, height = np.loadtxt(get_package_resource_path('tools21cm', 'input_data/central_geographic_position.txt'))
    
    # EarthLocation of the array center
    core = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    origin_ecef = np.array([core.x.value, core.y.value, core.z.value])  # in meters

    # Load ENU antenna positions
    path_to_file = get_package_resource_path('tools21cm', filename)
    ant_enu = np.loadtxt(path_to_file)  # shape (N_ant, 3)

    if verbose:
        print(f'{subarray_type} contains {ant_enu.shape[0]} antennae.')

    # Handle coordinate transformations
    if unit.lower() == 'enu':
        # local ENU positions
        return ant_enu * u.m  
    elif unit.lower() == 'ecef':
        # Convert ENU to ECEF
        if verbose:
            print('ENU -> ECEF')
        ant_ecef = enu_to_ecef(ant_enu, lat, lon, origin_ecef)
        return ant_ecef * u.m
    elif unit.lower() == 'xyz':
        # Convert ENU to ECEF to XYZ
        if verbose:
            print('ENU -> ECEF -> XYZ')
        ant_ecef = enu_to_ecef(ant_enu, lat, lon, origin_ecef)
        ant_xyz = ecef_to_xyz(ant_ecef, lon, origin_ecef)
        return ant_xyz * u.m
    elif unit in ['latlon', 'geodetic']:
        locations = EarthLocation.from_geocentric(ant_ecef[:, 0], ant_ecef[:, 1], ant_ecef[:, 2], unit=u.m)
        return locations.lat.deg, locations.lon.deg, locations.height.to(u.m).value
    else:
        raise ValueError(f"Unsupported unit '{unit}'. Must be one of 'enu', 'ecef', or 'xyz'.")

def get_latest_SKA_Low_layout(
                subarray_type='AA*',
                custom_stations=None,
                add_stations=None,
                exclude_stations=None,
                external_telescopes=None,
            ):
    """
    Retrieves the latest SKA Low layout using the official ska-ost-array-config tool.

    This function acts as a wrapper for the SKAO library, providing a convenient
    way to access up-to-date and standardized array configurations.

    Parameters
    ----------
    subarray_type : str, optional
        Name of the subarray configuration.
    custom_stations : list, optional
        A list of station names to create a custom subarray.
    add_stations : list, optional
        A list of station names to add to the specified subarray.
    exclude_stations : list, optional
        A list of station names to exclude from the specified subarray.
    external_telescopes : list, optional
        A list of external telescope configurations to include.

    Returns
    -------
    ska_ost_array_config.array_config.LowSubArray or None
        An object representing the SKA Low subarray configuration, or None if
        the required `ska_ost_array_config` library is not installed.
    """
    try:
        from ska_ost_array_config.array_config import LowSubArray, filter_array_by_distance
    except:
        print(f'Install the official tool from SKAO provided at')
        print(f'https://gitlab.com/ska-telescope/ost/ska-ost-array-config')
        return None
    
    low_all = LowSubArray(
                subarray_type=subarray_type,
                custom_stations=custom_stations,
                add_stations=add_stations,
                exclude_stations=exclude_stations,
                external_telescopes=external_telescopes,
            )
    return low_all

def SKA1_LowConfig_Sept2016():
	X = [1.167655398999999932e+02,1.167639199000000048e+02,1.167651233999999931e+02,1.167632707000000067e+02,1.167647362000000015e+02,1.167642596999999967e+02,1.167642823999999990e+02,1.167643800000000027e+02,1.167654335999999944e+02,1.167631211000000064e+02,1.167635745000000043e+02,1.167658808000000050e+02,1.167644358000000011e+02,1.167647771000000034e+02,1.167640540000000016e+02,1.167626123000000007e+02,1.167648029000000065e+02,1.167658351999999979e+02,1.167640291000000019e+02,1.167649085000000042e+02,1.167659199999999942e+02,1.167661626999999953e+02,1.167651634999999999e+02,1.167640078000000017e+02,1.167654026999999957e+02,1.167636229999999955e+02,1.167645437999999984e+02,1.167647060999999979e+02,1.167639855000000040e+02,1.167650722999999999e+02,1.167654382000000055e+02,1.167650862999999930e+02,1.167638584000000037e+02,1.167662325999999950e+02,1.167636825999999957e+02,1.167631401999999952e+02,1.167651603999999992e+02,1.167633533999999997e+02,1.167651051999999936e+02,1.167626254000000046e+02,1.167636904000000015e+02,1.167663275999999968e+02,1.167625929000000014e+02,1.167649674000000033e+02,1.167638366999999988e+02,1.167628655999999978e+02,1.167659002000000044e+02,1.167628385999999949e+02,1.167645657000000057e+02,1.167662167999999951e+02,1.167647726000000006e+02,1.167642471000000057e+02,1.167633445000000023e+02,1.167655101999999943e+02,1.167664458999999937e+02,1.167629455999999948e+02,1.167648417000000052e+02,1.167635398999999978e+02,1.167644183000000027e+02,1.167656886000000043e+02,1.167631796999999949e+02,1.167634831999999960e+02,1.167636839999999978e+02,1.167632454000000024e+02,1.167639195000000001e+02,1.167643659000000014e+02,1.167629960000000011e+02,1.167654433999999952e+02,1.167655697999999944e+02,1.167642254000000008e+02,1.167658689000000010e+02,1.167661604999999980e+02,1.167648360999999966e+02,1.167627719999999982e+02,1.167653454000000011e+02,1.167663943999999958e+02,1.167664467999999971e+02,1.167625601000000017e+02,1.167657815000000028e+02,1.167616289000000052e+02,1.167652462000000071e+02,1.167631212999999946e+02,1.167672824999999932e+02,1.167627999000000045e+02,1.167664591000000058e+02,1.167635965999999996e+02,1.167621414999999985e+02,1.167649980000000056e+02,1.167629746000000068e+02,1.167615334999999988e+02,1.167627927000000057e+02,1.167617536999999999e+02,1.167632495000000006e+02,1.167672551999999939e+02,1.167621608999999978e+02,1.167618740000000059e+02,1.167653964000000002e+02,1.167671021999999965e+02,1.167656369000000041e+02,1.167656084999999990e+02,1.167638069999999999e+02,1.167634192000000013e+02,1.167668948000000029e+02,1.167661372000000028e+02,1.167640186000000000e+02,1.167621211000000017e+02,1.167668842000000069e+02,1.167622668000000061e+02,1.167658279000000050e+02,1.167649911999999972e+02,1.167626226000000003e+02,1.167645538999999957e+02,1.167633663000000013e+02,1.167622453000000036e+02,1.167616692000000000e+02,1.167628236999999984e+02,1.167660285000000044e+02,1.167642421999999982e+02,1.167667019000000010e+02,1.167665894000000009e+02,1.167621203000000065e+02,1.167645772999999991e+02,1.167662301000000014e+02,1.167672749000000039e+02,1.167624308999999982e+02,1.167669259999999980e+02,1.167620172999999966e+02,1.167622932000000020e+02,1.167617213999999990e+02,1.167640444999999971e+02,1.167667126000000053e+02,1.167661492999999950e+02,1.167668223999999952e+02,1.167653755000000046e+02,1.167616431999999946e+02,1.167675196000000000e+02,1.167626336999999950e+02,1.167647103000000044e+02,1.167631223000000062e+02,1.167622778000000068e+02,1.167668196999999992e+02,1.167642399000000069e+02,1.167661542000000026e+02,1.167674226999999973e+02,1.167666547000000037e+02,1.167661875000000009e+02,1.167669635999999969e+02,1.167644075000000043e+02,1.167664853999999934e+02,1.167665176999999943e+02,1.167613343000000015e+02,1.167612311999999974e+02,1.167686377999999934e+02,1.167605929000000060e+02,1.167650950999999964e+02,1.167653222000000000e+02,1.167627564999999947e+02,1.167640370000000019e+02,1.167683598999999930e+02,1.167609994999999969e+02,1.167640258999999929e+02,1.167618693999999948e+02,1.167676603999999969e+02,1.167665496000000047e+02,1.167686127000000056e+02,1.167653518000000048e+02,1.167659311999999971e+02,1.167610827000000029e+02,1.167677769000000012e+02,1.167609613999999993e+02,1.167661935999999940e+02,1.167658203999999955e+02,1.167645573000000070e+02,1.167671459999999968e+02,1.167627073000000024e+02,1.167613300999999950e+02,1.167634316999999982e+02,1.167611986999999942e+02,1.167644733000000059e+02,1.167677046999999959e+02,1.167677573999999936e+02,1.167613807000000037e+02,1.167667629999999974e+02,1.167623195999999979e+02,1.167618808000000001e+02,1.167611495000000019e+02,1.167678961999999956e+02,1.167682370999999932e+02,1.167680438000000009e+02,1.167685012000000029e+02,1.167648745000000048e+02,1.167614764000000065e+02,1.167637109000000066e+02,1.167622887999999932e+02,1.167665419000000071e+02,1.167670628000000050e+02,1.167645423999999963e+02,1.167682532000000037e+02,1.167610607000000016e+02,1.167669630000000041e+02,1.167634067000000044e+02,1.167637098000000009e+02,1.167603370000000069e+02,1.167652939000000032e+02,1.167608787000000063e+02,1.167653751000000000e+02,1.167659601000000009e+02,1.167655450999999971e+02,1.167660134000000056e+02,1.167656620000000061e+02,1.167656702999999965e+02,1.167663144999999929e+02,1.167600867000000022e+02,1.167607957999999968e+02,1.167602056000000061e+02,1.167600324000000001e+02,1.167597561000000042e+02,1.167605132999999995e+02,1.167676510000000007e+02,1.167672890999999993e+02,1.167673297999999988e+02,1.167669669999999940e+02,1.167675715000000025e+02,1.167668773000000044e+02,1.167627662999999956e+02,1.167624284999999986e+02,1.167627393000000069e+02,1.167621333000000021e+02,1.167629950999999977e+02,1.167624133000000057e+02,1.167575812000000042e+02,1.167571410999999983e+02,1.167582742000000025e+02,1.167576853000000057e+02,1.167572466999999961e+02,1.167579644000000059e+02,1.167530505999999946e+02,1.167534117999999950e+02,1.167525791000000055e+02,1.167533348999999987e+02,1.167529681999999980e+02,1.167530614000000071e+02,1.167520465999999999e+02,1.167524496999999997e+02,1.167524927999999989e+02,1.167528934999999990e+02,1.167519147999999944e+02,1.167522364999999951e+02,1.167601496999999995e+02,1.167606990999999965e+02,1.167610568000000058e+02,1.167604848999999945e+02,1.167608436999999952e+02,1.167602442999999965e+02,1.167642048999999957e+02,1.167634233999999935e+02,1.167637744999999967e+02,1.167634227000000067e+02,1.167641981999999956e+02,1.167637390000000011e+02,1.167711220000000054e+02,1.167707261000000045e+02,1.167702907999999979e+02,1.167708843000000059e+02,1.167703810000000004e+02,1.167709195000000051e+02,1.167793211999999983e+02,1.167791813999999988e+02,1.167795291999999989e+02,1.167788500999999997e+02,1.167795128000000062e+02,1.167791201000000001e+02,1.167703630999999973e+02,1.167699891999999977e+02,1.167696118000000070e+02,1.167703549000000010e+02,1.167700985000000031e+02,1.167704837999999938e+02,1.167720829999999950e+02,1.167714744000000024e+02,1.167718145000000050e+02,1.167712388000000061e+02,1.167721587000000056e+02,1.167718593999999968e+02,1.167699709999999982e+02,1.167696284999999961e+02,1.167690852000000064e+02,1.167701157999999992e+02,1.167692767000000060e+02,1.167697275000000019e+02,1.167618649000000062e+02,1.167613391999999948e+02,1.167621651999999983e+02,1.167616862999999938e+02,1.167618862000000064e+02,1.167613223999999974e+02,1.167587488000000064e+02,1.167593112999999931e+02,1.167587160000000068e+02,1.167582898999999941e+02,1.167590691000000049e+02,1.167587759000000034e+02,1.167717769000000061e+02,1.167721822000000031e+02,1.167717075999999992e+02,1.167723117000000030e+02,1.167713075000000060e+02,1.167720742999999999e+02,1.167906677000000002e+02,1.167910262000000046e+02,1.167903222000000056e+02,1.167913484999999980e+02,1.167906526000000014e+02,1.167910011000000026e+02,1.168116744999999952e+02,1.168121998999999960e+02,1.168112332000000038e+02,1.168120439000000061e+02,1.168116773999999936e+02,1.168111896000000058e+02,1.168261114999999961e+02,1.168264826999999997e+02,1.168260810000000021e+02,1.168267996999999951e+02,1.168264779999999945e+02,1.168260118999999975e+02,1.167491272000000038e+02,1.167490547000000021e+02,1.167496873000000051e+02,1.167486614000000031e+02,1.167492924000000016e+02,1.167488723999999962e+02,1.167369383999999997e+02,1.167374566000000016e+02,1.167372028000000057e+02,1.167376328999999942e+02,1.167367779000000070e+02,1.167372457999999966e+02,1.167286485000000056e+02,1.167281763999999953e+02,1.167290074000000004e+02,1.167286519000000027e+02,1.167288958999999977e+02,1.167282369999999929e+02,1.167296735000000041e+02,1.167300575999999950e+02,1.167304058999999938e+02,1.167300922999999955e+02,1.167297104999999959e+02,1.167300098999999989e+02,1.167478811999999948e+02,1.167473976999999934e+02,1.167478698999999978e+02,1.167473075999999992e+02,1.167483165999999954e+02,1.167475354000000038e+02,1.167851002000000022e+02,1.167855277999999970e+02,1.167848766999999981e+02,1.167852766999999972e+02,1.167857920000000007e+02,1.167852249999999970e+02,1.167843351000000069e+02,1.167849020000000024e+02,1.167845193999999935e+02,1.167841326000000066e+02,1.167849202999999960e+02,1.167845937000000021e+02,1.167739720999999946e+02,1.167735765000000043e+02,1.167743154000000061e+02,1.167745857000000029e+02,1.167742106000000035e+02,1.167738521999999932e+02,1.167515432999999945e+02,1.167522850000000005e+02,1.167513538999999980e+02,1.167517880999999988e+02,1.167514498999999972e+02,1.167520839000000024e+02,1.167193486999999976e+02,1.167196165000000008e+02,1.167188220999999970e+02,1.167192399999999992e+02,1.167194779000000011e+02,1.167189309999999978e+02,1.168347370999999981e+02,1.168343480000000056e+02,1.168353120999999959e+02,1.168347631000000035e+02,1.168344672000000060e+02,1.168351963000000069e+02,1.168641578999999950e+02,1.168636118999999951e+02,1.168638423000000017e+02,1.168646104999999977e+02,1.168636347999999998e+02,1.168643040999999982e+02,1.168672039000000069e+02,1.168667745000000053e+02,1.168662077999999980e+02,1.168671252000000038e+02,1.168667427000000032e+02,1.168663363999999945e+02,1.169401076999999987e+02,1.169392885999999976e+02,1.169397308000000066e+02,1.169393103999999965e+02,1.169399659999999983e+02,1.169396636000000029e+02,1.169922322999999977e+02,1.169916867999999965e+02,1.169920481999999993e+02,1.169926363999999950e+02,1.169919191000000041e+02,1.169925288000000023e+02,1.170158578000000063e+02,1.170155881999999963e+02,1.170162531999999942e+02,1.170156053000000043e+02,1.170159924999999959e+02,1.170151704000000024e+02,1.171008345000000048e+02,1.171003946000000013e+02,1.171014442999999972e+02,1.171006011999999998e+02,1.171009887999999961e+02,1.171006102000000055e+02,1.166934693999999979e+02,1.166936476999999996e+02,1.166940412999999950e+02,1.166933202000000023e+02,1.166937032000000016e+02,1.166932108999999969e+02,1.166668403999999981e+02,1.166672183000000018e+02,1.166674160999999970e+02,1.166662459000000069e+02,1.166667465999999962e+02,1.166667281999999943e+02,1.166218060000000065e+02,1.166212838000000005e+02,1.166209024999999997e+02,1.166215547999999984e+02,1.166213153000000062e+02,1.166217019000000050e+02,1.166546550000000053e+02,1.166543053999999984e+02,1.166546701999999982e+02,1.166541229000000044e+02,1.166550110999999958e+02,1.166544358000000017e+02,1.166724451999999985e+02,1.166721744000000029e+02,1.166726361999999995e+02,1.166730036000000013e+02,1.166721143000000041e+02,1.166726134999999971e+02,1.167350928999999979e+02,1.167346517999999946e+02,1.167350832999999994e+02,1.167355045000000047e+02,1.167345238999999992e+02,1.167350732999999963e+02,1.167249632999999989e+02,1.167244781999999930e+02,1.167253556000000003e+02,1.167248093999999980e+02,1.167243430000000046e+02,1.167250602999999956e+02,1.167629117000000036e+02,1.167625574999999998e+02,1.167622003000000035e+02,1.167629697999999934e+02,1.167626113000000032e+02,1.167623293000000047e+02,1.168051832000000019e+02,1.168048251000000022e+02,1.168052279000000055e+02,1.168048880000000054e+02,1.168056073999999995e+02,1.168051392999999933e+02,1.167246454000000000e+02,1.167254413999999940e+02,1.167249869999999987e+02,1.167253591000000057e+02,1.167244720999999998e+02,1.167250108000000068e+02,1.166800132999999988e+02,1.166804197000000016e+02,1.166797939000000071e+02,1.166802069999999958e+02,1.166798232999999954e+02,1.166802390999999943e+02,1.166251826000000023e+02,1.166254959000000042e+02,1.166252036000000061e+02,1.166248264000000034e+02,1.166254433000000006e+02,1.166248820999999936e+02,1.165451233999999943e+02,1.165456468000000001e+02,1.165447980999999942e+02,1.165452092000000022e+02,1.165455271000000010e+02,1.165448236999999949e+02,1.164524468999999982e+02,1.164531216999999970e+02,1.164525555999999966e+02,1.164530242000000015e+02,1.164521976000000052e+02,1.164525842999999981e+02]
	Y = [-2.682598211999999904e+01,-2.682400458000000043e+01,-2.682519844000000120e+01,-2.682516319999999865e+01,-2.682463766000000049e+01,-2.682325862000000072e+01,-2.682416999000000146e+01,-2.682506289000000166e+01,-2.682419533999999928e+01,-2.682429412999999840e+01,-2.682428184000000115e+01,-2.682362068000000122e+01,-2.682555310000000048e+01,-2.682533563000000143e+01,-2.682459854000000021e+01,-2.682468756999999826e+01,-2.682399616000000009e+01,-2.682414453000000165e+01,-2.682512807999999893e+01,-2.682491591000000142e+01,-2.682458807000000078e+01,-2.682516717000000028e+01,-2.682559725999999856e+01,-2.682355978000000007e+01,-2.682351788000000070e+01,-2.682333347999999873e+01,-2.682367720999999960e+01,-2.682617452999999941e+01,-2.682568058000000022e+01,-2.682371811999999878e+01,-2.682539298000000016e+01,-2.682320405999999835e+01,-2.682607063000000025e+01,-2.682416668999999843e+01,-2.682531300000000130e+01,-2.682567879999999860e+01,-2.682466116999999883e+01,-2.682474063000000086e+01,-2.682620428999999973e+01,-2.682423371000000145e+01,-2.682458619000000155e+01,-2.682449107000000055e+01,-2.682524525999999909e+01,-2.682438176000000141e+01,-2.682640251000000120e+01,-2.682386629999999883e+01,-2.682575495000000032e+01,-2.682494529999999955e+01,-2.682298227000000068e+01,-2.682557407999999910e+01,-2.682341861000000094e+01,-2.682596995999999834e+01,-2.682382490000000175e+01,-2.682451285999999868e+01,-2.682485810000000015e+01,-2.682541349999999980e+01,-2.682577849000000114e+01,-2.682582645000000099e+01,-2.682448198999999889e+01,-2.682504806000000031e+01,-2.682605715000000046e+01,-2.682624136000000092e+01,-2.682495975000000143e+01,-2.682340203000000045e+01,-2.682303145999999927e+01,-2.682645225999999994e+01,-2.682465270999999873e+01,-2.682316336999999962e+01,-2.682382532000000097e+01,-2.682381442999999877e+01,-2.682541508999999991e+01,-2.682381893000000161e+01,-2.682648870999999957e+01,-2.682568999000000076e+01,-2.682494098000000093e+01,-2.682617564999999971e+01,-2.682650836999999910e+01,-2.682260188000000056e+01,-2.682657439000000110e+01,-2.682384838999999843e+01,-2.682734338999999935e+01,-2.682666532000000004e+01,-2.682346752000000123e+01,-2.682644618999999864e+01,-2.682314494999999965e+01,-2.682688522000000120e+01,-2.682546539999999879e+01,-2.682695037999999954e+01,-2.682285707999999858e+01,-2.682422313000000003e+01,-2.682686624000000108e+01,-2.682455841999999890e+01,-2.682217298999999855e+01,-2.682585337000000081e+01,-2.682416921000000087e+01,-2.682327722000000136e+01,-2.682272323000000114e+01,-2.682427994000000027e+01,-2.682703904999999978e+01,-2.682629731000000106e+01,-2.682221824000000154e+01,-2.682276292999999967e+01,-2.682564607000000123e+01,-2.682255358999999828e+01,-2.682263542000000101e+01,-2.682296638000000044e+01,-2.682385968000000176e+01,-2.682505448999999942e+01,-2.682239905999999863e+01,-2.682236118000000147e+01,-2.682598491999999979e+01,-2.682694784999999982e+01,-2.682718923999999916e+01,-2.682341097999999846e+01,-2.682577828000000153e+01,-2.682354013999999864e+01,-2.682622294999999824e+01,-2.682201411999999863e+01,-2.682340513000000115e+01,-2.682535036999999889e+01,-2.682373722999999899e+01,-2.682234714999999881e+01,-2.682342668000000074e+01,-2.682481891999999846e+01,-2.682653613000000092e+01,-2.682617013000000128e+01,-2.682651025999999916e+01,-2.682623596999999904e+01,-2.682541223000000130e+01,-2.682718982999999824e+01,-2.682415316999999888e+01,-2.682672155999999930e+01,-2.682295558999999940e+01,-2.682224409000000165e+01,-2.682507542999999828e+01,-2.682533616999999992e+01,-2.682301075000000168e+01,-2.682190813000000063e+01,-2.682255216000000075e+01,-2.682589630000000014e+01,-2.682653545999999878e+01,-2.682680214999999890e+01,-2.682704071999999940e+01,-2.682443615000000037e+01,-2.682590628999999893e+01,-2.682288295000000033e+01,-2.682463373999999945e+01,-2.682266231999999917e+01,-2.682271088999999975e+01,-2.682374225999999950e+01,-2.682451061000000081e+01,-2.682550190999999984e+01,-2.682529021999999941e+01,-2.682579734000000116e+01,-2.682117970999999912e+01,-2.682803679999999957e+01,-2.682784591000000063e+01,-2.682792141999999913e+01,-2.682601355999999981e+01,-2.682640912000000100e+01,-2.682096612000000135e+01,-2.682271340000000137e+01,-2.682675263000000143e+01,-2.682770799000000039e+01,-2.682458773999999835e+01,-2.682141498000000013e+01,-2.682734682999999976e+01,-2.682372738999999839e+01,-2.682592112000000029e+01,-2.682337974000000003e+01,-2.682755908000000034e+01,-2.682776266999999848e+01,-2.682851292000000143e+01,-2.682691589999999948e+01,-2.682148846000000120e+01,-2.682291113000000138e+01,-2.682799350999999888e+01,-2.682487697999999909e+01,-2.682135601999999963e+01,-2.682351119999999867e+01,-2.682414156000000105e+01,-2.682344168999999923e+01,-2.682156738000000118e+01,-2.682792622999999921e+01,-2.682684172000000089e+01,-2.682416534000000041e+01,-2.682551428000000016e+01,-2.682470260999999923e+01,-2.682339710000000110e+01,-2.682414079999999856e+01,-2.682773343999999938e+01,-2.682718221000000014e+01,-2.682115804000000026e+01,-2.682736391999999981e+01,-2.682181939999999898e+01,-2.682233542999999898e+01,-2.682761732999999893e+01,-2.682523962000000139e+01,-2.682257747999999964e+01,-2.682528072999999935e+01,-2.682842180999999826e+01,-2.682761299999999949e+01,-2.682469239999999999e+01,-2.682836179999999970e+01,-2.682458576000000150e+01,-2.682768902000000111e+01,-2.682093590999999932e+01,-2.682080716999999837e+01,-2.682141401000000158e+01,-2.682159569999999960e+01,-2.682117011000000062e+01,-2.682120898999999881e+01,-2.682541986000000023e+01,-2.682543757999999912e+01,-2.682500970999999979e+01,-2.682585369999999969e+01,-2.682560360000000088e+01,-2.682521484000000100e+01,-2.682781770000000066e+01,-2.682765479000000042e+01,-2.682721762999999982e+01,-2.682782358000000045e+01,-2.682745917000000091e+01,-2.682748782000000176e+01,-2.682017705000000163e+01,-2.681991547000000153e+01,-2.681976351000000136e+01,-2.681972558000000006e+01,-2.681950172999999893e+01,-2.681922924999999935e+01,-2.682106002000000089e+01,-2.682075517999999903e+01,-2.682063137000000097e+01,-2.682062054000000018e+01,-2.682034479000000005e+01,-2.682030267000000023e+01,-2.682574525000000065e+01,-2.682571809999999957e+01,-2.682539970999999923e+01,-2.682533873999999940e+01,-2.682530981000000025e+01,-2.682497726000000071e+01,-2.683431587000000107e+01,-2.683423445000000029e+01,-2.683386186999999978e+01,-2.683379150000000024e+01,-2.683370172000000053e+01,-2.683333486000000079e+01,-2.682900988999999825e+01,-2.682898022999999910e+01,-2.682891340999999841e+01,-2.682872649999999837e+01,-2.682863455000000030e+01,-2.682831961999999848e+01,-2.683235669999999828e+01,-2.683235249999999894e+01,-2.683221191000000161e+01,-2.683203243999999899e+01,-2.683189926000000014e+01,-2.683178841000000148e+01,-2.683354856000000055e+01,-2.683354453000000106e+01,-2.683336839000000040e+01,-2.683325586999999857e+01,-2.683298090999999985e+01,-2.683279832999999925e+01,-2.683009728000000038e+01,-2.682979153999999866e+01,-2.682971811999999900e+01,-2.682946553000000023e+01,-2.682936374000000157e+01,-2.682908632999999909e+01,-2.682615137999999888e+01,-2.682591782000000080e+01,-2.682585566999999926e+01,-2.682583525000000080e+01,-2.682545142000000027e+01,-2.682537765000000007e+01,-2.682189475999999928e+01,-2.682176822999999999e+01,-2.682163965000000161e+01,-2.682140138999999834e+01,-2.682131996000000029e+01,-2.682104471999999973e+01,-2.681601882000000003e+01,-2.681562856000000039e+01,-2.681555425999999898e+01,-2.681550478000000126e+01,-2.681523649000000020e+01,-2.681514360999999980e+01,-2.681117181000000116e+01,-2.681110182999999836e+01,-2.681092562999999984e+01,-2.681074687999999995e+01,-2.681045250000000024e+01,-2.681043869000000157e+01,-2.684394698999999918e+01,-2.684366272000000109e+01,-2.684360498999999933e+01,-2.684348559000000023e+01,-2.684327508999999878e+01,-2.684296827000000008e+01,-2.684975404000000054e+01,-2.684962038000000106e+01,-2.684925349999999966e+01,-2.684915399000000136e+01,-2.684908173999999903e+01,-2.684886159999999933e+01,-2.684866896000000125e+01,-2.684854895000000141e+01,-2.684848214999999882e+01,-2.684839303000000044e+01,-2.684806247999999940e+01,-2.684790263000000010e+01,-2.683634530000000140e+01,-2.683616165000000109e+01,-2.683609420000000156e+01,-2.683577632000000079e+01,-2.683574933999999956e+01,-2.683568570999999991e+01,-2.681023237999999864e+01,-2.680995352000000054e+01,-2.680979280000000031e+01,-2.680959315999999859e+01,-2.680945514000000074e+01,-2.680945385000000059e+01,-2.681152597000000171e+01,-2.681109842000000043e+01,-2.681091406000000177e+01,-2.681087983000000108e+01,-2.681082273000000171e+01,-2.681059630000000027e+01,-2.681849758999999978e+01,-2.681848799000000128e+01,-2.681825845999999913e+01,-2.681799621000000045e+01,-2.681796736000000081e+01,-2.681780759999999830e+01,-2.683374708999999925e+01,-2.683347982000000087e+01,-2.683342161999999931e+01,-2.683321420000000046e+01,-2.683294139999999928e+01,-2.683288694999999890e+01,-2.685609529999999978e+01,-2.685607410000000073e+01,-2.685574847000000176e+01,-2.685559479000000138e+01,-2.685548492999999937e+01,-2.685518160000000165e+01,-2.688073625999999905e+01,-2.688057617000000121e+01,-2.688033877000000160e+01,-2.688021957999999856e+01,-2.688001381999999850e+01,-2.687992913000000073e+01,-2.682014854999999898e+01,-2.682006489000000116e+01,-2.681983793999999932e+01,-2.681970066999999958e+01,-2.681951644000000101e+01,-2.681937909999999903e+01,-2.680705449999999956e+01,-2.680703232999999841e+01,-2.680677593000000059e+01,-2.680675068000000039e+01,-2.680671248000000162e+01,-2.680636044999999967e+01,-2.679281135000000091e+01,-2.679269800000000146e+01,-2.679269620999999901e+01,-2.679247190999999972e+01,-2.679234492000000145e+01,-2.679223100000000102e+01,-2.678285982999999959e+01,-2.678263645999999909e+01,-2.678258462000000151e+01,-2.678244409999999931e+01,-2.678225374000000159e+01,-2.678213682000000162e+01,-2.678450671999999955e+01,-2.678423906000000088e+01,-2.678418601999999993e+01,-2.678408510999999947e+01,-2.678376002999999983e+01,-2.678372956000000116e+01,-2.686176267999999823e+01,-2.686159646999999850e+01,-2.686155339999999825e+01,-2.686125592999999867e+01,-2.686094903000000045e+01,-2.686090598999999912e+01,-2.685878642999999855e+01,-2.685870578000000108e+01,-2.685834261000000112e+01,-2.685829027999999852e+01,-2.685805446999999901e+01,-2.685797243000000023e+01,-2.690680860999999879e+01,-2.690652953000000025e+01,-2.690646678999999963e+01,-2.690629901999999873e+01,-2.690614097999999998e+01,-2.690612716000000049e+01,-2.683064116999999982e+01,-2.683061409999999825e+01,-2.683051693000000171e+01,-2.683025639999999967e+01,-2.683021914000000052e+01,-2.683002687999999836e+01,-2.679902736000000019e+01,-2.679866777000000155e+01,-2.679863870000000148e+01,-2.679850684000000172e+01,-2.679821430000000149e+01,-2.679820227000000088e+01,-2.669732177999999934e+01,-2.669708801999999892e+01,-2.669691269999999861e+01,-2.669673155999999992e+01,-2.669667824000000067e+01,-2.669667331999999860e+01,-2.669055241999999950e+01,-2.669039417000000114e+01,-2.669031374000000056e+01,-2.669008061999999981e+01,-2.668998430999999982e+01,-2.668974898000000096e+01,-2.686364933999999849e+01,-2.686332064999999858e+01,-2.686328963000000059e+01,-2.686314137000000102e+01,-2.686286849000000032e+01,-2.686280413000000067e+01,-2.689933818000000088e+01,-2.689931557999999967e+01,-2.689901238999999933e+01,-2.689899700000000138e+01,-2.689886667000000031e+01,-2.689853735000000157e+01,-2.686551456000000115e+01,-2.686537586000000033e+01,-2.686524561999999960e+01,-2.686514601000000013e+01,-2.686482914000000122e+01,-2.686470904999999831e+01,-2.694980057000000073e+01,-2.694974299000000073e+01,-2.694945107999999934e+01,-2.694929703000000032e+01,-2.694917823999999840e+01,-2.694914685999999904e+01,-2.701275911999999835e+01,-2.701249221000000134e+01,-2.701237267999999858e+01,-2.701226611999999960e+01,-2.701213879999999889e+01,-2.701183478000000093e+01,-2.706405310999999969e+01,-2.706377415000000042e+01,-2.706373506999999989e+01,-2.706359356000000105e+01,-2.706347915999999998e+01,-2.706318397000000076e+01,-2.712774939999999901e+01,-2.712761601000000056e+01,-2.712752824000000018e+01,-2.712736265999999929e+01,-2.712729809000000003e+01,-2.712713915999999870e+01,-2.675105887000000138e+01,-2.675091735999999898e+01,-2.675086263000000031e+01,-2.675068531999999877e+01,-2.675056633999999889e+01,-2.675031407000000172e+01,-2.672196934999999840e+01,-2.672182176999999825e+01,-2.672161184999999861e+01,-2.672149780000000163e+01,-2.672145048000000145e+01,-2.672108151000000120e+01,-2.671301005999999845e+01,-2.671294577999999831e+01,-2.671276973000000154e+01,-2.671246972999999869e+01,-2.671244053999999934e+01,-2.671240137999999931e+01,-2.668502140000000011e+01,-2.668492387999999949e+01,-2.668475241000000153e+01,-2.668455841000000106e+01,-2.668435021000000162e+01,-2.668422316999999921e+01,-2.666182192999999856e+01,-2.666167061000000160e+01,-2.666138424000000029e+01,-2.666136020000000073e+01,-2.666107184999999902e+01,-2.666097462999999834e+01,-2.670516586000000103e+01,-2.670502262999999843e+01,-2.670481610999999944e+01,-2.670474630000000005e+01,-2.670460080999999875e+01,-2.670440535999999909e+01,-2.660137166999999891e+01,-2.660115099000000072e+01,-2.660097515000000001e+01,-2.660081122000000065e+01,-2.660076065000000156e+01,-2.660061336000000054e+01]
	return np.vstack((np.array(X),np.array(Y))).T
    