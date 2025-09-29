from scipy.integrate import quad, odeint
try:
    from scipy.integrate import trapz, cumtrapz, simps
except:
    from scipy.integrate import trapezoid as trapz
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    from scipy.integrate import simpson as simps

from scipy.interpolate import splrep, splev, interp1d
from scipy.special import gamma, erf
from scipy.signal import savgol_filter, windows
from scipy.optimize import fsolve
from scipy.ndimage import gaussian_filter

import numpy as np

def numpy_product(*args, **kwargs):
    try:
        return np.prod(*args, **kwargs)
    except:
        return np.product(*args, **kwargs)