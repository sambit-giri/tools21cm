'''
Tools21cm is a Python package for analysing simulations of the 21-cm signal
during Epoch of Reionization and Cosmic Dawn.
We incorpoarte its predecessor, c2raytools, into this package.

You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import tools21cm as t2c
>>> help(t2c.calc_dt)
'''

import sys
### From c2raytools
#Import sub-modules 
#import conv
from .conv import * #set_sim_constants
from .const import *
from .beam_convolve import *
from .density_file import *
from .irate_file import *
from .nbody_file import *
from .nbody_tools import *
from .xfrac_file import *
from .temper_file import *
from .vel_file import *
from .source_file import *
from .halo_list import *
from .statistics import *
from .power_spectrum import *
from .tau import *
from .lightcone import *
from .pv_mpm import *
from .temperature import *
from .helper_functions import *
from .cosmo import *
from .plotting import *
from .power_legendre import *
from .deprecated import *
from .angular_coordinates import *
from .smoothing import *
from .power_spectrum_noise import *
from .gaussian_random_field import *
from .corr_function import *
# from . import power_spect_fast

from .line_intensity import *
from .lyman_alpha_transmission import *

from .bubble_stats import *

from .foreground_model import *
from .radio_telescope_sensitivity import *
from .radio_telescope_layout import *
from .radio_telescope_calibration import *

#if 'numba' in sys.modules: 
#from numba_functions import *
from .usefuls import *
from .radio_telescope_noise import *
from .superpixels import *
try: from . import segmentation #from .segmentation import *
except: pass 
from .identify_regions import *
from .read_files import *
from .primary_beam import *
from .topology import *
from .box_counting import *
from .fitting_methods import *
from .percolation import *
from .compare_constraints import *
from .astro_tools import *

# from .power_spect_fast import *
### Cosmospectra
# from cosmospectra import * 

from .bispectrum import *
from .position_dependent_power_spectrum import *
from .test_case import *

# Interface to simulation codes
from . import model_21cmfast

#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')
