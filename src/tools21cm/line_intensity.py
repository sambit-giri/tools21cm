import numpy as np
from glob import glob
import astropy

from .nbody_file import halo_list_to_grid

def Mhi_on_grid(mass, pos_xyz, box_dim, n_grid):
	"""
	Assign HI line intensity to dark matter haloes and put them on a grid.

	Parameters
	----------
	mass : ndarray
		Mass of haloes.
	pos_xyz : ndarray
		Position of the haloes.
	box_dim : float
		Length of simulation in each direction.
	n_grid : int
		Number of grids along each direction.

	Returns
	----------
	lim_map : ndarray
		The line intensity put on grids of shape (nGrid, nGrid, nGrid).
	"""