# uv_map_cython.pyx

cimport numpy as np
# from cython.parallel import parallel, prange
from libc.math cimport pi

# Gridding function: Cython optimized, now accepting z_to_cdist as a parameter
def grid_uv_tracks_with_gains(Nbase, gain_vals, gain_uv_map, 
                              z_to_cdist, double z, int ncells, 
                              double boxsize, bint include_mirror_baselines):
    cdef int i, x, y
    cdef int nb_rows = Nbase.shape[0]
    cdef double theta_max = boxsize / z_to_cdist(z)  # Using the provided z_to_cdist function
    cdef np.ndarray[int, ndim=2] Nb = np.round(Nbase[:, :2] * theta_max).astype(np.int32)

    # Loop over baselines without parallelism
    for i in range(nb_rows):
        x, y = Nb[i]
        if (x >= -ncells / 2 and x < ncells / 2 and y >= -ncells / 2 and y < ncells / 2):
            gain_uv_map[x + ncells // 2, y + ncells // 2, 0] += 1
            gain_uv_map[x + ncells // 2, y + ncells // 2, 1] += gain_vals[i, 0]
            gain_uv_map[x + ncells // 2, y + ncells // 2, 2] += gain_vals[i, 1]

        if include_mirror_baselines:
            gain_uv_map[-x + ncells // 2, -y + ncells // 2, 0] += 1
            gain_uv_map[-x + ncells // 2, -y + ncells // 2, 1] += gain_vals[i, 0]
            gain_uv_map[-x + ncells // 2, -y + ncells // 2, 2] += gain_vals[i, 1]

