# CubeMap_cython.pyx

import numpy as np
cimport numpy as np
from cython cimport bint 

def CubeMap(np.ndarray[np.int64_t, ndim=3] arr, bint multi_marker=True):
    cdef int nx, ny, nz, Nx, Ny, Nz, i, j, k
    cdef np.ndarray[np.int64_t, ndim=3] cubemap
    cdef tuple markers

    nx = <int> np.PyArray_DIM(arr, 0)
    ny = <int> np.PyArray_DIM(arr, 1)
    nz = <int> np.PyArray_DIM(arr, 2)
    Nx, Ny, Nz = int(2 * nx), int(2 * ny), int(2 * nz)
    
    cubemap = np.empty((Nx, Ny, Nz), dtype=int)

    if multi_marker:
        markers = (1, 2, 3, 4)
    else:
        markers = (1, 1, 1, 1)

    # Vertices
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if arr[i, j, k]:
                    cubemap[i * 2, j * 2, k * 2] = markers[0]

    # Edges
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    if cubemap[(i - 1), j, k] == markers[0] and cubemap[(i + 1) % Nx, j, k] == markers[0]:
                        cubemap[i, j, k] = markers[1]
                    elif cubemap[i, (j - 1), k] == markers[0] and cubemap[i, (j + 1) % Ny, k] == markers[0]:
                        cubemap[i, j, k] = markers[1]
                    elif cubemap[i, j, (k - 1)] == markers[0] and cubemap[i, j, (k + 1) % Nz] == markers[0]:
                        cubemap[i, j, k] = markers[1]

    # Faces
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    if cubemap[(i - 1), j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1] and cubemap[i, (j - 1), k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1]:
                        cubemap[i, j, k] = markers[2]
                    elif cubemap[i, (j - 1), k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1] and cubemap[i, j, (k - 1)] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1]:
                        cubemap[i, j, k] = markers[2]
                    elif cubemap[i, j, (k - 1)] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1] and cubemap[(i - 1), j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1]:
                        cubemap[i, j, k] = markers[2]

    # Cubes
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    if cubemap[(i - 1), j, k] == markers[2] and cubemap[(i + 1) % Nx, j, k] == markers[2]:
                        if cubemap[i, (j - 1), k] == markers[2] and cubemap[i, (j + 1) % Ny, k] == markers[2]:
                            if cubemap[i, j, (k - 1)] == markers[2] and cubemap[i, j, (k + 1) % Nz] == markers[2]:
                                cubemap[i, j, k] = markers[3]
    
    return cubemap
