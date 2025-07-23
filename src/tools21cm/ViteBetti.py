# ViteBetti.py

import numpy as np
import os

# --- Optional Numba Support ---
try:
    from numba import jit
    numba_available = True
except ImportError:
    numba_available = False
    # Define a dummy decorator if numba is not available
    def jit(func, *args, **kwargs):
        return func

# --- Optional Cython Support ---
try:
    from .ViteBetti_cython import CubeMap as cython_CubeMap
    cython_available = True
except ImportError:
    cython_CubeMap = None
    cython_available = False

# --- Optional Joblib Support ---
try:
    from joblib import Parallel, delayed
    from multiprocessing import shared_memory
    joblib_available = True
except ImportError:
    joblib_available = False
    Parallel, delayed, shared_memory = None, None, None

# --- Optional PyTorch (GPU) Support ---
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    torch = None

# --- Core Algorithm (Pure Python) ---
def CubeMap(arr, multi_marker=True):
    """
    Generates a cubical complex map from a binary 3D array.
    This is the pure Python implementation which serves as a fallback.
    """
    nx, ny, nz = arr.shape
    Nx, Ny, Nz = 2 * nx, 2 * ny, 2 * nz
    cubemap = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    
    markers = (1, 1, 1, 1)
    if multi_marker:
        markers = (1, 2, 3, 4)

    # Vertices (1)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if arr[i, j, k]:
                    cubemap[i * 2, j * 2, k * 2] = markers[0]

    # Edges (2)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    # Check for neighbors along each axis using explicit periodic boundaries
                    if cubemap[(i - 1) % Nx, j, k] == markers[0] and cubemap[(i + 1) % Nx, j, k] == markers[0]:
                        cubemap[i, j, k] = markers[1]
                    elif cubemap[i, (j - 1) % Ny, k] == markers[0] and cubemap[i, (j + 1) % Ny, k] == markers[0]:
                        cubemap[i, j, k] = markers[1]
                    elif cubemap[i, j, (k - 1) % Nz] == markers[0] and cubemap[i, j, (k + 1) % Nz] == markers[0]:
                        cubemap[i, j, k] = markers[1]

    # Faces (3)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    if (cubemap[(i - 1) % Nx, j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1] and
                        cubemap[i, (j - 1) % Ny, k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1]):
                        cubemap[i, j, k] = markers[2]
                    elif (cubemap[i, (j - 1) % Ny, k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1] and
                          cubemap[i, j, (k - 1) % Nz] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1]):
                        cubemap[i, j, k] = markers[2]
                    elif (cubemap[i, j, (k - 1) % Nz] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1] and
                          cubemap[(i - 1) % Nx, j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1]):
                        cubemap[i, j, k] = markers[2]
    
    # Cubes (4)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if cubemap[i, j, k] == 0:
                    if (cubemap[(i - 1) % Nx, j, k] == markers[2] and cubemap[(i + 1) % Nx, j, k] == markers[2] and
                        cubemap[i, (j - 1) % Ny, k] == markers[2] and cubemap[i, (j + 1) % Ny, k] == markers[2] and
                        cubemap[i, j, (k - 1) % Nz] == markers[2] and cubemap[i, j, (k + 1) % Nz] == markers[2]):
                        cubemap[i, j, k] = markers[3]

    return cubemap

# --- Accelerated Versions ---
CubeMap_numba = jit(CubeMap, nopython=True) if numba_available else None
CubeMap_cython = cython_CubeMap # Assigned from the import attempt above

# --- Joblib Parallel Implementation ---
def _CubeMap_joblib_worker(shm_name, arr_shape, i_start, i_end, arr_for_vertices, markers, stage):
    """Worker function for joblib with the complete and correct logic."""
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cubemap = np.ndarray(arr_shape, dtype=np.int32, buffer=existing_shm.buf)
    
    Nx, Ny, Nz = arr_shape
    nx, ny, nz = arr_for_vertices.shape

    if stage == 'vertices':
        for i in range(i_start, i_end):
            for j in range(ny):
                for k in range(nz):
                    if arr_for_vertices[i, j, k]:
                        cubemap[i * 2, j * 2, k * 2] = markers[0]

    elif stage == 'edges':
        for i in range(i_start, i_end):
            for j in range(Ny):
                for k in range(Nz):
                    if cubemap[i, j, k] == 0:
                        # **FIX**: Added all elif branches
                        if cubemap[(i - 1) % Nx, j, k] == markers[0] and cubemap[(i + 1) % Nx, j, k] == markers[0]:
                            cubemap[i, j, k] = markers[1]
                        elif cubemap[i, (j - 1) % Ny, k] == markers[0] and cubemap[i, (j + 1) % Ny, k] == markers[0]:
                            cubemap[i, j, k] = markers[1]
                        elif cubemap[i, j, (k - 1) % Nz] == markers[0] and cubemap[i, j, (k + 1) % Nz] == markers[0]:
                            cubemap[i, j, k] = markers[1]

    elif stage == 'faces':
        for i in range(i_start, i_end):
            for j in range(Ny):
                for k in range(Nz):
                    if cubemap[i, j, k] == 0:
                        # **FIX**: Added all elif branches
                        if (cubemap[(i - 1) % Nx, j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1] and
                            cubemap[i, (j - 1) % Ny, k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1]):
                            cubemap[i, j, k] = markers[2]
                        elif (cubemap[i, (j - 1) % Ny, k] == markers[1] and cubemap[i, (j + 1) % Ny, k] == markers[1] and
                              cubemap[i, j, (k - 1) % Nz] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1]):
                            cubemap[i, j, k] = markers[2]
                        elif (cubemap[i, j, (k - 1) % Nz] == markers[1] and cubemap[i, j, (k + 1) % Nz] == markers[1] and
                              cubemap[(i - 1) % Nx, j, k] == markers[1] and cubemap[(i + 1) % Nx, j, k] == markers[1]):
                            cubemap[i, j, k] = markers[2]
    
    elif stage == 'cubes':
        for i in range(i_start, i_end):
            for j in range(Ny):
                for k in range(Nz):
                    if cubemap[i, j, k] == 0:
                        # **FIX**: This was already complete
                        if (cubemap[(i - 1) % Nx, j, k] == markers[2] and cubemap[(i + 1) % Nx, j, k] == markers[2] and
                            cubemap[i, (j - 1) % Ny, k] == markers[2] and cubemap[i, (j + 1) % Ny, k] == markers[2] and
                            cubemap[i, j, (k - 1) % Nz] == markers[2] and cubemap[i, j, (k + 1) % Nz] == markers[2]):
                            cubemap[i, j, k] = markers[3]

    existing_shm.close()


def CubeMap_joblib(arr, multi_marker=True, n_jobs=-1):
    """
    Generates a cubical complex map from a binary 3D array using joblib for parallelism.
    """
    if not joblib_available:
        raise ImportError("Joblib or its dependencies are not installed. Cannot use 'joblib' backend.")

    nx, ny, nz = arr.shape
    Nx, Ny, Nz = 2 * nx, 2 * ny, 2 * nz
    cubemap_shape = (Nx, Ny, Nz)
    markers = (1, 2, 3, 4) if multi_marker else (1, 1, 1, 1)

    shm = shared_memory.SharedMemory(create=True, size=np.dtype(np.int32).itemsize * Nx * Ny * Nz)
    shared_cubemap = np.ndarray(cubemap_shape, dtype=np.int32, buffer=shm.buf)
    shared_cubemap[:] = 0

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    try:
        for stage_name, domain_size in [('vertices', nx), ('edges', Nx), ('faces', Nx), ('cubes', Nx)]:
            chunk_size = (domain_size + n_jobs - 1) // n_jobs
            tasks = [
                delayed(_CubeMap_joblib_worker)(
                    shm.name, cubemap_shape, i * chunk_size, 
                    min((i + 1) * chunk_size, domain_size), 
                    arr, markers, stage_name
                )
                for i in range(n_jobs)
            ]
            Parallel(n_jobs=n_jobs)(tasks)
        
        result_cubemap = np.copy(shared_cubemap)

    finally:
        shm.close()
        shm.unlink()

    return result_cubemap

# --- PyTorch GPU Implementation ---
def CubeMap_torch(arr, multi_marker=True):
    """
    Generates a cubical complex map using PyTorch for GPU acceleration.
    """
    if not torch_available:
        raise ImportError("PyTorch is not installed. Cannot use 'torch' backend.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using PyTorch on device: {device}")

    arr_tensor = torch.tensor(arr, dtype=torch.int32, device=device)
    nx, ny, nz = arr_tensor.shape
    Nx, Ny, Nz = 2 * nx, 2 * ny, 2 * nz
    cubemap = torch.zeros((Nx, Ny, Nz), dtype=torch.int32, device=device)

    markers = (1, 1, 1, 1)
    if multi_marker:
        markers = (1, 2, 3, 4)

    # Vertices
    coords = torch.nonzero(arr_tensor, as_tuple=False)
    if coords.shape[0] > 0:
        cubemap[coords[:, 0] * 2, coords[:, 1] * 2, coords[:, 2] * 2] = markers[0]

    # Edges
    mask = cubemap == 0
    edge_mask = (
        (cubemap.roll(1, 0) == markers[0]) & (cubemap.roll(-1, 0) == markers[0]) |
        (cubemap.roll(1, 1) == markers[0]) & (cubemap.roll(-1, 1) == markers[0]) |
        (cubemap.roll(1, 2) == markers[0]) & (cubemap.roll(-1, 2) == markers[0])
    )
    cubemap[mask & edge_mask] = markers[1]

    # Faces
    mask = cubemap == 0
    face_mask = (
        (cubemap.roll(1, 0) == markers[1]) & (cubemap.roll(-1, 0) == markers[1]) &
        (cubemap.roll(1, 1) == markers[1]) & (cubemap.roll(-1, 1) == markers[1]) |
        (cubemap.roll(1, 1) == markers[1]) & (cubemap.roll(-1, 1) == markers[1]) &
        (cubemap.roll(1, 2) == markers[1]) & (cubemap.roll(-1, 2) == markers[1]) |
        (cubemap.roll(1, 2) == markers[1]) & (cubemap.roll(-1, 2) == markers[1]) &
        (cubemap.roll(1, 0) == markers[1]) & (cubemap.roll(-1, 0) == markers[1])
    )
    cubemap[mask & face_mask] = markers[2]

    # Cubes
    mask = cubemap == 0
    cube_mask = (
        (cubemap.roll(1, 0) == markers[2]) & (cubemap.roll(-1, 0) == markers[2]) &
        (cubemap.roll(1, 1) == markers[2]) & (cubemap.roll(-1, 1) == markers[2]) &
        (cubemap.roll(1, 2) == markers[2]) & (cubemap.roll(-1, 2) == markers[2])
    )
    cubemap[mask & cube_mask] = markers[3]

    return cubemap.cpu().numpy()