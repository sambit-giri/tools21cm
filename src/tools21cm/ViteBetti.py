import numpy as np

def CubeMap_torch(arr, multi_marker=True):
	import torch

	# Check if GPU is available and choose the device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Convert the input array to a PyTorch tensor and move it to the selected device
	arr_tensor = torch.tensor(arr, dtype=torch.int32, device=device)

	nx, ny, nz = arr_tensor.shape
	Nx, Ny, Nz = 2*nx, 2*ny, 2*nz
	cubemap = torch.zeros((Nx, Ny, Nz), dtype=torch.int32, device=device)

	# Define markers
	markers = (1, 1, 1, 1)
	if multi_marker:
		markers = (1, 2, 3, 4)

	# Set the vertices
	cubemap[2*arr_tensor.bool()] = markers[0]

	# Convert indices for easier manipulation
	i_indices = torch.arange(Nx, device=device)
	j_indices = torch.arange(Ny, device=device)
	k_indices = torch.arange(Nz, device=device)

	i_indices, j_indices, k_indices = torch.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

	# Define a mask to select non-zero values
	mask = cubemap == 0

	# Edges
	edge_mask_1 = (cubemap[(i_indices-1) % Nx, j_indices, k_indices] == markers[0]) & (cubemap[(i_indices+1) % Nx, j_indices, k_indices] == markers[0])
	edge_mask_2 = (cubemap[i_indices, (j_indices-1) % Ny, k_indices] == markers[0]) & (cubemap[i_indices, (j_indices+1) % Ny, k_indices] == markers[0])
	edge_mask_3 = (cubemap[i_indices, j_indices, (k_indices-1) % Nz] == markers[0]) & (cubemap[i_indices, j_indices, (k_indices+1) % Nz] == markers[0])

	cubemap[mask & (edge_mask_1 | edge_mask_2 | edge_mask_3)] = markers[1]

	# Faces
	face_mask_1 = (cubemap[(i_indices-1) % Nx, j_indices, k_indices] == markers[1]) & (cubemap[(i_indices+1) % Nx, j_indices, k_indices] == markers[1]) & \
					(cubemap[i_indices, (j_indices-1) % Ny, k_indices] == markers[1]) & (cubemap[i_indices, (j_indices+1) % Ny, k_indices] == markers[1])
	face_mask_2 = (cubemap[i_indices, (j_indices-1) % Ny, k_indices] == markers[1]) & (cubemap[i_indices, (j_indices+1) % Ny, k_indices] == markers[1]) & \
					(cubemap[i_indices, j_indices, (k_indices-1) % Nz] == markers[1]) & (cubemap[i_indices, j_indices, (k_indices+1) % Nz] == markers[1])
	face_mask_3 = (cubemap[i_indices, j_indices, (k_indices-1) % Nz] == markers[1]) & (cubemap[i_indices, j_indices, (k_indices+1) % Nz] == markers[1]) & \
					(cubemap[(i_indices-1) % Nx, j_indices, k_indices] == markers[1]) & (cubemap[(i_indices+1) % Nx, j_indices, k_indices] == markers[1])

	cubemap[mask & (face_mask_1 | face_mask_2 | face_mask_3)] = markers[2]

	# Cubes
	cube_mask = (cubemap[(i_indices-1) % Nx, j_indices, k_indices] == markers[2]) & (cubemap[(i_indices+1) % Nx, j_indices, k_indices] == markers[2]) & \
				(cubemap[i_indices, (j_indices-1) % Ny, k_indices] == markers[2]) & (cubemap[i_indices, (j_indices+1) % Ny, k_indices] == markers[2]) & \
				(cubemap[i_indices, j_indices, (k_indices-1) % Nz] == markers[2]) & (cubemap[i_indices, j_indices, (k_indices+1) % Nz] == markers[2])

	cubemap[mask & cube_mask] = markers[3]

	# Move the result back to the CPU and convert to numpy array
	return cubemap.cpu().numpy()

def CubeMap(arr, multi_marker=True):
	nx, ny, nz = arr.shape
	Nx, Ny, Nz = 2*nx,2*ny,2*nz#2*nx-1,2*ny-1,2*nz-1
	cubemap    = np.zeros((Nx,Ny,Nz))
	markers    = 1, 1, 1, 1
	if multi_marker: markers = 1, 2, 3, 4
	## Vertices
	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				if arr[i,j,k]: cubemap[i*2,j*2,k*2] = markers[0]

	## Edges 
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[0] and cubemap[(i+1)%Nx,j,k]==markers[0]: cubemap[i,j,k] = markers[1]
					elif cubemap[i,(j-1),k]==markers[0] and cubemap[i,(j+1)%Ny,k]==markers[0]: cubemap[i,j,k] = markers[1]
					elif cubemap[i,j,(k-1)]==markers[0] and cubemap[i,j,(k+1)%Nz]==markers[0]: cubemap[i,j,k] = markers[1]

	## Faces 
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[1] and cubemap[(i+1)%Nx,j,k]==markers[1] and cubemap[i,(j-1),k]==markers[1] and cubemap[i,(j+1)%Ny,k]==markers[1]: cubemap[i,j,k] = markers[2]
					elif cubemap[i,(j-1),k]==markers[1] and cubemap[i,(j+1)%Ny,k]==markers[1] and cubemap[i,j,(k-1)]==markers[1] and cubemap[i,j,(k+1)%Nz]==markers[1]: cubemap[i,j,k] = markers[2]
					elif cubemap[i,j,(k-1)]==markers[1] and cubemap[i,j,(k+1)%Nz]==markers[1] and cubemap[(i-1),j,k]==markers[1] and cubemap[(i+1)%Nx,j,k]==markers[1]: cubemap[i,j,k] = markers[2]
	
	## Cubes
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[2] and cubemap[(i+1)%Nx,j,k]==markers[2]: 
						if cubemap[i,(j-1),k]==markers[2] and cubemap[i,(j+1)%Ny,k]==markers[2]: 
							if cubemap[i,j,(k-1)]==markers[2] and cubemap[i,j,(k+1)%Nz]==markers[2]: cubemap[i,j,k] = markers[3]

	return cubemap	

def EulerCharacteristic_seq(A):
	chi = 0
	nx,ny,nz = A.shape
	for x in range(nx):
		for y in range(ny):
			for z in range(nz):
				if(A[x,y,z] == 1):
					if (x+y+z)%2 == 0: chi += 1
					else: chi -= 1
	return chi 


import jax
import jax.numpy as jnp

@jax.jit
def CubeMap_jax(arr, multi_marker=True):
    markers = (1, 1, 1, 1)
    if multi_marker:
        markers = (1, 2, 3, 4)

    nx, ny, nz = arr.shape
    Nx, Ny, Nz = 2*nx, 2*ny, 2*nz
    cubemap = jnp.zeros((Nx, Ny, Nz), dtype=jnp.int32)

    # Step 1: Vertices
    coords = jnp.argwhere(arr == 1)
    vert_indices = coords * 2
    cubemap = cubemap.at[tuple(vert_indices.T)].set(markers[0])

    # Step 2: Edges (use jnp.roll for periodic neighbor checking)
    mask = cubemap == 0
    m0 = markers[0]
    m1 = markers[1]

    def edge_mask_axis(cmap, axis):
        left = jnp.roll(cmap, 1, axis=axis)
        right = jnp.roll(cmap, -1, axis=axis)
        return (left == m0) & (right == m0)

    edge_mask = edge_mask_axis(cubemap, 0) | edge_mask_axis(cubemap, 1) | edge_mask_axis(cubemap, 2)
    cubemap = jnp.where(mask & edge_mask, m1, cubemap)

    # Further steps: similar idea, but masks get more complex

    return cubemap
