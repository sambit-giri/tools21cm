import numpy as np
from numba import jit, prange

@njit(parallel=True)
def CubeMap(arr, multi_marker=True):
	nx, ny, nz = arr.shape
	Nx, Ny, Nz = 2*nx,2*ny,2*nz#2*nx-1,2*ny-1,2*nz-1
	cubemap    = np.zeros((Nx,Ny,Nz))
	markers    = 1, 1, 1, 1
	if multi_marker: markers = 1, 2, 3, 4
	## Vertices
	for i in prange(nx):
		for j in range(ny):
			for k in range(nz):
				if arr[i,j,k]: cubemap[i*2,j*2,k*2] = markers[0]

	## Edges 
	for i in prange(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[0] and cubemap[(i+1)%Nx,j,k]==markers[0]: cubemap[i,j,k] = markers[1]
					elif cubemap[i,(j-1),k]==markers[0] and cubemap[i,(j+1)%Ny,k]==markers[0]: cubemap[i,j,k] = markers[1]
					elif cubemap[i,j,(k-1)]==markers[0] and cubemap[i,j,(k+1)%Nz]==markers[0]: cubemap[i,j,k] = markers[1]

	## Faces 
	for i in prange(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[1] and cubemap[(i+1)%Nx,j,k]==markers[1] and cubemap[i,(j-1),k]==markers[1] and cubemap[i,(j+1)%Ny,k]==markers[1]: cubemap[i,j,k] = markers[2]
					elif cubemap[i,(j-1),k]==markers[1] and cubemap[i,(j+1)%Ny,k]==markers[1] and cubemap[i,j,(k-1)]==markers[1] and cubemap[i,j,(k+1)%Nz]==markers[1]: cubemap[i,j,k] = markers[2]
					elif cubemap[i,j,(k-1)]==markers[1] and cubemap[i,j,(k+1)%Nz]==markers[1] and cubemap[(i-1),j,k]==markers[1] and cubemap[(i+1)%Nx,j,k]==markers[1]: cubemap[i,j,k] = markers[2]
	
	## Cubes
	for i in prange(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1),j,k]==markers[2] and cubemap[(i+1)%Nx,j,k]==markers[2]: 
						if cubemap[i,(j-1),k]==markers[2] and cubemap[i,(j+1)%Ny,k]==markers[2]: 
							if cubemap[i,j,(k-1)]==markers[2] and cubemap[i,j,(k+1)%Nz]==markers[2]: cubemap[i,j,k] = markers[3]

	return cubemap	


@jit
def EulerCharacteristic_seq(A):
	chi = 0;
	nx,ny,nz = A.shape
	for x in prange(nx):
		for y in range(ny):
			for z in range(nz):
				if(A[x,y,z] == 1):
					if (x+y+z)%2 == 0: chi += 1
					else: chi -= 1
	return chi 

