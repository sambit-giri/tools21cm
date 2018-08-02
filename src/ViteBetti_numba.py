import numpy as np
from scipy.ndimage import filters
from numba import autojit, prange

@autojit
def CubeMap(arr):
	nx, ny, nz = arr.shape
	Nx, Ny, Nz = 2*nx-1,2*ny-1,2*nz-1
	cubemap    = np.zeros((Nx,Ny,Nz))
	## Vertices
	for i in prange(nx):
		for j in xrange(ny):
			for k in xrange(nz):
				if arr[i,j,k]: cubemap[i*2,j*2,k*2] = 1

	## Edges 
	stele = np.zeros((3,3,3)); stele[:,1,1] = 1
	dummy = filters.convolve(cubemap, stele, mode='constant')*(1-cubemap)
	cubemap[dummy==2] = 1
	stele = np.zeros((3,3,3)); stele[1,:,1] = 1
	dummy = filters.convolve(cubemap, stele, mode='constant')
	cubemap[dummy==2] = 1
	stele = np.zeros((3,3,3)); stele[1,1,:] = 1
	dummy = filters.convolve(cubemap, stele, mode='constant')
	cubemap[dummy==2] = 1
	for i in prange(Nx):
		for j in xrange(Ny):
			for k in xrange(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1)%Nx,j,k] and cubemap[(i+1)%Nx,j,k]: cubemap[i,j,k] = 1
					elif cubemap[i,(j-1)%Ny,k] and cubemap[i,(j+1)%Ny,k]: cubemap[i,j,k] = 1
					elif cubemap[i,j,(k-1)%Nz] and cubemap[i,j,(k+1)%Nz]: cubemap[i,j,k] = 1

	## Faces 
	for i in prange(Nx):
		for j in xrange(Ny):
			for k in xrange(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1)%Nx,j,k] and cubemap[(i+1)%Nx,j,k] and cubemap[i,(j-1)%Ny,k]==1 and cubemap[i,(j+1)%Ny,k]: cubemap[i,j,k] = 1
					elif cubemap[i,(j-1)%Ny,k] and cubemap[i,(j+1)%Ny,k] and cubemap[i,j,(k-1)%Nz] and cubemap[i,j,(k+1)%Nz]: cubemap[i,j,k] = 1
					elif cubemap[i,j,(k-1)%Nz] and cubemap[i,j,(k+1)%Nz] and cubemap[(i-1)%Nx,j,k] and cubemap[(i+1)%Nx,j,k]: cubemap[i,j,k] = 1
	
	## Cubes
	for i in prange(Nx):
		for j in xrange(Ny):
			for k in xrange(Nz):
				if cubemap[i,j,k] == 0:
					if cubemap[(i-1)%Nx,j,k] and cubemap[(i+1)%Nx,j,k]: 
						if cubemap[i,(j-1)%Ny,k] and cubemap[i,(j+1)%Ny,k]: 
							if cubemap[i,j,(k-1)%Nz] and cubemap[i,j,(k+1)%Nz]: cubemap[i,j,k] = 1

	return cubemap	


@autojit
def EulerCharacteristic_seq(A):
	chi = 0;
	nx,ny,nz = A.shape
	for x in prange(nx):
		  for y in xrange(ny):
			for z in xrange(nz):
				if(A[x,y,z] == 1):
					if (x+y+z)%2 == 0: chi += 1
					else: chi -= 1
	return chi 

