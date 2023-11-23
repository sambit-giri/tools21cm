'''
Contains functions to create test cases.
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

import pickle, os, sys
from time import time 
from glob import glob

from astropy.convolution import convolve_fft
from skimage.morphology import ball

def paint_profile_in_cube(cube, positions, profile=None, kernel=None):
    assert profile is not None or kernel is not None
    if kernel is None: 
        x = np.arange(cube.shape[0])
        y = np.arange(cube.shape[1])
        z = np.arange(cube.shape[2])
        rx, ry, rz = np.meshgrid(x, y, z, sparse=True)
        rgrid  = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        kernel = profile(rgrid)
    
    positions = positions.squeeze()
    positions = np.array([positions]) if len(positions)==3 else np.array(positions)
    pos_grid = np.zeros_like(cube)
    # pos_grid[positions[:,0],positions[:,1],positions[:,2]] = 1
    for x,y,z in positions:
        pos_grid[x,y,z] = 1
    
    out = convolve_fft(
                       pos_grid, kernel, 
                       boundary='wrap', 
                       normalize_kernel=False, 
                       allow_huge=True
                      )
    return out+cube

def get_cube(n_cells, **kwargs):
    cube = kwargs.get('cube', np.zeros((n_cells,n_cells,n_cells)))
    # xv, yv, zv = np.meshgrid(np.arange(n_cells), np.arange(n_cells), np.arange(n_cells))
    # return cube, np.array([xv, yv, zv])
    return cube

def spherical_vol(r):
    return 4*np.pi/3*r**3
    
def spherical_bubble_model(n_cells, x_ion, r=10, 
                 source_distribution='Poisson',
                 **kwargs
                ):
    ''' 
        Create a cube with spherical bubbles.

        Parameters:
                n_cells (int): number of cells in the cube.
                x_ion (float): fraction of pixels filled with bubbles.
                r = 10 (float or array-like): radius of the bubbles in cell units.
                source_distribution = 'Poisson' (string): The distribution of 
                            bubbles/sources in the cube.

        Returns:
                The cube filled with bubbles.           
    '''
    # Implement working with r distribution
    
    batch_size = kwargs.get('batch_size', 1)
    max_iter   = kwargs.get('max_iter', 1000)
    
    frac_1bub  = spherical_vol(r)/n_cells**3
    if x_ion/(frac_1bub*batch_size)>max_iter:
        print(f'single bubble fraction x batch_size = {frac_1bub*batch_size:.3f}')
        print(f'(target fraction)/(single bubble fraction x batch_size) = {x_ion/(frac_1bub*batch_size):.1f}')
        print(f'batch_size, max_iter = {batch_size}, {max_iter}')
        print('Use a larger batch_size for number of iteration to remain close to the max_iter.')
        return None
    elif x_ion/(frac_1bub*batch_size)<2:
        print(f'single bubble fraction x batch_size = {frac_1bub*batch_size:.3f}')
        print(f'(target fraction)/(single bubble fraction x batch_size) = {x_ion/(frac_1bub*batch_size):.1f}')
        print(f'batch_size, max_iter = {batch_size}, {max_iter}')
        print('Use a smaller batch_size to prevent error in converging to the provided x_ion.')
        return None
    else:
        pass
    
    if source_distribution.lower()=='poisson':
        cube = get_cube(n_cells, **kwargs)
        # xv, yv, zv = np.meshgrid(np.arange(n_cells), np.arange(n_cells), np.arange(n_cells))
        # xyz_list  = np.array([xv, yv, zv]).T
        # np.random.shuffle(xyz_list)
        print('Cube intialized...')
        
        sphr = ball(r)
        cube2 = cube.copy()
        pbar  = tqdm(range(0, (n_cells**3), batch_size), total=max_iter*2)
        
        for batch_start in pbar:
            # batch = xyz_list[batch_start:batch_start + batch_size]
            batch = np.random.randint(0,n_cells,(batch_size,3))
            cube2 = paint_profile_in_cube(cube, batch, profile=None, kernel=sphr)
            cube2 = (cube2>=0.5).astype(float)
            if cube2.mean()>x_ion: break
            cube = cube2.copy()
            pbar.set_postfix({"Filling fraction": f"{cube.mean():.3f}"})
    else:
        print(f'{source_distribution} is not implemented.')
        cube = None
    print(f'done')
    return cube