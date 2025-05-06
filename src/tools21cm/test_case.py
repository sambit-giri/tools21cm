'''
Contains functions to create test cases.
'''

import numpy as np
from tqdm import tqdm
from skimage.morphology import ball

def paint_kernel(cube, positions, kernel):
    '''
    Applies a kernel at specified positions in a 3D cube.
    
    Parameters:
        cube (np.ndarray): The 3D array where the kernel will be applied.
        positions (np.ndarray): An array of shape (N, 3) containing positions in the cube.
        kernel (np.ndarray): The kernel to apply (should be a 3D array).
    '''
    kernel_shape = kernel.shape
    kernel_radius = np.array([k // 2 for k in kernel_shape])
    
    for pos in positions:
        # Roll the cube to center the kernel at the specified position
        rolled_cube = np.roll(cube, shift=-pos + kernel_radius, axis=(0, 1, 2))
        
        # Add the kernel to the rolled cube within the bounds
        rolled_cube[:kernel_shape[0], :kernel_shape[1], :kernel_shape[2]] += kernel
        
        # Roll it back to the original position
        cube += np.roll(rolled_cube, shift=pos - kernel_radius, axis=(0, 1, 2))

    return None
            
def spherical_bubble_model(n_cells, x_ion, r=10, source_distribution='Poisson', **kwargs):
    """
    Create a 3D cube with randomly placed spherical bubbles, filling a target ionization fraction.
    
    Parameters:
    n_cells (int): Number of cells in each dimension of the cube.
    x_ion (float): Target fraction of cells to be filled with bubbles (ionized).
    r (float): Radius of the spherical bubbles in grid cell units.
    source_distribution (str): Distribution type for placing bubbles; default is 'Poisson'.
                               Accepted values: ['Poisson'].
    batch_size (int, optional): Number of bubbles placed per iteration. Default is 1.
    max_iter (int, optional): Maximum number of iterations to achieve target ionization. Default is 1000.
    
    Returns:
    np.ndarray: 3D array representing the cube filled with bubbles, or None if conditions are not met.
    """
    # Validate source_distribution
    valid_distributions = ['Poisson']
    if source_distribution not in valid_distributions:
        print(f"Error: '{source_distribution}' is not implemented. Available options: {valid_distributions}")
        return None
    
    # Parameters
    batch_size = kwargs.get('batch_size', 1)
    max_iter = kwargs.get('max_iter', 1000)
    
    cube = kwargs.get('cube')
    if cube is None:
        print('Cube initialized to zeros...')
        cube = np.zeros((n_cells, n_cells, n_cells), dtype=float)
    else:
        print(f'Cube with filling fraction of {cube.mean():.3f} provided...')
    
    # Volume fraction of a single bubble
    frac_1bub = 4 * np.pi / 3 * r**3 
    if frac_1bub >= (x_ion-cube.mean())*n_cells**3:
        print(f'd(x_ion)={x_ion-cube.mean():.3f}, sphere volume={frac_1bub}')
        print("Bubble radius too large; each bubble exceeds target filling fraction.")
        return cube
    if frac_1bub*batch_size >= (x_ion-cube.mean())*n_cells**3:
        print(f'd(x_ion)={x_ion-cube.mean():.3f}, sphere volume={frac_1bub}, batch_size={batch_size}, max_iter={max_iter}')
        print("batch_size too large, each batch paints more bubbles than target filling fraction.")
        return cube
    
    # Initialize spherical kernel
    kernel = ball(r).astype(float)
    
    # Initialize progress bar to track ionization fraction
    pbar = tqdm(total=max_iter, desc="Painting bubble")
    current_fraction = cube.mean()
    
    for _ in range(max_iter):
        # Select positions based on source_distribution
        if source_distribution == 'Poisson':
            # positions = np.random.randint(0, n_cells, (batch_size, 3))
            arg_zero = np.argwhere(cube==0)
            positions = arg_zero[np.random.randint(0, arg_zero.shape[0], batch_size)]
        
        # Apply the kernel at selected positions
        paint_kernel(cube, positions, kernel)
        
        # Threshold and update ionization level
        cube = (cube >= 0.5).astype(float)
        new_fraction = cube.mean()
        
        # Update progress bar with increment in ionization fraction
        pbar.update(1)
        current_fraction = new_fraction
        
        if current_fraction >= x_ion:
            break
        pbar.set_postfix({"Filling fraction": f"{cube.mean():.3f}/{x_ion:.3f}"})
    
    # Complete the progress bar if the loop finishes early
    pbar.n = pbar.total  # Set to total iterations to fill to 100%
    pbar.refresh()  # Refresh to show the updated state
    pbar.close()  # Close the progress bar
    
    cube[cube>1.] = 1.
    print(f'min={cube.min():.2f}, mean={cube.mean():.2f}, max={cube.max():.2f} | Cube filling complete.')
    return cube