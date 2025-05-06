import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sys
from time import time, sleep
from tqdm import tqdm

def _ray3d(ar, xs, ys, zs, ls, ms, ns, verbose):

    info = ar.shape
    longest = max(info)
    num_sz  = np.zeros(longest)

    # Ensure coordinates are of type float64 for compatibility with interpolation
    xs, ys, zs = xs.astype(np.float64), ys.astype(np.float64), zs.astype(np.float64)

    interp_func = RegularGridInterpolator(
        (np.arange(info[0]), np.arange(info[1]), np.arange(info[2])),
        ar,
        bounds_error=False,
        fill_value=0
    )

    if verbose:
        print('Estimating ray lengths...')
    
    sleep(0.01)
    total_iterations = longest
    with tqdm(total=total_iterations, dynamic_ncols=False, disable=not verbose) as pbar:
        for rr in range(longest):
            xs += ls
            ys += ms
            zs += ns
            
            # Efficiently create points and interpolate values
            pts = np.column_stack((xs, ys, zs))
            vals = interp_func(pts)
            
            # Use boolean indexing instead of np.argwhere and np.delete
            valid = vals > 0.5
            num_sz[rr] = len(xs) - np.sum(valid)
            xs, ys, zs = xs[valid], ys[valid], zs[valid]
            ls, ms, ns = ls[valid], ms[valid], ns[valid]
            
            pbar.update(1)  # Increment the progress bar
            if len(xs) == 0:
                pbar.n = pbar.total  # Manually set the progress to 100%
                pbar.refresh()  # Refresh the bar to show the update
                break
        # pbar.set_postfix({'Completion': '100%'})

    size_px = np.arange(longest)
    return num_sz, size_px

def mfp3d(arr, xth=0.5, iterations=10000000, verbose=True, point='random'):
    iterations = int(iterations)

    if verbose: 
        print('Initialising random rays...', end=' ')
    
    info = arr.shape

    ar = np.zeros(info, dtype=np.float64)
    ar[arr >= xth] = 1

    thetas = np.random.randint(0, 360, size=iterations)
    phis   = np.random.randint(0, 360, size=iterations)
    
    # Precompute trigonometric values
    sin_thetas = np.sin(np.radians(thetas))
    cos_thetas = np.cos(np.radians(thetas))
    cos_phis   = np.cos(np.radians(phis))
    sin_phis   = np.sin(np.radians(phis))
    
    ls = sin_thetas * cos_phis
    ms = sin_thetas * sin_phis
    ns = cos_thetas

    if point == 'random':
        loc = np.argwhere(ar == 1)
        rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
        xs, ys, zs = loc[rand_loc, 0], loc[rand_loc, 1], loc[rand_loc, 2]
    else:
        xs, ys, zs = point
        if isinstance(xs, (int,float)):
            if ar[xs, ys, zs] == 0:
                print('Given point is outside the structure.')
                return None
            xs, ys, zs = np.full(iterations, xs, dtype=np.float64), np.full(iterations, ys, dtype=np.float64), np.full(iterations, zs, dtype=np.float64)
    if verbose:
        print('done')
    num_sz, size_px = _ray3d(ar, xs, ys, zs, ls, ms, ns, verbose)
    return num_sz, size_px

def mfp2d(arr, xth=0.5, iterations=1000000, verbose=True, point='random'):
    iterations = int(iterations)
    
    if verbose: print('Initializing random rays...', end=' ')
    
    info = arr.shape
    longest = int(np.sqrt(2) * max(info))
    num_sz  = np.zeros(longest)

    ar = np.zeros(info, dtype=np.float64)
    ar[arr >= xth] = 1

    thetas = np.random.randint(0, 360, size=iterations)
    
    # Precompute trigonometric values
    sin_thetas = np.sin(np.radians(thetas))
    cos_thetas = np.cos(np.radians(thetas))

    ls = sin_thetas
    ms = cos_thetas

    if point == 'random':
        loc = np.argwhere(ar == 1)
        rand_loc = np.random.randint(0, high=loc.shape[0], size=iterations)
        xs, ys = loc[rand_loc, 0], loc[rand_loc, 1]
    else:
        xs, ys = point
        if ar[xs, ys] == 0:
            print('Given point is outside the structure.')
            return None
        xs, ys = np.full(iterations, xs, dtype=np.float64), np.full(iterations, ys, dtype=np.float64)
    
    # Ensure coordinates are of type float64 for compatibility with interpolation
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)

    interp_func = RegularGridInterpolator(
        (np.arange(info[0]), np.arange(info[1])),
        ar,
        bounds_error=False,
        fill_value=0
    )

    if verbose:
        print('done')
        print('Estimating ray lengths...')
    
    sleep(0.01)
    total_iterations = longest
    with tqdm(total=total_iterations, dynamic_ncols=False, disable=not verbose) as pbar:
        for rr in range(longest):
            xs += ls
            ys += ms
            
            # Efficiently create points and interpolate values
            pts = np.column_stack((xs, ys))
            vals = interp_func(pts)
            
            # Use boolean indexing instead of np.argwhere and np.delete
            valid = vals > 0.5
            num_sz[rr] = len(xs) - np.sum(valid)
            xs, ys = xs[valid], ys[valid]
            ls, ms = ls[valid], ms[valid]
            
            pbar.update(1)  # Increment the progress bar
            if len(xs) == 0:
                pbar.n = pbar.total  # Manually set the progress to 100%
                pbar.refresh()  # Refresh the bar to show the update
                break
        # pbar.set_postfix({'Completion': '100%'})

    size_px = np.arange(longest)
    return num_sz, size_px

def loading_verbose(string):
	msg = ("Completed: " + string )
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()


