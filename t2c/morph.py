import numpy as np 
from scipy.signal import fftconvolve

"""
This script contains dilation and erosion implementations described in the following link.
https://stackoverflow.com/questions/25034259/scipy-ndimage-morphology-operators-saturate-my-computer-memory-ram-8gb
This implementation deals with memory error faced in scipy ones.
"""

def binary_dilation(A, B):
    return fftconvolve(A, B,'same')>0.5

def binary_erosion(A, B):
	return _erode_v2(A, B)

def _erode_v1(A,B,R):
    #R should be the radius of the spherical kernel, i.e. half the width of B
    A_inv = np.logical_not(A)
    A_inv = np.pad(A_inv, R, 'constant', constant_values=1)
    tmp = fftconvolve(A_inv, B, 'same') > 0.5
    #now we must un-pad the result, and invert it again
    return np.logical_not(tmp[R:-R, R:-R, R:-R])

def _erode_v2(A,B):
    thresh = np.count_nonzero(B)-0.5
    return fftconvolve(A,B,'same') > thresh


def binary_opening(image, structure=None):
    """Return fast binary morphological opening of an image.
    This function returns the same result as greyscale opening but performs
    faster for binary images.
    The morphological opening on an image is defined as an erosion followed by
    a dilation. Opening can remove small bright spots (i.e. "salt") and connect
    small dark cracks. This tends to "open" up (dark) gaps between (bright)
    features.
    Parameters
    ----------
    image : ndarray
        Binary input image.
    selem : ndarray, optional
        The neighborhood expressed as a 2-D array of 1's and 0's.
        If None, use a cross-shaped structuring element (connectivity=1).
    Returns
    -------
    opening : ndarray of bool
        The result of the morphological opening.
    """
    eroded = binary_erosion(image, structure)
    out = binary_dilation(eroded, structure)
    # eroded = erode_v2(image, structure)
    # out = dilate(eroded, structure)
    return out

