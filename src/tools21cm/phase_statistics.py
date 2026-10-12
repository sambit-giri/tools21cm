'''
Contains functions to estimate statistics of the Fourier phases of a field.

The central statistic is the phase gradient spectrum R(k), the mean resultant
length of the phase differences between neighbouring Fourier modes,

    Z(k) = < exp(i [Phi(k + dk) - Phi(k)]) >_{|k| in bin},    R(k) = |Z(k)|,

where dk is an increment of dk_grid fundamental modes along a coordinate axis.
For a Gaussian random field the phase gradients are uniformly distributed and
R(k) only shows the mode-counting noise floor sqrt(pi / (4 N_indep)). To leading
order in the non-Gaussianity, R(k) measures the angle-averaged normalised
squeezed bispectrum p3(k) = B(dk, k, -k-dk) / sqrt(V P(dk) P(k)^2):

    <R(k)> = sqrt( (pi/4) [1/N_indep + (pi^2/16) p3(k)^2] ),

which reduces to (pi^(3/2)/8) |p3(k)| when the non-Gaussian signal dominates.
Here N_indep = N_k/2 is the number of independent phase gradients in the bin,
as a mode and its conjugate partner give the same phase gradient.
'''

import numpy as np
from scipy import fftpack

from .scipy_func import numpy_product
from .power_spectrum import _get_dims, _get_k, _get_kbins, radial_average, apply_window


_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}


def _axis_list(axis, ndim):
    '''
    Convert the axis argument into a list of array axes.
    'xyz' (or 'all') uses every axis of the array.
    '''
    if isinstance(axis, str) and axis.lower() in ('xyz', 'all'):
        return list(range(ndim))
    if isinstance(axis, (list, tuple, np.ndarray)):
        return [_AXIS_INDEX[a] for a in axis]
    return [_AXIS_INDEX[axis]]


def _fourier_transform(input_array):
    '''FFT of the array with the zero mode at the centre, matching _get_k.'''
    return fftpack.fftshift(fftpack.fftn(np.asarray(input_array, dtype='float64')))


def phase_gradient_spectrum(input_array, kbins=10, box_dims=None, dk_grid=1, axis='xyz',
                            binning='log', window=None, return_n_modes=False, **kwargs):
    '''
    Calculate the phase gradient spectrum R(k) of a field.

    For each Fourier mode k, the phase gradient is the phase difference to its
    neighbour at k + dk, with dk = dk_grid fundamental modes along an axis. R(k) is
    the length of the mean of the unit vectors exp(i * phase gradient) over the
    modes in a k-bin. R = 0 for uniformly distributed phase gradients (a Gaussian
    random field in the limit of many modes) and R = 1 for perfectly coherent ones.

    Parameters:
            input_array (numpy array): the field. Can be of any dimensions.
            kbins = 10 (integer or array-like): the number of bins,
                    or a list containing the bin edges.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            dk_grid = 1 (integer): the increment in units of the fundamental mode.
            axis = 'xyz' : direction of the increment. 'x', 'y' or 'z' (or 0, 1, 2) use one axis.
                    'xyz' averages R over the increments along every axis. Each axis is
                    referenced to an independent long-wavelength mode, so the lengths of
                    the resultants are averaged rather than the resultants themselves.
            binning = 'log' : the type of binning in k-space, 'log' or 'linear'.
            window = None : taper applied to the field before the Fourier transform,
                    as in apply_window.
            return_n_modes = False (bool): if true, also return the number of modes
                    in each bin. The number of independent phase gradients is half of it.

    Returns:
            R(k) and the k bin centres, and the number of modes if return_n_modes is true.
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, input_array.shape)
    if window is not None:
        input_array = apply_window(np.array(input_array, dtype='float64'), window)

    ft = _fourier_transform(input_array)
    R_axes = []
    for ax in _axis_list(axis, input_array.ndim):
        # exp(i [Phi(k - dk) - Phi(k)]); the sign of the increment does not change |Z|
        phase_vectors = np.exp(1j * np.angle(np.roll(ft, dk_grid, axis=ax) * np.conj(ft)))
        Z, k_bins, n_modes = radial_average(phase_vectors, box_dims, kbins=kbins, binning=binning)
        R_axes.append(np.abs(Z))
    R = np.mean(R_axes, axis=0)

    if return_n_modes:
        return R, k_bins, n_modes
    return R, k_bins


def squeezed_bispectrum_normalized(input_array, kbins=10, box_dims=None, dk_grid=1, axis='xyz',
                                   binning='log', debias=True, return_n_modes=False, **kwargs):
    '''
    Calculate the angle-averaged normalised squeezed bispectrum

            p3(k) = B(dk, k, -k-dk) / sqrt( V P(dk) P(k)^2 ),

    in the triangle configuration probed by phase_gradient_spectrum: the long
    side dk = dk_grid fundamental modes along an axis, and the two short sides
    k and k + dk. p3 is dimensionless (Matsubara 2003).

    Parameters:
            input_array (numpy array): the field. Can be of any dimensions.
            kbins = 10 (integer or array-like): the number of bins,
                    or a list containing the bin edges.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. See phase_gradient_spectrum.
            dk_grid = 1 (integer): the long side in units of the fundamental mode.
            axis = 'xyz' : direction of the long side, as in phase_gradient_spectrum.
                    'xyz' averages |p3| over the axes.
            binning = 'log' : the type of binning in k-space, 'log' or 'linear'.
            debias = True (bool): subtract the noise bias of the modulus of the bin mean,
                    |B|^2 = max(|<b>|^2 - Var(b)/N_indep, 0) with N_indep = N_k/2.
            return_n_modes = False (bool): if true, also return the number of modes
                    in each bin.

    Returns:
            p3(k) and the k bin centres, and the number of modes if return_n_modes is true.
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, input_array.shape)
    boxvol = numpy_product(box_dims)
    pixelsize = boxvol / numpy_product(input_array.shape)

    ft = _fourier_transform(input_array)
    P_k, k_bins, n_modes = radial_average(np.abs(ft)**2 * pixelsize**2 / boxvol, box_dims,
                                          kbins=kbins, binning=binning)
    centre = [n // 2 for n in input_array.shape]   # the zero mode, as in _get_k

    p3_axes = []
    for ax in _axis_list(axis, input_array.ndim):
        idx = list(centre); idx[ax] += dk_grid
        ft_dk = ft[tuple(idx)]                                   # the long-wavelength mode
        P_dk = np.abs(ft_dk)**2 * pixelsize**2 / boxvol
        # closed triangle (dk, k, -k-dk): the roll maps every k to its neighbour k + dk
        b = ft_dk * ft * np.conj(np.roll(ft, -dk_grid, axis=ax)) * pixelsize**3 / boxvol
        B_mean, _, _ = radial_average(b, box_dims, kbins=kbins, binning=binning)
        if debias:
            b2_mean, _, _ = radial_average(np.abs(b)**2, box_dims, kbins=kbins, binning=binning)
            var_b = np.real(b2_mean) - np.abs(B_mean)**2
            B_abs = np.sqrt(np.maximum(np.abs(B_mean)**2 - var_b / (n_modes / 2.), 0.))
        else:
            B_abs = np.abs(B_mean)
        p3_axes.append(B_abs / np.sqrt(np.maximum(boxvol * P_dk * np.real(P_k)**2, 1e-300)))
    p3 = np.mean(p3_axes, axis=0)

    if return_n_modes:
        return p3, k_bins, n_modes
    return p3, k_bins


def phase_gradient_noise_floor(n_modes):
    '''
    Expected phase gradient spectrum of a Gaussian random field,
    sqrt(pi / (4 N_indep)) with N_indep = n_modes/2. This is the null expectation
    of the Rayleigh test of circular uniformity.

    Parameters:
            n_modes (float or array-like): the number of modes in each k-bin,
                    as returned by phase_gradient_spectrum.

    Returns:
            The noise floor in each bin.
    '''
    return np.sqrt(np.pi / (4. * np.asarray(n_modes, dtype=float) / 2.))


def phase_gradient_spectrum_from_bispectrum(p3, n_modes=None):
    '''
    Expected phase gradient spectrum for a normalised squeezed bispectrum p3,

            <R> = sqrt( (pi/4) [1/N_indep + (pi^2/16) p3^2] ),   N_indep = n_modes/2,

    valid to leading order in the non-Gaussianity.

    Parameters:
            p3 (float or array-like): the normalised squeezed bispectrum,
                    as returned by squeezed_bispectrum_normalized.
            n_modes = None (float or array-like): the number of modes in each k-bin.
                    If None, the noise term is left out (signal-dominated limit).

    Returns:
            The expected R in each bin.
    '''
    signal = (np.pi**2 / 16.) * np.asarray(p3, dtype=float)**2
    noise = 0. if n_modes is None else 1. / (np.asarray(n_modes, dtype=float) / 2.)
    return np.sqrt((np.pi / 4.) * (noise + signal))


def phase_gradients_per_bin(input_array, kbins=10, box_dims=None, dk_grid=1, axis='x',
                            binning='log', centre=True, **kwargs):
    '''
    Return the phase gradients of the modes in each k-bin, e.g. to plot their
    distribution. A von Mises distribution with concentration
    resultant_length_to_kappa(R) is a good description of it.

    Parameters:
            input_array (numpy array): the field. Can be of any dimensions.
            kbins = 10 (integer or array-like): the number of bins,
                    or a list containing the bin edges.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. See phase_gradient_spectrum.
            dk_grid = 1 (integer): the increment in units of the fundamental mode.
            axis = 'x' : direction of the increment, 'x', 'y' or 'z' (or 0, 1, 2).
            binning = 'log' : the type of binning in k-space, 'log' or 'linear'.
            centre = True (bool): if true, the phase gradients of each bin are shifted
                    by their circular mean, so that the distribution peaks at zero.

    Returns:
            A list with an array of phase gradients in [-pi, pi) for each bin,
            and the k bin edges.
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, input_array.shape)
    ax = _axis_list(axis, input_array.ndim)
    if len(ax) != 1:
        raise ValueError('phase_gradients_per_bin needs a single axis, got %r' % (axis,))

    ft = _fourier_transform(input_array)
    grad = np.angle(np.roll(ft, dk_grid, axis=ax[0]) * np.conj(ft)).flatten()
    _, k = _get_k(input_array, box_dims)
    k = k.flatten()
    edges = _get_kbins(kbins, box_dims, k, binning=binning)

    grads = []
    for i, (klo, khi) in enumerate(zip(edges[:-1], edges[1:])):
        # same bin membership as radial_average (np.histogram includes the last right edge)
        last = i == len(edges) - 2
        g = grad[(k >= klo) & ((k <= khi) if last else (k < khi))]
        if centre and g.size:
            g = np.angle(np.exp(1j * g) * np.exp(-1j * np.angle(np.mean(np.exp(1j * g)))))
        grads.append(g)
    return grads, edges


def resultant_length_to_kappa(R, method='sra'):
    '''
    Approximate concentration kappa of the von Mises distribution with mean
    resultant length R.

    Parameters:
            R (float or array-like): mean resultant length in [0, 1).
            method = 'sra' : 'sra' uses kappa = R (2 - R^2) / (1 - R^2) (Banerjee et al. 2005;
                    Sra 2012). 'best_fisher' uses the piecewise approximation of
                    Best & Fisher (1981).

    Returns:
            kappa with the shape of R.
    '''
    R = np.asarray(R, dtype=float)
    if method.lower() in ('sra', 'banerjee'):
        return R * (2 - R**2) / (1 - R**2)
    kappa = np.empty_like(R)
    m1 = R < 0.53
    m2 = (R >= 0.53) & (R < 0.85)
    m3 = R >= 0.85
    kappa[m1] = 2 * R[m1] + R[m1]**3 + (5 / 6) * R[m1]**5
    kappa[m2] = -0.4 + 1.39 * R[m2] + 0.43 / (1 - R[m2])
    kappa[m3] = 1 / (R[m3]**3 - 4 * R[m3]**2 + 3 * R[m3])
    return kappa


def von_mises_pdf(x, mu=0., kappa=1.):
    '''
    Probability density of the von Mises distribution on [-pi, pi),
    exp(kappa cos(x - mu)) / (2 pi I0(kappa)).
    '''
    from scipy.special import i0e
    # i0e(kappa) = exp(-kappa) I0(kappa) avoids overflow for large kappa
    return np.exp(kappa * (np.cos(np.asarray(x) - mu) - 1)) / (2 * np.pi * i0e(kappa))
