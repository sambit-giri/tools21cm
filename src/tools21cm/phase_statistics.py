'''
Contains functions to estimate statistics of the Fourier phases of a field.

The central statistic is the phase gradient spectrum R(k), the mean resultant
length of the phase differences between neighbouring Fourier modes,

    Z(k) = < exp(i [Phi(k + dk) - Phi(k)]) >_{|k| in bin},    R(k) = |Z(k)|,

where dk is the wavevector of a long-wavelength mode of the box. R(k) is averaged
over long modes, by default all of them with |dk| <= 3 x 2 pi / L (61 in 3D).
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
from scipy import fft as scipy_fft

from .scipy_func import numpy_product
from .power_spectrum import (_get_dims, _get_k, _get_kbins, radial_average, apply_window,
                             _import_backend, _as_backend_array, _scatter_add, _take)


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


def long_mode_vectors(n2_max=1, n2_min=1, ndim=3, exclude_axes=None):
    '''
    Integer increments n (in units of the fundamental mode) with n2_min <= |n|^2 <= n2_max,
    one of each pair n, -n (delta(-dk) = delta(dk)^*, so both give the same long mode).
    Each increment dk = n * 2 pi / L references the phase gradients to a different
    long-wavelength mode of the box. Averaging a statistic over many of them averages
    over independent long modes, as a k-bin averages over short modes.

    Parameters:
            n2_max = 1 (integer): largest |n|^2. 1 gives the three axis increments,
                    9 gives the 61 increments with |dk| <= 3 x 2 pi / L.
            n2_min = 1 (integer): smallest |n|^2, e.g. n2_min = n2_max for one shell.
            ndim = 3 (integer): number of dimensions.
            exclude_axes = None (int or list): drop increments that lie along these
                    axes only, e.g. exclude_axes=2 drops the k_perp = 0 increments
                    (0, 0, n), whose long mode is removed with the mean of each slice
                    along the line of sight.

    Returns:
            List of integer tuples, sorted by |n|^2.
    '''
    import itertools
    r = int(np.floor(np.sqrt(n2_max)))
    vecs = [v for v in itertools.product(range(-r, r + 1), repeat=ndim)
            if n2_min <= sum(x * x for x in v) <= n2_max and next(x for x in v if x != 0) > 0]
    if exclude_axes is not None:
        keep_axes = set(range(ndim)) - set(np.atleast_1d(exclude_axes).tolist())
        vecs = [v for v in vecs if any(v[a] != 0 for a in keep_axes)]
    return sorted(vecs, key=lambda v: (sum(x * x for x in v), [-x for x in v]))


def _shell_binner(shape, box_dims, kbins, binning):
    '''
    Bin index of every grid cell for the 'log'/'linear' tophat bins of radial_average
    (bins [a, b), last bin closed), computed once. Returns a function giving the bin
    means of an array, the bin centres and the number of modes per bin.
    '''
    if binning not in ('log', 'linear'):
        raise ValueError("dk_vectors supports binning='log' or 'linear' only, got %r" % (binning,))
    _, k = _get_k(np.empty(shape), box_dims)
    edges = np.asarray(_get_kbins(kbins, box_dims, k, binning=binning), dtype=float)
    kf = k.ravel()
    idx = np.searchsorted(edges, kf, side='right') - 1
    idx[kf == edges[-1]] = len(edges) - 2
    sel = np.flatnonzero((idx >= 0) & (idx < len(edges) - 1))
    idx = idx[sel]
    nb = len(edges) - 1
    n_modes = np.bincount(idx, minlength=nb).astype(float)

    def mean(arr):
        a = np.asarray(arr).ravel()[sel]
        if np.iscomplexobj(a):
            return (np.bincount(idx, a.real, nb) + 1j * np.bincount(idx, a.imag, nb)) / n_modes
        return np.bincount(idx, a, nb) / n_modes
    k_mid = edges[:-1] + (edges[1:] - edges[:-1]) / 2.
    return mean, k_mid, n_modes


_lag_cache = {}

def _lagged_shell_means(u, box_dims, kbins, binning, vecs):
    '''
    Bin means of np.roll(u, v) * conj(u) for every increment v, as _shell_binner gives them,
    but only over the binned modes: the modes are sorted by bin once (cached per grid), and
    a shift that does not wrap around the box is a constant offset of the flat index.
    Returns an array (n_increments, n_bins).
    '''
    shape = u.shape
    key = (shape, tuple(np.ravel(box_dims)), tuple(np.ravel(kbins)) if np.ndim(kbins) else kbins, binning)
    if key not in _lag_cache:
        binid, nb = _bin_index_grid(shape, box_dims, kbins, binning)
        flat = binid.ravel()
        order = np.argsort(flat, kind='stable')
        order = order[flat[order] >= 0]
        counts = np.bincount(flat[order], minlength=nb)
        coords = np.array(np.unravel_index(order, shape))
        _lag_cache[key] = (order, counts, coords, coords.min(axis=1), coords.max(axis=1))
    order, counts, coords, lo, hi = _lag_cache[key]
    nonempty = counts > 0
    starts = (np.cumsum(counts) - counts)[nonempty]
    strides = np.array([int(np.prod(shape[a + 1:])) for a in range(len(shape))])
    uf = u.ravel()
    u0c = np.conj(uf[order])
    out = np.full((len(vecs), len(counts)), np.nan, dtype=complex)
    for iv, v in enumerate(vecs):
        v = np.asarray(v)
        if np.all(lo - v >= 0) and np.all(hi - v < np.array(shape)):
            src = order - int(v @ strides)
        else:
            # wrapped flat index from small per-axis tables, ((i_a - v_a) mod n_a) * stride_a
            src = sum(((np.arange(n) - va) % n * s)[coords[a]]
                      for a, (n, va, s) in enumerate(zip(shape, v, strides)))
        out[iv, nonempty] = np.add.reduceat(uf[src] * u0c, starts) / counts[nonempty]
    return out


_bin_cache = {}

def _grid_key(shape, box_dims, kbins, binning):
    return (tuple(shape), tuple(np.ravel(box_dims)),
            tuple(np.ravel(kbins)) if np.ndim(kbins) else kbins, binning)


def _bin_index_grid(shape, box_dims, kbins, binning):
    '''k-bin index of every grid cell (-1 outside the bins), for the bins of _shell_binner.
    Cached per grid, as it only depends on the shape, box_dims and the bins.'''
    if binning not in ('log', 'linear'):
        raise ValueError("binning must be 'log' or 'linear' here, got %r" % (binning,))
    key = _grid_key(shape, box_dims, kbins, binning)
    if key not in _bin_cache:
        _, k = _get_k(np.empty(shape), box_dims)
        edges = np.asarray(_get_kbins(kbins, box_dims, k, binning=binning), dtype=float)
        idx = np.searchsorted(edges, k, side='right') - 1
        idx[k == edges[-1]] = len(edges) - 2
        idx[(idx < 0) | (idx >= len(edges) - 1)] = -1
        nb = len(edges) - 1
        n_modes = np.bincount(idx[idx >= 0], minlength=nb).astype(float)
        k_mid = edges[:-1] + (edges[1:] - edges[:-1]) / 2.
        _bin_cache[key] = (idx, nb, n_modes, k_mid)
    idx, nb = _bin_cache[key][:2]
    return idx, nb


def _bin_geometry(shape, box_dims, kbins, binning):
    '''(bin index grid, number of bins, modes per bin, bin centres), cached per grid.'''
    _bin_index_grid(shape, box_dims, kbins, binning)
    return _bin_cache[_grid_key(shape, box_dims, kbins, binning)]


def _fftn(x, name, xp, workers=None):
    '''n-dimensional FFT in the backend; numpy uses scipy.fft with `workers` threads.'''
    if name == 'numpy':
        return scipy_fft.fftn(x, workers=workers)
    return xp.fft.fftn(x)


def _ifftn(x, name, xp, workers=None):
    if name == 'numpy':
        return scipy_fft.ifftn(x, workers=workers)
    return xp.fft.ifftn(x)


def _shell_lag_means(a, c, binid, nb, n_modes, vecs, name='numpy', xp=np, workers=None):
    '''
    Bin means of a(k - v) conj(c(k)) over the modes k of each k-bin, for every increment v,
    with the periodic wrap of np.roll. For each bin this is a circular cross-correlation of a
    with the masked c, so all increments come from two FFTs per bin plus one for a, whatever
    their number. Works for the numpy, jax and torch backends (a, c backend arrays, binid a
    numpy array). Returns an array (n_increments, n_bins), NaN for empty bins.
    '''
    shape = np.array(binid.shape)
    lag = tuple(((-np.asarray(vecs, dtype=int)) % shape).T)      # C at index -v gives the lag v
    A = _fftn(a, name, xp, workers)
    cols = []
    for b in range(nb):
        if n_modes[b] == 0:
            cols.append(None)
            continue
        if name == 'numpy':
            cb = np.where(binid == b, c, 0)
        else:
            mask = _as_backend_array(binid == b, name, xp,
                                     device=c.device if name == 'torch' else None)
            cb = xp.where(mask, c, xp.zeros_like(c))
        C = _ifftn(A * xp.conj(_fftn(cb, name, xp, workers)), name, xp, workers)
        if name == 'torch':
            C = C[tuple(xp.as_tensor(l, dtype=xp.long, device=C.device) for l in lag)]
        else:
            C = C[lag]
        cols.append(C / n_modes[b])
    if name == 'numpy':
        out = np.full((len(vecs), nb), np.nan, dtype=complex)
        for b, col in enumerate(cols):
            if col is not None:
                out[:, b] = col
        return out
    if name == 'torch':
        nan = xp.full((len(vecs),), float('nan'), dtype=A.dtype, device=A.device)
    else:
        nan = xp.full((len(vecs),), xp.nan, dtype=A.dtype)
    return xp.stack([nan if col is None else col for col in cols], axis=1)


def _n_indep(binid, nb, vecs, sign=1):
    '''
    Effective number of independent terms in the shell mean of each increment. The term of
    mode k equals that of its conjugate partner (k' = dk - k for sign=1, the phase gradient
    convention, or k' = -k - dk for sign=-1, the bispectrum convention), so a pair inside the
    same bin enters twice. For N_k terms of which n_p belong to such pairs, the variance of
    the mean is that of N_eff = N_k^2 / (N_k + n_p) independent terms: N_k/2 if every term is
    paired, N_k if none is. Returns an array (n_increments, n_bins).
    '''
    ndim = binid.ndim
    c = np.array([n // 2 for n in binid.shape])
    # bin of the mode -k at every k (index i -> 2c - i); rolling by +-dk then gives the partner's bin
    flip = binid[np.ix_(*[(2 * c[a] - np.arange(binid.shape[a])) % binid.shape[a] for a in range(ndim)])]
    valid = binid >= 0
    N_k = np.bincount(binid[valid], minlength=nb).astype(float)
    out = []
    for v in vecs:
        partner = np.roll(flip, tuple(sign * x for x in v), axis=tuple(range(ndim)))
        same = valid & (partner == binid)
        n_p = np.bincount(binid[same], minlength=nb).astype(float)
        out.append(N_k**2 / np.maximum(N_k + n_p, 1))
    return np.array(out)


_n_indep_cache = {}

def _n_indep_cached(shape, box_dims, kbins, binning, vecs, sign=1):
    '''_n_indep for the bins of _bin_index_grid, cached per grid, increments and sign.'''
    key = (_grid_key(shape, box_dims, kbins, binning), tuple(tuple(int(x) for x in v) for v in vecs), sign)
    if key not in _n_indep_cache:
        binid, nb = _bin_index_grid(shape, box_dims, kbins, binning)
        _n_indep_cache[key] = _n_indep(binid, nb, vecs, sign=sign)
    return _n_indep_cache[key]


def _bin_means(arr, binid, nb, n_modes, name='numpy', xp=np):
    '''Bin means of a real or complex backend array over the k-bins of binid (NaN if empty).'''
    sel = np.flatnonzero(binid.ravel() >= 0)
    idx = binid.ravel()[sel]
    if name == 'numpy':
        a = np.asarray(arr).ravel()[sel]
        s = np.bincount(idx, a.real, nb) + (1j * np.bincount(idx, a.imag, nb) if np.iscomplexobj(a) else 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            return s / n_modes
    sums = _scatter_add(_take(arr.reshape(-1), sel, name, xp), idx, nb, name, xp)
    return sums / _as_backend_array(np.where(n_modes > 0, n_modes, np.nan), name, xp,
                                    device=sums.device if name == 'torch' else None)


def _take_nd(arr, index, name, xp):
    '''arr[index] for a tuple of numpy integer index arrays, in the backend.'''
    if name == 'torch':
        return arr[tuple(xp.as_tensor(i, dtype=xp.long, device=arr.device) for i in index)]
    return arr[index]


def _clip_min(x, value, name, xp):
    return xp.clamp(x, min=value) if name == 'torch' else xp.maximum(x, value)


def phase_gradient_n_indep(shape, kbins=10, box_dims=None, dk_grid=1, axis='xyz', dk_vectors=None,
                           dk_squared_max=9, exclude_axes=None,
                           binning='log', **kwargs):
    '''
    Effective number of independent phase gradients in each k-bin, for each increment.

    A mode k and its conjugate partner dk - k give the same phase gradient. N_k/2 assumes
    that every partner lies in the same k-bin. That holds for |dk| << k, but for longer
    increments at low k many partners fall in another bin and the terms are independent.
    This returns the variance-equivalent count N_eff = N_k^2 / (N_k + n_p), with n_p the
    number of modes whose partner is in the same bin, which lies between N_k/2 and N_k.

    Parameters:
            shape (tuple): shape of the field.
            kbins, box_dims, dk_grid, axis, dk_vectors, dk_squared_max, exclude_axes, binning:
                    as in phase_gradient_spectrum.

    Returns:
            Array (n_increments, n_bins). Pass it as n_indep to phase_gradient_noise_floor and
            phase_gradient_spectrum_from_bispectrum.
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, shape)
    return _n_indep_cached(shape, box_dims, kbins, binning, _increments(axis, dk_grid, dk_vectors, len(shape),
                                                                      dk_squared_max, exclude_axes))


SQUEEZED_WARN_RATIO = 0.5

def _warn_if_not_squeezed(vecs, box_dims, k_bins, ratio=None):
    '''
    Warn when the longest long mode is not much shorter than the centre of a k-bin.
    The link between R and the squeezed bispectrum assumes |dk| << k, so in those bins
    the triangles are not squeezed. The threshold is SQUEEZED_WARN_RATIO (|dk|_max / k).
    '''
    import warnings
    ratio = SQUEEZED_WARN_RATIO if ratio is None else ratio
    kf = 2 * np.pi / np.asarray(box_dims, dtype=float)
    dk_max = np.max(np.linalg.norm(np.asarray(vecs, dtype=float) * kf, axis=1))
    k_bins = np.asarray(k_bins, dtype=float)
    bad = np.flatnonzero(dk_max > ratio * k_bins)
    if bad.size:
        warnings.warn(
            'The longest long mode, |dk| = %.3g, exceeds %.2g x k in the k-bins centred at %s, '
            'so the triangles are not squeezed there and R no longer follows the squeezed '
            'bispectrum. Use shorter long modes (a smaller dk_squared_max, dk_grid or dk_vectors), '
            'or drop these bins.' % (dk_max, ratio, np.array2string(k_bins[bad], precision=3)), stacklevel=3)


def _increments(axis, dk_grid, dk_vectors, ndim, dk_squared_max=None, exclude_axes=None):
    '''
    Integer increment vectors: dk_vectors if given, else all increments with
    |n|^2 <= dk_squared_max (long_mode_vectors), else dk_grid along each axis.
    '''
    if dk_vectors is not None:
        return [tuple(int(x) for x in v) for v in dk_vectors]
    if dk_squared_max is not None:
        return long_mode_vectors(n2_max=dk_squared_max, ndim=ndim, exclude_axes=exclude_axes)
    vecs = []
    for ax in _axis_list(axis, ndim):
        v = [0] * ndim; v[ax] = dk_grid; vecs.append(tuple(v))
    return vecs


def phase_gradient_spectrum(input_array, kbins=10, box_dims=None, dk_grid=1, axis='xyz',
                            binning='log', window=None, return_n_modes=False, dk_vectors=None,
                            dk_squared_max=9, exclude_axes=None,
                            backend='numpy', method='auto', workers=None, **kwargs):
    '''
    Calculate the phase gradient spectrum R(k) of a field.

    For each Fourier mode k, the phase gradient is the phase difference to its
    neighbour at k + dk. The increment dk is the wavevector of a long-wavelength
    mode of the box. R(k) is the length of the mean of the unit vectors
    exp(i * phase gradient) over the modes in a k-bin, averaged over the long modes.
    R = 0 for uniformly distributed phase gradients (a Gaussian random field in the
    limit of many modes) and R = 1 for perfectly coherent ones.

    The long modes are, in order of precedence: dk_vectors if given, else every
    increment with |dk|^2 <= dk_squared_max (2 pi / L)^2 (the default, 61 long modes),
    else dk_grid fundamental modes along the axes in axis (dk_squared_max=None).

    Parameters:
            input_array (numpy array): the field. Can be of any dimensions.
            kbins = 10 (integer or array-like): the number of bins,
                    or a list containing the bin edges.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            dk_squared_max = 9 (integer or None): use every long mode with
                    |dk|^2 <= dk_squared_max in units of the squared fundamental mode
                    (2 pi / L)^2, one of each pair dk, -dk. 1, 2, 3, 4, 5, 6, 8 and 9 give
                    3, 9, 13, 16, 28, 40, 46 and 61 long modes; 9 includes all long modes
                    up to 3 fundamental modes in every direction. Larger values add long
                    modes but make the triangles less squeezed at low k: a warning is
                    issued for k-bins where |dk|_max > SQUEEZED_WARN_RATIO x k (0.5). None
                    uses axis and dk_grid instead.
            exclude_axes = None (int or list): with dk_squared_max, drop the long modes that
                    lie along these axes only, e.g. 2 drops the k_perp = 0 modes (0, 0, n),
                    which are removed with the mean of each slice along the line of sight.
            dk_grid = 1 (integer): with dk_squared_max=None, the increment in units of the
                    fundamental mode.
            axis = 'xyz' : with dk_squared_max=None, direction of the increment. 'x', 'y'
                    or 'z' (or 0, 1, 2) use one axis. 'xyz' averages R over the increments
                    along every axis.
            binning = 'log' : the type of binning in k-space, 'log' or 'linear'.
            window = None : taper applied to the field before the Fourier transform,
                    as in apply_window.
            return_n_modes = False (bool): if true, also return the number of modes
                    in each bin. phase_gradient_n_indep gives the number of independent
                    phase gradients, between half of it and all of it.
            dk_vectors = None (list of integer tuples): a custom list of long modes in units
                    of the fundamental mode, e.g. from long_mode_vectors. Overrides
                    dk_squared_max, axis and dk_grid. Each long mode is a separate reference
                    for the phases, so the lengths of the resultants are averaged rather
                    than the resultants themselves.
            backend = 'numpy' (string): 'numpy', 'jax' or 'torch'. With 'jax'/'torch' the
                    FFTs and array operations run in that framework, so R is differentiable
                    (jax.grad / torch autograd, except at modes of zero amplitude) and can run
                    on a GPU. Enable jax_enable_x64 for float64; jax uses float32 otherwise.
                    Only 'log' and 'linear' binning.
            method = 'auto' : how the bin means are computed for the numpy backend (the result
                    is the same). 'fft' computes all increments at once from cross-correlations,
                    two FFTs per k-bin, so its cost does not grow with the number of increments.
                    'gather' sums the products mode by mode, one pass per increment. 'auto' uses
                    'fft' when there are more increments than k-bins. The other backends use 'fft'.
            workers = None (int): threads for the numpy FFTs (scipy.fft), e.g. -1 for all cores.

    Returns:
            R(k) and the k bin centres, and the number of modes if return_n_modes is true.
            A field that vanishes everywhere has no phases, and its R is NaN (numpy backend).
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, input_array.shape)
    name, xp = _import_backend(backend)
    if window is not None:
        input_array = apply_window(np.array(input_array, dtype='float64') if name == 'numpy'
                                   else input_array, window, backend=backend)

    if binning in ('log', 'linear'):
        vecs = _increments(axis, dk_grid, dk_vectors, input_array.ndim, dk_squared_max, exclude_axes)
        binid, nb, n_modes, k_bins = _bin_geometry(input_array.shape, box_dims, kbins, binning)
        _warn_if_not_squeezed(vecs, box_dims, k_bins)
        if name == 'numpy':
            ft = np.fft.fftshift(_fftn(np.asarray(input_array, dtype='float64'), name, xp, workers))
            amp = np.abs(ft)
            u = ft / np.where(amp > 0, amp, 1)
            if method == 'auto':
                method = 'fft' if len(vecs) > nb else 'gather'
            if method == 'fft':
                Z = _shell_lag_means(u, u, binid, nb, n_modes, vecs, workers=workers)
            elif method == 'gather':
                Z = _lagged_shell_means(u, box_dims, kbins, binning, vecs)
            else:
                raise ValueError("method must be 'auto', 'fft' or 'gather', got %r" % (method,))
            # exp(i [Phi(k - dk) - Phi(k)]); the sign of the increment does not change |Z|
            R = np.mean(np.abs(Z), axis=0)
            if not np.any(amp):
                R = np.full(nb, np.nan)
        else:
            ft = xp.fft.fftshift(xp.fft.fftn(_as_backend_array(input_array, name, xp)))
            amp = xp.abs(ft)
            u = ft / xp.where(amp > 0, amp, xp.ones_like(amp))
            Z = _shell_lag_means(u, u, binid, nb, n_modes, vecs, name, xp)
            R = xp.mean(xp.abs(Z), axis=0)
        if return_n_modes:
            return R, k_bins, n_modes
        return R, k_bins

    if name != 'numpy' or dk_vectors is not None or dk_squared_max is not None:
        raise ValueError("backend=%r, dk_vectors and dk_squared_max need binning='log' or 'linear', "
                         "got %r (set dk_squared_max=None to use axis and dk_grid)" % (backend, binning))
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
                                   binning='log', debias=True, return_n_modes=False, dk_vectors=None,
                                   dk_squared_max=9, exclude_axes=None, average=True, squared=False, backend='numpy', method='auto',
                                   workers=None, **kwargs):
    '''
    Calculate the angle-averaged normalised squeezed bispectrum

            p3(k) = B(dk, k, -k-dk) / sqrt( V P(dk) P(k)^2 ),

    in the triangle configuration probed by phase_gradient_spectrum: the long
    side dk is a long-wavelength mode of the box, chosen as in phase_gradient_spectrum,
    and the two short sides k and k + dk. p3 is dimensionless (Matsubara 2003).

    Parameters:
            input_array (numpy array): the field. Can be of any dimensions.
            kbins = 10 (integer or array-like): the number of bins,
                    or a list containing the bin edges.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. See phase_gradient_spectrum.
            dk_squared_max = 9, exclude_axes = None, dk_grid = 1, axis = 'xyz': the long
                    sides, as in phase_gradient_spectrum. Use the same values there so that
                    p3 and R refer to the same long modes.
            binning = 'log' : the type of binning in k-space, 'log' or 'linear'.
            debias = True (bool): subtract the noise bias of the modulus of the bin mean,
                    |B|^2 = max(|<b>|^2 - Var(b)/N_indep, 0), with N_indep from
                    phase_gradient_n_indep for the long modes of dk_squared_max or dk_vectors,
                    and N_k/2 for those of axis and dk_grid.
            return_n_modes = False (bool): if true, also return the number of modes
                    in each bin.
            dk_vectors = None (list of integer tuples): a custom list of long sides, as in
                    phase_gradient_spectrum. Overrides dk_squared_max, axis and dk_grid.
            average = True (bool): average |p3| over the increments. If false, return one row
                    per increment, so that phase_gradient_spectrum_from_bispectrum can average
                    the prediction itself (it is not linear in p3).
            squared = False (bool): return the debiased p3^2 without clipping at zero. The
                    estimate uses the measured long mode, so |p3| scales with its amplitude and
                    its mean over realisations is sqrt(pi)/2 times the p3 of the bispectrum.
                    Average p3^2 over realisations and take the square root to get the latter,
                    which is what phase_gradient_spectrum_from_bispectrum expects.
            backend = 'numpy' (string): 'numpy', 'jax' or 'torch', as in phase_gradient_spectrum.
                    Not with dk_squared_max=None.
            method = 'auto' : 'fft' computes all long sides at once from cross-correlations,
                    'roll' loops over them. 'auto' uses 'fft' when there are more long sides
                    than k-bins. Numpy backend, not with dk_squared_max=None.
            workers = None (int): threads for the numpy FFTs (scipy.fft), e.g. -1 for all cores.

    Returns:
            p3(k) and the k bin centres, and the number of modes if return_n_modes is true.
    '''
    if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
    box_dims = _get_dims(box_dims, input_array.shape)
    boxvol = numpy_product(box_dims)
    pixelsize = boxvol / numpy_product(input_array.shape)

    if dk_vectors is not None or dk_squared_max is not None:
        name, xp = _import_backend(backend)
        shape = input_array.shape
        binid, nb, n_modes, k_bins = _bin_geometry(shape, box_dims, kbins, binning)
        vecs = np.asarray(_increments(axis, dk_grid, dk_vectors, input_array.ndim,
                                      dk_squared_max, exclude_axes), dtype=int)
        _warn_if_not_squeezed(vecs, box_dims, k_bins)
        if name == 'numpy':
            ft = np.fft.fftshift(_fftn(np.asarray(input_array, dtype='float64'), name, xp, workers))
            if method == 'auto':
                method = 'fft' if len(vecs) > nb else 'roll'
        else:
            ft = xp.fft.fftshift(xp.fft.fftn(_as_backend_array(input_array, name, xp)))
            method = 'fft'
        pw = xp.abs(ft)**2
        P_k = xp.real(_bin_means(pw, binid, nb, n_modes, name, xp)) * (pixelsize**2 / boxvol)
        centre = np.array([n // 2 for n in shape])
        ft_dk = _take_nd(ft, tuple((centre + vecs).T), name, xp)       # the long-wavelength modes
        P_dk = xp.abs(ft_dk)**2 * (pixelsize**2 / boxvol)
        scale = pixelsize**3 / boxvol
        # closed triangle (dk, k, -k-dk): b(k) = d(dk) d(k) d*(k + dk), averaged over each k-bin
        if method == 'fft':
            # <d(k) d*(k + dk)> and <|d(k)|^2 |d(k + dk)|^2> for all dk from cross-correlations
            M = xp.conj(_shell_lag_means(ft, ft, binid, nb, n_modes, -vecs, name, xp, workers))
            B_mean = ft_dk[:, None] * M * scale
            if debias:
                Q = xp.real(_shell_lag_means(pw, pw, binid, nb, n_modes, -vecs, name, xp, workers))
                b2_mean = (xp.abs(ft_dk)**2)[:, None] * Q * scale**2
        elif method == 'roll':
            mean, _, _ = _shell_binner(shape, box_dims, kbins, binning)
            all_axes = tuple(range(input_array.ndim))
            B_mean, b2_mean = [], []
            for iv, v in enumerate(vecs):
                b = ft_dk[iv] * ft * np.conj(np.roll(ft, tuple(-v), axis=all_axes)) * scale
                B_mean.append(mean(b))
                if debias:
                    b2_mean.append(np.real(mean(np.abs(b)**2)))
            B_mean, b2_mean = np.array(B_mean), np.array(b2_mean)
        else:
            raise ValueError("method must be 'auto', 'fft' or 'roll', got %r" % (method,))
        B2 = xp.abs(B_mean)**2
        if debias:   # b(k) = b(-k-dk): the variance of the bin mean uses the effective count
            n_eff = _as_backend_array(_n_indep_cached(shape, box_dims, kbins, binning, vecs, sign=-1),
                                      name, xp, device=B2.device if name == 'torch' else None)
            B2 = B2 - (b2_mean - B2) / n_eff
        norm = _clip_min(boxvol * P_dk[:, None] * P_k[None, :]**2, 1e-300, name, xp)
        p3_inc = B2 / norm if squared else xp.sqrt(_clip_min(B2, 0., name, xp) / norm)
        p3 = xp.mean(p3_inc, axis=0) if average else p3_inc
        if return_n_modes:
            return p3, k_bins, n_modes
        return p3, k_bins

    if backend not in (None, 'numpy'):
        raise ValueError("backend=%r needs long modes from dk_squared_max or dk_vectors" % (backend,))
    if binning in ('log', 'linear'):
        # the cached bins of _bin_index_grid hold the same modes as radial_average, and are faster
        ft = np.fft.fftshift(_fftn(np.asarray(input_array, dtype='float64'), 'numpy', np, workers))
        binid, nb, n_modes, k_bins = _bin_geometry(input_array.shape, box_dims, kbins, binning)
        shell_mean = lambda a: _bin_means(a, binid, nb, n_modes)
        P_k = shell_mean(np.abs(ft)**2 * pixelsize**2 / boxvol)
    else:
        ft = _fourier_transform(input_array)
        P_k, k_bins, n_modes = radial_average(np.abs(ft)**2 * pixelsize**2 / boxvol, box_dims,
                                              kbins=kbins, binning=binning)
        shell_mean = lambda a: radial_average(a, box_dims, kbins=kbins, binning=binning)[0]
    centre = [n // 2 for n in input_array.shape]   # the zero mode, as in _get_k

    p3_axes = []
    for ax in _axis_list(axis, input_array.ndim):
        idx = list(centre); idx[ax] += dk_grid
        ft_dk = ft[tuple(idx)]                                   # the long-wavelength mode
        P_dk = np.abs(ft_dk)**2 * pixelsize**2 / boxvol
        # closed triangle (dk, k, -k-dk): the roll maps every k to its neighbour k + dk
        b = ft_dk * ft * np.conj(np.roll(ft, -dk_grid, axis=ax)) * pixelsize**3 / boxvol
        B_mean = shell_mean(b)
        B2 = np.abs(B_mean)**2
        if debias:
            b2_mean = shell_mean(np.abs(b)**2)
            B2 = B2 - (np.real(b2_mean) - np.abs(B_mean)**2) / (n_modes / 2.)
        norm = np.maximum(boxvol * P_dk * np.real(P_k)**2, 1e-300)
        p3_axes.append(B2 / norm if squared else np.sqrt(np.maximum(B2, 0.) / norm))
    p3 = np.mean(p3_axes, axis=0)

    if return_n_modes:
        return p3, k_bins, n_modes
    return p3, k_bins


def phase_gradient_noise_floor(n_modes=None, n_indep=None):
    '''
    Expected phase gradient spectrum of a Gaussian random field,
    sqrt(pi / (4 N_indep)). This is the null expectation of the Rayleigh test of
    circular uniformity.

    Parameters:
            n_modes = None (float or array-like): the number of modes in each k-bin,
                    as returned by phase_gradient_spectrum. Gives N_indep = n_modes/2,
                    which assumes |dk| << k.
            n_indep = None (array-like): effective number of independent phase gradients,
                    from phase_gradient_n_indep. Overrides n_modes. A 2D array (one row per
                    increment) gives the floor averaged over the increments.

    Returns:
            The noise floor in each bin.
    '''
    if n_indep is not None:
        floor = np.sqrt(np.pi / (4. * np.asarray(n_indep, dtype=float)))
        return floor.mean(axis=0) if floor.ndim > 1 else floor
    return np.sqrt(np.pi / (4. * np.asarray(n_modes, dtype=float) / 2.))


def phase_gradient_spectrum_from_bispectrum(p3, n_modes=None, n_indep=None):
    '''
    Expected phase gradient spectrum for a normalised squeezed bispectrum p3,

            <R> = sqrt( (pi/4) [1/N_indep + (pi^2/16) p3^2] ),   N_indep = n_modes/2,

    valid to leading order in the non-Gaussianity.

    Parameters:
            p3 (float or array-like): the normalised squeezed bispectrum,
                    as returned by squeezed_bispectrum_normalized. A 2D array (one row per
                    increment, average=False) gives the prediction averaged over the rows,
                    matching R averaged over the same increments.
            n_modes = None (float or array-like): the number of modes in each k-bin,
                    giving N_indep = n_modes/2. If None (and no n_indep), the noise term
                    is left out (signal-dominated limit).
            n_indep = None (array-like): effective number of independent phase gradients
                    from phase_gradient_n_indep, overriding n_modes. A 2D array (one row per
                    increment) gives the prediction averaged over the increments.

    Returns:
            The expected R in each bin.
    '''
    p3 = np.asarray(p3, dtype=float)
    signal = (np.pi**2 / 16.) * p3**2
    if n_indep is not None:
        noise = 1. / np.asarray(n_indep, dtype=float)
    else:
        noise = 0. if n_modes is None else 1. / (np.asarray(n_modes, dtype=float) / 2.)
    R = np.sqrt((np.pi / 4.) * (noise + signal))
    return R.mean(axis=0) if R.ndim > 1 else R


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
