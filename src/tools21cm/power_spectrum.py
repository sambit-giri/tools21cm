'''
Contains functions to estimate various two point statistics.
'''

import numpy as np, gc
from . import const, conv, cosmo, smoothing
from .helper_functions import print_msg, get_eval
from .scipy_func import numpy_product
from scipy import fftpack, stats, interpolate

def apply_window(input_array, window, backend='numpy'):
    if window is None: return input_array
    from scipy.signal import windows
    if backend in (None, 'numpy'):
        if window.lower()=='blackmanharris':
                input_array *= windows.blackmanharris(input_array.shape[-1])[None,None,:]
        elif window.lower()=='tukey':
                input_array *= windows.tukey(input_array.shape[-1])[None,None,:]
        else:
                input_array *= window
        return input_array
    # Differentiable backends: out-of-place multiply, window built in numpy (static).
    name, xp = _import_backend(backend)
    if isinstance(window, str) and window.lower()=='blackmanharris':
            w = windows.blackmanharris(input_array.shape[-1])[None,None,:]
    elif isinstance(window, str) and window.lower()=='tukey':
            w = windows.tukey(input_array.shape[-1])[None,None,:]
    else:
            w = np.asarray(window)
    arr = _as_backend_array(input_array, name, xp)
    wb  = _as_backend_array(w, name, xp)
    if name == 'torch': wb = wb.to(arr.dtype)
    return arr * wb


def _import_backend(backend):
        '''
        Resolve a backend name to (name, array module).
        'numpy' → numpy, 'jax' → jax.numpy, 'torch' → torch.
        Note for jax: enable jax_enable_x64 for float64 precision, otherwise
        the jax path runs in float32 (jax default).
        '''
        if backend is None: return 'numpy', np
        name = str(backend).lower()
        if name == 'numpy':
                return name, np
        if name == 'jax':
                import jax.numpy as jnp
                return name, jnp
        if name == 'torch':
                import torch
                return name, torch
        raise ValueError("backend must be 'numpy', 'jax' or 'torch'; got %r" % (backend,))


def _as_backend_array(input_array, name, xp):
        '''Convert to the backend's array type without copying when already there.'''
        if name == 'torch':
                return input_array if xp.is_tensor(input_array) else xp.as_tensor(np.ascontiguousarray(input_array))
        return xp.asarray(input_array)


def _scatter_add(values, idx, n_bins, name, xp):
        '''
        Differentiable scatter-add: sum `values` (backend array) into `n_bins`
        bins according to the integer numpy index array `idx`.
        '''
        if name == 'jax':
                return xp.zeros(n_bins, dtype=values.dtype).at[xp.asarray(idx)].add(values)
        if name == 'torch':
                index = xp.as_tensor(idx, dtype=xp.long, device=values.device)
                return xp.zeros(n_bins, dtype=values.dtype, device=values.device).index_add(0, index, values)
        out = np.zeros(n_bins, dtype=values.dtype)
        np.add.at(out, idx, values)
        return out


def _take(values, idx, name, xp):
        '''Gather `values[idx]` for a numpy integer index array (differentiable).'''
        if name == 'torch':
                return values[xp.as_tensor(idx, dtype=xp.long, device=values.device)]
        return values[idx]


def power_spectrum_nd(input_array, box_dims=None, verbose=False, backend='numpy', **kwargs):
        ''' 
        Calculate the power spectrum of input_array and return it as an n-dimensional array.
        
        Parameters:
                input_array (numpy array): the array to calculate the 
                        power spectrum of. Can be of any dimensions.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                backend = 'numpy' (string): 'numpy' (default), 'jax' or 'torch'.
                        With 'jax'/'torch' the FFT and all array operations run in the
                        chosen framework, so the result is differentiable (jax.grad /
                        torch autograd) and can execute on GPU. Default path unchanged.

        Returns:
                The power spectrum in the same dimensions as the input array.
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)

        # scale
        boxvol = numpy_product(box_dims)
        pixelsize = boxvol/(numpy_product(input_array.shape))

        if backend not in (None, 'numpy'):
                name, xp = _import_backend(backend)
                arr = _as_backend_array(input_array, name, xp)
                ft = xp.fft.fftshift(xp.fft.fftn(arr))
                return xp.abs(ft)**2 * (pixelsize**2/boxvol)

        if(verbose): print( 'Calculating power spectrum...')
        ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
        power_spectrum = np.abs(ft)**2
        if(verbose): print( '...done')

        power_spectrum *= pixelsize**2/boxvol

        return power_spectrum


def cross_power_spectrum_nd(input_array1, input_array2, box_dims, backend='numpy', **kwargs):
        ''' 
        Calculate the cross power spectrum two arrays and return it as an n-dimensional array.
        
        Parameters:
                input_array1 (numpy array): the first array to calculate the 
                        power spectrum of. Can be of any dimensions.
                input_array2 (numpy array): the second array. Must have same 
                        dimensions as input_array1.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
        
        Returns:
                The cross power spectrum in the same dimensions as the input arrays.
                
        TODO:
                Also return k values.
        '''

        assert(input_array1.shape == input_array2.shape)

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1.shape)

        if backend not in (None, 'numpy'):
                name, xp = _import_backend(backend)
                a1 = _as_backend_array(input_array1, name, xp)
                a2 = _as_backend_array(input_array2, name, xp)
                ft1 = xp.fft.fftshift(xp.fft.fftn(a1))
                ft2 = xp.fft.fftshift(xp.fft.fftn(a2))
                power_spectrum = (ft1 * xp.conj(ft2)).real
                boxvol = numpy_product(box_dims)
                pixelsize = boxvol/(numpy_product(input_array1.shape))
                return power_spectrum * (pixelsize**2/boxvol)

        print_msg( 'Calculating power spectrum...')
        ft1 = fftpack.fftshift(fftpack.fftn(input_array1.astype('float64')))
        ft2 = fftpack.fftshift(fftpack.fftn(input_array2.astype('float64')))
        power_spectrum = np.real(ft1)*np.real(ft2)+np.imag(ft1)*np.imag(ft2)
        print_msg( '...done')

        # scale
        boxvol = numpy_product(box_dims)
        pixelsize = boxvol/(numpy_product(input_array1.shape))
        power_spectrum *= pixelsize**2/boxvol

        return power_spectrum


def _adaptive_bin_edges(modes, n_bins, log=True):
        '''
        Compute bin edges so that every bin contains at least one discrete mode.
        log=True  : edges at geometric midpoints (suitable for log-spaced bin centres).
        log=False : edges at arithmetic midpoints (suitable for linear bin centres).
        Returns an array of length min(len(modes), n_bins) + 1.
        '''
        n_modes = len(modes)
        if n_modes == 0:
                return None
        def _mid(a, b):
                return float(np.sqrt(a * b)) if log else float((a + b) / 2.0)
        if n_modes <= n_bins:
                edges = [modes[0] * 0.5]
                for i in range(n_modes - 1):
                        edges.append(_mid(modes[i], modes[i + 1]))
                edges.append(modes[-1] * 2.0)
        else:
                idx = np.unique(np.round(np.linspace(0, n_modes, n_bins + 1)).astype(int))
                idx = np.clip(idx, 0, n_modes)
                edges = [modes[0] * 0.5]
                for i in range(1, len(idx) - 1):
                        edges.append(_mid(modes[idx[i] - 1], modes[idx[i]]))
                edges.append(modes[-1] * 2.0)
        return np.array(edges)


def _radial_average_backend(power, k, box_dims, kbins, binning, kmax,
                            fill_empty_bins, name, xp):
        '''
        Differentiable radial average for jax/torch backend arrays.

        Bin geometry (k grid, bin edges, per-bin mode counts, empty-bin fill
        indices) depends only on the grid shape, box_dims and the binning
        scheme, so it is computed in numpy and is static under jax.jit /
        torch autograd. Only the power values carry gradients; they are
        summed per bin with a differentiable scatter-add.

        Supports binning='log' and 'linear' (matches the numpy tophat path).
        '''
        if binning not in ('log', 'linear'):
                raise NotImplementedError(
                        "backend='%s' supports binning='log' or 'linear'; got %r" % (name, binning))
        kbins_edges = _get_kbins(kbins, box_dims, k, binning=binning, kmax=kmax)
        k_flat = k.flatten()
        n_bins = len(kbins_edges) - 1
        idx = np.digitize(k_flat, kbins_edges) - 1
        idx[k_flat == kbins_edges[-1]] = n_bins - 1   # np.histogram includes the right edge
        valid = np.flatnonzero((idx >= 0) & (idx < n_bins))
        idx_v = idx[valid]
        n_modes = np.bincount(idx_v, minlength=n_bins).astype(float)

        pw   = _take(power.reshape(-1), valid, name, xp)
        sums = _scatter_add(pw, idx_v, n_bins, name, xp)
        counts = _as_backend_array(np.where(n_modes > 0, n_modes, 1.0), name, xp)
        if name == 'torch': counts = counts.to(sums.dtype)
        outdata = sums / counts

        empty = n_modes == 0
        if empty.any() and not empty.all():
                if fill_empty_bins:
                        # static nearest-neighbour gather, mirrors the numpy fill
                        gather = np.arange(n_bins)
                        non_empty = np.flatnonzero(~empty)
                        for ni in np.flatnonzero(empty):
                                gather[ni] = non_empty[np.argmin(np.abs(non_empty - ni))]
                        outdata = _take(outdata, gather, name, xp)
                else:
                        nan_vec = _as_backend_array(np.where(empty, np.nan, 0.0), name, xp)
                        if name == 'torch': nan_vec = nan_vec.to(outdata.dtype)
                        outdata = outdata + nan_vec

        dk = (kbins_edges[1:] - kbins_edges[:-1]) / 2.
        k_mid = kbins_edges[:-1] + dk
        return outdata, k_mid, n_modes


def radial_average(input_array, box_dims, kbins=10, binning='log', kmax=None,
                   fill_empty_bins=True, backend='numpy', **kwargs):
        '''
        Radially average data. Mostly for internal use.

        Parameters:
                input_array (numpy array): the data array
                box_dims (float or array-like): box dimensions in Mpc.
                kbins = 10 (integer or array-like): number of bins or explicit bin edges.
                binning = 'log' : binning scheme ('log', 'linear', 'adaptive', 'adaptive_linear', 'kde', 'kde_linear').
                kmax = None (float): upper k bin edge (from _resolve_k_limit).
                fill_empty_bins = True : fill zero-mode bins via 1D nearest-neighbour.
                backend = 'numpy' : 'numpy' (default), 'jax' or 'torch'. Differentiable
                        scatter-add binning for jax/torch arrays ('log'/'linear' only).

        Returns:
                (data, bin_centers, n_modes)
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')

        if backend not in (None, 'numpy'):
                name, xp = _import_backend(backend)
                _, k_np = _get_k(input_array, box_dims)   # numpy k grid: uses only .shape
                return _radial_average_backend(input_array, k_np, box_dims, kbins,
                                               binning, kmax, fill_empty_bins, name, xp)

        k_comp, k = _get_k(input_array, box_dims)

        # ---- KDE path (log-k or linear-k) ----
        if binning in ('kde', 'kde_linear'):
                use_log = (binning == 'kde')
                n_bins = kbins if isinstance(kbins, int) else len(kbins) - 1
                valid = k.flatten() > 0
                kv = np.log10(k.flatten()[valid]) if use_log else k.flatten()[valid]
                kmax_s = (np.log10(kmax) if use_log else kmax) if kmax is not None else kv.max()
                k_mid_s = np.linspace(kv.min(), kmax_s, n_bins)
                bw = float(np.ptp(k_mid_s)) / max(n_bins - 1, 1)
                pw = input_array.flatten()[valid]
                ps_out  = np.empty(n_bins)
                nm_out  = np.empty(n_bins)
                for i, kc in enumerate(k_mid_s):
                        w = np.exp(-(kv - kc)**2 / (2.0 * bw**2))
                        wsum = w.sum()
                        ps_out[i] = (pw * w).sum() / wsum if wsum > 0 else np.nan
                        nm_out[i] = float(np.sum(np.abs(kv - kc) <= bw))
                k_mid_out = np.power(10, k_mid_s) if use_log else k_mid_s
                return ps_out, k_mid_out, nm_out

        # ---- Adaptive / tophat path ----
        if binning in ('adaptive', 'adaptive_linear'):
                k_modes = np.unique(k[k > 0])
                if kmax is not None:
                        k_modes = k_modes[k_modes <= kmax]
                n_bins = kbins if isinstance(kbins, int) else len(kbins) - 1
                kbins_edges = _adaptive_bin_edges(k_modes, n_bins, log=(binning == 'adaptive'))
        else:
                kbins_edges = _get_kbins(kbins, box_dims, k, binning=binning, kmax=kmax)

        dk = (kbins_edges[1:] - kbins_edges[:-1]) / 2.
        outdata = np.histogram(k.flatten(), bins=kbins_edges,
                               weights=input_array.flatten())[0].astype(float)
        n_modes = np.histogram(k.flatten(), bins=kbins_edges)[0].astype(float)
        outdata /= n_modes

        if fill_empty_bins:
                nan_mask = np.isnan(outdata)
                if nan_mask.any() and not nan_mask.all():
                        non_nan = np.where(~nan_mask)[0]
                        for ni in np.where(nan_mask)[0]:
                                nearest = non_nan[np.argmin(np.abs(non_nan - ni))]
                                outdata[ni] = outdata[nearest]

        if binning == 'adaptive':
                k_mid = np.sqrt(kbins_edges[:-1] * kbins_edges[1:])
        elif binning == 'adaptive_linear':
                k_mid = (kbins_edges[:-1] + kbins_edges[1:]) / 2.0
        else:
                k_mid = kbins_edges[:-1] + dk

        return outdata, k_mid, n_modes
        

def power_spectrum_1d(input_array_nd, kbins=100, box_dims=None, return_n_modes=False,
                      binning='log', window=None, k_limit=None, fill_empty_bins=True,
                      backend='numpy', **kwargs):
        ''' Calculate the spherically averaged power spectrum of an array
        and return it as a one-dimensional array.

        Parameters:
                input_array_nd (numpy array): the data array
                kbins = 100 (integer or array-like): number of bins or explicit bin edges.
                box_dims = None (float or array-like): box dimensions in Mpc.
                return_n_modes = False (bool): if True, also return n_modes array.
                binning = 'log' : binning scheme. Options:
                        'log'             — uniform log-spaced bins (default)
                        'linear'          — uniform linear-spaced bins
                        'adaptive'        — edges at geometric midpoints between discrete
                                            k-modes, centers at geometric means; no empty bins
                        'adaptive_linear' — same but arithmetic midpoints/means (linear-k)
                        'kde'             — Gaussian kernel in log-k space (bandwidth = 1 bin
                                            spacing); avoids empty bins, mild cross-bin mixing
                        'kde_linear'      — Gaussian kernel in linear-k space
                window = None : taper ('blackmanharris', 'tukey', or array).
                k_limit = None : upper k limit (None, 'nyquist', or 'aliasing').
                fill_empty_bins = True : fill zero-mode bins via 1D nearest-neighbour
                        after tophat/adaptive binning. No effect for kde variants.
                backend = 'numpy' : 'numpy' (default), 'jax' or 'torch'. With 'jax'/'torch'
                        the estimator is end-to-end differentiable (jax.grad / torch
                        autograd) and GPU-capable; binning must be 'log' or 'linear'.
                        The numpy path is unchanged.

        Returns:
                (Pk, bins) or (Pk, bins, n_modes) if return_n_modes is True.
        '''

        input_array_nd = apply_window(input_array_nd, window, backend=backend)

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array_nd.shape)

        kmax = _resolve_k_limit(k_limit, box_dims, input_array_nd.shape)
        input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims, backend=backend)

        ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims=box_dims,
                                           binning=binning, kmax=kmax,
                                           fill_empty_bins=fill_empty_bins, backend=backend)
        if return_n_modes:
                return ps, bins, n_modes
        return ps, bins


def _power_spectrum_2d_backend(power, kp, kz, kbins, binning, kmax_per, kmax_par,
                               fill_empty_bins, return_modes, name, xp):
        '''
        Differentiable cylindrical (kper, kpar) binning for jax/torch backend
        arrays. Static bin geometry in numpy, differentiable scatter-add for
        the power values. Supports binning='log' and 'linear'.
        '''
        if binning not in ('log', 'linear'):
                raise NotImplementedError(
                        "backend='%s' supports binning='log' or 'linear'; got %r" % (name, binning))
        if binning == 'log':
                with np.errstate(divide='ignore'):
                        kp_b, kz_b = np.log10(kp), np.log10(kz)
                edges_per = np.linspace(np.log10(kp[kp != 0].min()), np.log10(kmax_per), kbins[0] + 1)
                edges_par = np.linspace(np.log10(kz[kz != 0].min()), np.log10(kmax_par), kbins[1] + 1)
        else:
                kp_b, kz_b = kp, kz
                edges_per = np.linspace(kp[kp != 0].min(), kmax_per, kbins[0] + 1)
                edges_par = np.linspace(kz[kz != 0].min(), kmax_par, kbins[1] + 1)

        n_per, n_par = kbins[0], kbins[1]
        ip = np.digitize(kp_b.flatten(), edges_per) - 1
        iz = np.digitize(kz_b.flatten(), edges_par) - 1
        ip[kp_b.flatten() == edges_per[-1]] = n_per - 1
        iz[kz_b.flatten() == edges_par[-1]] = n_par - 1
        valid = np.flatnonzero((ip >= 0) & (ip < n_per) & (iz >= 0) & (iz < n_par))
        flat = ip[valid] * n_par + iz[valid]
        n_modes = np.bincount(flat, minlength=n_per * n_par).astype(float)

        pw   = _take(power.reshape(-1), valid, name, xp)
        sums = _scatter_add(pw, flat, n_per * n_par, name, xp)
        counts = _as_backend_array(np.where(n_modes > 0, n_modes, 1.0), name, xp)
        if name == 'torch': counts = counts.to(sums.dtype)
        ps_flat = sums / counts

        empty = n_modes == 0
        if empty.any() and not empty.all():
                if fill_empty_bins:
                        from scipy.ndimage import distance_transform_edt
                        nan_mask = empty.reshape(n_per, n_par)
                        _, (ri, ci) = distance_transform_edt(nan_mask, return_indices=True)
                        gather = (ri * n_par + ci).flatten()
                        ps_flat = _take(ps_flat, gather, name, xp)
                else:
                        nan_vec = _as_backend_array(np.where(empty, np.nan, 0.0), name, xp)
                        if name == 'torch': nan_vec = nan_vec.to(ps_flat.dtype)
                        ps_flat = ps_flat + nan_vec
        ps_out = ps_flat.reshape(n_per, n_par)

        if binning == 'log':
                kper_mid = np.power(10, 0.5 * (edges_per[:-1] + edges_per[1:]))
                kpar_mid = np.power(10, 0.5 * (edges_par[:-1] + edges_par[1:]))
        else:
                kper_mid = (edges_per[:-1] + edges_per[1:]) / 2.
                kpar_mid = (edges_par[:-1] + edges_par[1:]) / 2.

        if return_modes:
                return ps_out, kper_mid, kpar_mid, n_modes.reshape(n_per, n_par)
        return ps_out, kper_mid, kpar_mid


def power_spectrum_2d(input_array, kbins=10, binning='log', box_dims=244/.7, return_modes=False,
                      nu_axis=2, window=None, k_limit=None, fill_empty_bins=True,
                      backend='numpy', **kwargs):
        '''
        Calculate the power spectrum and bin it in kper and kpar.

        Parameters:
                input_array (numpy array): the data array
                nu_axis = 2 (integer): the line-of-sight axis
                kbins = 10 (integer or [n_kper, n_kpar]): number of bins per direction.
                box_dims = 244/.7 (float or array-like): box dimensions in Mpc.
                return_modes = False (bool): if True, also return n_modes array.
                binning = 'log' : binning scheme. Options:
                        'log'             — uniform log-spaced bins (default)
                        'linear'          — uniform linear-spaced bins
                        'adaptive'        — edges at geometric midpoints between discrete
                                            k-modes, centers at geometric means; no empty bins
                        'adaptive_linear' — same but arithmetic midpoints/means (linear-k)
                        'kde'             — Gaussian kernel in log-k space (bandwidth = 1 bin
                                            spacing); avoids empty bins, mild cross-bin mixing
                        'kde_linear'      — Gaussian kernel in linear-k space
                window = None : taper ('blackmanharris', 'tukey', or array).
                k_limit = None : upper k limit (None, 'nyquist', or 'aliasing').
                fill_empty_bins = True : after tophat/adaptive binning, fill any remaining
                        NaN (zero-mode) bins via 2D nearest-neighbour in index space.
                        Set False to keep NaN. Ignored for kde variants.
                backend = 'numpy' : 'numpy' (default), 'jax' or 'torch'. With 'jax'/'torch'
                        the estimator is differentiable and GPU-capable; binning must be
                        'log' or 'linear'. The numpy path is unchanged.

        Returns:
                (Pk, kper_mid, kpar_mid) or (Pk, kper_mid, kpar_mid, n_modes).
        '''
        input_array = apply_window(input_array, window, backend=backend)

        if type(kbins) == list:
                binning = None
        elif np.array(kbins).size == 1:
                kbins = [kbins, kbins]
        elif not isinstance(kbins[0], int):
                binning = None

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)
        power = power_spectrum_nd(input_array, box_dims, backend=backend)
        k_xyz, k = _get_k(input_array, box_dims)
        xy_axis = [0, 1, 2]
        xy_axis.remove(nu_axis)
        kz = np.abs(k_xyz[nu_axis])
        kp = np.sqrt(k_xyz[xy_axis[0]]**2 + k_xyz[xy_axis[1]]**2)
        del k_xyz, k
        gc.collect()

        # Per-direction Nyquist limits
        if k_limit is not None:
                shape = np.array(input_array.shape)
                bdims = np.array(box_dims, dtype=float)
                k_ny_all = np.pi * shape / bdims
                kl = k_limit.lower()
                if kl == 'nyquist':
                        factor = 1.0
                elif kl == 'aliasing':
                        factor = 0.5
                else:
                        raise ValueError(f"k_limit must be None, 'nyquist', or 'aliasing'; got {k_limit!r}")
                kmax_per = float(k_ny_all[xy_axis].min()) * factor
                kmax_par = float(k_ny_all[nu_axis]) * factor
                kmax_per = min(kmax_per, kp.max())
                kmax_par = min(kmax_par, kz.max())
        else:
                kmax_per = kp.max()
                kmax_par = kz.max()
        del xy_axis

        # ---- Differentiable backend path (jax/torch) ----
        if backend not in (None, 'numpy'):
                name, xp = _import_backend(backend)
                return _power_spectrum_2d_backend(power, kp, kz, kbins, binning,
                                                  kmax_per, kmax_par, fill_empty_bins,
                                                  return_modes, name, xp)

        # ---- KDE path (log-k or linear-k) ----
        if binning in ('kde', 'kde_linear'):
                use_log = (binning == 'kde')
                valid = (kp.flatten() > 0) & (kz.flatten() > 0)
                kpv = np.log10(kp.flatten()[valid]) if use_log else kp.flatten()[valid]
                kzv = np.log10(kz.flatten()[valid]) if use_log else kz.flatten()[valid]
                kper_max_s = np.log10(kmax_per) if use_log else kmax_per
                kpar_max_s = np.log10(kmax_par) if use_log else kmax_par
                kper_mid_s = np.linspace(kpv.min(), kper_max_s, kbins[0])
                kpar_mid_s = np.linspace(kzv.min(), kpar_max_s, kbins[1])
                bw_per = float(np.ptp(kper_mid_s)) / max(kbins[0] - 1, 1)
                bw_par = float(np.ptp(kpar_mid_s)) / max(kbins[1] - 1, 1)
                pw = power.flatten()[valid]
                ps_out = np.empty((kbins[0], kbins[1]))
                for i, kpc in enumerate(kper_mid_s):
                        w_per = np.exp(-(kpv - kpc)**2 / (2.0 * bw_per**2))
                        for j, kzc in enumerate(kpar_mid_s):
                                w = w_per * np.exp(-(kzv - kzc)**2 / (2.0 * bw_par**2))
                                wsum = w.sum()
                                ps_out[i, j] = (pw * w).sum() / wsum if wsum > 0 else np.nan
                kper_mid = np.power(10, kper_mid_s) if use_log else kper_mid_s
                kpar_mid = np.power(10, kpar_mid_s) if use_log else kpar_mid_s
                if return_modes:
                        edges_per = np.linspace(kper_mid_s[0] - bw_per / 2,
                                                kper_mid_s[-1] + bw_per / 2, kbins[0] + 1)
                        edges_par = np.linspace(kpar_mid_s[0] - bw_par / 2,
                                                kpar_mid_s[-1] + bw_par / 2, kbins[1] + 1)
                        nm = stats.binned_statistic_2d(kpv, kzv, None,
                                                       statistic='count',
                                                       bins=[edges_per, edges_par])
                        return ps_out, kper_mid, kpar_mid, nm.statistic
                return ps_out, kper_mid, kpar_mid

        # ---- Tophat / adaptive path ----
        if binning is None:
                kper = np.array(kbins[0])
                kpar = np.array(kbins[1])
        elif binning == 'log':
                kper = np.linspace(np.log10(kp[kp != 0].min()), np.log10(kmax_per), kbins[0] + 1)
                kpar = np.linspace(np.log10(kz[kz != 0].min()), np.log10(kmax_par), kbins[1] + 1)
                kp, kz = np.log10(kp), np.log10(kz)
        elif binning == 'linear':
                kper = np.linspace(kp[kp != 0].min(), kmax_per, kbins[0] + 1)
                kpar = np.linspace(kz[kz != 0].min(), kmax_par, kbins[1] + 1)
        elif binning in ('adaptive', 'adaptive_linear'):
                _log = (binning == 'adaptive')
                kper = _adaptive_bin_edges(np.unique(kp[(kp > 0) & (kp <= kmax_per)]), kbins[0], log=_log)
                kpar = _adaptive_bin_edges(np.unique(kz[(kz > 0) & (kz <= kmax_par)]), kbins[1], log=_log)

        kp_f, kz_f, pw_f = kp.flatten(), kz.flatten(), power.flatten()
        ps = stats.binned_statistic_2d(x=kp_f, y=kz_f, values=pw_f,
                                       statistic='mean', bins=[kper, kpar])

        if binning == 'log':
                kper_mid = np.power(10, 0.5 * (kper[:-1] + kper[1:]))
                kpar_mid = np.power(10, 0.5 * (kpar[:-1] + kpar[1:]))
        elif binning == 'adaptive':
                kper_mid = np.sqrt(kper[:-1] * kper[1:])
                kpar_mid = np.sqrt(kpar[:-1] * kpar[1:])
        elif binning == 'adaptive_linear':
                kper_mid = (kper[:-1] + kper[1:]) / 2.0
                kpar_mid = (kpar[:-1] + kpar[1:]) / 2.0
        else:
                kper_mid = (kper[:-1] + kper[1:]) / 2.
                kpar_mid = (kpar[:-1] + kpar[1:]) / 2.

        ps_out = ps.statistic
        if fill_empty_bins:
                nan_mask = np.isnan(ps_out)
                if nan_mask.any() and not nan_mask.all():
                        from scipy.ndimage import distance_transform_edt
                        _, (ri, ci) = distance_transform_edt(nan_mask, return_indices=True)
                        ps_out = ps_out.copy()
                        ps_out[nan_mask] = ps_out[ri[nan_mask], ci[nan_mask]]

        if return_modes:
                n_modes = stats.binned_statistic_2d(x=kp_f, y=kz_f, values=None,
                                                    statistic='count', bins=[kper, kpar])
                return ps_out, kper_mid, kpar_mid, n_modes.statistic
        return ps_out, kper_mid, kpar_mid


def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, kbins=100, box_dims=None,
                            return_n_modes=False, binning='log', k_limit=None,
                            fill_empty_bins=True, backend='numpy', **kwargs):
        ''' Calculate the spherically averaged cross power spectrum of two arrays
        and return it as a one-dimensional array.

        Parameters:
                input_array1_nd (numpy array): the first data array
                input_array2_nd (numpy array): the second data array
                kbins = 100 (integer or array-like): number of bins or explicit bin edges.
                box_dims = None (float or array-like): box dimensions in Mpc.
                return_n_modes = False (bool): if True, also return n_modes array.
                binning = 'log' : binning scheme ('log', 'linear', 'adaptive', 'kde').
                k_limit = None : upper k limit (None, 'nyquist', or 'aliasing').
                fill_empty_bins = True : fill zero-mode bins via 1D nearest-neighbour.
                backend = 'numpy' : 'numpy' (default), 'jax' or 'torch' (differentiable,
                        GPU-capable; binning must be 'log' or 'linear').

        Returns:
                (Pk, bins) or (Pk, bins, n_modes) if return_n_modes is True.
        '''
        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1_nd.shape)

        kmax = _resolve_k_limit(k_limit, box_dims, input_array1_nd.shape)
        input_array = cross_power_spectrum_nd(input_array1_nd, input_array2_nd,
                                              box_dims=box_dims, backend=backend)

        ps, bins, n_modes = radial_average(input_array, kbins=kbins, box_dims=box_dims,
                                           binning=binning, kmax=kmax,
                                           fill_empty_bins=fill_empty_bins, backend=backend)
        if return_n_modes:
                return ps, bins, n_modes
        return ps, bins


def power_spectrum_mu(input_array, los_axis=0, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, k_limit=None, **kwargs):
        '''
        Calculate the power spectrum and bin it in mu=cos(theta) and k.

        Parameters:
                input_array (numpy array): the data array
                los_axis = 0 (integer): the line-of-sight axis
                mubins = 20 (integer): the number of mu bins
                kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                return_n_modes = False (bool): if true, also return the
                        number of modes in each bin
                exlude_zero_modes = True (bool): if true, modes with any components
                        of k equal to zero will be excluded.
                absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1]
                k_limit = None : Upper k limit for the bins. Options:
                                    None       — use the grid maximum (default, no restriction)
                                    'nyquist'  — cap at k_Nyquist = π·min(N_i/L_i)
                                    'aliasing' — cap at k_Nyquist/2 (alias-free limit)

        Returns:
                A tuple with (Pk, mubins, kbins), where Pk is an array with the
                power spectrum of dimensions (n_mubins x n_kbins),
                mubins is an array with the mu bin centers and
                kbins is an array with the k bin centers.

        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array.shape)

        kmax = _resolve_k_limit(k_limit, box_dims, input_array.shape)
        powerspectrum = power_spectrum_nd(input_array, box_dims=box_dims)

        ps, mu_bins, k_bins, n_modes = mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes, absolute_mus=absolute_mus, kmax=kmax)

        if return_n_modes:
                return ps, mu_bins, k_bins, n_modes
        return ps, mu_bins, k_bins


def cross_power_spectrum_mu(input_array1, input_array2, los_axis=0, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, k_limit=None, **kwargs):
        '''
        Calculate the cross power spectrum and bin it in mu=cos(theta) and k.
        
        Parameters: 
                input_array1 (numpy array): the first data array
                input_array2 (numpy array): the second data array
                los_axis = 0 (integer): the line-of-sight axis
                mubins = 20 (integer): the number of mu bins
                kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.
                box_dims = None (float or array-like): the dimensions of the 
                        box in Mpc. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                return_n_modes = False (bool): if true, also return the
                        number of modes in each bin
                exlude_zero_modes = True (bool): if true, modes with any components
                        of k equal to zero will be excluded.
                absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1] 
                
        Returns: 
                A tuple with (Pk, mubins, kbins), where Pk is an array with the 
                cross power spectrum of dimensions (n_mubins x n_kbins), 
                mubins is an array with the mu bin centers and
                kbins is an array with the k bin centers.
                
        TODO:
                Add support for (non-numpy) lists for the bins
        '''

        if kwargs.get('boxsize') is not None: box_dims = kwargs.get('boxsize')
        box_dims = _get_dims(box_dims, input_array1.shape)

        kmax = _resolve_k_limit(k_limit, box_dims, input_array1.shape)
        powerspectrum = cross_power_spectrum_nd(input_array1, input_array2, box_dims=box_dims)

        ps, mu_bins, k_bins, n_modes = mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes, absolute_mus=absolute_mus, kmax=kmax)
        if return_n_modes:
                return ps, mu_bins, k_bins, n_modes
        return ps, mu_bins, k_bins


def mu_binning(powerspectrum, los_axis = 0, mubins=20, kbins=10, box_dims=None, weights=None,
                        exclude_zero_modes=True, binning='log', absolute_mus=True, kmax=None):
        '''
        This function is for internal use only.
        '''

        if weights != None:
                powerspectrum *= weights

        assert(len(powerspectrum.shape)==3)

        k_comp, k = _get_k(powerspectrum, box_dims)

        mu = _get_mu(k_comp, k, los_axis, absolute_mus)

        #Calculate k values, and make k bins
        kbins = _get_kbins(kbins, box_dims, k, binning=binning, kmax=kmax)
        dk = (kbins[1:]-kbins[:-1])/2.
        n_kbins = len(kbins)-1
                
        #Exclude k_perp = 0 modes
        if exclude_zero_modes:
                good_idx = _get_nonzero_idx(powerspectrum.shape, los_axis)
        else:
                good_idx = np.ones_like(powerspectrum)

        #Make mu bins
        min_mu=0.0 if absolute_mus else -1.0
        if isinstance(mubins,int):
                mubins = np.linspace(min_mu, 1.0 , mubins+1)
        dmu = (mubins[1:]-mubins[:-1])/2.
        n_mubins = len(mubins)-1

        #Remove the zero component from the power spectrum. mu is undefined here
        powerspectrum[tuple((np.array(powerspectrum.shape)/2).astype(int))] = 0.

        #Bin the data
        print_msg('Binning data...')
        outdata = np.zeros((n_mubins,n_kbins))
        n_modes = np.zeros((n_mubins,n_kbins))
        for ki in range(n_kbins):
                print_msg('Bin %d of %d' % (ki, n_kbins))
                kmin_bin = kbins[ki]
                kmax_bin = kbins[ki+1]
                kidx = (k >= kmin_bin) & (k < kmax_bin)
                kidx = kidx*good_idx
                for i in range(n_mubins):
                        mu_min = mubins[i]
                        mu_max = mubins[i+1]
                        idx = (mu >= mu_min) & (mu < mu_max) & kidx.astype(bool)
                        outdata[i,ki] = np.mean(powerspectrum[idx])
                        n_modes[i,ki] = np.size(powerspectrum[idx])
                        if weights != None:
                                outdata[i,ki] /= weights[idx].mean()

        return outdata, mubins[:-1]+dmu, kbins[:-1]+dk, n_modes

def get_k(input_array, box_dims):
        box_dims = _get_dims(box_dims, input_array.shape)
        return _get_k(input_array, box_dims)
        
def get_kbins(kbins, box_dims, k=None, array=None, binning='log'):
        assert k is not None or array is not None
        box_dims = _get_dims(box_dims, array.shape)
        if k is None: k_comp, k = _get_k(array, box_dims)
        return _get_kbins(kbins, box_dims, k, binning=binning)

#Some methods for internal use

def _get_k(input_array, box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
    dim = len(input_array.shape)
    if dim == 1:
        x = np.arange(len(input_array))
        center = x.max()/2.
        kx = 2.*np.pi*(x-center)/box_dims[0]
        return [kx], kx
    elif dim == 2:
        x,y = np.indices(input_array.shape, dtype='int32')
        center = np.array([(x.max()-x.min())/2, (y.max()-y.min())/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        k = np.sqrt(kx**2 + ky**2)
        return [kx, ky], k
    elif dim == 3:
        nx,ny,nz = input_array.shape
        x,y,z  = np.indices(input_array.shape, dtype='int32')
        center = np.array([nx/2 if nx%2==0 else (nx-1)/2, ny/2 if ny%2==0 else (ny-1)/2, \
                            nz/2 if nz%2==0 else (nz-1)/2])
        kx = 2.*np.pi * (x-center[0])/box_dims[0]
        ky = 2.*np.pi * (y-center[1])/box_dims[1]
        kz = 2.*np.pi * (z-center[2])/box_dims[2]

        k = np.sqrt(kx**2 + ky**2 + kz**2 )     
        return [kx,ky,kz], k


def _get_mu(k_comp, k, los_axis, absolute_mus):
        '''
        Get the mu values for given k values and 
        a line-of-sight axis.
        For internal use
        '''
                
        #Line-of-sight distance from center 
        if los_axis == 0:
                los_dist = k_comp[0]
        elif los_axis == 1:
                los_dist = k_comp[1]
        elif los_axis == 2:
                los_dist = k_comp[2]
        else:
                raise Exception('Your space is not %d-dimensional!' % los_axis)

        #mu=cos(theta) = k_par/k
        mu = np.abs(los_dist/k) if absolute_mus else los_dist/np.abs(k)
        mu[np.where(k < 0.001)] = np.nan
        
        return mu


def _get_kbins(kbins, box_dims, k, binning='log', kmin=None, kmax=None):
        '''
        Make a list of bin edges if kbins is an integer,
        otherwise return it as it is.
        '''
        if isinstance(kbins,int):
                kmin = 2.*np.pi/min(box_dims) if kmin is None else kmin
                kmax = k.max() if kmax is None else kmax
                if binning=='linear': kbins = np.linspace(kmin, kmax, kbins+1)
                else: kbins = 10**np.linspace(np.log10(kmin), np.log10(kmax), kbins+1)
        return kbins


def _get_dims(box_dims, ashape):
        '''
        If box dims is a scalar, assume that dimensions
        are cubic and make a list
        If it's not given, assume it's the default value of the box
        size
        Otherwise, return as it is
        '''
        if box_dims == None:
                return [conv.LB]*len(ashape)
        if not hasattr(box_dims, '__iter__'):
                return [box_dims]*len(ashape)
        return box_dims


def _resolve_k_limit(k_limit, box_dims, array_shape):
        '''
        Convert a k_limit string to a kmax value.

        Parameters:
            k_limit : None, 'nyquist', or 'aliasing'
                None         → return None (use the grid maximum)
                'nyquist'    → π * min(N_i / L_i)  (Nyquist wavenumber)
                'aliasing'   → k_nyquist / 2  (alias-free limit)
            box_dims  : list of box lengths [Mpc] per axis
            array_shape : shape of the data array

        Returns:
            float or None
        '''
        if k_limit is None:
                return None
        shape = np.array(array_shape[:len(box_dims)])
        k_ny_axes = np.pi * shape / np.array(box_dims, dtype=float)
        k_nyquist = k_ny_axes.min()
        kl = k_limit.lower()
        if kl == 'nyquist':
                return float(k_nyquist)
        elif kl == 'aliasing':
                return float(k_nyquist) / 2.0
        raise ValueError(f"k_limit must be None, 'nyquist', or 'aliasing'; got {k_limit!r}")

def dimensionless_ps(data, kbins=100, box_dims=None, binning='log', factor=10, k_limit=None):
        r'''
        Dimensionless power spectrum is P(k)*k^3/(2pi^2)

        Parameters
        ----------
        data    : ndarray
                The numpy data whose power spectrum is to be determined.
        kbins   : int
                Number of bins for in the k-space (Default: 100).
        box_dims: float
                The size of the box in Mpc (Default: Takes the value from the set_sim_constants).
        binning : str
                The type of binning to be used for the k-space (Default: 'log').
        factor  : int
                The factor multiplied to the given kbins to smooth the spectrum from (Default: 10).
        k_limit : None, 'nyquist', or 'aliasing'
                Upper k limit passed to power_spectrum_1d (Default: None).

        Returns
        -------
        (\Delta^2, ks)
        '''
        from scipy.interpolate import interp1d
        Pk, ks = power_spectrum_1d(data, kbins=kbins*factor, box_dims=box_dims, binning=binning, k_limit=k_limit)
        f_Dlta = interp1d(ks, Pk*ks**3/2/np.pi**2)
        knew   = 10**np.linspace(np.log10(ks[1]),np.log10(ks[-1]), kbins) if binning=='log' else np.linspace(ks[1],ks[-1], kbins)
        return f_Dlta(knew), knew

def _get_nonzero_idx(ps_shape, los_axis):
        '''
        Get the indices where k_perp != 0
        '''
        x,y,z = np.indices(ps_shape)
        if los_axis == 0:
                zero_idx = (y == ps_shape[1]/2)*(z == ps_shape[2]/2)
        elif los_axis == 1:
                zero_idx = (x == ps_shape[0]/2)*(z == ps_shape[2]/2)
        else:
                zero_idx = (x == ps_shape[0]/2)*(y == ps_shape[1]/2)
        good_idx = np.invert(zero_idx)
        return good_idx


def anisotropy_ratio_r_mu(input_array, los_axis=0, mu_cut=0.5, mubins=20, kbins=10, box_dims=None, weights=None, exclude_zero_modes=True, return_n_modes=False, absolute_mus=True, k_limit=None, **kwargs):
    '''
    Calculate the anisotropy ratio r_mu(k).

    Parameters:
            input_array (numpy array): the data array
            los_axis = 0 (integer): the line-of-sight axis
            mu_cut = 0.5 (float): the value of mu to split the P(k,mu) data.
            mubins = 20 (integer): the number of mu bins
            kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            box_dims = None (float or array-like): the dimensions of the
                    box in Mpc. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            return_n_modes = False (bool): if true, also return the
                    number of modes in each bin
            exlude_zero_modes = True (bool): if true, modes with any components
                    of k equal to zero will be excluded.
            absolute_mus = True (boolean): if true, use the absolute values of mu, range [0,1]. If false, use the range [-1,1]
            k_limit = None : Upper k limit for the bins. Options:
                    None       — use the grid maximum (default, no restriction)
                    'nyquist'  — cap at k_Nyquist = π·min(N_i/L_i)
                    'aliasing' — cap at k_Nyquist/2 (alias-free limit)

    Returns:
            A tuple with (rmu, kbins).

    '''
    Pk, mubins, kbins = power_spectrum_mu(input_array, los_axis=los_axis, mubins=mubins, kbins=kbins, box_dims=box_dims, weights=weights, exclude_zero_modes=exclude_zero_modes, return_n_modes=return_n_modes, absolute_mus=absolute_mus, k_limit=k_limit, **kwargs)
    Pup = Pk[mubins>=mu_cut,:]
    Pdn = Pk[mubins<mu_cut,:]
    rup = np.array([pp[np.isfinite(pp)].mean() for pp in Pup.T])
    rdn = np.array([pp[np.isfinite(pp)].mean() for pp in Pdn.T])
    rmu = rup/rdn-1.0
    return rmu, kbins


def horizon_wedge_equation(z, fov_deg=90.0):
    '''
    Horizon wedge boundary in the cylindrical power spectrum.
    Equation 14 in arXiv:2201.10798.

    Parameters:
        z (float): redshift
        fov_deg (float): field of view in degrees (default 90)

    Returns:
        A lambda function k_parallel(k_perpendicular).
    '''
    f_kpar = lambda kper: kper * np.sin(fov_deg * np.pi / 180) / (1 + z) * \
                          cosmo.hubble_parameter(z) / const.c * cosmo.z_to_cdist(z)
    return f_kpar


def plot_2d_power(ps, xlabel=r'$k_\perp$', ylabel=r'$k_\parallel$',
                  ps_label=r'$P(k_\perp,k_\parallel)$',
                  fig=None, ax=None,
                  plotting_scale=None,
                  draw_wedge=None, **kwargs):
    '''
    Plot the 2D cylindrical power spectrum.

    Parameters:
        ps (tuple or dict): (Pk, kper_bins, kpar_bins) or
                            {'Pk': Pk, 'kper': kper_bins, 'kpar': kpar_bins}
        xlabel, ylabel, ps_label: axis and colorbar labels
        fig, ax: matplotlib figure and axes (created if None)
        plotting_scale (dict): axis scales, e.g. {'x': 'log', 'y': 'log', 'z': 'log'}
        draw_wedge (dict or None): wedge params e.g. {'z': 9.0, 'fov_deg': 90.0, 'ls': '--', 'color': 'k'}

    Returns:
        matplotlib figure
    '''
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    if plotting_scale is None:
        plotting_scale = {'x': 'log', 'y': 'log', 'z': 'log'}
    if draw_wedge is None:
        draw_wedge = {'z': 9.0, 'fov_deg': 90.0, 'ls': '--', 'color': 'k'}

    try:
        pp, kper, kpar = (ps[ke] for ke in ['Pk', 'kper', 'kpar'])
    except TypeError:
        pp, kper, kpar = ps

    fp = interpolate.RegularGridInterpolator(
        (kper, kpar), pp, method='linear', bounds_error=False, fill_value=None
    )

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    elif fig is None:
        pass
    else:
        ax = fig.axes[0]

    X, Y = kper, kpar
    Xg, Yg = np.meshgrid(X, Y, indexing='ij')
    C = fp(np.stack([Xg, Yg], axis=-1)).T
    norm = colors.LogNorm(vmin=C[np.isfinite(C)].min(), vmax=C[np.isfinite(C)].max()) \
           if plotting_scale.get('z') == 'log' else None
    pcm = ax.pcolormesh(X, Y, C, norm=norm, **kwargs)

    if draw_wedge is not None:
        f_kpar = horizon_wedge_equation(draw_wedge['z'], fov_deg=draw_wedge.get('fov_deg', 90.0))
        ax.plot(X, f_kpar(X), ls=draw_wedge.get('ls', '--'), color=draw_wedge.get('color', 'k'))
        ax.axis([X.min(), X.max(), Y.min(), Y.max()])

    plt.colorbar(pcm, ax=ax, label=ps_label, pad=0.01)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(plotting_scale.get('x', 'log'))
    ax.set_yscale(plotting_scale.get('y', 'log'))
    return fig