import numpy as np
import pytest
import tools21cm as t2c

box_dims = 300.
n_cells = 64
kbins = np.linspace(2 * np.pi / box_dims, np.pi * n_cells / box_dims, 6)


def _gaussian_field(seed):
    return np.random.default_rng(seed).normal(size=(n_cells, n_cells, n_cells))


def _lognormal_field(seed, gamma=1.0):
    '''Log-normal transform of a smoothed Gaussian random field.'''
    from scipy.ndimage import gaussian_filter
    g = gaussian_filter(_gaussian_field(seed), sigma=2, mode='wrap')
    g /= g.std()
    return np.exp(gamma * g)


def test_radial_average_keeps_complex():
    '''
    A complex field must give a complex shell mean. Casting to float would keep
    only the real part.
    '''
    field = np.full((32, 32, 32), np.exp(1j * np.pi / 4))
    avg, kk, nm = t2c.radial_average(field, [100.] * 3, kbins=5, binning='linear')
    assert np.iscomplexobj(avg)
    assert np.allclose(avg, np.exp(1j * np.pi / 4))


def test_phase_gradient_spectrum_gaussian_noise_floor():
    '''
    For Gaussian random fields R(k) follows the Rayleigh noise floor sqrt(pi / (4 N_indep)).
    '''
    R = []
    for seed in range(10):
        rr, kk, nm = t2c.phase_gradient_spectrum(_gaussian_field(seed), kbins=kbins, box_dims=box_dims,
                                                 binning='linear', return_n_modes=True, dk_squared_max=1)
        R.append(rr)
    ratio = np.mean(R, axis=0) / t2c.phase_gradient_noise_floor(nm)
    assert np.all(np.abs(ratio - 1) < 0.15)


def test_phase_gradient_spectrum_lognormal_matches_bispectrum():
    '''
    For a non-Gaussian field R(k) rises above the noise floor and agrees with
    the prediction from the normalised squeezed bispectrum.
    '''
    R, P3 = [], []
    for seed in range(10):
        field = _lognormal_field(seed)
        rr, kk, nm = t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims,
                                                 binning='linear', return_n_modes=True, dk_squared_max=1)
        p3, _ = t2c.squeezed_bispectrum_normalized(field, kbins=kbins, box_dims=box_dims, binning='linear',
                                                   dk_squared_max=1)
        R.append(rr); P3.append(p3)
    R, pred = np.mean(R, axis=0), t2c.phase_gradient_spectrum_from_bispectrum(np.mean(P3, axis=0), nm)
    assert np.all(R > t2c.phase_gradient_noise_floor(nm))
    assert np.all(np.abs(R / pred - 1) < 0.5)


def test_phase_gradient_spectrum_axes():
    '''A single axis and the average over all axes give values of the same size.'''
    field = _lognormal_field(0)
    rx, _ = t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', axis='x',
                                        dk_squared_max=None)
    rxyz, _ = t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', axis='xyz',
                                          dk_squared_max=None)
    assert rx.shape == rxyz.shape
    assert np.all(rxyz > 0)


def test_phase_gradients_per_bin_and_von_mises():
    '''Per-bin phase gradients are centred on zero and the von Mises pdf integrates to one.'''
    grads, edges = t2c.phase_gradients_per_bin(_lognormal_field(0), kbins=kbins, box_dims=box_dims,
                                               binning='linear', axis='x')
    assert len(grads) == len(edges) - 1
    for g in grads:
        assert np.all((g >= -np.pi) & (g <= np.pi))
        assert abs(np.angle(np.mean(np.exp(1j * g)))) < 1e-6
    x = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    kappa = t2c.resultant_length_to_kappa(0.3)
    assert abs(np.sum(t2c.von_mises_pdf(x, kappa=kappa)) * (x[1] - x[0]) - 1) < 1e-6


@pytest.mark.filterwarnings('ignore:The longest long mode')
def test_long_mode_methods_agree():
    '''
    The cross-correlation ('fft') and mode-by-mode ('gather'/'roll') bin means give the same
    R and p3 for many long modes, with the periodic wrap of np.roll.
    '''
    field = _lognormal_field(5)
    vecs = t2c.long_mode_vectors(9)
    kw = dict(kbins=kbins, box_dims=box_dims, binning='linear', dk_vectors=vecs)
    R_fft, _ = t2c.phase_gradient_spectrum(field, method='fft', **kw)
    R_gather, _ = t2c.phase_gradient_spectrum(field, method='gather', **kw)
    assert np.allclose(R_fft, R_gather, rtol=1e-10, atol=0)
    p3_fft, _ = t2c.squeezed_bispectrum_normalized(field, method='fft', average=False, squared=True, **kw)
    p3_roll, _ = t2c.squeezed_bispectrum_normalized(field, method='roll', average=False, squared=True, **kw)
    assert np.allclose(p3_fft, p3_roll, rtol=1e-8, atol=1e-14)


@pytest.mark.filterwarnings('ignore:The longest long mode')
def test_phase_gradient_spectrum_zero_field_is_nan():
    '''A field that vanishes everywhere has no phases.'''
    R, _ = t2c.phase_gradient_spectrum(np.zeros((32, 32, 32)), kbins=4, box_dims=100., binning='linear')
    assert np.all(np.isnan(R))


def test_phase_statistics_jax_backend():
    '''The jax backend reproduces the numpy result (float64) and can be differentiated.'''
    import pytest
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    field = _lognormal_field(6)
    kw = dict(kbins=kbins, box_dims=box_dims, binning='linear', dk_vectors=t2c.long_mode_vectors(2))
    R_np, _ = t2c.phase_gradient_spectrum(field, **kw)
    R_jx, _ = t2c.phase_gradient_spectrum(jnp.asarray(field), backend='jax', **kw)
    assert np.allclose(np.asarray(R_jx), R_np, rtol=1e-10)
    p3_np, _ = t2c.squeezed_bispectrum_normalized(field, **kw)
    p3_jx, _ = t2c.squeezed_bispectrum_normalized(jnp.asarray(field), backend='jax', **kw)
    assert np.allclose(np.asarray(p3_jx), p3_np, rtol=1e-8)
    grad = jax.grad(lambda x: jnp.sum(t2c.phase_gradient_spectrum(x, backend='jax', **kw)[0]))(jnp.asarray(field))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_phase_statistics_torch_backend():
    '''The torch backend reproduces the numpy result (float64) and supports autograd.'''
    import pytest
    torch = pytest.importorskip('torch')
    field = _lognormal_field(7)
    kw = dict(kbins=kbins, box_dims=box_dims, binning='linear', dk_vectors=t2c.long_mode_vectors(2))
    R_np, _ = t2c.phase_gradient_spectrum(field, **kw)
    x = torch.tensor(field, dtype=torch.float64, requires_grad=True)
    R_t, _ = t2c.phase_gradient_spectrum(x, backend='torch', **kw)
    assert np.allclose(R_t.detach().numpy(), R_np, rtol=1e-10)
    R_t.sum().backward()
    assert torch.all(torch.isfinite(x.grad))
    p3_np, _ = t2c.squeezed_bispectrum_normalized(field, **kw)
    p3_t, _ = t2c.squeezed_bispectrum_normalized(torch.tensor(field), backend='torch', **kw)
    assert np.allclose(p3_t.numpy(), p3_np, rtol=1e-8)


def test_warning_when_long_modes_are_not_squeezed():
    '''A warning is issued only for k-bins where |dk|_max > SQUEEZED_WARN_RATIO x k.'''
    import warnings
    field = _gaussian_field(8)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', dk_squared_max=1)
    with pytest.warns(UserWarning, match='not squeezed'):
        t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', dk_squared_max=25)
    with pytest.warns(UserWarning, match='not squeezed'):
        t2c.squeezed_bispectrum_normalized(field, kbins=kbins, box_dims=box_dims, binning='linear',
                                           dk_squared_max=25)
