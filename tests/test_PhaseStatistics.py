import numpy as np
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
                                                 binning='linear', return_n_modes=True)
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
                                                 binning='linear', return_n_modes=True)
        p3, _ = t2c.squeezed_bispectrum_normalized(field, kbins=kbins, box_dims=box_dims, binning='linear')
        R.append(rr); P3.append(p3)
    R, pred = np.mean(R, axis=0), t2c.phase_gradient_spectrum_from_bispectrum(np.mean(P3, axis=0), nm)
    assert np.all(R > t2c.phase_gradient_noise_floor(nm))
    assert np.all(np.abs(R / pred - 1) < 0.5)


def test_phase_gradient_spectrum_axes():
    '''A single axis and the average over all axes give values of the same size.'''
    field = _lognormal_field(0)
    rx, _ = t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', axis='x')
    rxyz, _ = t2c.phase_gradient_spectrum(field, kbins=kbins, box_dims=box_dims, binning='linear', axis='xyz')
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
