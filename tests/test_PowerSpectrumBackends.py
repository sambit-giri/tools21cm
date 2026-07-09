'''
Tests for the differentiable jax/torch backends of the power spectrum
estimators (issue #81). The numpy path is the reference; jax/torch must
reproduce it and be differentiable via jax.grad / torch autograd.
'''
import numpy as np
import pytest

import tools21cm as t2c

jax = pytest.importorskip('jax', reason='jax not installed')
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

box_dims = 200
dims = [32, 32, 32]
rng = np.random.RandomState(42)
field = rng.normal(loc=0., scale=1., size=dims)
field2 = rng.normal(loc=0., scale=1., size=dims)
kbins = 8


def test_power_spectrum_1d_jax_matches_numpy():
    pk_np, k_np = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims)
    pk_jx, k_jx = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims,
                                        backend='jax')
    assert np.allclose(k_np, k_jx)
    assert np.allclose(pk_np, np.asarray(pk_jx), rtol=1e-8)


def test_power_spectrum_1d_linear_binning_jax():
    pk_np, k_np = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims,
                                        binning='linear')
    pk_jx, k_jx = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims,
                                        binning='linear', backend='jax')
    assert np.allclose(k_np, k_jx)
    assert np.allclose(pk_np, np.asarray(pk_jx), rtol=1e-8)


@pytest.mark.skipif(not TORCH, reason='torch not installed')
def test_power_spectrum_1d_torch_matches_numpy():
    pk_np, k_np = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims)
    pk_th, k_th = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims,
                                        backend='torch')
    assert np.allclose(k_np, k_th)
    assert np.allclose(pk_np, pk_th.detach().cpu().numpy(), rtol=1e-8)


def test_cross_power_spectrum_1d_backends():
    pk_np, k_np = t2c.cross_power_spectrum_1d(field, field2, kbins=kbins,
                                              box_dims=box_dims)
    pk_jx, k_jx = t2c.cross_power_spectrum_1d(field, field2, kbins=kbins,
                                              box_dims=box_dims, backend='jax')
    assert np.allclose(k_np, k_jx)
    assert np.allclose(pk_np, np.asarray(pk_jx), rtol=1e-8)
    if TORCH:
        pk_th, _ = t2c.cross_power_spectrum_1d(field, field2, kbins=kbins,
                                               box_dims=box_dims, backend='torch')
        assert np.allclose(pk_np, pk_th.detach().cpu().numpy(), rtol=1e-8)


def test_power_spectrum_2d_backends():
    pk_np, kper_np, kpar_np = t2c.power_spectrum_2d(field, kbins=kbins,
                                                    box_dims=box_dims)
    pk_jx, kper_jx, kpar_jx = t2c.power_spectrum_2d(field, kbins=kbins,
                                                    box_dims=box_dims,
                                                    backend='jax')
    assert np.allclose(kper_np, kper_jx)
    assert np.allclose(kpar_np, kpar_jx)
    assert np.allclose(pk_np, np.asarray(pk_jx), rtol=1e-8, equal_nan=True)
    if TORCH:
        pk_th, _, _ = t2c.power_spectrum_2d(field, kbins=kbins,
                                            box_dims=box_dims, backend='torch')
        assert np.allclose(pk_np, pk_th.detach().cpu().numpy(), rtol=1e-8,
                           equal_nan=True)


def test_power_spectrum_1d_grad_jax():
    def loss(delta):
        pk, _ = t2c.power_spectrum_1d(delta, kbins=kbins, box_dims=box_dims,
                                      backend='jax')
        return pk.sum()

    g = jax.grad(loss)(jnp.asarray(field))
    g = np.asarray(g)
    assert g.shape == tuple(dims)
    assert np.all(np.isfinite(g))
    assert np.abs(g).max() > 0

    # finite-difference check along one direction
    eps = 1e-4
    v = rng.normal(size=dims)
    fd = (loss(jnp.asarray(field + eps * v)) - loss(jnp.asarray(field - eps * v))) / (2 * eps)
    assert np.isclose(float(fd), float(np.sum(g * v)), rtol=1e-4)


def test_power_spectrum_1d_jit_jax():
    f = jax.jit(lambda d: t2c.power_spectrum_1d(d, kbins=kbins,
                                                box_dims=box_dims,
                                                backend='jax')[0])
    pk_jit = np.asarray(f(jnp.asarray(field)))
    pk_np, _ = t2c.power_spectrum_1d(field, kbins=kbins, box_dims=box_dims)
    assert np.allclose(pk_np, pk_jit, rtol=1e-8)


@pytest.mark.skipif(not TORCH, reason='torch not installed')
def test_power_spectrum_1d_grad_torch():
    delta = torch.as_tensor(field, dtype=torch.float64).requires_grad_(True)
    pk, _ = t2c.power_spectrum_1d(delta, kbins=kbins, box_dims=box_dims,
                                  backend='torch')
    pk.sum().backward()
    g = delta.grad.detach().cpu().numpy()
    assert g.shape == tuple(dims)
    assert np.all(np.isfinite(g))
    assert np.abs(g).max() > 0

    # cross-check torch gradient against the jax gradient
    g_jax = np.asarray(jax.grad(
        lambda d: t2c.power_spectrum_1d(d, kbins=kbins, box_dims=box_dims,
                                        backend='jax')[0].sum()
    )(jnp.asarray(field)))
    assert np.allclose(g, g_jax, rtol=1e-6, atol=1e-12)
